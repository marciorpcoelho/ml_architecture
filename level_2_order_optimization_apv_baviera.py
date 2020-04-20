import time
import sys
import logging
from traceback import format_exc
import level_2_order_optimization_apv_baviera_options as options_file
from modules.level_1_a_data_acquisition import dw_data_retrieval, autoline_data_retrieval, read_csv
from modules.level_1_b_data_processing import apv_dataset_treatment
from modules.level_1_c_data_modelling import apv_stock_evolution_calculation, part_ref_selection, part_ref_ta_definition, apv_last_stock_calculation, apv_photo_stock_treatment, solver_dataset_preparation
from modules.level_1_e_deployment import time_tags, sql_inject
from modules.level_0_performance_report import log_record, project_dict, error_upload, performance_info, performance_info_append

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S @ %d/%m/%y', filename=options_file.log_files['full_log'], filemode='a')
logging.Logger('errors')
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # Allows the stdout to be seen in the console
logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))  # Allows the stderr to be seen in the console


def main():
    log_record('Projeto: {}'.format(project_dict[options_file.project_id]), options_file.project_id)
    current_date, _ = time_tags(format_date='%Y%m%d')

    last_processed_date, preprocessed_data_exists_flag = apv_last_stock_calculation(options_file.min_date, current_date, options_file.pse_code, options_file.project_id)
    print('Processing data from {} to {}'.format(last_processed_date, current_date))

    df_sales, df_product_group_dw, df_history = data_acquisition(options_file, last_processed_date, current_date)
    df_sales_cleaned = data_processing(df_sales, options_file)
    df_solver, df_part_ref_ta = data_modelling(options_file.pse_code, df_sales_cleaned, df_history, last_processed_date, current_date, preprocessed_data_exists_flag, options_file.project_id)
    deployment(df_solver, df_part_ref_ta)

    performance_info(options_file.project_id, options_file, model_choice_message='N/A', unit_count=df_solver.shape[0])

    log_record('Conclusão com sucesso - Projeto: {} .\n'.format(project_dict[options_file.project_id]), options_file.project_id)


def data_acquisition(options_info, last_processed_date, current_date):
    performance_info_append(time.time(), 'Section_A_Start')
    log_record('Início Secção A...', options_file.project_id)

    pse_code = options_info.pse_code
    start = time.time()

    df_sales, df_history, df_product_group_dw = dw_data_retrieval(pse_code, current_date, options_info, last_processed_date)

    print('Elapsed time: {:.2f} seconds.'.format(time.time() - start))

    log_record('Fim Secção A.', options_file.project_id)
    # return df_sales, df_purchases, df_stock, df_reg, df_reg_al_clients, df_history
    return df_sales, df_product_group_dw, df_history


def data_processing(df_sales, options_info):
    performance_info_append(time.time(), 'Section_B_Start')
    log_record('Início Secção B...', options_file.project_id)
    start_treatment = time.time()

    df_sales = apv_dataset_treatment(df_sales, options_info.pse_code, options_info.urgent_purchases_flags, options_info.project_id)

    print('Elapsed time: {:.2f} seconds.'.format(time.time() - start_treatment))

    log_record('Fim Secção B.', options_file.project_id)
    return df_sales


def data_modelling(pse_code, df_sales, df_history, min_date, max_date, preprocessed_data_exists_flag, project_id):
    performance_info_append(time.time(), 'Section_C_Start')
    log_record('Início Secção C', options_file.project_id)
    start = time.time()

    if pse_code == '0I':
        selected_parts = ['BM83.21.2.405.675', 'BM07.12.9.952.104', 'BM07.14.9.213.164', 'BM83.19.2.158.851', 'BM64.11.9.237.555']  # PSE_Code = 0I, Lisboa - Expo
    if pse_code == '0B':
        selected_parts = ['BM83.21.0.406.573', 'BM83.13.9.415.965', 'BM51.18.1.813.017', 'BM11.42.8.507.683', 'BM64.11.9.237.555']  # PSE_Code = 0B, Gaia

    selected_parts = part_ref_selection(df_sales, min_date, max_date, options_file.project_id)
    results = apv_photo_stock_treatment(df_sales, df_history, selected_parts, preprocessed_data_exists_flag, min_date, max_date, pse_code, project_id)
    part_ref_matchup_df = part_ref_ta_definition(df_sales, selected_parts, pse_code, max_date, [options_file.bmw_ta_mapping, options_file.mini_ta_mapping], options_file.regex_dict, options_file.bmw_original_oil_words, options_file.project_id)  # This function deliberately uses the full amount of data, while i don't have a reliable source of TA - the more information, the less likely it is for the TA to be wrong
    df_solver = solver_dataset_preparation(results, part_ref_matchup_df, options_file.group_goals['dtss_goal'], max_date)

    print('Elapsed time: {:.2f}'.format(time.time() - start))

    log_record('Fim Secção C', options_file.project_id)
    return df_solver, part_ref_matchup_df


def deployment(df_solver, df_part_ref_ta):
    performance_info_append(time.time(), 'Section_E_Start')
    log_record('Início Secção E...', options_file.project_id)

    sql_inject(df_solver, options_file.DSN_MLG, options_file.sql_info['database_final'], options_file.sql_info['final_table'], options_file, columns=list(options_file.column_sql_renaming.values()), truncate=1, check_date=1)
    sql_inject(df_part_ref_ta, options_file.DSN, options_file.sql_info['database_final'], options_file.sql_info['ta_table'], options_file, columns=list(df_part_ref_ta), truncate=1, check_date=1)

    log_record('Fim Secção E.', options_file.project_id)
    performance_info_append(time.time(), 'Section_E_End')
    return


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        project_identifier, exception_desc = options_file.project_id, str(sys.exc_info()[1])
        log_record(exception_desc, project_identifier, flag=2)
        error_upload(options_file, project_identifier, format_exc(), exception_desc, error_flag=1)
        log_record('Falhou - Projeto: ' + str(project_dict[project_identifier]) + '.', project_identifier)
