import time
import sys
import os
import logging
from traceback import format_exc
import pandas as pd
import level_2_order_optimization_apv_baviera_options as options_file
from modules.level_1_a_data_acquisition import dw_data_retrieval, autoline_data_retrieval, read_csv
from modules.level_1_b_data_processing import apv_dataset_treatment, column_rename
from modules.level_1_c_data_modelling import apv_stock_evolution_calculation, part_ref_selection, part_ref_ta_definition, apv_last_stock_calculation, apv_photo_stock_treatment, solver_dataset_preparation
from modules.level_1_e_deployment import time_tags, sql_inject, sql_truncate
from modules.level_0_performance_report import log_record, project_dict, error_upload, performance_info, performance_info_append

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S @ %d/%m/%y', filename=options_file.log_files['full_log'], filemode='a')
logging.Logger('errors')
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # Allows the stdout to be seen in the console
logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))  # Allows the stderr to be seen in the console


def main():
    log_record('Projeto: {}'.format(project_dict[options_file.project_id]), options_file.project_id)

    for pse_group in options_file.pse_codes_groups:
        current_date, _ = time_tags(format_date='%Y%m%d')

        last_processed_date, second_to_last_processed_date, preprocessed_data_exists_flag = apv_last_stock_calculation(options_file.min_date, current_date, pse_group[0], options_file.project_id)  # Considering all PSE Groups were processed in the same day
        print('Processing data from {} to {}'.format(last_processed_date, current_date))
        print('Deleting data for the day of {}'.format(second_to_last_processed_date))

        df_sales_group, df_product_group_dw, df_history_group = data_acquisition(options_file, pse_group, last_processed_date, current_date)

        for pse_code in pse_group:
            df_sales = df_sales_group.loc[df_sales_group['PSE_Code'] == pse_code, :]
            df_history = df_history_group.loc[df_history_group['SO_Code'] == pse_code, :]

            log_record('Começou PSE = {}'.format(pse_code), options_file.project_id)

            df_sales_cleaned = data_processing(df_sales, pse_code, options_file)
            df_solver, df_part_ref_ta = data_modelling(pse_code, df_sales_cleaned, df_history, last_processed_date, current_date, preprocessed_data_exists_flag, options_file.project_id)
            deployment(df_solver, df_part_ref_ta, pse_code)

            log_record('Terminou PSE = {}'.format(pse_code), options_file.project_id)

    performance_info(options_file.project_id, options_file, model_choice_message='N/A')

    delete_temp_files(options_file.pse_codes_groups, second_to_last_processed_date)

    log_record('Conclusão com sucesso - Projeto: {} .\n'.format(project_dict[options_file.project_id]), options_file.project_id)


def data_acquisition(options_info, pse_group, last_processed_date, current_date):
    performance_info_append(time.time(), 'Section_A_Start')
    log_record('Início Secção A...', options_file.project_id)

    start = time.time()

    df_sales, df_history, df_product_group_dw = dw_data_retrieval(pse_group, current_date, options_info, last_processed_date)

    print('Elapsed time: {:.2f} seconds.'.format(time.time() - start))

    log_record('Fim Secção A.', options_file.project_id)
    performance_info_append(time.time(), 'Section_A_End')
    return df_sales, df_product_group_dw, df_history


def data_processing(df_sales, pse_code, options_info):
    performance_info_append(time.time(), 'Section_B_Start')
    log_record('Início Secção B...', options_file.project_id)
    start_treatment = time.time()

    df_sales = apv_dataset_treatment(df_sales, pse_code, options_info.urgent_purchases_flags, options_info.project_id)

    print('Elapsed time: {:.2f} seconds.'.format(time.time() - start_treatment))

    log_record('Fim Secção B.', options_file.project_id)
    performance_info_append(time.time(), 'Section_B_End')
    return df_sales


def data_modelling(pse_code, df_sales, df_history, min_date, max_date, preprocessed_data_exists_flag, project_id):
    performance_info_append(time.time(), 'Section_C_Start')
    log_record('Início Secção C', options_file.project_id)
    start = time.time()

    selected_parts = part_ref_selection(df_sales, min_date, max_date, options_file.project_id)
    results = apv_photo_stock_treatment(df_sales, df_history, selected_parts, preprocessed_data_exists_flag, min_date, max_date, pse_code, project_id)
    part_ref_matchup_df = part_ref_ta_definition(df_sales, selected_parts, pse_code, max_date, [options_file.bmw_ta_mapping, options_file.mini_ta_mapping], options_file.regex_dict, options_file.bmw_original_oil_words, options_file.project_id)  # This function deliberately uses the full amount of data, while i don't have a reliable source of TA - the more information, the less likely it is for the TA to be wrong
    df_solver = solver_dataset_preparation(results, part_ref_matchup_df, options_file.group_goals['dtss_goal'], pse_code, max_date)

    print('Elapsed time: {:.2f}'.format(time.time() - start))

    log_record('Fim Secção C', options_file.project_id)
    performance_info_append(time.time(), 'Section_C_End')
    return df_solver, part_ref_matchup_df


def deployment(df_solver, df_part_ref_ta, pse_code):
    performance_info_append(time.time(), 'Section_E_Start')
    log_record('Início Secção E...', options_file.project_id)

    df_solver = column_rename(df_solver, list(options_file.column_sql_renaming.keys()), list(options_file.column_sql_renaming.values()))
    df_solver = df_solver.dropna(subset=[options_file.column_sql_renaming['Group']])
    df_solver['Cost'] = pd.to_numeric(df_solver['Cost'], errors='coerce')
    df_solver.dropna(axis=0, subset=['Cost'], inplace=True)

    df_part_ref_ta = column_rename(df_part_ref_ta, ['Group'], [options_file.column_sql_renaming['Group']])

    sql_truncate(options_file.DSN_MLG, options_file, options_file.sql_info['database_final'], options_file.sql_info['final_table'], query=options_file.truncate_table_query.format(options_file.sql_info['final_table'], pse_code))
    sql_inject(df_solver, options_file.DSN_MLG, options_file.sql_info['database_final'], options_file.sql_info['final_table'], options_file, columns=list(options_file.column_sql_renaming.values()), check_date=1)

    sql_truncate(options_file.DSN_MLG, options_file, options_file.sql_info['database_final'], options_file.sql_info['ta_table'], query=options_file.truncate_table_query.format(options_file.sql_info['ta_table'], pse_code))
    sql_inject(df_part_ref_ta, options_file.DSN_MLG, options_file.sql_info['database_final'], options_file.sql_info['ta_table'], options_file, columns=list(df_part_ref_ta), check_date=1)

    log_record('Fim Secção E.', options_file.project_id)
    performance_info_append(time.time(), 'Section_E_End')
    return


def delete_temp_files(pse_groups, second_last_processed_date):
    for pse_group in pse_groups:
        pse_group_file_name = str('_'.join(pse_group))

        os.remove('dbs/stock_history_{}_{}.csv'.format(pse_group_file_name, second_last_processed_date))
        os.remove('dbs/df_product_group_dw_{}_{}.csv'.format(pse_group_file_name, second_last_processed_date))
        os.remove('dbs/df_sales_{}_{}.csv'.format(pse_group_file_name, second_last_processed_date))

        for pse in pse_group:
            os.remove('dbs/df_sales_cleaned_{}_{}.csv'.format(pse, second_last_processed_date))
            os.remove('dbs/df_sales_processed_{}_{}.csv'.format(pse, second_last_processed_date))
            os.remove('output/df_solver_{}_{}.csv'.format(pse, second_last_processed_date))
            os.remove('output/part_ref_ta_{}_{}.csv'.format(pse, second_last_processed_date))
            os.remove('output/results_merge_{}_{}.csv'.format(pse, second_last_processed_date))

    return



if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        project_identifier, exception_desc = options_file.project_id, str(sys.exc_info()[1])
        log_record(exception_desc, project_identifier, flag=2)
        error_upload(options_file, project_identifier, format_exc(), exception_desc, error_flag=1)
        log_record('Falhou - Projeto: ' + str(project_dict[project_identifier]) + '.', project_identifier)
