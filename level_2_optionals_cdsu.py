import sys
import time
import logging
import pandas as pd
from traceback import format_exc
import level_2_optionals_cdsu_options

from level_2_optionals_cdsu_options import project_id
from modules.level_1_a_data_acquisition import project_units_count_checkup, read_csv, sql_retrieve_df, sql_mapping_retrieval
from modules.level_1_b_data_processing import null_analysis, options_scraping_v2, zero_analysis, constant_columns_removal, remove_zero_price_total_vhe, lowercase_column_conversion, remove_rows, remove_columns, string_replacer, date_cols, options_scraping, color_replacement, new_column_creation, score_calculation, duplicate_removal, total_price, margin_calculation, col_group, new_features, ohe, global_variables_saving, dataset_split, column_rename, feature_selection, sell_place_parametrization
from modules.level_1_d_model_evaluation import performance_evaluation_classification, model_choice, plot_roc_curve, feature_contribution, multiprocess_model_evaluation, data_grouping_by_locals_temp
from modules.level_1_e_deployment import sql_inject, sql_date_comparison, sql_mapping_upload
from modules.level_0_performance_report import performance_info_append, performance_info, error_upload, log_record, project_dict
pd.set_option('display.expand_frame_repr', False)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S @ %d/%m/%y', filename=level_2_optionals_cdsu_options.log_files['full_log'], filemode='a')
logging.Logger('errors')
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # Allows the stdout to be seen in the console
logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))  # Allows the stderr to be seen in the console

configuration_parameters = level_2_optionals_cdsu_options.selected_configuration_parameters

dict_sql_upload_flag = 0


def main():
    log_record('Projeto: Sugestão Encomenda CDSU - Viaturas', project_id)

    query_filters = {'NLR_CODE': '4R0', 'Franchise_Code_DW': '43'}

    df = data_acquisition(query_filters)
    control_prints(df, 'after getting data')

    df = data_processing(df)

    _, df, _ = data_grouping_by_locals_temp(df, configuration_parameters, level_2_optionals_cdsu_options.project_id)

    control_prints(df, 'before deployment', head=1)
    deployment(df, level_2_optionals_cdsu_options.sql_info['database'], level_2_optionals_cdsu_options.sql_info['final_table_temp'])

    # performance_info(level_2_optionals_cdsu_options.project_id, level_2_optionals_cdsu_options, model_choice_message, vehicle_count)

    log_record('Conclusão com sucesso - Projeto {}.\n'.format(project_dict[project_id]), project_id)


def data_acquisition(query_filters):
    performance_info_append(time.time(), 'Section_A_Start')
    log_record('Início Secção A...', project_id)

    df = sql_retrieve_df(level_2_optionals_cdsu_options.DSN_MLG_PRD, level_2_optionals_cdsu_options.sql_info['database'], level_2_optionals_cdsu_options.sql_info['initial_table'], level_2_optionals_cdsu_options, list(level_2_optionals_cdsu_options.sql_to_code_renaming.keys()), query_filters, column_renaming=1, parse_dates=['Purchase_Date', 'Sell_Date'])
    # project_units_count_checkup(df, 'Nº Stock', level_2_optionals_cdsu_options, sql_check=0)

    log_record('Fim Secção A.', project_id)
    performance_info_append(time.time(), 'Section_A_End')

    return df


def control_prints(df, tag, head=0, save=0):

    print('{}\n{}'.format(tag, df.shape))
    try:
        print('Unique Vehicles: {}'.format(df['Nº Stock'].nunique()))
    except KeyError:
        print('Unique Vehicles: {}'.format(df['VHE_Number'].nunique()))

    if head:
        print(df.head())
    if save:
        df.to_csv('dbs/cdsu_control_save_tag_{}.csv'.format(tag))

    return


def data_processing(df):
    performance_info_append(time.time(), 'Section_B_Start')
    log_record('Início Secção B...', project_id)

    if sql_date_comparison(level_2_optionals_cdsu_options.DSN_MLG_PRD, level_2_optionals_cdsu_options, level_2_optionals_cdsu_options.sql_info['database'], level_2_optionals_cdsu_options.sql_info['checkpoint_b_table'], 'Date', level_2_optionals_cdsu_options.update_frequency_days):
        log_record('Checkpoint não encontrado ou demasiado antigo. A processar dados...', project_id)

        df = lowercase_column_conversion(df, ['Opcional', 'Cor', 'Interior', 'Versão'])  # Lowercases the strings of these columns

        dict_strings_to_replace = {('Modelo', ' - não utilizar'): '', ('Interior', '\\|'): '/', ('Cor', '|'): '', ('Interior', 'ind.'): '', ('Interior', ']'): '/', ('Interior', '.'): ' ', ('Interior', '\'merino\''): 'merino', ('Interior', '\' merino\''): 'merino', ('Interior', '\'vernasca\''): 'vernasca', ('Interior', 'leder'): 'leather',
                                   ('Interior', 'p '): 'pele', ('Interior', 'pelenevada'): 'pele nevada', ('Opcional', 'bi-xénon'): 'bixénon', ('Opcional', 'bi-xenon'): 'bixénon', ('Opcional', 'vidro'): 'vidros', ('Opcional', 'dacota'): 'dakota', ('Opcional', 'whites'): 'white', ('Opcional', 'beige'): 'bege', ('Interior', '\'dakota\''): 'dakota', ('Interior', 'dacota'): 'dakota',
                                   ('Interior', 'mokka'): 'mocha', ('Interior', 'beige'): 'bege', ('Interior', 'dakota\''): 'dakota', ('Interior', 'antracite/cinza/p'): 'antracite/cinza/preto', ('Interior', 'antracite/cinza/pretoreto'): 'antracite/cinza/preto', ('Interior', 'nevada\''): 'nevada',
                                   ('Interior', '"nappa"'): 'nappa', ('Interior', 'anthrazit'): 'antracite', ('Interior', 'antracito'): 'antracite', ('Interior', 'preto/laranja/preto/lara'): 'preto/laranja', ('Interior', 'anthtacite'): 'antracite',
                                   ('Interior', 'champag'): 'champagne', ('Interior', 'cri'): 'crimson', ('Modelo', 'Enter Model Details'): '', ('Registration_Number', '\.'): '', ('Interior', 'preto/m '): 'preto ', ('Interior', 'congnac/preto'): 'cognac/preto',
                                   ('Local da Venda', 'DCN'): 'DCP', ('Cor', 'oceanao'): 'oceano', ('Cor', 'ocenao'): 'oceano', ('Interior', 'reto'): 'preto', ('Cor', 'banco'): 'branco', ('Cor', 'catanho'): 'castanho', ('Cor', 'petrìleo'): 'petróleo', ('Interior', 'ecido'): 'tecido',
                                   ('Interior', 'ege'): 'bege', ('Interior', 'inza'): 'cinza', ('Interior', 'inzento'): 'cinzento', ('Interior', 'teciso'): 'tecido', ('Opcional', 'autmático'): 'automático', ('Opcional', 'esctacionamento'): 'estacionamento',
                                   ('Opcional', 'estacionamernto'): 'estacionamento', ('Opcional', 'pct'): 'pacote', ('Opcional', 'navegaçãp'): 'navegação', ('Opcional', '\\+'): ''}

        control_prints(df, '1')
        df = string_replacer(df, dict_strings_to_replace)  # Replaces the strings mentioned in dict_strings_to_replace which are typos, useless information, etc

        null_analysis(df)
        df.dropna(subset=['Cor', 'Colour_Ext_Code', 'Modelo', 'Interior', 'Tipo Encomenda'], axis=0, inplace=True)  # Removes all remaining NA's
        control_prints(df, '2')

        df = new_column_creation(df, [x for x in level_2_optionals_cdsu_options.configuration_parameters_full if x != 'Modelo'], 0)  # Creates new columns filled with zeros, which will be filled in the future

        # dict_cols_to_take_date_info = {'buy_': 'Data Compra'}
        # df = date_cols(df, dict_cols_to_take_date_info)  # Creates columns for the datetime columns of dict_cols_to_take_date_info, with just the day, month and year
        df = total_price(df)  # Creates a new column with the total cost for each configuration;
        control_prints(df, '3a', head=0)

        df = remove_zero_price_total_vhe(df, project_id)  # Removes VHE with a price total of 0; ToDo: keep checking up if this is still necessary
        control_prints(df, '3b', head=0)

        # df = remove_rows(df, [df[df.Modelo.str.contains('Série|Z4|i3|MINI')].index], project_id)  # No need for Prov filtering, as it is already filtered in the data source;
        # df = remove_rows(df, [df[df.Franchise_Code.str.contains('T|Y|R|G')].index], project_id)  # This removes Toyota Vehicles that aren't supposed to be in this model
        df = remove_rows(df, [df[(df.Colour_Ext_Code == ' ') & (df.Cor == ' ')].index], project_id, warning=1)

        df = options_scraping_v2(df, level_2_optionals_cdsu_options, 'Modelo')  # Scrapes the optionals columns for information regarding the GPS, Auto Transmission, Posterior Parking Sensors, External and Internal colours, Model and Rim's Size
        control_prints(df, '4', head=0, save=1)

        # df = remove_rows(df, [df[df.Modelo.isnull()].index], project_id, warning=1)
        df = remove_columns(df, ['Colour_Ext_Code'], project_id)  # This column was only needed for some very specific cases where no Colour_Ext_Code was available;
        df.to_csv('dbs/df_cdsu.csv', index=False)
        control_prints(df, '5', head=0, save=1)

        # project_units_count_checkup(df, 'Nº Stock', level_2_optionals_cdsu_options, sql_check=1)

        df = color_replacement(df, level_2_optionals_cdsu_options.colors_to_replace_dict, project_id)  # Translates all english colors to portuguese
        control_prints(df, '6', head=0, save=1)

        df = duplicate_removal(df, subset_col='Nº Stock')  # Removes duplicate rows, based on the Stock number. This leaves one line per configuration;
        control_prints(df, '7')

        df = remove_columns(df, ['Cor', 'Interior', 'Opcional', 'Custo', 'Versão', 'Tipo Encomenda', 'Franchise_Code'], project_id)  # Remove columns not needed atm;
        # Will probably need to also remove: stock_days, stock_days_norm, and one of the scores

        # df = remove_rows(df, [df.loc[df['Local da Venda'] == 'DCV - Viat.Toy Viseu', :].index], project_id)  # Removes the vehicles sold here, as they are from another brand (Toyota)

        df = margin_calculation(df)  # Calculates the margin in percentage of the total price
        control_prints(df, '8')

        df = score_calculation(df, [level_2_optionals_cdsu_options.stock_days_threshold], level_2_optionals_cdsu_options.margin_threshold, level_2_optionals_cdsu_options.project_id)  # Classifies the stockdays and margin based in their respective thresholds in tow classes (0 or 1) and then creates a new_score metric,
        control_prints(df, '9')
        # where only configurations with 1 in both dimension, have 1 as new_score

        # df = new_column_creation(df, ['Local da Venda_v2'], df['Local da Venda'])
        # control_prints(df, '10')

        # cols_to_group_layer_2 = ['Local da Venda']
        # mapping_dictionaries, _ = sql_mapping_retrieval(level_2_optionals_cdsu_options.DSN_MLG_PRD, level_2_optionals_cdsu_options.sql_info['database'], level_2_optionals_cdsu_options.sql_info['mappings_temp'], 'Mapped_Value', level_2_optionals_cdsu_options)
        # df = sell_place_parametrization(df, 'Local da Venda', 'Local da Venda_Fase2', mapping_dictionaries[2], level_2_optionals_cdsu_options.project_id)

        # df = col_group(df, cols_to_group_layer_2[0:2], mapping_dictionaries[0:2], project_id)  # Based on the information provided by Manuel some entries were grouped as to remove small groups. The columns grouped are mentioned in cols_to_group, and their respective groups are shown in level_2_optionals_cdsu_options

        df = new_features(df, configuration_parameters, project_id)  # Creates a series of new features, explained in the provided pdf

        # global_variables_saving(df, level_2_optionals_cdsu_options.project_id)  # Small functions to save 2 specific global variables which will be needed later

        log_record('Checkpoint B.1...', project_id)
        # performance_info_append(time.time(), 'checkpoint_b1')
        df = column_rename(df, list(level_2_optionals_cdsu_options.column_checkpoint_sql_renaming.keys()), list(level_2_optionals_cdsu_options.column_checkpoint_sql_renaming.values()))
        # sql_inject(df, level_2_optionals_cdsu_options.DSN_MLG_PRD, level_2_optionals_cdsu_options.sql_info['database'], level_2_optionals_cdsu_options.sql_info['checkpoint_b_table'], level_2_optionals_cdsu_options, list(level_2_optionals_cdsu_options.column_checkpoint_sql_renaming.values()), truncate=1, check_date=1)
        df = column_rename(df, list(level_2_optionals_cdsu_options.column_checkpoint_sql_renaming.values()), list(level_2_optionals_cdsu_options.column_checkpoint_sql_renaming.keys()))
        df = remove_columns(df, ['Date'], project_id)

    else:
        log_record('Checkpoint Found. Retrieving data...', project_id)
        df = sql_retrieve_df(level_2_optionals_cdsu_options.DSN_MLG_PRD, level_2_optionals_cdsu_options.sql_info['database'], level_2_optionals_cdsu_options.sql_info['checkpoint_b_table'], level_2_optionals_cdsu_options, list(level_2_optionals_cdsu_options.column_checkpoint_sql_renaming.values()))
        df = column_rename(df, list(level_2_optionals_cdsu_options.column_checkpoint_sql_renaming.values()), list(level_2_optionals_cdsu_options.column_checkpoint_sql_renaming.keys()))

    log_record('Fim Secção B.', project_id)

    performance_info_append(time.time(), 'Section_B_End')

    return df


def deployment(df, db, view):
    performance_info_append(time.time(), 'Section_E_Start')
    log_record('Início Secção E...', project_id)

    if df is not None:
        # df = column_rename(df, list(level_2_optionals_cdsu_options.column_sql_renaming.keys()), list(level_2_optionals_cdsu_options.column_sql_renaming.values()))
        df = df.rename(columns=level_2_optionals_cdsu_options.column_sql_renaming)
        control_prints(df, 'before deployment, after renaming', head=1)
        sql_inject(df, level_2_optionals_cdsu_options.DSN_MLG_DEV, db, view, level_2_optionals_cdsu_options, list(level_2_optionals_cdsu_options.column_checkpoint_sql_renaming.values()), truncate=1, check_date=1)

    log_record('Fim Secção E.', project_id)
    performance_info_append(time.time(), 'Section_E_End')
    return


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        project_identifier, exception_desc = level_2_optionals_cdsu_options.project_id, str(sys.exc_info()[1])
        log_record(exception_desc, project_identifier, flag=2)
        # error_upload(level_2_optionals_cdsu_options, project_identifier, format_exc(), exception_desc, error_flag=1)
        log_record('Falhou - Projeto: {}.'.format(str(project_dict[project_identifier])), project_identifier)
