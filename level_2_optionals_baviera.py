import time
import sys
import logging
import pandas as pd
import level_2_optionals_baviera_options
from level_2_optionals_baviera_options import project_id
from level_1_a_data_acquisition import vehicle_count_checkup, read_csv, sql_retrieve_df, sql_mapping_retrieval
from level_1_b_data_processing import datasets_dictionary_function, constant_columns_removal, remove_zero_price_total_vhe, lowercase_column_convertion, remove_rows, remove_columns, string_replacer, date_cols, options_scraping, color_replacement, new_column_creation, score_calculation, duplicate_removal, total_price, margin_calculation, col_group, new_features_optionals_baviera, ohe, global_variables_saving, dataset_split, column_rename, feature_selection
from level_1_c_data_modelling import model_training, save_model
from level_1_d_model_evaluation import performance_evaluation, model_choice, plot_roc_curve, feature_contribution, multiprocess_model_evaluation
from level_1_e_deployment import sql_inject, sql_age_comparison
from level_0_performance_report import performance_info_append, performance_info, error_upload, log_record, project_dict
pd.set_option('display.expand_frame_repr', False)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S @ %d/%m/%y', filename=level_2_optionals_baviera_options.log_files['full_log'], filemode='a')
logging.Logger('errors')
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # Allows the stdout to be seen in the console
logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))  # Allows the stderr to be seen in the console

configuration_parameters = level_2_optionals_baviera_options.selected_configuration_parameters

running_times_upload_flag = 1


def main():
    log_record('Project: Baviera Stock Optimization', project_id)

    ### Options:
    input_file = 'dbs/' + 'full_data_bmw.csv'

    target_variable = ['new_score']  # possible targets = ['stock_class1', 'stock_class2', 'margem_class1', 'score_class', 'new_score']
    oversample_check = 0
    models = ['xgb', 'lgb']
    query_filters = {'NLR_CODE': '701'}
    # possible_evaluation_metrics: 'ROC_Curve', 'Micro_F1', 'Average_F1', 'Macro_F1', 'Accuracy', 'Precision'
    ###

    number_of_features = 'all'
    df = data_acquistion(input_file, query_filters, local=0)
    df, datasets = data_processing(df, target_variable, oversample_check, number_of_features)
    classes, best_models, running_times = data_modelling(df, datasets, models)
    model_choice_message, best_model, vehicle_count = model_evaluation(df, models, best_models, running_times, classes, datasets, number_of_features, level_2_optionals_baviera_options, project_id)
    deployment(best_model, level_2_optionals_baviera_options.sql_info['database'], level_2_optionals_baviera_options.sql_info['final_table'])

    performance_info(level_2_optionals_baviera_options.project_id, level_2_optionals_baviera_options, model_choice_message, vehicle_count, running_times_upload_flag)

    log_record('Finished Successfully - Project: Baviera Order Optimization.\n', project_id)


def data_acquistion(input_file, query_filters, local=0):
    performance_info_append(time.time(), 'start_section_a')
    log_record('Started Step A...', project_id)

    if local:
        try:
            df = read_csv(input_file, encoding='utf-8', parse_dates=['Purchase_Date', 'Sell_Date'], usecols=level_2_optionals_baviera_options.sql_to_code_renaming.keys(), infer_datetime_format=True, decimal='.')
            column_rename(df, list(level_2_optionals_baviera_options.sql_to_code_renaming.keys()), list(level_2_optionals_baviera_options.sql_to_code_renaming.values()))
        except UnicodeDecodeError:
            df = read_csv(input_file, encoding='latin-1', parse_dates=['Purchase_Date', 'Sell_Date'], usecols=level_2_optionals_baviera_options.sql_to_code_renaming.keys(), infer_datetime_format=True, decimal='.')
            column_rename(df, list(level_2_optionals_baviera_options.sql_to_code_renaming.keys()), list(level_2_optionals_baviera_options.sql_to_code_renaming.values()))
    else:
        df = sql_retrieve_df(level_2_optionals_baviera_options.DSN_MLG, level_2_optionals_baviera_options.sql_info['database'], level_2_optionals_baviera_options.sql_info['initial_table'], level_2_optionals_baviera_options, list(level_2_optionals_baviera_options.sql_to_code_renaming.keys()), query_filters, column_renaming=1, parse_dates=['Purchase_Date', 'Sell_Date'])
        vehicle_count_checkup(df, level_2_optionals_baviera_options, sql_check=0)

    log_record('Finished Step A.', project_id)
    performance_info_append(time.time(), 'end_section_a')

    return df


def data_processing(df, target_variable, oversample_check, number_of_features):
    performance_info_append(time.time(), 'start_section_b')
    log_record('Started Step B...', project_id)

    if sql_age_comparison(level_2_optionals_baviera_options.DSN_MLG, level_2_optionals_baviera_options, level_2_optionals_baviera_options.sql_info['database'], level_2_optionals_baviera_options.sql_info['checkpoint_b_table'], level_2_optionals_baviera_options.update_frequency_days):
        log_record('Checkpoint not found or too old. Preprocessing data...', project_id)

        df = lowercase_column_convertion(df, ['Opcional', 'Cor', 'Interior', 'Versão'])  # Lowercases the strings of these columns

        dict_strings_to_replace = {('Modelo', ' - não utilizar'): '', ('Interior', '|'): '/', ('Cor', '|'): '', ('Interior', 'ind.'): '', ('Interior', ']'): '/', ('Interior', '.'): ' ', ('Interior', '\'merino\''): 'merino', ('Interior', '\' merino\''): 'merino', ('Interior', '\'vernasca\''): 'vernasca', ('Interior', 'leder'): 'leather',
                                   ('Interior', 'p '): 'pele', ('Interior', 'pelenevada'): 'pele nevada', ('Opcional', 'bi-xénon'): 'bixénon', ('Opcional', 'vidro'): 'vidros', ('Opcional', 'dacota'): 'dakota', ('Opcional', 'whites'): 'white', ('Opcional', 'beige'): 'bege', ('Interior', '\'dakota\''): 'dakota', ('Interior', 'dacota'): 'dakota',
                                   ('Interior', 'mokka'): 'mocha', ('Interior', 'beige'): 'bege', ('Interior', 'dakota\''): 'dakota', ('Interior', 'antracite/cinza/p'): 'antracite/cinza/preto', ('Interior', 'antracite/cinza/pretoreto'): 'antracite/cinza/preto', ('Interior', 'nevada\''): 'nevada',
                                   ('Interior', '"nappa"'): 'nappa', ('Interior', 'anthrazit'): 'antracite', ('Interior', 'antracito'): 'antracite', ('Interior', 'preto/laranja/preto/lara'): 'preto/laranja', ('Interior', 'anthtacite'): 'antracite',
                                   ('Interior', 'champag'): 'champagne', ('Interior', 'cri'): 'crimson', ('Modelo', 'Enter Model Details'): '', ('Registration_Number', '\.'): '', ('Interior', 'preto/m '): 'preto ', ('Interior', 'congnac/preto'): 'cognac/preto'}

        df = string_replacer(df, dict_strings_to_replace)  # Replaces the strings mentioned in dict_strings_to_replace which are typos, useless information, etc
        df.dropna(axis=0, inplace=True)  # Removes all remaining NA's

        df = new_column_creation(df, [x for x in level_2_optionals_baviera_options.configuration_parameters_full if x != 'Modelo'], 0)  # Creates new columns filled with zeros, which will be filled in the future

        dict_cols_to_take_date_info = {'buy_': 'Data Compra'}
        df = date_cols(df, dict_cols_to_take_date_info)  # Creates columns for the datetime columns of dict_cols_to_take_date_info, with just the day, month and year
        df = total_price(df)  # Creates a new column with the total cost for each configuration;
        df = remove_zero_price_total_vhe(df)  # Removes VHE with a price total of 0; ToDo: keep checking up if this is still necessary
        df = remove_rows(df, [df[df.Modelo.str.contains('Série|Z4|i3|MINI')].index])  # No need for Prov filtering, as it is already filtered in the data source;
        df = remove_rows(df, [df[df.Franchise_Code.str.contains('T|Y|R|G')].index])  # This removes Toyota Vehicles that aren't supposed to be in this model

        df = options_scraping(df)  # Scrapes the optionals columns for information regarding the GPS, Auto Transmission, Posterior Parking Sensors, External and Internal colours, Model and Rim's Size

        vehicle_count_checkup(df, level_2_optionals_baviera_options, sql_check=1)

        df = color_replacement(df)  # Translates all english colors to portuguese

        df = duplicate_removal(df, subset_col='Nº Stock')  # Removes duplicate rows, based on the Stock number. This leaves one line per configuration;

        df = remove_columns(df, ['Cor', 'Interior', 'Opcional', 'Custo', 'Versão', 'Tipo Encomenda', 'Franchise_Code'])  # Remove columns not needed atm;
        # Will probably need to also remove: stock_days, stock_days_norm, and one of the scores

        df = remove_rows(df, [df.loc[df['Local da Venda'] == 'DCV - Viat.Toy Viseu', :].index])  # Removes the vehicles sold here, as they are from another brand (Toyota)

        df = margin_calculation(df)  # Calculates the margin in percentage of the total price
        df = score_calculation(df, level_2_optionals_baviera_options.stock_days_threshold, level_2_optionals_baviera_options.margin_threshold)  # Classifies the stockdays and margin based in their respective thresholds in tow classes (0 or 1) and then creates a new_score metric,
        # where only configurations with 1 in both dimension, have 1 as new_score
        df = new_column_creation(df, ['Local da Venda_v2'], df['Local da Venda'])

        cols_to_group_layer_2 = ['Jantes', 'Local da Venda', 'Local da Venda_v2', 'Modelo', 'Versao', 'Tipo_Interior', 'Cor_Exterior', 'Cor_Interior', 'Motor']
        dictionaries = sql_mapping_retrieval(level_2_optionals_baviera_options.DSN_MLG, level_2_optionals_baviera_options.sql_info['database'], level_2_optionals_baviera_options.sql_info['mappings'], level_2_optionals_baviera_options)
        df = col_group(df, cols_to_group_layer_2, dictionaries)  # Based on the information provided by Manuel some entries were grouped as to remove small groups. The columns grouped are mentioned in cols_to_group, and their respective
        # groups are shown in level_2_optionals_baviera_options

        df = new_features_optionals_baviera(df, sel_cols=configuration_parameters)  # Creates a series of new features, explained in the provided pdf

        global_variables_saving(df, project='optionals_baviera')  # Small functions to save 2 specific global variables which will be needed later

        log_record('Checkpoint B.1...', project_id)
        # performance_info_append(time.time(), 'checkpoint_b1')
        df = column_rename(df, list(level_2_optionals_baviera_options.column_checkpoint_sql_renaming.keys()), list(level_2_optionals_baviera_options.column_checkpoint_sql_renaming.values()))
        sql_inject(df, level_2_optionals_baviera_options.DSN_MLG, level_2_optionals_baviera_options.sql_info['database'], level_2_optionals_baviera_options.sql_info['checkpoint_b_table'], level_2_optionals_baviera_options, list(level_2_optionals_baviera_options.column_checkpoint_sql_renaming.values()), truncate=1, check_date=1)
        df = column_rename(df, list(level_2_optionals_baviera_options.column_checkpoint_sql_renaming.values()), list(level_2_optionals_baviera_options.column_checkpoint_sql_renaming.keys()))
        df = remove_columns(df, ['Date'])

    else:
        log_record('Checkpoint Found. Retrieving data...', project_id)
        df = sql_retrieve_df(level_2_optionals_baviera_options.DSN_MLG, level_2_optionals_baviera_options.sql_info['database'],  level_2_optionals_baviera_options.sql_info['checkpoint_b_table'], level_2_optionals_baviera_options, list(level_2_optionals_baviera_options.column_checkpoint_sql_renaming.values()))
        df = column_rename(df, list(level_2_optionals_baviera_options.column_checkpoint_sql_renaming.values()), list(level_2_optionals_baviera_options.column_checkpoint_sql_renaming.keys()))

    ohe_cols = configuration_parameters + ['Local da Venda', 'buy_day', 'buy_month', 'buy_year']

    df_ohe = df.copy(deep=True)  # Creates a copy of the original df
    df_ohe = ohe(df_ohe, ohe_cols)  # Creates the OHE for columns in ohe_cols

    if type(number_of_features) != str:
        sel_columns, removed_columns = feature_selection(df_ohe, configuration_parameters, target_variable, number_of_features)
    else:
        removed_columns = []

    df_ohe = remove_columns(df_ohe, removed_columns)
    df_ohe = constant_columns_removal(df_ohe)

    train_x, train_y, test_x, test_y = dataset_split(df_ohe[[x for x in df_ohe if x not in ['Local da Venda_v2', 'Registration_Number', 'score_euros', 'days_stock_price', 'Data Venda', 'Data Compra', 'Margem', 'Nº Stock', 'margem_percentagem', 'margin_class', 'stock_days', 'stock_days_class']]], target_variable, oversample_check)
    # Dataset split in train/test datasets, at the ratio of 0.75/0.25, while also ensuring both classes are evenly distributed

    datasets = datasets_dictionary_function(train_x, train_y, test_x, test_y)

    log_record('Finished Step B.', project_id)

    performance_info_append(time.time(), 'end_section_b')

    return df, datasets


def data_modelling(df, datasets, models):
    performance_info_append(time.time(), 'start_section_c')
    log_record('Started Step C...', project_id)

    df.sort_index(inplace=True)

    classes, best_models, running_times = model_training(models, datasets['train_x'], datasets['train_y'])  # Training of each referenced model
    save_model(best_models, models)

    log_record('Finished Step C.', project_id)
    performance_info_append(time.time(), 'end_section_c')

    return classes, best_models, running_times


def model_evaluation(df, models, best_models, running_times, classes, datasets, number_of_features, options_file, proj_id):
    performance_info_append(time.time(), 'start_section_d')
    log_record('Started Step D...', proj_id)

    results_training, results_test, predictions = performance_evaluation(models, best_models, classes, running_times, datasets, options_file, proj_id)  # Creates a df with the performance of each model evaluated in various metrics, explained
    # in the provided pdf
    plot_roc_curve(best_models, models, datasets, 'roc_curve_temp_' + str(number_of_features), save_dir='plots/')

    df_model_dict = multiprocess_model_evaluation(df, models, datasets, best_models, predictions, configuration_parameters, proj_id)
    model_choice_message, best_model_name, _, section_e_upload_flag = model_choice(options_file.DSN_MLG, options_file, results_test)

    if not section_e_upload_flag:
        best_model = None
    else:
        best_model = df_model_dict[best_model_name]
        feature_contribution(best_model, configuration_parameters, options_file, proj_id)

    log_record('Finished Step D.', proj_id)

    performance_info_append(time.time(), 'end_section_d')
    return model_choice_message, best_model, df.shape[0]


def deployment(df, db, view):
    performance_info_append(time.time(), 'start_section_e')
    log_record('Started Step E...', project_id)

    if df is not None:
        df = column_rename(df, list(level_2_optionals_baviera_options.column_sql_renaming.keys()), list(level_2_optionals_baviera_options.column_sql_renaming.values()))
        sql_inject(df, level_2_optionals_baviera_options.DSN_MLG, db, view, level_2_optionals_baviera_options, level_2_optionals_baviera_options.columns_for_sql, truncate=1, check_date=1)

    log_record('Finished Step E.', project_id)
    performance_info_append(time.time(), 'end_section_e')
    return


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        project_identifier = 2162
        log_record(exception.args[0], project_identifier, flag=2)
        error_upload(level_2_optionals_baviera_options, project_identifier, level_2_optionals_baviera_options.log_files['full_log'], error_flag=1)
        log_record('Failed - Project: {}.'.format(str(project_dict[project_identifier])), project_identifier)
