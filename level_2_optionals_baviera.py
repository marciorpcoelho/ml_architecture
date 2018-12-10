import time
import sys
import schedule
import logging
import warnings
import os.path
import pandas as pd
import level_2_optionals_baviera_options
from level_1_a_data_acquisition import read_csv
from level_1_b_data_processing import lowercase_column_convertion, remove_rows, remove_columns, string_replacer, date_cols, options_scraping, color_replacement, new_column_creation, score_calculation, duplicate_removal, total_price, margin_calculation, col_group, new_features_optionals_baviera, ohe, global_variables_saving, dataset_split, column_rename, feature_selection
from level_1_c_data_modelling import model_training, save_model
from level_1_d_model_evaluation import performance_evaluation, probability_evaluation, model_choice, model_comparison, plot_roc_curve, add_new_columns_to_df, df_decimal_places_rounding, feature_contribution
from level_1_e_deployment import save_csv, sql_inject, sql_age_comparison, sql_retrieve_df
from level_2_optionals_baviera_performance_report_info import performance_info_append, performance_info
pd.set_option('display.expand_frame_repr', False)
warnings.filterwarnings('ignore')  # ToDO: remove this line

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S @ %d/%m/%y', filename='logs/optionals_baviera.txt', filemode='a')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

configuration_parameters = ['7_Lug', 'AC Auto', 'Alarme', 'Barras_Tej', 'Caixa Auto', 'Cor_Exterior_new', 'Cor_Interior_new', 'Farois_LED', 'Farois_Xenon', 'Jantes_new', 'Modelo_new', 'Navegação', 'Prot.Solar', 'Sensores', 'Teto_Abrir', 'Tipo_Interior_new', 'Versao_new']
performance_list = []


def main():
    logging.info('Project: Baviera Stock Optimization')
    # log_files('optional_baviera')

    ### Options:
    input_file = 'dbs/' + 'ENCOMENDA.csv'
    output_file = 'output/' + 'db_full_baviera.csv'

    target_variable = ['new_score']  # possible targets = ['stock_class1', 'stock_class2', 'margem_class1', 'score_class', 'new_score']
    oversample_check = 0
    models = ['dt', 'rf', 'lr', 'ab', 'gc', 'ann', 'voting']
    # models = ['gc']
    k = 10  # Stratified Cross-Validation number of Folds
    gridsearch_score = 'recall'  # Metric on which to optimize GridSearchCV
    metric, metric_threshold = 'roc_auc_curve', 0.75
    # possible_evaluation_metrics: 'roc_auc_curve', 'micro', 'average', 'macro', 'accuracy', 'precision', 'recall', 'classification_report'
    development = 1
    ###

    number_of_features = 'all'
    df = data_acquistion(input_file)
    df, train_x, train_y, test_x, test_y = data_processing(df, target_variable, oversample_check, number_of_features)
    classes, best_models, running_times = data_modelling(df, train_x, train_y, test_x, models, k, gridsearch_score)
    best_model = model_evaluation(df, models, best_models, running_times, classes, metric, metric_threshold, train_x, train_y, test_x, test_y, development, number_of_features)
    # best_model = 0
    vehicle_count = deployment(best_model, level_2_optionals_baviera_options.sql_info['database'], level_2_optionals_baviera_options.sql_info['final_table'], output_file)

    performance_info(vehicle_count)

    # sys.stdout.flush()
    logging.info('Finished - Project: Baviera Stock Optimization\n')
    # return schedule.CancelJob


def data_acquistion(input_file):
    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Started Step A...')
    performance_info_append(time.time(), 'start_section_a')
    logging.info('Started Step A...')

    try:
        df = read_csv(input_file, delimiter=';', encoding='utf-8', parse_dates=['Data Compra', 'Data Venda'], infer_datetime_format=True, decimal=',')
    except UnicodeDecodeError:
        df = read_csv(input_file, delimiter=';', encoding='latin-1', parse_dates=['Data Compra', 'Data Venda'], infer_datetime_format=True, decimal=',')

    logging.info('Finished Step A.')
    performance_info_append(time.time(), 'end_section_a')
    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Finished Step A.')

    return df


def data_processing(df, target_variable, oversample_check, number_of_features):
    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Started Step B...')
    performance_info_append(time.time(), 'start_section_b')
    logging.info('Started Step B...')

    if sql_age_comparison(level_2_optionals_baviera_options.sql_info['database'], level_2_optionals_baviera_options.sql_info['checkpoint_b_table'], level_2_optionals_baviera_options.update_frequency_days):
        print('Checkpoint not found or too old. Preprocessing data...')

        df = lowercase_column_convertion(df, ['Opcional', 'Cor', 'Interior'])  # Lowercases the strings of these columns
        df = remove_rows(df, [df.loc[df['Opcional'] == 'preço de venda', :].index])  # Removes the rows with "Preço de Venda"

        dict_strings_to_replace = {('Modelo', ' - não utilizar'): '', ('Interior', '|'): '/', ('Cor', '|'): '', ('Interior', 'ind.'): '', ('Interior', ']'): '/', ('Interior', '.'): ' ', ('Interior', '\'merino\''): 'merino', ('Interior', '\' merino\''): 'merino', ('Interior', '\'vernasca\''): 'vernasca', ('Interior', 'leder'): 'leather', ('Interior', 'p '): 'pele', ('Interior', 'pelenevada'): 'pele nevada',
                                   ('Opcional', 'bi-xénon'): 'bixénon', ('Opcional', 'vidro'): 'vidros', ('Opcional', 'dacota'): 'dakota', ('Opcional', 'whites'): 'white', ('Opcional', 'beige'): 'bege', ('Interior', '\'dakota\''): 'dakota', ('Interior', 'dacota'): 'dakota',
                                   ('Interior', 'mokka'): 'mocha', ('Interior', 'beige'): 'bege', ('Interior', 'dakota\''): 'dakota', ('Interior', 'antracite/cinza/p'): 'antracite/cinza/preto', ('Interior', 'antracite/cinza/pretoreto'): 'antracite/cinza/preto', ('Interior', 'nevada\''): 'nevada',
                                   ('Interior', '"nappa"'): 'nappa', ('Interior', 'anthrazit'): 'antracite', ('Interior', 'antracito'): 'antracite', ('Interior', 'preto/laranja/preto/lara'): 'preto/laranja', ('Interior', 'anthtacite'): 'antracite',
                                   ('Interior', 'champag'): 'champagne', ('Interior', 'cri'): 'crimson'}

        df = string_replacer(df, dict_strings_to_replace)  # Replaces the strings mentioned in dict_strings_to_replace which are typos, useless information, etc
        df = remove_columns(df, ['CdInt', 'CdCor'])  # Columns that have missing values which are needed
        df.dropna(axis=0, inplace=True)  # Removes all remaining NA's

        df = new_column_creation(df, ['Versao', 'Navegação', 'Sensores', 'Cor_Interior', 'Tipo_Interior', 'Caixa Auto', 'Cor_Exterior', 'Jantes', 'Farois_LED', 'Farois_Xenon', 'Barras_Tej', '7_Lug', 'Alarme', 'Prot.Solar', 'AC Auto', 'Teto_Abrir'])  # Creates new columns filled with zeros, which will be filled in the future

        dict_cols_to_take_date_info = {'buy_': 'Data Compra'}
        df = date_cols(df, dict_cols_to_take_date_info)  # Creates columns for the datetime columns of dict_cols_to_take_date_info, with just the day, month and year
        df = options_scraping(df)  # Scrapes the optionals columns for information regarding the GPS, Auto Transmission, Posterior Parking Sensors, External and Internal colours, Model and Rim's Size
        df = color_replacement(df)  # Translates all english colors to portuguese

        df = total_price(df)  # Creates a new column with the total cost for each configuration;
        df = duplicate_removal(df, subset_col='Nº Stock')  # Removes duplicate rows, based on the Stock number. This leaves one line per configuration;
        df = remove_columns(df, ['Cor', 'Interior', 'Opcional', 'A', 'S', 'Custo', 'Versão', 'Vendedor', 'Canal de Venda', 'Tipo Encomenda'])  # Remove columns not needed atm;
        # Will probably need to also remove: stock_days, stock_days_norm, and one of the scores

        # df = remove_rows(df, [df[df.Modelo.str.contains('Série')].index, df[df.Modelo.str.contains('Z4')].index, df[df.Modelo.str.contains('MINI')].index, df[df['Prov'] == 'Demonstração'].index, df[df['Prov'] == 'Em utilização'].index])
        # Removes entries of motorcycles (Série), recent car models (Z4), MINI models (MINI) and those whose order is Demonstração and Em Utilização
        df = margin_calculation(df)  # Calculates the margin in percentage of the total price
        df = score_calculation(df, level_2_optionals_baviera_options.stock_days_threshold, level_2_optionals_baviera_options.margin_threshold)  # Classifies the stockdays and margin based in their respective thresholds in tow classes (0 or 1) and then creates a new_score metric,
        # where only configurations with 1 in both dimension, have 1 as new_score

        cols_to_group_layer_1 = ['Cor_Exterior', 'Cor_Interior']
        dictionaries_layer_1 = [level_2_optionals_baviera_options.color_ext_dict_layer_1, level_2_optionals_baviera_options.color_int_dict_layer_1]
        df = col_group(df, cols_to_group_layer_1, dictionaries_layer_1)

        column_rename(df, ['Cor_Interior_new'], ['Cor_Interior'])

        cols_to_group_layer_2 = ['Jantes', 'Local da Venda', 'Modelo', 'Versao', 'Tipo_Interior', 'Cor_Interior']
        dictionaries = [level_2_optionals_baviera_options.jantes_dict, level_2_optionals_baviera_options.sales_place_dict, level_2_optionals_baviera_options.model_dict, level_2_optionals_baviera_options.versao_dict, level_2_optionals_baviera_options.tipo_int_dict, level_2_optionals_baviera_options.color_int_dict_layer_2]
        df = col_group(df, cols_to_group_layer_2, dictionaries)  # Based on the information provided by Manuel some entries were grouped as to remove small groups. The columns grouped are mentioned in cols_to_group, and their respective
        # groups are shown in level_2_optionals_baviera_options

        df = remove_columns(df, ['Prov'])
        df = new_features_optionals_baviera(df, sel_cols=['7_Lug', 'AC Auto', 'Alarme', 'Barras_Tej', 'Farois_LED', 'Farois_Xenon', 'Prot.Solar', 'Teto_Abrir', 'Tipo_Interior_new', 'Versao_new', 'Navegação', 'Sensores', 'Caixa Auto', 'Cor_Exterior_new', 'Cor_Interior_new', 'Jantes_new', 'Modelo_new'])  # Creates a series of new features, explained in the provided pdf

        global_variables_saving(df, project='optionals_baviera')  # Small functions to save 2 specific global variables which will be needed later
        # df = z_scores_function(df, cols_to_normalize=['price_total', 'number_prev_sales', 'last_margin', 'last_stock_days'])  # Converts all the mentioned columns to their respective Z-Score

        logging.info('Checkpoint B.1...')
        performance_info_append(time.time(), 'checkpoint_b1')
        df = column_rename(df, list(level_2_optionals_baviera_options.column_checkpoint_sql_renaming.keys()), list(level_2_optionals_baviera_options.column_checkpoint_sql_renaming.values()))
        sql_inject(df, level_2_optionals_baviera_options.sql_info['database'], level_2_optionals_baviera_options.sql_info['checkpoint_b_table'], list(level_2_optionals_baviera_options.column_checkpoint_sql_renaming.values()), truncate=1, check_date=1)
        df = column_rename(df, list(level_2_optionals_baviera_options.column_checkpoint_sql_renaming.values()), list(level_2_optionals_baviera_options.column_checkpoint_sql_renaming.keys()))
        df = remove_columns(df, ['Date'])

    else:
        print('Checkpoint Found. Retrieving data...')
        df = sql_retrieve_df(level_2_optionals_baviera_options.sql_info['database'], level_2_optionals_baviera_options.sql_info['checkpoint_b_table'], list(level_2_optionals_baviera_options.column_checkpoint_sql_renaming.values()))
        df = column_rename(df, list(level_2_optionals_baviera_options.column_checkpoint_sql_renaming.values()), list(level_2_optionals_baviera_options.column_checkpoint_sql_renaming.keys()))

    ohe_cols = ['Jantes_new', 'Cor_Interior_new', 'Cor_Exterior_new', 'Local da Venda_new', 'Modelo_new', 'buy_day', 'buy_month', 'buy_year', 'Versao_new', 'Tipo_Interior_new']

    df_ohe = df.copy(deep=True)  # Creates a copy of the original df
    df_ohe = ohe(df_ohe, ohe_cols)  # Creates the OHE for columns in ohe_cols

    sel_columns, removed_columns = feature_selection(df_ohe, configuration_parameters, target_variable, number_of_features)

    df_ohe = remove_columns(df_ohe, removed_columns)

    train_x, train_y, test_x, test_y = dataset_split(df_ohe[[x for x in df_ohe if x not in ['score_euros', 'days_stock_price', 'Data Venda', 'Data Compra', 'Margem', 'Nº Stock', 'margem_percentagem', 'margin_class', 'stock_days', 'stock_days_class']]], target_variable, oversample_check)
    # Dataset split in train/test datasets, at the ratio of 0.75/0.25, while also ensuring both classes are evenly distributed

    # save_csv([df], ['output/' + 'ENCOMENDA_checkpoint_end_b'])  # Saves a first version of the DF after treatment
    logging.info('Finished Step B.')

    performance_info_append(time.time(), 'end_section_b')
    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Finished Step B.')

    return df, train_x, train_y, test_x, test_y


def data_modelling(df, train_x, train_y, test_x, models, k, score):
    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Started Step C...')
    performance_info_append(time.time(), 'start_section_c')
    logging.info('Started Step C...')

    df.sort_index(inplace=True)

    classes, best_models, running_times = model_training(models, train_x, train_y, k, score)  # Training of each referenced model
    save_model(best_models, models)

    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Finished Step C.')
    logging.info('Finished Step C.')
    performance_info_append(time.time(), 'end_section_c')

    return classes, best_models, running_times


def model_evaluation(df, models, best_models, running_times, classes, metric, metric_threshold, train_x, train_y, test_x, test_y, development, number_of_features):
    print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Started Step D...')
    performance_info_append(time.time(), 'start_section_d')
    logging.info('Started Step D...')

    results_training, results_test, predictions = performance_evaluation(models, best_models, classes, running_times, train_x, train_y, test_x, test_y)  # Creates a df with the performance of each model evaluated in various metrics, explained
    # in the provided pdf
    # save_csv([results_training, results_test], ['output/' + 'model_performance_train_df_' + str(number_of_features), 'output/' + 'model_performance_test_df_' + str(number_of_features)])
    plot_roc_curve(best_models, models, train_x, train_y, test_x, test_y, 'roc_curve_temp_' + str(number_of_features), save_dir='plots/')

    if not development:
        best_model_name, best_model_value = model_choice(results_test, metric, metric_threshold)  # Chooses the best model based a chosen metric/threshold
        if model_comparison(best_model_name, best_model_value, metric):  # Compares the best model from the previous step with the already existing result - only compares within the same metric
            proba_training, proba_test = probability_evaluation(best_model_name, best_models, train_x, test_x)
            df_model = add_new_columns_to_df(df, proba_training, proba_test, predictions[best_model_name], train_x, train_y, test_x, test_y, configuration_parameters)
            df_model = df_decimal_places_rounding(df_model, {'proba_0': 2, 'proba_1': 2})
    elif development:
        for model_name in models:
            start = time.time()
            print('Evaluating model ' + str(model_name) + '@ ' + time.strftime("%H:%M:%S @ %d/%m/%y") + '...')
            train_x_copy, test_x_copy = train_x.copy(deep=True), test_x.copy(deep=True)
            proba_training, proba_test = probability_evaluation(model_name, best_models, train_x_copy, test_x_copy)
            df_model = add_new_columns_to_df(df, proba_training, proba_test, predictions[model_name], train_x_copy, train_y, test_x_copy, test_y, configuration_parameters)
            df_model = df_decimal_places_rounding(df_model, {'proba_0': 2, 'proba_1': 2})
            save_csv([df_model], ['output/' + 'db_final_classification_' + model_name])
            print('Elapsed time: %f' % (time.time() - start))

    feature_contribution(df_model, configuration_parameters)

    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Finished Step D.')
    logging.info('Finished Step D.')
    performance_info_append(time.time(), 'end_section_d')
    return df_model


def deployment(df, db, view, output_file):
    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Started Step E...')
    performance_info_append(time.time(), 'start_section_e')
    logging.info('Started Step E...')

    df = pd.read_csv('output/' + 'db_final_classification_gc.csv', index_col=0)

    df = column_rename(df, list(level_2_optionals_baviera_options.column_sql_renaming.keys()), list(level_2_optionals_baviera_options.column_sql_renaming.values()))
    # sql_truncate(db, view)

    sql_inject(df, db, view, level_2_optionals_baviera_options.columns_for_sql, truncate=1)

    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Finished Step E.')
    logging.info('Finished Step E.')
    performance_info_append(time.time(), 'end_section_e')

    return df.shape[0]


if __name__ == '__main__':
    main()
