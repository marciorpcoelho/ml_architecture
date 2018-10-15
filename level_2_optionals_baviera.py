import time
import sys
import schedule
import logging
import warnings
import pandas as pd
import level_2_optionals_baviera_options
from level_1_a_data_acquisition import read_csv, log_files
from level_1_b_data_processing import lowercase_column_convertion, remove_rows, remove_columns, string_replacer, date_cols, options_scraping, color_replacement, new_column_creation, score_calculation, duplicate_removal, reindex, total_price, margin_calculation, col_group, new_features_optionals_baviera, z_scores_function, ohe, global_variables_saving, prov_replacement, dataset_split, null_analysis, zero_analysis
from level_1_c_data_modelling import model_training, save_model
from level_1_d_model_evaluation import performance_evaluation, probability_evaluation, model_choice, model_comparison, plot_roc_curve, add_new_columns_to_df, df_decimal_places_rounding
from level_1_e_deployment import save_csv
pd.set_option('display.expand_frame_repr', False)
warnings.filterwarnings('ignore')  # ToDO: remove this line

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S @ %d/%m/%y', filename='logs/optionals_baviera.txt', filemode='a')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def main():
    logging.info('Project: Baviera Stock Optimization')
    # log_files('optional_baviera')

    ### Options:
    input_file = 'dbs/' + 'ENCOMENDA.csv'
    # input_file = 'dbs/' + 'testing_ENCOMENDA.csv'
    # input_file = 'dbs/' + 'teste_s2_gran.csv'
    output_file = 'output/' + 'db_full_baviera.csv'
    stockdays_threshold, margin_threshold = 45, 3.5
    target_variable = ['new_score']  # possible targets = ['stock_class1', 'stock_class2', 'margem_class1', 'score_class', 'new_score']
    oversample_check = 0
    models = ['dt', 'rf', 'lr', 'ab', 'gc', 'ann', 'voting']
    k = 10
    gridsearch_score = 'recall'
    metric = 'accuracy'
    metric_threshold = 0.8
    # evaluation_metrics = ['micro', 'average', 'macro', 'accuracy', 'precision', 'recall', 'classification_report']
    ###

    df = data_acquistion(input_file)
    df, train_x, train_y, test_x, test_y = data_processing(df, stockdays_threshold, margin_threshold, target_variable, oversample_check)
    classes, best_models, running_times = data_modelling(df, train_x, train_y, test_x, models, k, gridsearch_score, oversample_check)
    model_evaluation(df, models, best_models, running_times, classes, metric, metric_threshold, train_x, train_y, test_x, test_y)
    deployment()

    # df = pd.DataFrame()
    # save_csv(df, 'logs/optionals_baviera_ran')

    # sys.stdout.flush()
    logging.info('Finished - Project: Baviera Stock Optimization\n')
    # return schedule.CancelJob


def data_acquistion(input_file):
    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Started Step A...')
    logging.info('Started Step A...')

    # df = read_csv(input_file, delimiter=';', parse_dates=['Data Compra', 'Data Venda'], infer_datetime_format=True, decimal=',')
    try:
        df = read_csv(input_file, delimiter=';', encoding='utf-8', parse_dates=['Data Compra', 'Data Venda'], infer_datetime_format=True, decimal=',')
    except UnicodeDecodeError:
        df = read_csv(input_file, delimiter=';', encoding='latin-1', parse_dates=['Data Compra', 'Data Venda'], infer_datetime_format=True, decimal=',')

    logging.info('Finished Step A.')
    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Finished Step A.')

    return df


def data_processing(df, stockdays_threshold, margin_threshold, target_variable, oversample_check):
    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Started Step B...')
    logging.info('Started Step B...')

    df = lowercase_column_convertion(df, ['Opcional', 'Cor', 'Interior'])  # Lowercases the strings of these columns
    df = remove_rows(df, [df.loc[df['Opcional'] == 'preço de venda', :].index])  # Removes the rows with "Preço de Venda"

    dict_strings_to_replace = {('Modelo', ' - não utilizar'): '', ('Interior', '|'): '/', ('Cor', '|'): '', ('Interior', 'ind.'): '', ('Interior', ']'): '/', ('Interior', '.'): ' ', ('Interior', '\'merino\''): 'merino', ('Interior', '\' merino\''): 'merino', ('Interior', '\'vernasca\''): 'vernasca', ('Interior', 'leder'): 'leather', ('Interior', 'p '): 'pele', ('Interior', 'pelenevada'): 'pele nevada', ('Opcional', 'bi-xénon'): 'bixénon'}
    df = string_replacer(df, dict_strings_to_replace)  # Replaces the strings mentioned in dict_strings_to_replace which are typos, useless information, etc
    df = remove_columns(df, ['CdInt', 'CdCor'])  # Columns that have missing values which are needed
    df.dropna(axis=0, inplace=True)  # Removes all remaining NA's

    df = new_column_creation(df, ['Versão', 'Navegação', 'Sensores', 'Cor_Interior', 'Tipo_Interior', 'Caixa Auto', 'Cor_Exterior', 'Jantes', 'Farois_LED', 'Farois_Xenon'])  # Creates new columns filled with zeros, which will be filled in the future

    dict_cols_to_take_date_info = {'buy_': 'Data Compra'}
    df = date_cols(df, dict_cols_to_take_date_info)  # Creates columns for the datetime columns of dict_cols_to_take_date_info, with just the day, month and year
    df = options_scraping(df)  # Scrapes the optionals columns for information regarding the GPS, Auto Transmission, Posterior Parking Sensors, External and Internal colours, Model and Rim's Size
    null_analysis(df)
    zero_analysis(df)
    df.to_csv('output/' + 'testing_new_parameters.csv')
    sys.exit()
    df = color_replacement(df)  # Translates all english colors to portuguese

    df = total_price(df)  # Creates a new column with the total cost for each configuration;
    df = duplicate_removal(df, subset_col='Nº Stock')  # Removes duplicate rows, based on the Stock number. This leaves one line per configuration;
    df = remove_columns(df, ['Cor', 'Interior', 'Versão', 'Opcional', 'A', 'S', 'Custo', 'Vendedor', 'Canal de Venda', 'Tipo Encomenda'])  # Remove columns not needed atm;
    # Will probably need to also remove: stock_days, stock_days_norm, and one of the scores
    # df = reindex(df)  # Creates a new order index - after removing duplicate rows, the index loses its sequence/order

    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Checkpoint B.1...')
    save_csv([df], ['output/' + 'ENCOMENDA_checkpoint_b1'])  # Saves a first version of the DF after treatment
    logging.info('Checkpoint B.1...')
    # ToDO: Checkpoint B.1 - this should be the first savepoint of the df. If an error is found after this point, the code should check for the df of this checkpoint

    df = remove_rows(df, [df[df.Modelo.str.contains('Série')].index, df[df.Modelo.str.contains('Z4')].index, df[df.Modelo.str.contains('MINI')].index, df[df['Prov'] == 'Demonstração'].index, df[df['Prov'] == 'Em utilização'].index])
    # Removes entries of motorcycles (Série), recent car models (Z4), MINI models (MINI) and those whose order is Demonstração and Em Utilização
    df = margin_calculation(df)  # Calculates the margin in percentage of the total price
    df = score_calculation(df, stockdays_threshold, margin_threshold)  # Classifies the stockdays and margin based in their respective thresholds in tow classes (0 or 1) and then creates a new_score metric,
    # where only configurations with 1 in both dimension, have 1 as new_score

    cols_to_group = ['Cor_Exterior', 'Cor_Interior', 'Jantes', 'Local da Venda', 'Modelo']
    dictionaries = [level_2_optionals_baviera_options.color_ext_dict, level_2_optionals_baviera_options.color_int_dict, level_2_optionals_baviera_options.jantes_dict, level_2_optionals_baviera_options.sales_place_dict, level_2_optionals_baviera_options.model_dict]
    df = col_group(df, cols_to_group, dictionaries)  # Based on the information provided by Manuel some entries were grouped as to remove small groups. The columns grouped are mentioned in cols_to_group, and their respective
    # groups are shown in level_2_optionals_baviera_options
    df = prov_replacement(df)  # Replaces all entries with order type of Viaturas Km 0 as Novos
    df = new_features_optionals_baviera(df, sel_cols=['Navegação', 'Sensores', 'Caixa Auto', 'Cor_Exterior_new', 'Cor_Interior_new', 'Jantes_new', 'Modelo_new'])  # Creates a series of new features, explained in the provided pdf

    global_variables_saving(df, project='optionals_baviera')  # Small functions to save 2 specific global variables which will be needed later
    # df = z_scores_function(df, cols_to_normalize=['price_total', 'number_prev_sales', 'last_margin', 'last_stock_days'])  # Converts all the mentioned columns to their respective Z-Score

    # df = df_copy(df)
    ohe_cols = ['Jantes_new', 'Cor_Interior_new', 'Cor_Exterior_new', 'Local da Venda_new', 'Modelo_new', 'Prov_new', 'buy_day', 'buy_month', 'buy_year']
    df_ohe = df.copy(deep=True)  # Creates a copy of the original df
    df_ohe = ohe(df_ohe, ohe_cols)  # Creates the OHE for columns in ohe_cols
    train_x, train_y, test_x, test_y = dataset_split(df_ohe[[x for x in df_ohe if x not in ['Data Venda', 'Data Compra', 'Margem', 'Nº Stock', 'margem_percentagem', 'margin_class', 'stock_days', 'stock_days_class']]], target_variable, oversample_check)
    # Dataset split in train/test datasets, at the ratio of 0.75/0.25, while also ensuring both classes are evenly distributed

    logging.info('Finished Step B.')
    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Finished Step B.')

    return df, train_x, train_y, test_x, test_y


def data_modelling(df, train_x, train_y, test_x, models, k, score, oversample_check):
    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Started Step C...')
    logging.info('Started Step C...')

    df.sort_index(inplace=True)

    classes, best_models, running_times = model_training(models, train_x, train_y, k, score)  # Training of each referenced model
    save_model(best_models, models)

    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Finished Step C.')
    logging.info('Finished Step C.')

    return classes, best_models, running_times


def model_evaluation(df, models, best_models, running_times, classes, metric, metric_threshold, train_x, train_y, test_x, test_y):
    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Started Step D...')
    logging.info('Started Step D...')

    results_training, results_test, predictions = performance_evaluation(models, best_models, classes, running_times, train_x, train_y, test_x, test_y)  # Creates a df with the performance of each model evaluated in various metrics, explained
    # in the provided pdf
    save_csv([results_training, results_test], ['output/' + 'model_performance_train_df', 'output/' + 'model_performance_test_df'])
    plot_roc_curve(best_models, models, train_x, train_y, test_x, test_y, 'roc_curve_temp', save_dir='plots/')

    best_model_name, best_model_value = model_choice(results_test, metric, metric_threshold)  # Chooses the best model based a chosen metric/threshold
    if model_comparison(best_model_name, best_model_value, metric):  # Compares the best model from the previous step with the already existing result - only compares within the same metric
        proba_training, proba_test = probability_evaluation(best_model_name, best_models, train_x, test_x)
        # print(proba_training.shape, proba_test.shape)
        df_best_model = add_new_columns_to_df(df, proba_training, proba_test, predictions[best_model_name], train_x, train_y, test_x, test_y)
        df_best_model = df_decimal_places_rounding(df_best_model, {'proba_0': 1, 'proba_1': 2})

        print(best_model_name)
        print(df_best_model)
        print(df_best_model.shape)
    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Finished Step D.')
    logging.info('Finished Step D.')


def deployment():
    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Started Step E...')
    logging.info('Started Step E...')

    # save_csv  # ToDo: Save df locally

    # print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Finished Step E.')
    logging.info('Finished Step E.')


if __name__ == '__main__':
    main()
