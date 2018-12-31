import pandas as pd
import numpy as np
import os
import multiprocessing
import time
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score, silhouette_samples, silhouette_score, mean_squared_error, r2_score, roc_curve, auc, roc_auc_score
from level_1_e_deployment import sql_inject, save_csv, sql_second_highest_date_checkup
from level_2_optionals_baviera_options import sql_info, pool_workers_count, dict_models_name_conversion
from level_2_optionals_baviera_performance_report_info import log_record
pd.set_option('display.expand_frame_repr', False)

my_dpi = 96


class ClassificationEvaluation(object):
    def __init__(self, groundtruth, prediction):
        self.micro = f1_score(groundtruth, prediction, average='micro', labels=np.unique(prediction))
        self.average = f1_score(groundtruth, prediction, average='weighted', labels=np.unique(prediction))
        self.macro = f1_score(groundtruth, prediction, average='macro', labels=np.unique(prediction))
        self.accuracy = accuracy_score(groundtruth, prediction)
        self.precision = precision_score(groundtruth, prediction, average=None)
        self.recall = recall_score(groundtruth, prediction, average=None)
        self.classification_report = classification_report(groundtruth, prediction)
        self.roc_auc_curve = roc_auc_score(groundtruth, prediction)


class ClusterEvaluation(object):
    def __init__(self, x, cluster_clients):
        self.silh_score = silhouette_score(x, cluster_clients)
        _, self.clusters_size = np.unique(cluster_clients, return_counts=True)
        self.sample_silh_score = silhouette_samples(x, cluster_clients)


class RegressionEvaluation(object):
    def __init__(self, groundtruth, prediction):
        self.mse = mean_squared_error(groundtruth, prediction)  # Mean Square Error
        self.score = r2_score(groundtruth, prediction)


def performance_evaluation(models, best_models, classes, running_times, train_x, train_y, test_x, test_y):

    results_train, results_test = [], []
    predictions, feat_importance = {}, pd.DataFrame(index=list(train_x), columns={'Importance'})
    for model in models:
        prediction_train = best_models[model].predict(train_x)
        prediction_test = best_models[model].predict(test_x)
        evaluation_training = ClassificationEvaluation(groundtruth=train_y, prediction=prediction_train)
        evaluation_test = ClassificationEvaluation(groundtruth=test_y, prediction=prediction_test)
        predictions[model] = [prediction_train.astype(int, copy=False), prediction_test.astype(int, copy=False)]
        try:
            feat_importance['Importance'] = best_models[model].feature_importances_
            feat_importance.sort_values(by='Importance', ascending=False, inplace=True)
            feat_importance.to_csv('output/' + 'feature_importance_' + str(model) + '.csv')
        except AttributeError:
            pass

        row_train = {'Micro_F1': getattr(evaluation_training, 'micro'),
                     'Average_F1': getattr(evaluation_training, 'average'),
                     'Macro_F1': getattr(evaluation_training, 'macro'),
                     'Accuracy': getattr(evaluation_training, 'accuracy'),
                     'ROC_Curve': getattr(evaluation_training, 'roc_auc_curve'),
                     ('Precision_Class_' + str(classes[0])): getattr(evaluation_training, 'precision')[0],
                     ('Precision_Class_' + str(classes[1])): getattr(evaluation_training, 'precision')[1],
                     ('Recall_Class_' + str(classes[0])): getattr(evaluation_training, 'recall')[0],
                     ('Recall_Class_' + str(classes[1])): getattr(evaluation_training, 'recall')[1],
                     'Running_Time': running_times[model]}

        row_test = {'Micro_F1': getattr(evaluation_test, 'micro'),
                    'Average_F1': getattr(evaluation_test, 'average'),
                    'Macro_F1': getattr(evaluation_test, 'macro'),
                    'Accuracy': getattr(evaluation_test, 'accuracy'),
                    'ROC_Curve': getattr(evaluation_test, 'roc_auc_curve'),
                    ('Precision_Class_' + str(classes[0])): getattr(evaluation_test, 'precision')[0],
                    ('Precision_Class_' + str(classes[1])): getattr(evaluation_test, 'precision')[1],
                    ('Recall_Class_' + str(classes[0])): getattr(evaluation_test, 'recall')[0],
                    ('Recall_Class_' + str(classes[1])): getattr(evaluation_test, 'recall')[1],
                    'Running_Time': running_times[model]}

        results_train.append(row_train)
        results_test.append(row_test)

    df_results_train = pd.DataFrame(results_train, index=models)
    df_results_train['Algorithms'] = df_results_train.index
    df_results_train['Dataset'] = ['Train'] * df_results_train.shape[0]
    df_results_test = pd.DataFrame(results_test, index=models)
    df_results_test['Algorithms'] = df_results_test.index
    df_results_test['Dataset'] = ['Test'] * df_results_train.shape[0]

    # df_results_train.to_csv('output/performance_report_wo_versao_protsolar_train.csv')
    # df_results_test.to_csv('output/performance_report_wo_versao_protsolar_test.csv')

    sql_inject(pd.concat([df_results_train, df_results_test]), sql_info['database'], sql_info['performance_algorithm_results'], list(df_results_train), time_to_last_update=0, check_date=1)

    return df_results_train, df_results_test, predictions


def probability_evaluation(models_name, models, train_x, test_x):
    proba_train = models[models_name].predict_proba(train_x)
    proba_test = models[models_name].predict_proba(test_x)

    return proba_train, proba_test


def feature_contribution(df, configuration_parameters):
    # configuration_parameters.remove('Modelo_new')
    configuration_parameters.remove('Modelo')
    boolean_parameters = [x for x in configuration_parameters if list(df[x].unique()) == [0, 1] or list(df[x].unique()) == [1, 0]]
    non_boolean_parameters = [x for x in configuration_parameters if x not in boolean_parameters]
    df_feature_contribution_total = pd.DataFrame()

    # for model in df['Modelo_new'].unique():
    for model in df['Modelo'].unique():
        # model_mask = df['Modelo_new'] == model
        model_mask = df['Modelo'] == model
        df_model = df.loc[df[model_mask].index, :]

        mask_class_1 = df_model['score_class_gt'] == 1
        mask_class_0 = df_model['score_class_gt'] == 0
        class_1 = df_model.loc[df_model[mask_class_1].index, :]
        class_0 = df_model.loc[df_model[mask_class_0].index, :]
        differences_boolean, differences_non_boolean, features_boolean, features_non_boolean = [], [], [], []
        differences_feature, features, model_tag = [], [], []

        for feature in configuration_parameters:
            if feature in boolean_parameters:
                c1_f1 = class_1.loc[class_1[feature] == 1, :].shape[0]
                c1_f0 = class_1.loc[class_1[feature] == 0, :].shape[0]
                c0_f1 = class_0.loc[class_0[feature] == 1, :].shape[0]
                c0_f0 = class_0.loc[class_0[feature] == 0, :].shape[0]
                f1 = c1_f1 + c0_f1
                f0 = c1_f0 + c0_f0

                try:
                    p_c1_f1 = c1_f1 / f1 * 1.
                    p_c1_f0 = c1_f0 / f0 * 1.
                    differences_boolean.append(p_c1_f1 - p_c1_f0)
                    features_boolean.append(feature + '_sim')
                except ZeroDivisionError:
                    continue

            elif feature in non_boolean_parameters:
                for value in df_model[feature].unique():
                    if value == 'outros':
                        continue

                    c1_f1 = class_1.loc[class_1[feature] == value, :].shape[0]
                    c1_f0 = class_1.loc[class_1[feature] != value, :].shape[0]
                    c0_f1 = class_0.loc[class_0[feature] == value, :].shape[0]
                    c0_f0 = class_0.loc[class_0[feature] != value, :].shape[0]

                    # ToDo: There might be cases where only one value for a feature is available if the df is too small (Only Preto as Cor_Interior, e.g.). I should add a try/exception to catch these for the conditions there feature != value

                    f1 = c1_f1 + c0_f1
                    f0 = c1_f0 + c0_f0

                    try:
                        p_c1_f1 = c1_f1 / f1 * 1.
                        p_c1_f0 = c1_f0 / f0 * 1.
                    except ZeroDivisionError:
                        # logging.warning('Insufficient data for feature ' + str(feature) + ' and value ' + str(value) + '.')
                        log_record('Insufficient data for feature ' + str(feature) + ' and value ' + str(value) + '.', sql_info['database'], sql_info['log_record'], flag=1)
                        continue

                    differences_non_boolean.append(p_c1_f1 - p_c1_f0)
                    features_non_boolean.append(feature + '_' + value)

        differences_feature.extend(differences_boolean)
        differences_feature.extend(differences_non_boolean)
        features.extend(features_boolean)
        features.extend(features_non_boolean)
        model_tag.extend([model] * (len(differences_boolean) + len(differences_non_boolean)))

        df_feature_contribution = pd.DataFrame()
        df_feature_contribution['Features'] = features
        df_feature_contribution['Differences'] = differences_feature
        df_feature_contribution['Model_Code'] = model_tag

        if abs(df_feature_contribution['Differences'].min()) > df_feature_contribution['Differences'].max():
            max_range_value = abs(df_feature_contribution['Differences'].min())
            min_range_value = df_feature_contribution['Differences'].min()
        else:
            max_range_value = df_feature_contribution['Differences'].max()
            min_range_value = df_feature_contribution['Differences'].max() * -1
        df_feature_contribution['Differences_Normalized'] = 2 * df_feature_contribution['Differences'] / (max_range_value - min_range_value)

        df_feature_contribution_total = pd.concat([df_feature_contribution_total, df_feature_contribution])

    sql_inject(df_feature_contribution_total, sql_info['database'], sql_info['feature_contribution'], list(df_feature_contribution_total), truncate=1)


def add_new_columns_to_df(df, proba_training, proba_test, predictions, train_x, train_y, test_x, test_y, configuration_parameters):
    train_x['proba_0'] = [x[0] for x in proba_training]
    train_x['proba_1'] = [x[1] for x in proba_training]
    train_x['score_class_gt'] = train_y
    train_x['score_class_pred'] = predictions[0]

    test_x['proba_0'] = [x[0] for x in proba_test]
    test_x['proba_1'] = [x[1] for x in proba_test]
    test_x['score_class_gt'] = test_y
    test_x['score_class_pred'] = predictions[1]

    train_test_datasets = pd.concat([train_x, test_x])
    train_test_datasets.sort_index(inplace=True)

    df['proba_0'] = train_test_datasets['proba_0']
    df['proba_1'] = train_test_datasets['proba_1']
    df['score_class_gt'] = train_test_datasets['score_class_gt'].astype(int, copy=False)
    df['score_class_pred'] = train_test_datasets['score_class_pred']

    df_grouped = df.groupby(configuration_parameters)
    df = df_grouped.apply(additional_info)
    # df_grouped2 = df.groupby(configuration_parameters + ['Local da Venda_new'])
    df_grouped2 = df.groupby(configuration_parameters + ['Local da Venda'])
    df = df_grouped2.apply(additional_info_local)

    return df


def additional_info_local(x):
    x['nr_cars_sold_local'] = len(x)
    x['average_percentage_margin_local'] = x['margem_percentagem'].mean()
    x['average_stock_days_local'] = x['stock_days'].mean()
    x['average_score_local'] = x['score_class_gt'].mean()
    x['average_score_pred_local'] = x['score_class_pred'].mean()
    x['average_score_euros_local'] = x['score_euros'].mean()

    return x


def additional_info(x):
    x['nr_cars_sold'] = len(x)
    x['average_percentage_margin'] = x['margem_percentagem'].mean()
    x['average_stock_days'] = x['stock_days'].mean()
    x['average_score'] = x['score_class_gt'].mean()
    x['average_score_pred'] = x['score_class_pred'].mean()
    x['average_score_euros'] = x['score_euros'].mean()
    return x


def df_decimal_places_rounding(df, dictionary):

    df = df.round(dictionary)

    return df


def model_choice(df_results, metric, threshold):
    step_e_upload_flag = 0

    try:
        # makes sure there are results above minimum threshold
        best_model_name = df_results[df_results.loc[:, metric].gt(threshold)][[metric, 'Running_Time']].idxmax().head(1).values[0]
        best_model_value = df_results[df_results.loc[:, metric].gt(threshold)][[metric, 'Running_Time']].max().head(1).values[0]
        # logging.info('There are values (%.4f' % best_model_value + ') from algorithm ' + str(best_model_name) + ' above minimum threshold (' + str(threshold) + '). Will compare with last result in SQL Server...')
        log_record('There are values (%.4f' % best_model_value + ') from algorithm ' + str(best_model_name) + ' above minimum threshold (' + str(threshold) + '). Will compare with last result in SQL Server...', sql_info['database'], sql_info['log_record'])

        df_previous_performance_results = sql_second_highest_date_checkup(sql_info['database'], sql_info['performance_algorithm_results'])
        if df_previous_performance_results.loc[df_previous_performance_results[metric].gt(best_model_value)].shape[0]:
            # logging.info('Older values have better results in the same metric: %.4f' % df_previous_performance_results.loc[df_previous_performance_results[metric].gt(best_model_value)][metric].max() + ' > %.4f' % best_model_value + ' in model ' + df_previous_performance_results.loc[df_previous_performance_results[metric].gt(best_model_value)][metric].idxmax() + ' so will not upload in section E...')
            log_record('Older values have better results in the same metric: %.4f' % df_previous_performance_results.loc[df_previous_performance_results[metric].gt(best_model_value)][metric].max() + ' > %.4f' % best_model_value + ' in model ' + df_previous_performance_results.loc[df_previous_performance_results[metric].gt(best_model_value)][metric].idxmax() + ' so will not upload in section E...', sql_info['database'], sql_info['log_record'], flag=1)
            model_choice_flag = 1
        else:
            step_e_upload_flag = 1
            # logging.info('New value is: %.4f' % best_model_value + ' and greater than the last value which was: %.4f' % df_previous_performance_results[metric].max() + 'for model' + df_previous_performance_results[metric].idxmax() + 'so will upload in section E...')
            log_record('New value is: %.4f' % best_model_value + ' and greater than the last value which was: %.4f' % df_previous_performance_results[metric].max() + ' for model ' + df_previous_performance_results[metric].idxmax() + 'so will upload in section E...', sql_info['database'], sql_info['log_record'])
            model_choice_flag = 2

    except ValueError:
        # logging.info('No value above minimum threshold (%.4f' % threshold + ') found. Will maintain previous result - No upload in Section E to SQL Server.')
        log_record('No value above minimum threshold (%.4f' % threshold + ') found. Will maintain previous result - No upload in Section E to SQL Server.', sql_info['database'], sql_info['log_record'], flag=1)
        model_choice_flag, best_model_name, best_model_value = 0, 0, 0

    model_choice_message = model_choice_upload(model_choice_flag, best_model_name, best_model_value, metric)

    return model_choice_message, best_model_name, best_model_value, step_e_upload_flag


def model_choice_upload(flag, name, value, metric):
    df_model_result = pd.DataFrame(columns={'Model_Choice_Flag', 'Chosen_Model', 'Metric', 'Value', 'Message'})

    df_model_result['Model_Choice_Flag'] = [flag]
    if not flag:
        message = 'Nenhum dos modelos treinados atinge os valores mínimos definidos.'
        df_model_result['Chosen_Model'] = [0]
        df_model_result['Metric'] = [0]
        df_model_result['Value'] = [0]
        df_model_result['Message'] = [message]
    elif flag == 1:
        message = 'Modelo anterior com melhor performance do que o atual.'
        df_model_result['Chosen_Model'] = [name]
        df_model_result['Metric'] = [metric]
        df_model_result['Value'] = [value]
        df_model_result['Message'] = [message]
    elif flag == 2:
        message = 'Modelo anterior substituído pelo atual.'
        df_model_result['Chosen_Model'] = [name]
        df_model_result['Metric'] = [metric]
        df_model_result['Value'] = [value]
        df_model_result['Message'] = [message]

    sql_inject(df_model_result, sql_info['database'], sql_info['model_choices'], list(df_model_result), time_to_last_update=0, check_date=1)

    return message


def plot_roc_curve(models, models_name, train_x, train_y, test_x, test_y, save_name, save_dir):
    plt.subplots(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
    for model in models_name:
        prob_train_init = models[model].fit(train_x, train_y[list(train_y)[0]]).predict_proba(test_x)
        prob_test_1 = [x[1] for x in prob_train_init]
        fpr, tpr, _ = roc_curve(test_y, prob_test_1, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc + ' ' + str(dict_models_name_conversion[model][0]))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic per Model')
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    save_fig(save_name, save_dir=save_dir)
    plt.clf()
    # plt.close()


def save_fig(name, save_dir='output/'):
    # Saves plot in at least two formats, png and pdf

    if os.path.exists(save_dir + str(name) + '.png'):
        os.remove(save_dir + str(name) + '.png')

    plt.savefig(save_dir + str(name) + '.png')
    plt.savefig(save_dir + str(name) + '.pdf')


def multiprocess_model_evaluation(df, models, train_x, train_y, test_x, test_y, best_models, predictions, configuration_parameters):
    start = time.time()
    workers = pool_workers_count
    pool = multiprocessing.Pool(processes=workers)
    results = pool.map(multiprocess_evaluation, [(df, model_name, train_x, train_y, test_x, test_y, best_models, predictions, configuration_parameters) for model_name in models])
    pool.close()
    df_model_dict = {key: value for (key, value) in results}

    print('D - Total Elapsed time: %f' % (time.time() - start))

    return df_model_dict


def multiprocess_evaluation(args):
    df, model_name, train_x, train_y, test_x, test_y, best_models, predictions, configuration_parameters = args

    start = time.time()
    # logging.info('Evaluating model ' + str(model_name) + ' @ ' + time.strftime("%H:%M:%S @ %d/%m/%y") + '...')
    log_record('Evaluating model ' + str(model_name) + ' @ ' + time.strftime("%H:%M:%S @ %d/%m/%y") + '...', sql_info['database'], sql_info['log_record'])
    train_x_copy, test_x_copy = train_x.copy(deep=True), test_x.copy(deep=True)
    proba_training, proba_test = probability_evaluation(model_name, best_models, train_x_copy, test_x_copy)
    df_model = add_new_columns_to_df(df, proba_training, proba_test, predictions[model_name], train_x_copy, train_y, test_x_copy, test_y, configuration_parameters)
    df_model = df_decimal_places_rounding(df_model, {'proba_0': 2, 'proba_1': 2})
    save_csv([df_model], ['output/' + 'db_final_classification_' + model_name])
    # logging.info(model_name + ' - Elapsed time: %f' % (time.time() - start))
    log_record(model_name + ' - Elapsed time: %f' % (time.time() - start), sql_info['database'], sql_info['log_record'])

    return model_name, df_model
