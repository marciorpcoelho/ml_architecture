import os
import time
import itertools
import numpy as np
from math import pi
import pandas as pd
import seaborn as sns
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import modules.level_1_e_deployment as level_1_e_deployment
import modules.level_0_performance_report as level_0_performance_report

from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score, silhouette_samples, silhouette_score, mean_squared_error, r2_score, roc_curve, auc, roc_auc_score
pd.set_option('display.expand_frame_repr', False)
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))

my_dpi = 96


class ClassificationEvaluation(object):
    def __init__(self, groundtruth, prediction):
        self.micro = f1_score(groundtruth, np.round(prediction), average='micro', labels=np.unique(np.round(prediction)))
        self.average = f1_score(groundtruth, np.round(prediction), average='weighted', labels=np.unique(np.round(prediction)))
        self.macro = f1_score(groundtruth, np.round(prediction), average='macro', labels=np.unique(np.round(prediction)))
        self.accuracy = accuracy_score(groundtruth, np.round(prediction))
        self.precision = precision_score(groundtruth, np.round(prediction), average=None)
        self.recall = recall_score(groundtruth, np.round(prediction), average=None)
        self.classification_report = classification_report(groundtruth, np.round(prediction))
        # self.roc_auc_curve = roc_auc_score(groundtruth, np.round(prediction))  # Note: this implementation is restricted to the binary classification task or multilabel classification task in label indicator format.

        self.precision_multiclass_micro = precision_score(groundtruth, np.round(prediction), average='micro')
        self.precision_multiclass_macro = precision_score(groundtruth, np.round(prediction), average='macro')
        self.precision_multiclass_average = precision_score(groundtruth, np.round(prediction), average='weighted')
        self.recall_multiclass_micro = recall_score(groundtruth, np.round(prediction), average='micro')
        self.recall_multiclass_macro = recall_score(groundtruth, np.round(prediction), average='macro')
        self.recall_multiclass_average = recall_score(groundtruth, np.round(prediction), average='weighted')


class ClusterEvaluation(object):
    def __init__(self, x, cluster_clients):
        self.silh_score = silhouette_score(x, cluster_clients)
        _, self.clusters_size = np.unique(cluster_clients, return_counts=True)
        self.sample_silh_score = silhouette_samples(x, cluster_clients)


class RegressionEvaluation(object):
    def __init__(self, groundtruth, prediction):
        self.mse = mean_squared_error(groundtruth, prediction)  # Mean Square Error
        self.r2_score = r2_score(groundtruth, prediction)


def performance_evaluation_classification(models, best_models, running_times, datasets, options_file, project_id):
    # models -> list with models names;
    # best_models -> dict with models name as key and the best clf after gridsearch as values;
    # classes -> clf.classes_
    # running_times -> dict with models name as key and the training time as values
    # datasets -> dict with the datasets required - train_x, test_x, train_y, test_y

    results_train, results_test = [], []
    predictions, feat_importance = {}, pd.DataFrame(index=list(datasets['train_x']), columns={'Importance'})
    for model in models:
        prediction_train = best_models[model].predict(datasets['train_x'])
        prediction_test = best_models[model].predict(datasets['test_x'])
        evaluation_training = ClassificationEvaluation(groundtruth=datasets['train_y'], prediction=prediction_train)
        evaluation_test = ClassificationEvaluation(groundtruth=datasets['test_y'], prediction=prediction_test)
        predictions[model] = [prediction_train.astype(int, copy=False), prediction_test.astype(int, copy=False)]

        # plot_conf_matrix(datasets['train_y'], prediction_train, classes, model, project_id)
        # plot_conf_matrix(datasets['test_y'], prediction_test, classes, model, project_id)
        try:
            feat_importance['Importance'] = best_models[model].feature_importances_
            feat_importance.sort_values(by='Importance', ascending=False, inplace=True)
            feat_importance.to_csv(base_path + '/output/' + 'feature_importance_' + str(model) + '.csv')
        except AttributeError:
            pass

        row_train = {'Micro_F1': getattr(evaluation_training, 'micro'),
                     'Average_F1': getattr(evaluation_training, 'average'),
                     'Macro_F1': getattr(evaluation_training, 'macro'),
                     'Accuracy': getattr(evaluation_training, 'accuracy'),
                     'ROC_Curve': getattr(evaluation_training, 'roc_auc_curve'),
                     # ('Precision_Class_' + str(classes[0])): getattr(evaluation_training, 'precision')[0],
                     ('Precision_Class_' + str(best_models[model].classes_[0])): getattr(evaluation_training, 'precision')[0],
                     ('Precision_Class_' + str(best_models[model].classes_[1])): getattr(evaluation_training, 'precision')[1],
                     ('Recall_Class_' + str(best_models[model].classes_[0])): getattr(evaluation_training, 'recall')[0],
                     ('Recall_Class_' + str(best_models[model].classes_[1])): getattr(evaluation_training, 'recall')[1],
                     # 'Precision_Micro_Class_' + str(classes[0]): getattr(evaluation_training, 'precision_multiclass_micro')[0],
                     # 'Precision_Micro_Class_' + str(classes[1]): getattr(evaluation_training, 'precision_multiclass_micro')[1],
                     # 'Precision_Macro_Class_' + str(classes[0]): getattr(evaluation_training, 'precision_multiclass_macro')[0],
                     # 'Precision_Macro_Class_' + str(classes[1]): getattr(evaluation_training, 'precision_multiclass_macro')[1],
                     # 'Precision_Average_Class_' + str(classes[0]): getattr(evaluation_training, 'precision_multiclass_average')[0],
                     # 'Precision_Average_Class_' + str(classes[1]): getattr(evaluation_training, 'precision_multiclass_average')[1],
                     #
                     # 'Recall_Micro_Class_' + str(classes[0]): getattr(evaluation_training, 'recall_multiclass_micro')[0],
                     # 'Recall_Micro_Class_' + str(classes[1]): getattr(evaluation_training, 'recall_multiclass_micro')[1],
                     # 'Recall_Macro_Class_' + str(classes[0]): getattr(evaluation_training, 'recall_multiclass_macro')[0],
                     # 'Recall_Macro_Class_' + str(classes[1]): getattr(evaluation_training, 'recall_multiclass_macro')[1],
                     # 'Recall_Average_Class_' + str(classes[0]): getattr(evaluation_training, 'recall_multiclass_average')[0],
                     # 'Recall_Average_Class_' + str(classes[1]): getattr(evaluation_training, 'recall_multiclass_average')[1],
                     'Running_Time': running_times[model]}

        row_test = {'Micro_F1': getattr(evaluation_test, 'micro'),
                    'Average_F1': getattr(evaluation_test, 'average'),
                    'Macro_F1': getattr(evaluation_test, 'macro'),
                    'Accuracy': getattr(evaluation_test, 'accuracy'),
                    'ROC_Curve': getattr(evaluation_test, 'roc_auc_curve'),
                    ('Precision_Class_' + str(best_models[model].classes_[0])): getattr(evaluation_test, 'precision')[0],
                    ('Precision_Class_' + str(best_models[model].classes_[1])): getattr(evaluation_test, 'precision')[1],
                    ('Recall_Class_' + str(best_models[model].classes_[0])): getattr(evaluation_test, 'recall')[0],
                    ('Recall_Class_' + str(best_models[model].classes_[1])): getattr(evaluation_test, 'recall')[1],
                    # 'Precision_Micro_Class_' + str(classes[0]): getattr(evaluation_test, 'precision_multiclass_micro')[0],
                    # 'Precision_Micro_Class_' + str(classes[1]): getattr(evaluation_test, 'precision_multiclass_micro')[1],
                    # 'Precision_Macro_Class_' + str(classes[0]): getattr(evaluation_test, 'precision_multiclass_macro')[0],
                    # 'Precision_Macro_Class_' + str(classes[1]): getattr(evaluation_test, 'precision_multiclass_macro')[1],
                    # 'Precision_Average_Class_' + str(classes[0]): getattr(evaluation_test, 'precision_multiclass_average')[0],
                    # 'Precision_Average_Class_' + str(classes[1]): getattr(evaluation_test, 'precision_multiclass_average')[1],
                    #
                    # 'Recall_Micro_Class_' + str(classes[0]): getattr(evaluation_test, 'recall_multiclass_micro')[0],
                    # 'Recall_Micro_Class_' + str(classes[1]): getattr(evaluation_test, 'recall_multiclass_micro')[1],
                    # 'Recall_Macro_Class_' + str(classes[0]): getattr(evaluation_test, 'recall_multiclass_macro')[0],
                    # 'Recall_Macro_Class_' + str(classes[1]): getattr(evaluation_test, 'recall_multiclass_macro')[1],
                    # 'Recall_Average_Class_' + str(classes[0]): getattr(evaluation_test, 'recall_multiclass_average')[0],
                    # 'Recall_Average_Class_' + str(classes[1]): getattr(evaluation_test, 'recall_multiclass_average')[1],
                    'Running_Time': running_times[model]}

        results_train.append(row_train)
        results_test.append(row_test)

    df_results_train = pd.DataFrame(results_train, index=models)
    df_results_train['Algorithms'] = df_results_train.index
    df_results_train['Dataset'] = ['Train'] * df_results_train.shape[0]
    df_results_train['Project_Id'] = [project_id] * df_results_train.shape[0]
    df_results_test = pd.DataFrame(results_test, index=models)
    df_results_test['Algorithms'] = df_results_test.index
    df_results_test['Dataset'] = ['Test'] * df_results_train.shape[0]
    df_results_test['Project_Id'] = [project_id] * df_results_train.shape[0]

    metric_bar_plot(df_results_train, 'project_{}_train_dataset'.format(project_id))
    metric_bar_plot(df_results_test, 'project_{}_test_dataset'.format(project_id))

    model_performance_saving(pd.concat([df_results_train, df_results_test]), options_file)
    level_1_e_deployment.sql_inject(pd.concat([df_results_train, df_results_test]), level_0_performance_report.performance_sql_info['DSN'], level_0_performance_report.performance_sql_info['DB'], level_0_performance_report.performance_sql_info['performance_algorithm_results'], options_file, list(df_results_train), check_date=1)

    return df_results_train, df_results_test, predictions


def algorithms_performance_dataset_creation(results_list, dataset_type_str, models_name_list, project_id):

    df_results = pd.DataFrame(results_list.items()).set_index([0])
    df_results = df_results.transpose()
    df_results['Algorithms'] = [models_name_list] * df_results.shape[0]
    df_results['Project_Id'] = [project_id] * df_results.shape[0]
    df_results['Dataset'] = [dataset_type_str] * df_results.shape[0]

    return df_results


def model_performance_saving(df, options_file):

    level_1_e_deployment.sql_inject(df, level_0_performance_report.DSN_MLG_PRD, level_0_performance_report.performance_sql_info['DB'], level_0_performance_report.performance_sql_info['performance_algorithm_results'], options_file, list(df), check_date=1)

    return


def performance_evaluation_regression(models, best_models, running_times, datasets, datasets_non_ohe, options_file, project_id):

    results_train, results_test = [], []
    predictions, feat_importance = {}, pd.DataFrame(index=list(datasets['train_x']), columns={'Importance'})
    for model in models:
        if model == 'lgb':
            train_x, test_x = datasets_non_ohe['train_x'], datasets_non_ohe['test_x']
            train_y, test_y = datasets_non_ohe['train_y'], datasets_non_ohe['test_y']
        else:
            train_x, test_x = datasets['train_x'], datasets['test_x']
            train_y, test_y = datasets['train_y'], datasets['test_y']

        prediction_train = best_models[model].predict(train_x)
        prediction_test = best_models[model].predict(test_x)
        evaluation_training = RegressionEvaluation(groundtruth=train_y, prediction=prediction_train)
        evaluation_test = RegressionEvaluation(groundtruth=test_y, prediction=prediction_test)
        predictions[model] = [prediction_train.astype(int, copy=False), prediction_test.astype(int, copy=False)]

        # try:
        #     feat_importance['Importance'] = best_models[model].feature_importances_
        #     feat_importance.sort_values(by='Importance', ascending=False, inplace=True)
        #     feat_importance.to_csv(base_path + '/output/' + 'feature_importance_' + str(model) + '.csv')
        # except AttributeError:
        #     pass

        row_train = {'R2': getattr(evaluation_training, 'r2_score'),
                     'MSE': getattr(evaluation_training, 'mse'),
                     'RMSE': np.sqrt(getattr(evaluation_training, 'mse')),
                     'Running_Time': running_times[model]}

        row_test = {'R2': getattr(evaluation_test, 'r2_score'),
                    'MSE': getattr(evaluation_test, 'mse'),
                    'RMSE': np.sqrt(getattr(evaluation_test, 'mse')),
                    'Running_Time': running_times[model]}

        results_train.append(row_train)
        results_test.append(row_test)

    df_results_train = pd.DataFrame(results_train, index=models)
    df_results_train['Algorithms'] = df_results_train.index
    df_results_train['Dataset'] = ['Train'] * df_results_train.shape[0]
    df_results_train['Project_Id'] = [project_id] * df_results_train.shape[0]
    df_results_test = pd.DataFrame(results_test, index=models)
    df_results_test['Algorithms'] = df_results_test.index
    df_results_test['Dataset'] = ['Test'] * df_results_train.shape[0]
    df_results_test['Project_Id'] = [project_id] * df_results_train.shape[0]

    # metric_bar_plot(df_results_train, 'project_{}_train_dataset'.format(project_id))
    # metric_bar_plot(df_results_test, 'project_{}_test_dataset'.format(project_id))

    level_1_e_deployment.sql_inject(pd.concat([df_results_train, df_results_test]), level_0_performance_report.performance_sql_info['DSN'], level_0_performance_report.performance_sql_info['DB'], level_0_performance_report.performance_sql_info['performance_algorithm_results'], options_file, list(df_results_train), check_date=1)

    return df_results_train, df_results_test, predictions


def probability_evaluation(models_name, models, train_x, test_x):
    probabilities = {'proba_train': models[models_name].predict_proba(train_x), 'proba_test': models[models_name].predict_proba(test_x)}

    return probabilities


def feature_contribution(df, configuration_parameters, col_to_group_by, options_file, project_id):
    configuration_parameters.remove(col_to_group_by)

    boolean_parameters = [x for x in configuration_parameters if list(df[x].unique()) == [0, 1] or list(df[x].unique()) == [1, 0]]
    non_boolean_parameters = [x for x in configuration_parameters if x not in boolean_parameters]
    df_feature_contribution_total = pd.DataFrame()

    for model in df[col_to_group_by].unique():
        model_mask = df[col_to_group_by] == model
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
                        # log_record('Insufficient data for feature ' + str(feature) + ' and value ' + str(value) + '.', project_id, flag=1)
                        level_0_performance_report.log_record('Dados insuficientes para a feature {} com valor {}.'.format(feature, value), project_id, flag=1)

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

    level_1_e_deployment.sql_inject(df_feature_contribution_total, options_file.DSN_MLG_PRD, options_file.sql_info['database_final'], options_file.sql_info['feature_contribution'], options_file, list(df_feature_contribution_total), truncate=1)


def add_new_columns_to_df(df, probabilities, predictions, datasets, configuration_parameters, oversample_check, project_id):

    if oversample_check:
        train_x = datasets['train_x_oversampled_original']
    else:
        train_x = datasets['train_x']

    test_x = datasets['test_x']

    train_x['proba_0'] = [x[0] for x in probabilities['proba_train']]
    train_x['proba_1'] = [x[1] for x in probabilities['proba_train']]
    train_x['score_class_gt'] = datasets['train_y']
    train_x['score_class_pred'] = predictions[0]

    # Train_x oversample fix:
    if oversample_check:
        train_x.drop_duplicates(subset='original_index', keep='first', inplace=True)

    test_x['proba_0'] = [x[0] for x in probabilities['proba_test']]
    test_x['proba_1'] = [x[1] for x in probabilities['proba_test']]
    test_x['score_class_gt'] = datasets['test_y']
    test_x['score_class_pred'] = predictions[1]

    train_test_datasets = pd.concat([train_x, test_x])
    train_test_datasets.sort_index(inplace=True)

    df['proba_0'] = train_test_datasets['proba_0']
    df['proba_1'] = train_test_datasets['proba_1']
    df['score_class_gt'] = train_test_datasets['score_class_gt'].astype(int, copy=False)
    df['score_class_pred'] = train_test_datasets['score_class_pred']

    if project_id == 2162:
        df_grouped = df.groupby(configuration_parameters)
        df = df_grouped.apply(additional_info, ('',))
        df_grouped2 = df.groupby(configuration_parameters + ['Local da Venda'])
        # df = df_grouped2.apply(additional_info_local)
        df = df_grouped2.apply(additional_info, ('_local',))
        df_grouped3 = df.groupby(configuration_parameters + ['Local da Venda_v2'])
        df = df_grouped3.apply(additional_info, ('_local_v2',))

    return df


def additional_info(x, tag):
    x['nr_cars_sold' + str(tag[0])] = len(x)
    x['average_percentage_margin' + str(tag[0])] = x['margem_percentagem'].mean()
    x['average_stock_days' + str(tag[0])] = x['stock_days'].mean()
    x['average_score' + str(tag[0])] = x['score_class_gt'].mean()
    x['average_score_pred' + str(tag[0])] = x['score_class_pred'].mean()
    x['average_score_euros' + str(tag[0])] = x['score_euros'].mean()
    return x


def additional_info_temp(x, tag):
    # This function is very similar to additional_info with exception to average_score_pred, as this function was created to deal with a non-trained approach;

    x['nr_cars_sold' + str(tag[0])] = len(x)
    x['average_percentage_margin' + str(tag[0])] = x['margem_percentagem'].mean().round(4)
    x['average_stock_days' + str(tag[0])] = x['stock_days'].mean().round(4)
    x['average_score' + str(tag[0])] = x['score_class_gt'].mean().round(4)
    x['average_score_euros' + str(tag[0])] = x['score_euros'].mean().round(4)
    return x


def df_decimal_places_rounding(df, dictionary):

    df = df.round(dictionary)

    return df


def heatmap_correlation_function(df, target_col, heatmap_name):

    df = df[[x for x in list(df) if x != target_col] + [target_col]]  # Changes the order of the df columns, so its easier to identify the target column in the heat map
    corr = df.corr()

    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    save_fig(heatmap_name)
    # plt.show()


def model_choice(dsn, options_file, df_results):
    step_e_upload_flag = 0
    performance_threshold_interval = 0.02

    try:
        # makes sure there are results above minimum threshold
        best_model_name = df_results[df_results.loc[:, options_file.metric].gt(options_file.metric_threshold)][[options_file.metric, 'Running_Time']].idxmax().head(1).values[0]
        best_model_value = df_results[df_results.loc[:, options_file.metric].gt(options_file.metric_threshold)][[options_file.metric, 'Running_Time']].max().head(1).values[0]
        level_0_performance_report.log_record('Existem resultados ({:.4f}) do algoritmo {} superiores ao limite mínimo ({}). A comparar com os últimos resultados guardados...'.format(best_model_value, best_model_name, options_file.metric_threshold), options_file.project_id)

        df_previous_performance_results = level_1_e_deployment.sql_second_highest_date_checkup(dsn, options_file, level_0_performance_report.performance_sql_info['DB'], level_0_performance_report.performance_sql_info['performance_algorithm_results'])
        if df_previous_performance_results.loc[df_previous_performance_results[options_file.metric].gt(best_model_value)].shape[0]:
            level_0_performance_report.log_record('Resultados mais antigos são melhores na métrica em avaliação: {:.4f} > {:.4f} para o algoritmo {}.'.format(df_previous_performance_results.loc[df_previous_performance_results[options_file.metric].gt(best_model_value)][options_file.metric].max(), best_model_value, df_previous_performance_results.loc[df_previous_performance_results[options_file.metric].gt(best_model_value)][options_file.metric].idxmax()), options_file.project_id, flag=1)

            model_choice_flag = 1
            if df_previous_performance_results.loc[df_previous_performance_results[options_file.metric].gt(best_model_value)][options_file.metric].max() - best_model_value < performance_threshold_interval:
                level_0_performance_report.log_record('Apesar de os resultados mais antigos serem melhores, a diferença é demasiado pequena (<2%). Como tal, os dados serão atualizados de acordo com os novos resultados.', options_file.project_id)
                step_e_upload_flag = 1
                model_choice_flag = 3
        elif df_previous_performance_results.loc[df_previous_performance_results[options_file.metric].eq(best_model_value)].shape[0]:
            level_0_performance_report.log_record('Os novos resultados são iguais aos resultados antigos ({:.4f}) na métrica em avaliação. Os dados serão atualizados em SQL de forma a garantir os dados mais recentes.'.format(best_model_value), options_file.project_id)
            step_e_upload_flag = 1
            model_choice_flag = 4
        else:
            step_e_upload_flag = 1
            try:
                level_0_performance_report.log_record('O novo resultado é: {:.4f} e é superior ao último resultado que era {:.4f} para o modelo {}. Como tal, os dados serão atualizados...'.format(best_model_value, df_previous_performance_results[options_file.metric].max(), df_previous_performance_results[options_file.metric].idxmax()), options_file.project_id)
            except TypeError:
                level_0_performance_report.log_record('O novo resultado é: {:.4f} e não foram encontrados resultados mais antigos. Como tal, os dados serão atualizados...'.format(best_model_value), options_file.project_id)
            model_choice_flag = 2

    except ValueError:
        level_0_performance_report.log_record('Sem resultados superiores ao limite mínimo ({:.4f}). Os últimos dados serão mantidos - Não haverá update dos mesmos.'.format(options_file.metric_threshold), options_file.project_id, flag=1)

        model_choice_flag, best_model_name, best_model_value = 0, 0, 0

    model_choice_message = model_choice_upload(model_choice_flag, best_model_name, best_model_value, options_file)

    return model_choice_message, best_model_name, best_model_value, step_e_upload_flag


def model_choice_upload(flag, name, value, options_file):
    df_model_result = pd.DataFrame(columns={'Model_Choice_Flag', 'Chosen_Model', 'Metric', 'Value', 'Message'})
    message = None

    df_model_result['Model_Choice_Flag'] = [flag]
    df_model_result['Project_Id'] = [options_file.project_id]
    if not flag:
        message = 'Nenhum dos modelos treinados atinge os valores mínimos definidos.'
        df_model_result['Chosen_Model'] = [0]
        df_model_result['Metric'] = [0]
        df_model_result['Value'] = [0]
    elif flag:
        if flag == 1:
            message = 'Modelo anterior com melhor performance do que o atual.'
        if flag == 2:
            message = 'Modelo anterior substituído pelo atual.'
        if flag == 3:
            message = 'Modelo anterior substituído pelo atual, com pequenas variações de performance.'
        if flag == 4:
            message = 'Novo modelo com performance igual ao anterior.'
        df_model_result['Chosen_Model'] = [name]
        df_model_result['Metric'] = [options_file.metric]
        df_model_result['Value'] = [value]
    df_model_result['Message'] = [message]
    level_1_e_deployment.sql_inject(df_model_result, options_file.DSN_MLG_PRD, level_0_performance_report.performance_sql_info['DB'], level_0_performance_report.performance_sql_info['model_choices'], options_file, list(df_model_result), check_date=1)

    return message


def plot_roc_curve(models, models_name, datasets, save_name):
    plt.subplots(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)

    for model in models_name:
        prob_train_init = models[model].fit(datasets['train_x'], np.ravel(datasets['train_y'])).predict_proba(datasets['test_x'])
        prob_test_1 = [x[1] for x in prob_train_init]
        fpr, tpr, _ = roc_curve(datasets['test_y'], prob_test_1, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc + ' ' + str(level_0_performance_report.dict_models_name_conversion[model][0]))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic per Model')
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    save_fig(save_name)
    plt.clf()


def save_fig(name, save_dir='plots/'):
    # Saves plot in at least two formats, .png and .pdf

    if os.path.exists(save_dir + str(name) + '.png'):
        os.remove(save_dir + str(name) + '.png')  # This is needed as png format is not overwritten

    plt.savefig(save_dir + str(name) + '.png')
    plt.savefig(save_dir + str(name) + '.pdf')


def multiprocess_model_evaluation(df, models, datasets, best_models, predictions, configuration_parameters, oversample_check, project_id):
    start = time.time()
    workers = level_0_performance_report.pool_workers_count
    pool = multiprocessing.Pool(processes=workers)
    results = pool.map(multiprocess_evaluation, [(df, model_name, datasets, best_models, predictions, configuration_parameters, oversample_check, project_id) for model_name in models])
    pool.close()
    df_model_dict = {key: value for (key, value) in results}

    print('D - Total Elapsed time: {:.2f}'.format(time.time() - start))

    return df_model_dict


def multiprocess_evaluation(args):
    df, model_name, datasets, best_models, predictions, configuration_parameters, oversample_check, project_id = args
    level_0_performance_report.log_record('A avaliar o modelo {} @ {}...'.format(level_0_performance_report.dict_models_name_conversion[model_name][0], time.strftime("%H:%M:%S @ %d/%m/%y")), project_id)

    train_x_copy, test_x_copy = datasets['train_x'].copy(deep=True), datasets['test_x'].copy(deep=True)

    probabilities = probability_evaluation(model_name, best_models, train_x_copy, test_x_copy)
    df_model = add_new_columns_to_df(df, probabilities, predictions[model_name], datasets, configuration_parameters, oversample_check, project_id)
    df_model = df_decimal_places_rounding(df_model, {'proba_0': 2, 'proba_1': 2})

    level_1_e_deployment.save_csv([df_model], [base_path + '/output/' + 'db_final_classification_' + model_name])

    level_0_performance_report.log_record('Modelo {} terminou @ {}'.format(level_0_performance_report.dict_models_name_conversion[model_name][0], time.strftime("%H:%M:%S @ %d/%m/%y")), project_id)
    return model_name, df_model


def cluster_metrics_plots(number_of_models_trained, scores, score_names):

    fig, ax = plt.subplots(2, 2, figsize=(18, 12))
    plt.setp(ax, xticks=range(0, number_of_models_trained), xticklabels=range(2, number_of_models_trained + 2))

    ax[0, 0].plot(scores[0])
    ax[0, 0].set_title(score_names[0])
    ax[0, 0].grid()

    ax[1, 0].plot(scores[1])
    ax[1, 0].set_title(score_names[1])
    ax[1, 0].grid()

    ax[0, 1].plot(scores[2])
    ax[0, 1].set_title(score_names[2])
    ax[0, 1].grid()

    ax[1, 1].plot(scores[3])
    ax[1, 1].set_title(score_names[3])
    ax[1, 1].grid()

    plt.show()


def make_spider(df, row, title, color):
    categories = list(df)
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(2, 2, row + 1, polar=True, )

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # plt.xticks(angles[:-1], categories, color='grey', size=8)
    # plt.xticks(angles[:-1], list(string.ascii_uppercase)[:N], color='grey', size=8)
    plt.xticks(angles[:-1], list(range(0, N)), color='grey', size=8)

    ax.set_rlabel_position(0)
    # plt.yticks([0.1, 0.5, 0.9], ["0.1", "0.5", "0.9"], color="grey", size=7)
    # plt.ylim(0, 1)

    values = df.loc[row].values.flatten().tolist()
    values += values[:1]
    # print(categories)
    # print(angles)
    print(values)
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    plt.title(title, size=11, color=color, y=1.1)


def radial_chart_preprocess(df, model):
    df_cluster_center = pd.DataFrame(index=np.unique(model.labels_), columns=list(df)[:-1])
    for label in np.unique(model.labels_):
        df_cluster_center.loc[label, :] = model.cluster_centers_[label]

    return df_cluster_center


def metric_bar_plot(df, tag):
    algorithms = df['Algorithms'].values
    algorithms_count = len(algorithms)

    i, j = 0, 0
    metrics_available = ['Micro_F1', 'Average_F1', 'Macro_F1', 'Accuracy', 'ROC_Curve', 'Precision_Class_0', 'Precision_Class_1', 'Recall_Class_0', 'Recall_Class_1', 'Running_Time']

    fig, ax = plt.subplots(2, 5, figsize=(1400 / my_dpi, 1000 / my_dpi), dpi=my_dpi)

    for metric in metrics_available:
        ax[j, i].bar(range(0, algorithms_count), df[metric].values)
        ax[j, i].set_title(metric)
        ax[j, i].grid()
        plt.setp(ax[j, i], xticks=range(0, 5), xticklabels=algorithms)
        if metric != 'Running_Time':
            ax[j, i].set_ylim(0, 1.01)
        k = 0
        for value in df[metric].values:
            ax[j, i].text(k - 0.45, round(value, 2) + 0.01, '{:.2f}'.format(value), color='red')
            k += 1

        i += 1
        if i == 5:
            i = 0
            j += 1

    plt.tight_layout()
    save_fig('bar_plot_' + tag)
    # plt.show()


def plot_conf_matrix(groundtruth, prediction, classes, model, project_id):

    cm = confusion_matrix(groundtruth, prediction)

    # w/o normalization
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title('Confusion matrix - ' + str(type))
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=70)
    # plt.yticks(tick_marks, classes)
    #
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    # 	plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    #
    # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')

    # plt.savefig(directory + 'confusion_matrix_' + str(type) + '_' + str(method) + '_Set_' + str(set) + '_' + str(class_weight) + '_Fold=' + '_' + identifier + '.png')
    # plt.show()
    # plt.clf()
    # plt.close()

    # w/ normalization
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix - ' + str(type))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=70)
    plt.yticks(tick_marks, classes)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    save_fig('project_{}_confusion_matrix_{}'.format(project_id, model))
    # plt.show()
    plt.clf()
    plt.close()


def data_grouping_by_locals_temp(df, configuration_parameters, project_id):
    df['score_class_gt'] = df['new_score']

    df_grouped = df.groupby(configuration_parameters)
    df = df_grouped.apply(additional_info_temp, ('',))

    df_grouped2 = df.groupby(configuration_parameters + ['Local da Venda'])
    df = df_grouped2.apply(additional_info_temp, ('_local',))

    if project_id == 2162:
        df_grouped3 = df.groupby(configuration_parameters + ['Local da Venda_v2'])
        df = df_grouped3.apply(additional_info_temp, ('_local_v2',))

        # df_grouped4 = df.groupby(configuration_parameters + ['Local da Venda_Fase2_level_1'])
        # df = df_grouped4.apply(additional_info_temp, ('_local_Fase2_level_1',))

        df_grouped5 = df.groupby(configuration_parameters + ['Local da Venda_Fase2_level_2'])
        df = df_grouped5.apply(additional_info_temp, ('_local_Fase2_level_2',))

        df.to_csv(base_path + '/output/bmw_dataset.csv')

    return 'N/A', df, df.shape[0]


def update_labels(df, new_df, key_col, target_col):

    left_a = df.set_index(key_col)
    right_a = new_df.set_index(key_col)
    res = left_a.reindex(columns=left_a.columns.union(right_a.columns))
    res.update(right_a)
    res.reset_index(inplace=True)
    res['Classification_Flag'] = np.where(res['prediction'].isnull(), 0, 1)
    res['prediction'].fillna(res[target_col], inplace=True)
    res.rename(index=str, columns={target_col: 'old_{}'.format(target_col)}, inplace=True)
    res.drop(['old_{}'.format(target_col)], axis=1, inplace=True)
    res.rename(index=str, columns={'prediction': target_col}, inplace=True)

    return res
