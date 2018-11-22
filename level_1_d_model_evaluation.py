import pandas as pd
import numpy as np
import logging
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score, silhouette_samples, silhouette_score, mean_squared_error, r2_score, roc_curve, auc, roc_auc_score
from level_1_e_deployment import sql_inject, sql_date_comparison
from level_2_optionals_baviera_options import sql_info
pd.set_option('display.expand_frame_repr', False)

my_dpi = 96

dict_models_name_conversion = {
    'dt': ['Decision Tree'],
    'rf': ['Random Forest'],
    'lr': ['Logistic Regression'],
    'knn': ['KNN'],
    'svm': ['SVM'],
    'ab': ['Adaboost'],
    'gc': ['Gradient'],
    'bayes': ['Bayesian'],
    'ann': ['ANN'],
    'voting': ['Voting']
}


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

    sql_inject(pd.concat([df_results_train, df_results_test]), sql_info['database'], sql_info['performance_algorithm_results'], list(df_results_train), time_to_last_update=0, check_date=1)

    return df_results_train, df_results_test, predictions


def probability_evaluation(models_name, models, train_x, test_x):
    proba_train = models[models_name].predict_proba(train_x)
    proba_test = models[models_name].predict_proba(test_x)

    return proba_train, proba_test


def feature_contribution(df, configuration_parameters):
    configuration_parameters.remove('Modelo_new')
    boolean_parameters = [x for x in configuration_parameters if list(df[x].unique()) == [0, 1] or list(df[x].unique()) == [1, 0]]
    non_boolean_parameters = [x for x in configuration_parameters if x not in boolean_parameters]
    df_feature_contribution_total = pd.DataFrame()

    for model in df['Modelo_new'].unique():
        model_mask = df['Modelo_new'] == model
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
                        print('Insufficient data for feature ' + str(feature) + ' and value ' + str(value) + '.')
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
    df_grouped2 = df.groupby(configuration_parameters + ['Local da Venda_new'])
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

    # try:
    best_model_name = df_results[df_results.loc[:, metric].gt(threshold)][[metric, 'running_time']].idxmax().head(1).values[0]
    best_model_value = df_results[df_results.loc[:, metric].gt(threshold)][[metric, 'running_time']].max().head(1).values[0]
    return best_model_name, best_model_value
    #  ToDo Need to include the try/except conditions again
    #  ToDO Need to add an exit condition when there's no better model/below minimum threshold
    # except ValueError:
    #     logging.info('No models above minimum performance threshold.')


def plot_roc_curve(models, models_name, train_x, train_y, test_x, test_y, save_name, save_dir):
    plt.subplots(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
    for model in models_name:
        prob_train_init = models[model].fit(train_x, train_y).predict_proba(test_x)
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
    plt.close()


def save_fig(name, save_dir='output/'):
    # Saves plot in at least two formats, png and pdf

    if os.path.exists(save_dir + str(name) + '.png'):
        os.remove(save_dir + str(name) + '.png')

    plt.savefig(save_dir + str(name) + '.png')
    plt.savefig(save_dir + str(name) + '.pdf')


def model_comparison(best_model_name, best_model_value, metric):
    # Assumes comparison between the same metrics;

    name = 'place_holder'

    try:
        old_results = pd.read_csv('output/' + name)  # ToDo: The name of the file should reflect the last time it was ran
        if old_results[old_results.loc[:, metric].gt(best_model_value)].shape[0]:
            logging.warning('Previous results are better than current ones - Will maintain.')
            return 0
        else:
            logging.info('Current results are better than previous ones - Will replace.')
            return 1
    except FileNotFoundError:
        logging.info('No previous results found.')
        return 1






