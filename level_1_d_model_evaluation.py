import pandas as pd
import numpy as np
import logging
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score, silhouette_samples, silhouette_score, mean_squared_error, r2_score
pd.set_option('display.expand_frame_repr', False)


class ClassificationEvaluation(object):
    def __init__(self, groundtruth, prediction):
        self.micro = f1_score(groundtruth, prediction, average='micro', labels=np.unique(prediction))
        self.average = f1_score(groundtruth, prediction, average='weighted', labels=np.unique(prediction))
        self.macro = f1_score(groundtruth, prediction, average='macro', labels=np.unique(prediction))
        self.accuracy = accuracy_score(groundtruth, prediction)
        self.precision = precision_score(groundtruth, prediction, average=None)
        self.recall = recall_score(groundtruth, prediction, average=None)
        self.classification_report = classification_report(groundtruth, prediction)


class ClusterEvaluation(object):
    def __init__(self, x, cluster_clients):
        self.silh_score = silhouette_score(x, cluster_clients)
        _, self.clusters_size = np.unique(cluster_clients, return_counts=True)
        self.sample_silh_score = silhouette_samples(x, cluster_clients)


class RegressionEvaluation(object):
    def __init__(self, groundtruth, prediction):
        self.mse = mean_squared_error(groundtruth, prediction)  # Mean Square Error
        self.score = r2_score(groundtruth, prediction)


def performance_evaluation(models, classes, model_predictions, running_times, train_y, test_y):

    results_train, results_test = [], []
    for model in models:
        evaluation_training = ClassificationEvaluation(groundtruth=train_y, prediction=model_predictions[model][0])
        evaluation_test = ClassificationEvaluation(groundtruth=test_y, prediction=model_predictions[model][1])

        row_train = {'micro': getattr(evaluation_training, 'micro'),
                     'average': getattr(evaluation_training, 'average'),
                     'macro': getattr(evaluation_training, 'macro'),
                     'accuracy': getattr(evaluation_training, 'accuracy'),
                     ('precision_class_' + str(classes[0])): getattr(evaluation_training, 'precision')[0],
                     ('precision_class_' + str(classes[1])): getattr(evaluation_training, 'precision')[1],
                     ('recall_class_' + str(classes[0])): getattr(evaluation_training, 'recall')[0],
                     ('recall_class_' + str(classes[1])): getattr(evaluation_training, 'recall')[1],
                     'running_time': running_times[model]}

        row_test = {'micro': getattr(evaluation_test, 'micro'),
                    'average': getattr(evaluation_test, 'average'),
                    'macro': getattr(evaluation_test, 'macro'),
                    'accuracy': getattr(evaluation_test, 'accuracy'),
                    ('precision_class_' + str(classes[0])): getattr(evaluation_test, 'precision')[0],
                    ('precision_class_' + str(classes[1])): getattr(evaluation_test, 'precision')[1],
                    ('recall_class_' + str(classes[0])): getattr(evaluation_test, 'recall')[0],
                    ('recall_class_' + str(classes[1])): getattr(evaluation_test, 'recall')[1],
                    'running_time': running_times[model]}

        results_train.append(row_train)
        results_test.append(row_test)

    df_results_train = pd.DataFrame(results_train, index=models)
    df_results_test = pd.DataFrame(results_test, index=models)

    return df_results_train, df_results_test


def model_choice(df_results, metric, threshold):

    try:
        best_model_name = df_results[df_results.loc[:, metric].gt(threshold)][[metric, 'running_time']].idxmax().head(1).values[0]
        best_model_value = df_results[df_results.loc[:, metric].gt(threshold)][[metric, 'running_time']].max().head(1).values[0]
        return best_model_name, best_model_value
    except ValueError:
        logging.info('No models above minimum performance threshold.')


def model_comparison(best_model_name, best_model_value, metric):
    # Assumes comparison between the same metrics;

    name = 'place_holder'

    try:
        old_results = pd.read_csv('output/' + name)
        if old_results[old_results.loc[:, metric].gt(best_model_value)].shape[0]:
            logging.warning('Previous results are better than current ones - Will maintain.')
        else:
            logging.info('Current results are better than previous ones - Will replace.')
    except FileNotFoundError:
        logging.info('No previous results found.')







