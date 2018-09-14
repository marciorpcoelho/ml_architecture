import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GridSearchCV
from gap_statistic import OptimalK
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score, silhouette_samples, silhouette_score, mean_squared_error, r2_score
from level_2_optionals_baviera_options import classification_models
pd.set_option('display.expand_frame_repr', False)


class ClassificationTraining(object):
    def __init__(self, clf, params=None):
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf()

    def grid_search(self, parameters, k, score):
        self.grid = GridSearchCV(estimator=self.clf, param_grid=parameters, cv=k, scoring=score)

    def clf_fit(self, x, y):
        self.grid.fit(x, y)

    def predict(self, x):
        # return self.clf.predict(x)
        self.grid.predict(x)

    def grid_performance(self, prediction, y):
        self.micro = f1_score(y, prediction, average='micro', labels=np.unique(prediction))
        # self.average = f1_score(y, prediction, average='weighted', labels=np.unique(prediction))
        self.macro = f1_score(y, prediction, average='macro', labels=np.unique(prediction))
        self.accuracy = accuracy_score(y, prediction)
        # self.precision = precision_score(y, prediction, average=None)
        # self.recall = recall_score(y, prediction, average=None)
        self.class_report = classification_report(y, prediction)

    def feature_importance(self):
        return self.clf.feature_importances_


class ClusterTraining(object):
    def __init__(self, clf, params=None):
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf()

    def cluster_optimal_number(self, matrix):
        optimalk = OptimalK(parallel_backend='joblib')
        k = optimalk(matrix, cluster_array=np.arange(1, 30))
        print('\nOptimal number of clusters is:', k)
        self.optimal_cluster_nb = k
        return self.optimal_cluster_nb

    def df_standardization(self, df):
        scaler = StandardScaler()
        scaler.fit(df)
        scaled_matrix = scaler.transform(df)
        self.scaled_matrix = scaled_matrix
        return self.scaled_matrix

    def clf_fit(self, x):
        self.clf.fit(x)

    def labels(self):
        return self.clf.labels_

    def cluster_centers(self):
        return self.clf.cluster_centers_


class RegressionTraining(object):
    def __init__(self, clf, params=None):
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf()

    def predict(self, x):
        return self.clf.predict(x)

    def grid_search(self, parameters, k, score):
        self.grid = GridSearchCV(estimator=self.clf, param_grid=parameters, cv=k, scoring=score)

    def clf_grid_fit(self, x, y):
        self.grid.fit(x, y)

    def clf_fit(self, x, y):
        self.clf.fit(x, y)


def model_training(models, train_x, train_y, test_x, k, score, voting=0):

    print(train_x.head())
    print()
    print(test_x.head())

    predictions, running_times = {}, {}
    for model in models:
        start = time.time()
        clf = ClassificationTraining(clf=classification_models[model][0])
        clf.grid_search(parameters=classification_models[model][1], k=k, score=score)
        clf.clf_fit(x=train_x, y=train_y.values.ravel())
        clf_best = clf.grid.best_estimator_

        if not voting:
            clf_best.fit(train_x, train_y.values.ravel())
            prediction_trainer, prediction_test = clf_best.predict(train_x), clf_best.predict(test_x)
            predictions[model] = [prediction_trainer, prediction_test]
            running_times[model] = time.time() - start
        elif voting:
            predictions[model] = clf_best

        classes = clf.grid.classes_

    return predictions, classes, running_times
