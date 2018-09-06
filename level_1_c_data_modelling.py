import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from gap_statistic import OptimalK
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score, silhouette_samples, silhouette_score, mean_squared_error, r2_score
pd.set_option('display.expand_frame_repr', False)


class ClassFit(object):
    def __init__(self, clf, params=None):
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf()

    def predict(self, x):
        return self.clf.predict(x)

    def grid_search(self, parameters, k, score):
        self.grid = GridSearchCV(estimator=self.clf, param_grid=parameters, cv=k, scoring=score)

    def clf_fit(self, x, y):
        self.grid.fit(x, y)

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


class ClusterFit(object):
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

    def predict(self, x):
        return self.clf.predict(x)

    def silhouette_score_avg(self, x, cluster_clients):
        self.silh_score = silhouette_score(x, cluster_clients)
        print('\nThe Avg. Silhouette Score is: %.3f' % self.silh_score)

    def cluster_clients_counts(self, cluster_clients):
        _, self.clusters_size = np.unique(cluster_clients, return_counts=True)
        print('Client Counts per cluster:', self.clusters_size)

    def sample_silhouette_score(self, x, cluster_clients):
        self.sample_silh_score = silhouette_samples(x, cluster_clients)


class RegFit(object):
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

    def mse_func(self, prediction, groundtruth):
        self.mse = mean_squared_error(groundtruth, prediction)  # Mean Square Error
        return self.mse

    def score_func(self, prediction, groundtruth):
        # return self.clf.score(groundtruth, prediction)  # R^2
        self.score = r2_score(groundtruth, prediction)
        return self.score

    def coefficients(self):
        return self.clf.coef_
