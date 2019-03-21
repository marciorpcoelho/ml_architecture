import pandas as pd
import numpy as np
import datetime
import pickle
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from scipy.spatial.distance import cdist, pdist
from sklearn.model_selection import GridSearchCV
from gap_statistic import OptimalK
from sklearn.preprocessing import StandardScaler
from level_0_performance_report import log_record
from level_2_optionals_baviera_options import classification_models, sql_info, k, gridsearch_score, project_id
pd.set_option('display.expand_frame_repr', False)


class ClassificationTraining(object):
    def __init__(self, clf, params=None):
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf()

    def grid_search(self, parameters, k, score):
        self.grid = GridSearchCV(estimator=self.clf, param_grid=parameters, cv=k, scoring=score, iid=True, n_jobs=-1)

    def clf_fit(self, x, y):
        self.grid.fit(x, y)

    def predict(self, x):
        self.grid.predict(x)

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
        print('\nOptimal number of clusters is: {}'.format(k))
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
        self.grid = GridSearchCV(estimator=self.clf, param_grid=parameters, cv=k, iid=True, scoring=score)

    def clf_grid_fit(self, x, y):
        self.grid.fit(x, y)

    def clf_fit(self, x, y):
        self.clf.fit(x, y)


def model_training(models, train_x, train_y):
    best_models_pre_fit, best_models_pos_fit, predictions, running_times = {}, {}, {}, {}
    clf, classes = None, None

    for model in models:
        log_record('MODEL: {}',format(model), project_id)
        if model == 'voting':
            start = time.time()
            parameters = {'estimators': [(x, y) for x, y in zip(best_models_pre_fit.keys(), best_models_pre_fit.values())]}
            vote_clf = ClassificationTraining(clf=classification_models['voting'][0], params=parameters)
            vote_clf.grid_search(parameters=classification_models['voting'][1], k=k, score=gridsearch_score)
            vote_clf.clf_fit(x=train_x, y=train_y.values.ravel())
            vote_clf_best = vote_clf.grid.best_estimator_
            vote_clf_best.fit(train_x, train_y.values.ravel())
            best_models_pos_fit['voting'] = vote_clf_best
            running_times['voting'] = time.time() - start
        else:
            start = time.time()
            clf = ClassificationTraining(clf=classification_models[model][0])
            clf.grid_search(parameters=classification_models[model][1], k=k, score=gridsearch_score)
            clf.clf_fit(x=train_x, y=train_y.values.ravel())
            clf_best = clf.grid.best_estimator_

            best_models_pre_fit[model] = clf_best
            clf_best.fit(train_x, train_y.values.ravel())
            best_models_pos_fit[model] = clf_best

        running_times[model] = time.time() - start

        classes = clf.grid.classes_

    return classes, best_models_pos_fit, running_times


def clustering_training(df, max_n_clusters):
    print('Testing different numbers of clusters...')

    silhouette_scores, calinski_scores, inertia_scores, elbow_scores, centroids = [], [], [], [], []

    models = [KMeans(n_clusters=n, init='k-means++', max_iter=10, n_init=100, n_jobs=-1).fit(df) for n in range(2, max_n_clusters+1)]

    for m in models:
        print('Evaluating for {} clusters...'.format(len(np.unique(list(m.labels_)))))
        centroids.append(m.cluster_centers_)

        s = silhouette_score(df, m.labels_, random_state=42)
        ch = calinski_harabaz_score(df, m.labels_)
        elbow = between_clusters_distance(df, m.cluster_centers_)

        inertia_scores.append(m.inertia_)
        silhouette_scores.append(s)
        calinski_scores.append(ch)
        elbow_scores.append(elbow)

    # dist = [np.min(cdist(df, c, 'euclidean'), axis=1) for c in centroids]
    # totss = sum(pdist(df) ** 2) / df.shape[0]
    # totwithinss = [sum(d ** 2) for d in dist]
    # between_clusters = (totss - totwithinss) / totss * 100

    scores = [silhouette_scores, calinski_scores, inertia_scores, elbow_scores]
    score_names = ['Silhouette Scores', 'Calinski Scores', 'Inertia Scores', 'Elbow Method']

    return models, scores, score_names


def between_clusters_distance(df, centroid):

    dist = np.min(cdist(df, centroid, 'euclidean'), axis=1)
    totss = sum(pdist(df) ** 2) / df.shape[0]
    totwithinss = sum(dist ** 2)
    between_clusters = (totss - totwithinss) / totss * 100

    return between_clusters


def save_model(clfs, model_name):

    timestamp = str(datetime.datetime.now().day) + '_' + str(datetime.datetime.now().month) + '_' + str(datetime.datetime.now().year)

    i = 0
    for clf in clfs.values():
        file_name = 'models/' + str(model_name[i]) + '_best_' + str(timestamp) + '.sav'
        i += 1

        file_handler = open(file_name, 'wb')
        pickle.dump(clf, file_handler)
        file_handler.close()
