import pandas as pd
import numpy as np
import datetime
import pickle
import time
import logging
from sklearn.model_selection import GridSearchCV
from gap_statistic import OptimalK
from sklearn.preprocessing import StandardScaler
from level_2_optionals_baviera_options import classification_models
pd.set_option('display.expand_frame_repr', False)


class ClassificationTraining(object):
    def __init__(self, clf, params=None):
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf()

    def grid_search(self, parameters, k, score):
        self.grid = GridSearchCV(estimator=self.clf, param_grid=parameters, cv=k, scoring=score, n_jobs=4)

    def clf_fit(self, x, y):
        self.grid.fit(x, y)

    def predict(self, x):
        self.grid.predict(x)

    # ToDo: Remove following comments?
    # def grid_performance(self, prediction, y):
    #     self.micro = f1_score(y, prediction, average='micro', labels=np.unique(prediction))
    #     # self.average = f1_score(y, prediction, average='weighted', labels=np.unique(prediction))
    #     self.macro = f1_score(y, prediction, average='macro', labels=np.unique(prediction))
    #     self.accuracy = accuracy_score(y, prediction)
    #     # self.precision = precision_score(y, prediction, average=None)
    #     # self.recall = recall_score(y, prediction, average=None)
    #     self.class_report = classification_report(y, prediction)

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


def model_training(models, train_x, train_y, k, score):
    best_models_pre_fit, best_models_pos_fit, predictions, running_times = {}, {}, {}, {}

    for model in models:
        logging.info('MODEL:', model)
        if model != 'voting':
            start = time.time()
            clf = ClassificationTraining(clf=classification_models[model][0])
            clf.grid_search(parameters=classification_models[model][1], k=k, score=score)
            clf.clf_fit(x=train_x, y=train_y.values.ravel())
            clf_best = clf.grid.best_estimator_

            best_models_pre_fit[model] = clf_best
            clf_best.fit(train_x, train_y.values.ravel())
            best_models_pos_fit[model] = clf_best

        if model == 'voting':
            start = time.time()
            parameters = {'estimators': [(x, y) for x, y in zip(best_models_pre_fit.keys(), best_models_pre_fit.values())]}
            vote_clf = ClassificationTraining(clf=classification_models['voting'][0], params=parameters)
            vote_clf.grid_search(parameters=classification_models['voting'][1], k=k, score=score)
            vote_clf.clf_fit(x=train_x, y=train_y.values.ravel())
            vote_clf_best = vote_clf.grid.best_estimator_
            vote_clf_best.fit(train_x, train_y.values.ravel())
            best_models_pos_fit['voting'] = vote_clf_best
            running_times['voting'] = time.time() - start

        running_times[model] = time.time() - start

        classes = clf.grid.classes_

    return classes, best_models_pos_fit, running_times


def save_model(clfs, model_name):

    timestamp = str(datetime.datetime.now().day) + '_' + str(datetime.datetime.now().month) + '_' + str(datetime.datetime.now().year)

    i = 0
    for clf in clfs.values():
        file_name = 'models/' + str(model_name[i]) + '_best_' + str(timestamp) + '.sav'
        i += 1

        file_handler = open(file_name, 'wb')
        pickle.dump(clf, file_handler)
        file_handler.close()
