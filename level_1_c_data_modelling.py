import pandas as pd
import numpy as np
import datetime
import pickle
import time
import nltk
import operator
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from scipy.spatial.distance import cdist, pdist
from sklearn.model_selection import GridSearchCV
from gap_statistic import OptimalK
from sklearn.preprocessing import StandardScaler
from level_0_performance_report import log_record
from level_1_a_data_acquisition import sql_mapping_retrieval
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
        log_record('MODEL: {}'.format(model), project_id)
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


def new_request_type(df, df_top_words, options_file):
    keyword_dict = sql_mapping_retrieval(options_file.DSN_MLG, options_file.sql_info['database_final'], ['SDK_Setup_Keywords'], 'Keyword_Group', options_file, multiple_columns=1)[0]

    stemmer_pt = SnowballStemmer('porter')
    user_dict, requests_dict_2 = {}, {}

    df_top_words['Label'] = 'Não Definido'
    for label in keyword_dict.keys():
        # print('Label: {}'.format(label))
        for keywords in keyword_dict[label]:
            # print('Keywords: {}'.format(keywords))
            consecutive_flag = 1
            # multiple words not consecutive
            if ';' in keywords:
                keywords = keywords.replace(';', ' ')
                consecutive_flag = 0

            if 'User:' in keywords:
                user_id = keywords.replace('User:', '')
                user_dict[label] = user_id
                continue

            if ' ' in keywords:
                rank = 1 + consecutive_flag
                tokenized_key_word = nltk.tokenize.word_tokenize(keywords)
                try:
                    selected_cols = df_top_words[tokenized_key_word]
                except KeyError:
                    tokenized_key_word = [stemmer_pt.stem(x) for x in tokenized_key_word]
                    try:
                        selected_cols = df_top_words[tokenized_key_word]
                    except KeyError:
                        log_record('Keywords {} which were stemmed to {} were not found consecutive.'.format(keywords, tokenized_key_word), options_file.project_id, flag=1)
                        continue

                matched_index = selected_cols[selected_cols == 1].dropna(axis=0).index.values  # returns the requests with the keyword present
                if consecutive_flag:
                    matched_index = consecutive_keyword_testing(df, matched_index, tokenized_key_word)  # out of all the requests with the keywords present, searches them for those where they keywords are consecutive

                if matched_index is not None:
                    requests_dict_2 = request_matches(label, keywords, rank, matched_index, requests_dict_2)

                    if len(matched_index):
                        df_top_words.loc[df_top_words.index.isin(matched_index), 'Label'] = label
                else:
                    log_record('Keywords {} which were stemmed to {} were not found consecutive.'.format(keywords, tokenized_key_word), options_file.project_id, flag=1)

            # Single word
            elif ' ' not in keywords:
                rank = 0
                try:
                    df_top_words.loc[df_top_words[keywords] == 1, 'Label'] = label
                    requests_dict_2 = request_matches(label, keywords, rank, df_top_words[df_top_words[keywords] == 1].index.values, requests_dict_2)
                except KeyError:
                    try:
                        df_top_words.loc[df_top_words[stemmer_pt.stem(keywords)] == 1, 'Label'] = label
                        requests_dict_2 = request_matches(label, keywords, rank, df_top_words[df_top_words[stemmer_pt.stem(keywords)] == 1].index.values, requests_dict_2)
                    except KeyError:
                        log_record('Keyword not found: {}'.format(keywords), options_file.project_id, flag=1)
                        continue

    df = requests_draw_handling(df, requests_dict_2)

    user_label_assignment(df, df_top_words, user_dict)

    df.sort_values(by='Request_Num', inplace=True)
    df_top_words.sort_index(inplace=True)

    if [(x, y) for (x, y) in zip(df['Request_Num'].values, df_top_words.index.values) if x != y]:
        unique_requests_df = df['Request_Num'].unique()
        unique_requests_df_top_words = df_top_words.index.values
        log_record('Requests have missing Labels!', options_file.project_id, flag=1)
        log_record('Missing requests in the original dataset: {}'.format([x for x in unique_requests_df if x not in unique_requests_df_top_words]), options_file.project_id, flag=1)
        log_record('Missing requests in the top words dataset: {}'.format([x for x in unique_requests_df_top_words if x not in unique_requests_df]), options_file.project_id, flag=1)
        raise ValueError('Requests have missing Labels!')

    print(df['Label'].value_counts())
    log_record('{:.2f}% de pedidos Não Definidos'.format((df[df['Label'] == 'Não Definido'].shape[0] / df['Request_Num'].nunique()) * 100), options_file.project_id)

    return df


# The goal of this function is to check for consecutive presence of keywords, by comparing their index position;
def consecutive_keyword_testing(df, matched_index, keywords):

    matched_requests = []
    for request in matched_index:
        # print('testing request: {}'.format(request))
        description = nltk.tokenize.word_tokenize(df[df['Request_Num'] == request]['StemmedDescription'].values[0])  # Note: this line will raise and IndexError when a request present in the matched index (from df_top_words) is not present in the df
        # print(description)
        keyword_idxs_total = []

        for keyword in keywords:
            keyword_idxs = [i for i, x in enumerate(description) if x == keyword]
            keyword_idxs_total.append(keyword_idxs)

        control_value = 1
        for i in range(len(keyword_idxs_total)):
            for value in keyword_idxs_total[i]:
                try:
                    if value + 1 in keyword_idxs_total[i + 1]:
                        # print('original value was {} and i found {}'.format(value, value + 1))
                        control_value += 1
                    # else:
                        # print('original value was {} and i did NOT found {}'.format(value, value + 1))
                except IndexError:
                    # print('Last List')
                    continue

        if control_value >= len(keyword_idxs_total):
            matched_requests.append(request)
        else:
            continue

    if len(matched_requests):
        return matched_requests
    else:
        return None


def user_label_assignment(df, df_top_words, user_dict):
    for key in user_dict.keys():
        matched_requests = df[df['Contact_Customer_Id'] == int(user_dict[key])]['Request_Num']
        df_top_words.loc[df_top_words.index.isin(matched_requests), 'Label'] = key
    return df_top_words


def requests_draw_handling(df, requests_dict):

    df['Label'] = 'Não Definido'
    for request in requests_dict.keys():
        matches_count = len(requests_dict[request])
        unique_labels_count = len(set([x[0] for x in requests_dict[request]]))
        labels = [x[0] for x in requests_dict[request]]
        unique_labels = set(labels)
        unique_ranks_count = len(set([x[1] for x in requests_dict[request]]))
        highest_rank_label = max(requests_dict[request], key=operator.itemgetter(1))[0]

        if matches_count > 1 and unique_labels_count > 1:
            if 'Workspace' in unique_labels:  # Workspace has priority over any other label
                df.loc[df['Request_Num'] == request, 'Label'] = 'Workspace'
            else:
                if highest_rank_label == 'Demonstração Resultados' and 'Importador' in unique_labels:  # Importador has priority over any rank of Demonstração Resultados
                    df.loc[df['Request_Num'] == request, 'Label'] = 'Importador'
                else:
                    if unique_ranks_count == 1 and unique_labels_count != len(labels):
                        label_counter = Counter(labels)
                        df.loc[df['Request_Num'] == request, 'Label'] = label_counter.most_common(1)[0][0]
                    elif unique_ranks_count == 1 and unique_labels_count == len(labels):
                        # print('DRAW:', request, requests_dict[request])
                        df.loc[df['Request_Num'] == request, 'Label'] = 'Draw: ' + '+'.join([x[0] for x in requests_dict[request]])
                    else:
                        df.loc[df['Request_Num'] == request, 'Label'] = highest_rank_label
        else:
            df.loc[df['Request_Num'] == request, 'Label'] = requests_dict[request][0][0]

    return df


def request_matches(label, keywords, rank, requests_list, dictionary):

    for request in requests_list:
        try:
            dictionary[request].append((label, rank))
        except KeyError:
            dictionary[request] = [(label, rank)]

    return dictionary
