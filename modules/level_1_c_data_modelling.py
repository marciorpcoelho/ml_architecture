import re
import os
import nltk
import time
import pickle
import operator
import datetime
import numpy as np
import pandas as pd
import datetime as dt
from os import listdir
from multiprocessing import Pool
from sklearn.cluster import KMeans
from gap_statistic import OptimalK
from nltk.stem.snowball import SnowballStemmer
from scipy.spatial.distance import cdist, pdist
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta
from sklearn.metrics import silhouette_score, calinski_harabaz_score
import modules.level_1_a_data_acquisition as level_1_a_data_acquisition
import modules.level_1_b_data_processing as level_1_b_data_processing
import modules.level_0_performance_report as level_0_performance_report
pd.set_option('display.expand_frame_repr', False)
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))


class ClassificationTraining(object):
    def __init__(self, clf, params=None):
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf()

    def grid_search(self, parameters, k, score):
        self.grid = GridSearchCV(estimator=self.clf, param_grid=parameters, cv=k, scoring=score, iid=True, n_jobs=8, error_score=np.nan)

    def clf_grid_fit(self, x, y):
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
        self.grid = GridSearchCV(estimator=self.clf, param_grid=parameters, cv=k, iid=True, scoring=score, n_jobs=-1, error_score=np.nan)

    def clf_grid_fit(self, x, y):
        self.grid.fit(x, y)

    def clf_fit(self, x, y):
        self.clf.fit(x, y)


def classification_model_training(models, train_x, train_y, classification_models, k, gridsearch_score, project_id):
    best_models_pre_fit, best_models_pos_fit, predictions, running_times = {}, {}, {}, {}
    clf, classes = None, None

    for model in models:
        level_0_performance_report.log_record('Modelo: {}'.format(level_0_performance_report.dict_models_name_conversion[model][0]), project_id)
        if model == 'voting':
            start = time.time()
            parameters = {'estimators': [(x, y) for x, y in zip(best_models_pre_fit.keys(), best_models_pre_fit.values())]}
            vote_clf = ClassificationTraining(clf=classification_models['voting'][0], params=parameters)
            vote_clf.grid_search(parameters=classification_models['voting'][1], k=k, score=gridsearch_score)
            vote_clf.clf_grid_fit(x=train_x, y=train_y.values.ravel())
            vote_clf_best = vote_clf.grid.best_estimator_
            vote_clf_best.fit(train_x, train_y.values.ravel())
            best_models_pos_fit['voting'] = vote_clf_best
            running_times['voting'] = time.time() - start
        else:
            start = time.time()
            clf = ClassificationTraining(clf=classification_models[model][0])
            clf.grid_search(parameters=classification_models[model][1], k=k, score=gridsearch_score)
            try:
                clf.clf_grid_fit(x=train_x, y=train_y.values.ravel())
            except AttributeError:
                clf.clf_grid_fit(x=train_x, y=train_y)
            clf_best = clf.grid.best_estimator_

            best_models_pre_fit[model] = clf_best
            try:
                clf_best.fit(train_x, train_y.values.ravel())
            except AttributeError:
                clf_best.fit(train_x, train_y)
            best_models_pos_fit[model] = clf_best

        running_times[model] = time.time() - start

        classes = clf.grid.classes_

    return classes, best_models_pos_fit, running_times


def regression_model_training(models, train_x, train_x_non_ohe, train_y, train_y_non_ohe, regression_models, k, gridsearch_score, project_id):
    best_models_pre_fit, best_models_pos_fit, predictions, running_times = {}, {}, {}, {}
    clf, classes = None, None

    for model in models:
        level_0_performance_report.log_record('Modelo: {}'.format(level_0_performance_report.dict_models_name_conversion[model][0]), project_id)
        start = time.time()
        clf = RegressionTraining(clf=regression_models[model][0])
        print('gridsearching...')
        clf.grid_search(parameters=regression_models[model][1], k=k, score=gridsearch_score)

        if model == 'lgb':
            clf.clf_grid_fit(x=train_x_non_ohe, y=train_y_non_ohe.values.ravel())
        else:
            clf.clf_grid_fit(x=train_x, y=train_y.values.ravel())
        clf_best = clf.grid.best_estimator_

        df_cv = pd.DataFrame(clf.grid.cv_results_)
        df_cv.to_csv(base_path + '/output/gridsearch_results_{}_{}.csv'.format(model, '18_10_19'))

        best_models_pre_fit[model] = clf_best
        if model == 'lgb':
            clf_best.fit(train_x_non_ohe, train_y_non_ohe.values.ravel())
        else:
            clf_best.fit(train_x, train_y.values.ravel())
        best_models_pos_fit[model] = clf_best

        running_times[model] = time.time() - start

    return best_models_pos_fit, running_times


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


def save_model(clfs, model_name, project_id):

    timestamp = str(datetime.datetime.now().day) + '_' + str(datetime.datetime.now().month) + '_' + str(datetime.datetime.now().year)

    i = 0
    for clf in clfs.values():
        file_name = 'models/project_{}_{}_best_{}.sav'.format(project_id, model_name[i], timestamp)
        i += 1

        file_handler = open(file_name, 'wb')
        pickle.dump(clf, file_handler)
        file_handler.close()


def new_request_type(df, df_top_words, df_manual_classification, options_file):
    keyword_dict, ranking_dict = level_1_a_data_acquisition.sql_mapping_retrieval(options_file.DSN_MLG, options_file.sql_info['database_final'], options_file.sql_info['keywords_table'], 'Keyword_Group', options_file, multiple_columns=1)
    keyword_dict = keyword_dict[0]

    stemmer_pt = SnowballStemmer('porter')
    user_dict, force_dict, requests_dict_2 = {}, {}, {}
    verbose, request_num = 0, 'RE-000000'

    df_top_words['Label'] = 'Não Definido'
    for label in keyword_dict.keys():
        for keywords in keyword_dict[label]:
            consecutive_flag = 1
            # multiple words not consecutive
            if ';' in keywords:
                keywords = keywords.replace(';', ' ')
                consecutive_flag = 0

            if 'User:' in keywords:
                user_id = keywords.replace('User:', '')
                if label in user_dict.keys():
                    user_dict[label].append(user_id)
                else:
                    user_dict[label] = [user_id]
                continue

            elif "Forced:" in keywords:
                cleaned_keyword = keywords.replace('Forced:', '')
                if label in force_dict.keys():
                    force_dict[label].append(cleaned_keyword)
                else:
                    force_dict[label] = [cleaned_keyword]
                continue

            if ' ' in keywords:
                rank = 1 + consecutive_flag
                keywords = level_1_b_data_processing.string_digit_removal(keywords)
                keywords = level_1_b_data_processing.unidecode_function(keywords)
                keywords = level_1_b_data_processing.string_punctuation_removal(keywords)
                tokenized_key_word = nltk.tokenize.word_tokenize(keywords)
                try:
                    tokenized_key_word = [stemmer_pt.stem(x) for x in tokenized_key_word]
                    selected_cols = df_top_words[tokenized_key_word]
                except KeyError:
                    level_0_performance_report.log_record('Palavras chave {} que foram reduzidas a {} não foram encontradas de forma consecutiva.'.format(keywords, tokenized_key_word), options_file.project_id, flag=1)
                    continue

                matched_index = selected_cols[selected_cols == 1].dropna(axis=0).index.values  # returns the requests with the keyword present
                if consecutive_flag:
                    matched_index = consecutive_keyword_testing(df, matched_index, tokenized_key_word)  # out of all the requests with the keywords present, searches them for those where they keywords are consecutive

                if matched_index is not None:
                    requests_dict_2 = request_matches(label, keywords, rank, matched_index, requests_dict_2)

                    if len(matched_index):
                        df_top_words.loc[df_top_words.index.isin(matched_index), 'Label'] = label
                else:
                    level_0_performance_report.log_record('Palavras chave {} que foram reduzidas a {} não foram encontradas de forma consecutiva.'.format(keywords, tokenized_key_word), options_file.project_id, flag=1)

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
                        level_0_performance_report.log_record('Palavra chave não encontrada: {}'.format(keywords), options_file.project_id, flag=1)
                        continue

    df = requests_draw_handling(df, requests_dict_2, ranking_dict)

    if verbose:
        print('A \n', df[df['Request_Num'] == request_num]['Label'])
    df, df_top_words = force_label_assignment(df, df_top_words, force_dict, options_file)
    if verbose:
        print('B \n', df[df['Request_Num'] == request_num]['Label'])
    df, _ = user_label_assignment(df, df_top_words, user_dict)
    if verbose:
        print('C \n', df[df['Request_Num'] == request_num]['Label'])

    df.sort_values(by='Request_Num', inplace=True)
    df_top_words.sort_index(inplace=True)

    if [(x, y) for (x, y) in zip(df['Request_Num'].values, df_top_words.index.values) if x != y]:
        unique_requests_df = df['Request_Num'].unique()
        unique_requests_df_top_words = df_top_words.index.values
        level_0_performance_report.log_record('Existem pedidos sem classificação!', options_file.project_id, flag=1)
        level_0_performance_report.log_record('Pedidos não classificados no dataset original: {}'.format([x for x in unique_requests_df if x not in unique_requests_df_top_words]), options_file.project_id, flag=1)
        level_0_performance_report.log_record('Pedidos não classificados no dataset Top_Words: {}'.format([x for x in unique_requests_df_top_words if x not in unique_requests_df]), options_file.project_id, flag=1)
        raise ValueError('Existem pedidos sem classificação!')

    if df_manual_classification.shape[0]:  # No need to call this function if there are no manually classified requests;
        df = manual_classified_requests(df, df_manual_classification)

    level_0_performance_report.log_record('{:.2f}% de pedidos Não Definidos'.format((df[df['Label'] == 'Não Definido'].shape[0] / df['Request_Num'].nunique()) * 100), options_file.project_id)

    return df


# The goal of this function is to check for consecutive presence of keywords, by comparing their index position;
def consecutive_keyword_testing(df, matched_index, keywords):

    matched_requests = []
    for request in matched_index:
        # print('testing request: {}'.format(request))
        description = nltk.tokenize.word_tokenize(df[df['Request_Num'] == request]['StemmedDescription'].values[0])  # Note: this line will raise an IndexError when a request present in the matched index (from df_top_words) is not present in the df
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


def manual_classified_requests(df, df_manual_classification):
    # Project ID = 2244
    # The goal of this function is to replace the "Não Definidos" labels by the manual classified requests via streamlit dashboard. It considers there is only one row per request;
    # The filtering for the "Não Definidos" is enforced via df_manual_classification, as those are only for that class of requests;

    for request_num in df_manual_classification['Request_Num'].unique():
        label = df_manual_classification.loc[df_manual_classification['Request_Num'] == request_num, 'Label'].values[0]

        df.loc[df['Request_Num'] == request_num, 'Label'] = label

    return df


def user_label_assignment(df, df_top_words, user_dict):
    for key in user_dict.keys():
        matched_requests = df[df['Contact_Customer_Id'].isin([int(user) for user in user_dict[key]])]['Request_Num']
        df_top_words.loc[df_top_words.index.isin(matched_requests), 'Label'] = key
        df.loc[df['Request_Num'].isin(matched_requests), 'Label'] = key
    return df, df_top_words


def force_label_assignment(df, df_top_words, force_dict, options_file):
    for key in force_dict.keys():
        for value in force_dict[key]:
            try:
                matched_requests = df[df['Description'].str.contains(value)]['Request_Num']
                df.loc[df['Request_Num'].isin(matched_requests), 'Label'] = key
                df_top_words.loc[df_top_words.index.isin(matched_requests), 'Label'] = key

            except KeyError:
                level_0_performance_report.log_record('Palavra chave não encontrada: {}'.format(force_dict[key]), options_file.project_id, flag=1)

    return df, df_top_words


# New Version, with priority
def requests_draw_handling(df, requests_dict, ranking_dict):

    df['Label'] = 'Não Definido'
    for request in requests_dict.keys():
        matches_count = len(requests_dict[request])
        unique_labels_count = len(set([x[0] for x in requests_dict[request]]))
        labels = [x[0] for x in requests_dict[request]]

        if matches_count > 1 and unique_labels_count > 1:
            sel_dict = {key: ranking_dict[key] for key in labels if key in ranking_dict}
            unique_sel_labels_count = len(set(sel_dict.values()))

            if unique_sel_labels_count > 1:
                max_key = max(sel_dict.items(), key=operator.itemgetter(1))[0]
                df.loc[df['Request_Num'] == request, 'Label'] = max_key
            else:
                # print('Draw found in Request {} for categories {}'.format(request, [x[0] for x in requests_dict[request]]))
                # print('With the following dict: {}'.format(sel_dict))
                df.loc[df['Request_Num'] == request, 'Label'] = 'Draw: ' + '+'.join([x[0] for x in requests_dict[request]])

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


def part_ref_selection(df_al, min_date, max_date, project_id, last_year_flag=1):
    # Proj_ID = 2259
    level_0_performance_report.log_record('Selection of Part References...', project_id)

    if last_year_flag:
        min_date = pd.to_datetime(max_date) - relativedelta(years=1)

    df_al_filtered = df_al[(df_al['Movement_Date'] > min_date) & (df_al['Movement_Date'] <= max_date)]

    all_unique_part_refs = df_al_filtered['Part_Ref'].unique()

    all_unique_part_refs_bm = [x for x in all_unique_part_refs if x.startswith('BM')]
    all_unique_part_refs_mn = [x for x in all_unique_part_refs if x.startswith('MN')]

    all_unique_part_refs = all_unique_part_refs_bm + all_unique_part_refs_mn

    all_unique_part_refs_at = [x for x in all_unique_part_refs if x.endswith('AT')]

    all_unique_part_refs = [x for x in all_unique_part_refs if x not in all_unique_part_refs_at]

    [level_0_performance_report.log_record('{} has a weird size!'.format(x), project_id, flag=1) for x in all_unique_part_refs if len(x) > 17 or len(x) < 13]

    level_0_performance_report.log_record('{} unique part_refs between {} and {}.'.format(len(all_unique_part_refs), min_date, max_date), project_id)

    return all_unique_part_refs


def apv_last_stock_calculation(min_date_str, current_date, pse_code, project_id):
    # Proj_ID = 2259
    results_files = [datetime.datetime.strptime(f[17:25], format('%Y%m%d')) for f in listdir(base_path + '/output/') if f.startswith('results_merge_{}_'.format(pse_code))]
    min_date_datetime = datetime.datetime.strptime(min_date_str, format('%Y%m%d'))
    max_date_datetime = datetime.datetime.strptime(current_date, format('%Y%m%d'))
    preprocessed_data_exists_flag = 0

    top_2_idx = np.argsort(results_files)[-2:]
    results_files = [results_files[i] for i in top_2_idx]
    second_to_last_processed_date = results_files[0]

    try:
        max_file_date = np.max([results_files[-1]])
        if min_date_datetime < max_file_date:
            if max_file_date == max_date_datetime:
                raise Exception('All data has been processed already up to date {}.'.format(max_file_date))
            else:
                preprocessed_data_exists_flag = 1
                level_0_performance_report.log_record('Data already processed from {} to {} for PSE {}. Will adjust accordingly: minimum date to process is now: {}.'.format(min_date_str, max_file_date, pse_code, max_file_date), project_id)
                # print('Data already processed from {} to {} for PSE {}. Will adjust accordingly: minimum date to process is now: {}.'.format(min_date_str, max_file_date, pse_code, max_file_date))
            return datetime.datetime.strftime(max_file_date, format('%Y%m%d')), datetime.datetime.strftime(second_to_last_processed_date, format('%Y%m%d')), preprocessed_data_exists_flag
        else:  # Do nothing
            return min_date_str, '', preprocessed_data_exists_flag
    except ValueError:  # No Files found
        return min_date_str, '', preprocessed_data_exists_flag


def apv_stock_evolution_calculation(pse_code, selected_parts, df_sales, df_al, df_stock, df_reg_al_clients, df_purchases, min_date, max_date, project_id):
    # PRJ-2259

    try:
        results = level_1_a_data_acquisition.read_csv(base_path + '/output/results_merge_{}_{}.csv'.format(pse_code, max_date), index_col='index')

    except FileNotFoundError:
        level_0_performance_report.log_record('Ficheiro results_merge_{}_{} não encontrado. A processar...'.format(pse_code, max_date), project_id)
        min_date, _, preprocessed_data_exists_flag = apv_last_stock_calculation(min_date, max_date, pse_code, project_id)

        df_stock.set_index('Record_Date', inplace=True)
        df_purchases.set_index('Movement_Date', inplace=True)
        df_al['Unit'] = df_al['Unit'] * (-1)  # Turn the values to their symmetrical so it matches the other dfs

        dataframes_list = [df_sales, df_al, df_stock, df_reg_al_clients, df_purchases]
        datetime_index = pd.date_range(start=min_date, end=max_date)
        results = pd.DataFrame()
        positions = []

        if preprocessed_data_exists_flag:
            selected_parts = part_ref_selection(df_al, min_date, max_date, project_id, last_year_flag=0)
            df_last_processed = level_1_a_data_acquisition.read_csv(base_path + '/output/results_merge_{}_{}.csv'.format(pse_code, min_date), index_col=0, parse_dates=['index'])
            dataframes_list.append(df_last_processed)

        # i, parts_count = 1, len(selected_parts)
        for part_ref in selected_parts:
            result_part_ref, stock_evolution_correct_flag, offset, last_processed_stock = sql_data([part_ref], pse_code, datetime.datetime.strptime(min_date, format('%Y%m%d')), max_date, dataframes_list, preprocessed_data_exists_flag)

            if result_part_ref.shape[0]:
                result_part_ref = result_part_ref.reindex(datetime_index).reset_index().rename(columns={'Unnamed: 0': 'Movement_Date'})

                result_part_ref = fill_cols_function(result_part_ref, last_processed_stock)

                if result_part_ref[result_part_ref['Part_Ref'].isnull()].shape[0]:
                    level_0_performance_report.log_record('null values found for part_ref: \n{}'.format(part_ref), project_id, flag=1)
                    level_0_performance_report.log_record('Number of null rows: {}'.format(result_part_ref[result_part_ref['Part_Ref'].isnull()].shape), project_id, flag=1)

                result_part_ref.loc[:, 'Stock_Evolution_Correct_Flag'] = stock_evolution_correct_flag
                result_part_ref.loc[:, 'Stock_Evolution_Offset'] = offset

                # Just so the variation matches with the rest: positive regularization means an increase in stock, while negative is a decrease; Equivalent for cost;
                result_part_ref['Qty_Regulated_sum'] = result_part_ref['Qty_Regulated_sum'] * (-1)
                result_part_ref['Cost_Reg_avg'] = result_part_ref['Cost_Reg_avg'] * (-1)

                results = results.append(result_part_ref)
                # results.to_csv(base_path + '/output/testing_results.csv'.format(part_ref))

            # position = int((i / parts_count) * 100)
            # if not position % 1:
            #     if position not in positions:
            #         print('{}% completed'.format(position))
            #         positions.append(position)
            #
            # i += 1
        if preprocessed_data_exists_flag:
            results = results_preprocess_merge(pse_code, results, min_date, max_date)

        results.to_csv(base_path + '/output/results_merge_{}_{}.csv'.format(pse_code, max_date))

    return results


def apv_photo_stock_treatment(df_sales, df_history, selected_parts, preprocessed_data_exists_flag, min_date, max_date, pse_code, project_id):
    # PRJ-2259

    # DW Sales Preparation
    df_sales = df_sales[['Movement_Date', 'Part_Ref', 'Part_Desc', 'Product_Group', 'PVP_1', 'Cost_Sale_1', 'Qty_Sold_sum_mov']]
    df_sales = df_sales.rename(columns={'Qty_Sold_sum_mov': 'Qty_Sold'})
    df_sales = df_sales.sort_values(by='Movement_Date')
    df_sales = df_sales.drop_duplicates(subset=['Movement_Date', 'Part_Ref'])

    datetime_index = pd.date_range(start=df_sales['Movement_Date'].min(), end=df_sales['Movement_Date'].max())
    df_sales.set_index(['Movement_Date'], drop=True, inplace=True)

    df_history['Movement_Date'] = pd.to_datetime(df_history['Movement_Date'], format='%Y-%m-%d')
    pool = Pool(processes=int(level_0_performance_report.pool_workers_count))
    results = pool.map(stock_and_sales_reconstitution, [(part_ref, df_sales, df_history, datetime_index) for part_ref in selected_parts])
    pool.close()
    final_df = pd.concat([result for result in results if result is not None])

    if preprocessed_data_exists_flag:
        old_results = level_1_a_data_acquisition.read_csv('output/results_merge_{}_{}.csv'.format(pse_code, min_date), index_col=0)
        final_df.rename(columns={'PVP_1': 'PVP_avg', 'Cost_Sale_1': 'Cost_Sale_avg', 'Qty_Sold': 'Qty_Sold_sum_al', 'Qty_Stock': 'Stock_Qty_al', 'Movement_Date': 'index'}, inplace=True)
        final_df = pd.concat([old_results, final_df], ignore_index=True)

    final_df.to_csv('output/results_merge_{}_{}.csv'.format(pse_code, max_date))
    return final_df


def stock_and_sales_reconstitution(args):
    sel_part, df_sales, df_stock, datetime_index = args

    zero_fill_cols = ['Qty_Sold', 'PVP_1', 'Cost_Sale_1']

    df_stock_filtered = df_stock.loc[df_stock['Part_Ref'] == sel_part, :]
    df_sales_dw_filtered = df_sales.loc[df_sales['Part_Ref'] == sel_part, :]

    result_part_ref = df_sales_dw_filtered.reindex(datetime_index).reset_index()
    [result_part_ref[x].fillna(0, inplace=True) for x in zero_fill_cols]
    result_part_ref['Part_Ref'] = sel_part

    result_part_ref = pd.merge(result_part_ref, df_stock_filtered[['Qty_Stock', 'Movement_Date']], how='inner', left_on='index', right_on='Movement_Date', validate='one_to_one').drop(['index'], axis=1)
    return result_part_ref


def fill_cols_function(result_part_ref, last_processed_stock):

    ffill_and_zero_fill_cols = ['Stock_Qty_al', 'Sales Evolution_al', 'Purchases Evolution', 'Regulated Evolution', 'Purchases Urgent Evolution', 'Purchases Non Urgent Evolution']
    zero_fill_cols = ['Qty_Purchased_sum', 'Qty_Regulated_sum', 'Qty_Sold_sum_al', 'Qty_Purchased_urgent_sum', 'Qty_Purchased_non_urgent_sum', 'Cost_Purchase_avg', 'Cost_Reg_avg', 'Cost_Sale_avg', 'PVP_avg']
    ffill_and_bfill_cols = ['Part_Ref', 'Stock_Qty']

    [result_part_ref[x].fillna(method='ffill', inplace=True) for x in ffill_and_zero_fill_cols + ffill_and_bfill_cols]
    [result_part_ref[x].fillna(method='bfill', inplace=True) for x in ffill_and_zero_fill_cols + ffill_and_bfill_cols]
    [result_part_ref[x].fillna(0, inplace=True) for x in zero_fill_cols + ffill_and_zero_fill_cols]

    return result_part_ref


def results_preprocess_merge(pse_code, results, min_date, max_date):

    old_results = level_1_a_data_acquisition.read_csv(base_path + '/output/results_merge_{}_{}.csv'.format(pse_code, min_date), index_col=0, parse_dates=['index'])
    new_results = pd.concat([old_results, results])

    # shape_before = new_results.shape
    new_results.drop_duplicates(subset=['index', 'Part_Ref'], inplace=True)
    # shape_after = new_results.shape
    # shape_diff = shape_before[0] - shape_after[0]

    # print('Removed {} repeated rows.'.format(shape_diff))
    return new_results


def sql_data(selected_part, pse_code, min_date, max_date, dataframes_list, preprocessed_data_exists_flag):
    # Proj_ID = 2259

    df_sales, df_al, df_stock, df_reg_al_clients, df_purchases = dataframes_list[0], dataframes_list[1], dataframes_list[2], dataframes_list[3], dataframes_list[4]

    result, stock_evolution_correct_flag, offset, last_stock_qty_al = pd.DataFrame(), 0, 0, 0

    df_sales_filtered = df_sales[(df_sales['Part_Ref'].isin(selected_part)) & (df_sales['Movement_Date'] >= min_date) & (df_sales['Movement_Date'] <= max_date)]
    df_al_filtered = df_al[(df_al['Part_Ref'].isin(selected_part)) & (df_al['Movement_Date'] >= min_date) & (df_al['Movement_Date'] <= max_date)]
    df_purchases_filtered = df_purchases[(df_purchases['Part_Ref'].isin(selected_part)) & (df_purchases.index > min_date) & (df_purchases.index <= max_date)]
    df_stock_filtered = df_stock[(df_stock['Part_Ref'].isin(selected_part)) & (df_stock.index >= min_date) & (df_stock.index <= max_date)]
    if preprocessed_data_exists_flag:
        df_last_processed = dataframes_list[5]
        df_last_processed_filtered_last_row = df_last_processed[df_last_processed['Part_Ref'].isin(selected_part)].sort_values(by='index').tail(1)
        df_last_stock_filtered_first_row = df_stock_filtered[df_stock_filtered['Part_Ref'].isin(selected_part)].sort_values(by='Record_Date').tail(1)

    df_al_filtered = auto_line_dataset_cleaning(df_sales_filtered, df_al_filtered, df_purchases_filtered, df_reg_al_clients, pse_code)
    df_sales_filtered = df_sales_filtered.drop_duplicates(subset=['Movement_Date', 'Part_Ref']).sort_values(by='Movement_Date')
    df_sales_filtered.set_index('Movement_Date', inplace=True)

    if not df_al_filtered.shape[0]:
        # raise ValueError('No data found for part_ref {} and/or selected period {}/{}'.format(selected_part[0], min_date, max_date))
        # print('\nNo data found for part_ref {} and/or selected period {}/{}.\n'.format(selected_part[0], min_date, max_date))
        # no_data_flag = 1
        return pd.DataFrame(), stock_evolution_correct_flag, offset, last_stock_qty_al
    # elif df_al_filtered.shape[0] == 1:
    #     print('im here!')
        # raise ValueError('Only 1 row found for part_ref {} and/or selected period {}/{}'.format(selected_part[0], min_date, max_date))
        # print('\nOnly 1 row found for part_ref {} and/or selected period {}/{}. Ignored.\n'.format(selected_part[0], min_date, max_date))
        # one_row_only_flag = 1
        # return pd.DataFrame(), stock_evolution_correct_flag, offset

    df_al_filtered['Qty_Sold_sum_al'], df_al_filtered['Cost_Sale_avg'], df_al_filtered['PVP_avg'] = 0, 0, 0  # Placeholder for cases without sales
    df_al_grouped = df_al_filtered[df_al_filtered['regularization_flag'] == 0].groupby(['Movement_Date', 'Part_Ref'])
    for key, row in df_al_grouped:
        rows_selection = (df_al_filtered['Movement_Date'] == key[0]) & (df_al_filtered['Part_Ref'] == key[1])
        df_al_filtered.loc[rows_selection, 'Qty_Sold_sum_al'] = row['Unit'].sum()
        df_al_filtered.loc[rows_selection, 'Cost_Sale_avg'] = row['Preço de custo'].mean()
        df_al_filtered.loc[rows_selection, 'PVP_avg'] = row['P. V. P'].mean()

    df_al_filtered = df_al_filtered.assign(Qty_Regulated_sum=0)  # Placeholder for cases without regularizations
    df_al_filtered = df_al_filtered.assign(Cost_Reg_avg=0)  # Placeholder for cases without regularizations

    df_al_grouped_reg_flag = df_al_filtered[df_al_filtered['regularization_flag'] == 1].groupby(['Movement_Date', 'Part_Ref'])
    for key, row in df_al_grouped_reg_flag:
        rows_selection = (df_al_filtered['Movement_Date'] == key[0]) & (df_al_filtered['Part_Ref'] == key[1])
        df_al_filtered.loc[rows_selection, 'Qty_Regulated_sum'] = row['Unit'].sum()
        df_al_filtered.loc[rows_selection, 'Cost_Reg_avg'] = row['Cost_Reg'].sum()

    df_al_filtered.drop(['Unit', 'Preço de custo', 'P. V. P', 'regularization_flag'], axis=1, inplace=True)

    df_al_filtered = df_al_filtered.drop_duplicates(subset=['Movement_Date'])
    df_al_filtered.set_index('Movement_Date', inplace=True)

    df_purchases_filtered = df_purchases_filtered.loc[~df_purchases_filtered.index.duplicated(keep='first')]

    qty_sold_al = df_al_filtered[df_al_filtered.index > min_date]['Qty_Sold_sum_al'].sum()
    qty_purchased = df_purchases_filtered['Qty_Purchased_sum'].sum()

    try:
        if not preprocessed_data_exists_flag:
            stock_start = df_stock_filtered[df_stock_filtered.index == min_date]['Stock_Qty'].values[0]
        elif preprocessed_data_exists_flag:
            stock_start = df_last_processed_filtered_last_row['Stock_Qty'].values[0]  #
    except IndexError:
        stock_start = 0  # When stock is 0, it is not saved in SQL, hence why the previous line doesn't return any value;

    try:
        if not preprocessed_data_exists_flag:
            stock_end = df_stock_filtered[df_stock_filtered.index == max_date]['Stock_Qty'].values[0]
        elif preprocessed_data_exists_flag:
            stock_end = df_last_stock_filtered_first_row['Stock_Qty'].values[0]
            last_stock_qty_al = df_last_processed_filtered_last_row['Stock_Qty_al'].values[0]
    except IndexError:
        stock_end = 0  # When stock is 0, it is not saved in SQL, hence why the previous line doesn't return any value;
        last_stock_qty_al = 0

    reg_value = df_al_filtered['Qty_Regulated_sum'].sum()

    if not reg_value:
        reg_value = 0

    result = pd.concat([df_purchases_filtered[['Qty_Purchased_sum', 'Qty_Purchased_urgent_sum', 'Qty_Purchased_non_urgent_sum', 'Cost_Purchase_avg']], df_al_filtered[['Qty_Regulated_sum', 'Cost_Reg_avg', 'Qty_Sold_sum_al', 'Cost_Sale_avg', 'PVP_avg']]], axis=1, sort=False)
    result['Part_Ref'] = selected_part * result.shape[0]
    try:
        # result['Stock_Qty'] = df_stock_filtered['Stock_Qty'].head(1).values[0]
        result['Stock_Qty'] = stock_start
    except IndexError:
        result['Stock_Qty'] = 0  # Cases when there is no stock information

    if not preprocessed_data_exists_flag:
        result_al = stock_start + qty_purchased - qty_sold_al - reg_value
    else:
        try:
            stock_start_preprocessed = df_last_processed_filtered_last_row['Stock_Qty_al'].values[0]
        except IndexError:
            stock_start_preprocessed = 0
        result_al = stock_start_preprocessed + qty_purchased - qty_sold_al - reg_value

    if result_al != stock_end:
        offset = stock_end - result_al
        # print('Selected Part: {} - Values dont match for AutoLine values - Stock has an offset of {:.2f} \n'.format(selected_part, offset))
    else:
        # print('Selected Part: {} - Values for AutoLine are correct :D \n'.format(selected_part))
        stock_evolution_correct_flag = 1

    result['Stock_Qty'].fillna(method='ffill', inplace=True)
    result['Part_Ref'].fillna(method='ffill', inplace=True)
    result.fillna(0, inplace=True)

    # The filters in the evolution columns is to compensate for sales/purchases/etc in the first day, even though the stock for that same day already has those movements into consideration.
    # The stock_qty_al initial value is to compensate for the previous line.
    min_date_next_day = min_date + relativedelta(days=1)
    try:
        if preprocessed_data_exists_flag:
            try:
                sales_evolution_offset = df_last_processed_filtered_last_row['Sales Evolution_al'].values[0]
                purchases_evolution_offset = df_last_processed_filtered_last_row['Purchases Evolution'].values[0]
                purchases_urgent_evolution_offset = df_last_processed_filtered_last_row['Purchases Urgent Evolution'].values[0]
                purchases_non_urgent_evolution_offset = df_last_processed_filtered_last_row['Purchases Non Urgent Evolution'].values[0]
                regulated_evolution_offset = df_last_processed_filtered_last_row['Regulated Evolution'].values[0]
            except IndexError:
                sales_evolution_offset = 0
                purchases_evolution_offset = 0
                purchases_urgent_evolution_offset = 0
                purchases_non_urgent_evolution_offset = 0
                regulated_evolution_offset = 0

            result['Sales Evolution_al'] = result[result.index > min_date]['Qty_Sold_sum_al'].cumsum() + sales_evolution_offset
            result['Purchases Evolution'] = result[result.index > min_date]['Qty_Purchased_sum'].cumsum() + purchases_evolution_offset
            result['Purchases Urgent Evolution'] = result[result.index > min_date]['Qty_Purchased_urgent_sum'].cumsum() + purchases_urgent_evolution_offset
            result['Purchases Non Urgent Evolution'] = result[result.index > min_date]['Qty_Purchased_non_urgent_sum'].cumsum() + purchases_non_urgent_evolution_offset
            result['Regulated Evolution'] = result[result.index > min_date]['Qty_Regulated_sum'].cumsum() + regulated_evolution_offset
        elif not preprocessed_data_exists_flag:
            result['Sales Evolution_al'] = result[result.index > min_date]['Qty_Sold_sum_al'].cumsum()
            result['Purchases Evolution'] = result[result.index > min_date]['Qty_Purchased_sum'].cumsum()
            result['Purchases Urgent Evolution'] = result[result.index > min_date]['Qty_Purchased_urgent_sum'].cumsum()
            result['Purchases Non Urgent Evolution'] = result[result.index > min_date]['Qty_Purchased_non_urgent_sum'].cumsum()
            result['Regulated Evolution'] = result[result.index > min_date]['Qty_Regulated_sum'].cumsum()
    except KeyError:
        print(selected_part, min_date_next_day, '\n', result)

    result['Stock_Qty_al'] = result['Stock_Qty'] - result['Sales Evolution_al'] + result['Purchases Evolution'] - result['Regulated Evolution']
    if not preprocessed_data_exists_flag:
        result.loc[result.index[0], 'Stock_Qty_al'] = stock_start
    elif preprocessed_data_exists_flag:
        result.loc[result.index[0], 'Stock_Qty_al'] = last_stock_qty_al
    result.loc[result['Qty_Purchased_sum'] == 0, 'Cost_Purchase_avg'] = 0

    return result, stock_evolution_correct_flag, offset, last_stock_qty_al


def auto_line_dataset_cleaning(df_sales, df_al, df_purchases, df_reg_al_clients, pse_code):
    # print('AutoLine and PSE_Sales Lines comparison started...')

    # ToDo Martelanço
    if pse_code == '0B' and df_purchases['Part_Ref'].unique() == 'BM83.21.0.406.573':
        if '2019-02-05' in df_purchases.index:
            df_purchases.drop(df_purchases[df_purchases['PLR_Document'] == 0].index, inplace=True)

    purchases_unique_plr = df_purchases['PLR_Document'].unique().tolist()
    reg_unique_slr = df_reg_al_clients['SLR_Account'].unique().tolist() + ['@Check']

    duplicated_rows = df_al[df_al.duplicated(subset='Encomenda', keep=False)]
    if duplicated_rows.shape[0]:
        duplicated_rows_grouped = duplicated_rows.groupby(['Movement_Date', 'Part_Ref', 'WIP_Number'])

        df_al = df_al.drop(duplicated_rows.index, axis=0)

        pool = Pool(processes=int(level_0_performance_report.pool_workers_count))
        results = pool.map(sales_cleaning, [(key, group, df_sales, pse_code) for (key, group) in duplicated_rows_grouped])
        pool.close()
        df_al_merged = pd.concat([df_al, pd.concat([result for result in results if result is not None])], axis=0)

        # ToDo Martelanço
        if pse_code == '0B':
            df_al_merged.loc[(df_al_merged['Part_Ref'] == 'BM11.42.8.507.683') & (df_al_merged['Movement_Date'] == '2018-12-04') & (df_al_merged['WIP_Number'] == 41765) & (df_al_merged['Unit'] == 0) & (df_al_merged['SLR_Document_Account'] == 'd077612'), 'Unit'] = -1

        df_al_cleaned = purchases_reg_cleaning(df_al_merged, purchases_unique_plr, reg_unique_slr)
    else:
        df_al_cleaned = purchases_reg_cleaning(df_al, purchases_unique_plr, reg_unique_slr)

    return df_al_cleaned


def sales_cleaning(args):
    key, group, df_sales, pse_code = args
    group_size = group.shape[0]

    # Note: There might not be a match!
    if group_size > 1 and group['Audit_Number'].nunique() < group_size:

        matching_sales = df_sales[(df_sales['Movement_Date'] == key[0]) & (df_sales['Part_Ref'] == key[1]) & (df_sales['WIP_Number'] == key[2])]

        number_matched_lines, group_size = matching_sales.shape[0], group.shape[0]
        if 0 < number_matched_lines <= group_size:

            group = group[group['SLR_Document_Number'].isin(matching_sales['SLR_Document'].unique())]

        # elif number_matched_lines > 0 and number_matched_lines == group_size:
            # print('matched lines equal to group size')
            # print('sales = autoline: no rows to remove')
            # pass  # ToDo will need to handle these exceptions better
        elif number_matched_lines > 0 and number_matched_lines > group_size:
            # print('number_matched_lines > group_size')
            # print('matched lines over group size')
            # print('sales > autoline - weird case?')
            pass  # ToDo will need to handle these exceptions better
        elif number_matched_lines == 0:
            # print('number_matched_lines == 0')
            # print('NO MATCHED ROWS?!?')
            # print(group, '\n', matching_sales)
            group = group.tail(1)  # ToDo Needs to be confirmed

        # ToDo Martelanço:
        if pse_code == '0I':
            if group['Part_Ref'].unique() == 'BM83.19.2.158.851' and key[2] == 38381:
                group = group[group['SLR_Document_Number'] != 44446226]
            if group['Part_Ref'].unique() == 'BM83.21.2.405.675' and key[2] == 63960:
                group = group[group['SLR_Document_Number'] != 44462775]

    return group


def purchases_reg_cleaning(df_al, purchases_unique_plr, reg_unique_slr):

    matched_rows_purchases = df_al[df_al['SLR_Document_Number'].isin(purchases_unique_plr)]
    if matched_rows_purchases.shape[0]:
        df_al = df_al.drop(matched_rows_purchases.index, axis=0)

    matched_rows_reg = df_al[df_al['SLR_Document_Account'].isin(reg_unique_slr)].index

    df_al = df_al.assign(regularization_flag=0)
    df_al.loc[df_al.index.isin(matched_rows_reg), 'regularization_flag'] = 1
    df_al.loc[df_al['regularization_flag'] == 1, 'Cost_Reg'] = df_al['Unit'] * df_al['Preço de custo']

    return df_al


def part_ref_ta_definition(df_sales, selected_parts, pse_code, max_date, mappings, regex_dict, bmw_original_oil_words, project_id):  # From PSE_Sales
    level_0_performance_report.log_record('A procurar TA para cada referência...', project_id)
    start = time.time()

    try:
        df_part_ref_ta_grouped = level_1_a_data_acquisition.read_csv(base_path + '/output/part_ref_ta_{}_{}.csv'.format(pse_code, max_date), index_col=0)

    except FileNotFoundError:
        level_0_performance_report.log_record('Não foi encontrado o ficheiro part_ref_ta_{}_{}. A processar...'.format(pse_code, max_date), project_id)
        df_part_ref_ta = pd.DataFrame(columns={'Part_Ref', 'TA'})

        df_sales = df_sales[df_sales['Product_Group'].notnull()]
        df_sales_sel_parts_grouped = df_sales.loc[df_sales['Part_Ref'].isin(selected_parts), :].groupby('Part_Ref')

        pool = Pool(processes=level_0_performance_report.pool_workers_count)
        results = pool.map(ta_selection, [(part_ref, df_sales_per_part_ref, regex_dict, bmw_original_oil_words, project_id, 'Product_Group') for (part_ref, df_sales_per_part_ref) in df_sales_sel_parts_grouped])
        pool.close()
        df_part_ref_ta['Part_Ref'] = [result[0] for result in results if result is not None]
        df_part_ref_ta['TA'] = [result[1] for result in results if result is not None]
        df_part_ref_ta['Group'] = df_part_ref_ta['TA']
        df_part_ref_ta['PSE_Code'] = pse_code

        df_bmw = level_1_b_data_processing.col_group(df_part_ref_ta[df_part_ref_ta['Part_Ref'].str.startswith('BM')].copy(), ['Group'], [mappings[0]], project_id)
        df_mini = level_1_b_data_processing.col_group(df_part_ref_ta[df_part_ref_ta['Part_Ref'].str.startswith('MN')].copy(), ['Group'], [mappings[1]], project_id)

        df_part_ref_ta_grouped = pd.concat([df_bmw, df_mini])

        df_part_ref_ta_grouped.to_csv(base_path + '/output/part_ref_ta_{}_{}.csv'.format(pse_code, max_date))

    print('TA Def v2: Elapsed Time: {:.2f}'.format(time.time() - start))
    return df_part_ref_ta_grouped


def ta_selection(args):
    part_ref, df, regex_dict, bmw_original_oil_words, project_id, product_group_col = args

    part_ref_ta = 'NO_TA'
    part_ref_desc = df['Part_Desc'].value_counts().head(1).index  # Gets the most common Part_Desc, for the part_ref which have multiple descriptions;
    part_ref_unique_ta = df[product_group_col].unique()

    if len(part_ref_unique_ta) == 1:
        try:
            part_ref_ta = part_ref_unique_ta[0][1]
        except (IndexError, TypeError):
            return 'NO_TA'
            # Cases where only the part_ref only has one TA but it is only a letter and not letter + number, which provides no information

    elif len(part_ref_unique_ta) > 1:
        part_ref_ta_value_counts = df[df['Part_Ref'] == part_ref][product_group_col].value_counts()

        try:
            part_ref_ta = part_ref_ta_value_counts.head(1).index.values[0][1]  # Get the second character of the most common TA
        except (TypeError, IndexError):
            if len(re.findall(regex_dict['bmw_part_ref_format'], part_ref)):
                if len(part_ref_desc.values[0]) and any(x in part_ref_desc for x in bmw_original_oil_words):
                    part_ref_ta = '1'
            else:
                level_0_performance_report.log_record('Problematic Part - {} - {}'.format(part_ref, part_ref_desc), project_id, flag=1)
                return 'NO_TA'

    elif not part_ref_unique_ta:
        return 'NO_TA'

    return part_ref, part_ref_ta


def load_model(model_name, project_id):
    print('Loading model(s) for Project {} - {}...'.format(project_id, level_1_b_data_processing.project_dict[project_id]))

    last_date = '4_9_2019'
    # ToDo: this function needs to fetch the max date available

    file_name = 'models/project_{}_{}_best_{}.sav'.format(project_id, model_name, last_date)

    model = pickle.load(file_name)
    print(model)

    return


def solver_dataset_preparation(df_sales, df_part_refs_ta, dtss_goal, pse_code, current_date):
    # start = time.time()

    df_sales['index'] = pd.to_datetime(df_sales['index'], format='%Y-%m-%d')
    df_sales['weekday'] = df_sales['index'].dt.dayofweek

    last_year_date = pd.to_datetime(current_date) - relativedelta(years=1)
    dts_interval_sales = df_sales.tail(dtss_goal)['index'].min()

    weekdays_count = np.busday_count(
        np.array([last_year_date]).astype('datetime64[D]'),
        np.array([dt.datetime.strptime(current_date, '%Y%m%d')]).astype('datetime64[D]')
    )[0]

    df_sales = df_sales[(df_sales['weekday'] >= 0) & (df_sales['weekday'] < 5)]  # Removal of weekend data
    df_sales.set_index('Part_Ref', inplace=True)

    df_sales_grouped = df_sales.groupby('Part_Ref')
    pool = Pool(processes=level_0_performance_report.pool_workers_count)
    results = pool.map(solver_metrics_per_part_ref, [(part_ref, group, last_year_date, dts_interval_sales, weekdays_count) for (part_ref, group) in df_sales_grouped])
    pool.close()
    df_solve = pd.concat([result for result in results if result is not None])

    df_solve = level_1_b_data_processing.df_join_function(df_solve, df_part_refs_ta[['Part_Ref', 'Group']].set_index('Part_Ref'), on='Part_Ref')  # Addition of the Groups Description
    df_solve['PSE_Code'] = pse_code

    df_solve.to_csv('output/df_solver_{}_{}.csv'.format(pse_code, current_date))

    # print('Elapsed Time: {:.2f}'.format(time.time() - start))
    return df_solve


def solver_metrics_per_part_ref(args):
    part_ref, df_sales_filtered, last_year_date, dts_interval_sales, weekdays_count = args
    df_solve = pd.DataFrame(columns=['Part_Ref', 'Cost', 'PVP', 'Margin', 'Last Stock', 'Last Stock Value', 'Last Year Sales', 'Last Year Sales Mean', 'Last Year COGS', 'DII Year', 'DII Year weekdays', 'DII Year weekdays v2', 'DaysToSell_1_Part', 'DaysToSell_1_Part_v2', 'DTS_Total_Qty_Sold', 'DTS_Min_Total_Qty_Sold', 'DTS_Max_Total_Qty_Sold', 'DaysToSell_1_Part_v2_mean', 'DaysToSell_1_Part_v2_median'])

    if df_sales_filtered['Qty_Sold_sum_al'].sum() >= 0 and df_sales_filtered[df_sales_filtered['Qty_Sold_sum_al'] > 0].shape[0] > 1:  # This checkup is needed as I can not enforce it before when processing a time interval, only when processing all data
        df_sales_filtered.set_index('index', inplace=True)  # I should use last purchase date/cost, but first: not all parts have purchases (e.g. BM83.13.9.415.965) and second, they should give the same value.

        last_sale_date = df_sales_filtered[(df_sales_filtered['Qty_Sold_sum_al'] > 0)].index.max()

        if last_sale_date < last_year_date:  # No sales in last year considering the current date
            return None

        df_last_sale = df_sales_filtered.loc[last_sale_date, ['Cost_Sale_avg', 'PVP_avg']].values
        last_cost, last_pvp = df_last_sale[0], df_last_sale[1]

        dts_sales = df_sales_filtered.loc[dts_interval_sales::, 'Qty_Sold_sum_al']
        dts_total_qty_sold = dts_sales.sum()
        dts_min_total_qty_sold = dts_sales[dts_sales >= 0].min()
        dts_max_total_qty_sol = dts_sales.max()

        last_stock = df_sales_filtered['Stock_Qty_al'].tail(1).values[0]
        if last_stock >= 0:
            last_stock_value = last_stock * last_cost

            df_last_year = df_sales_filtered.loc[last_year_date::, 'Qty_Sold_sum_al']
            last_year_sales = df_last_year.sum()
            last_year_sales_avg = df_last_year.mean()

            if last_year_sales:
                last_year_cogs = last_year_sales * last_cost
                margin = (last_pvp - last_cost)

                dii = last_stock_value / last_year_cogs
                dii_year = dii * 365

                avg_sales_per_day = last_stock / last_year_sales_avg  # The diff between dii_year and avg_sales_per_day is caused by the number of non-weekdays;

                avg_sales_per_day_v2 = last_year_sales / weekdays_count

                days_to_sell_1_part = 1 / last_year_sales_avg

                # days_to_sell_1_part_v2 = 1 / avg_sales_per_day_v2

                df_sales_day_diff = df_sales_filtered[df_sales_filtered['Qty_Sold_sum_al'] > 0].index.to_series().diff().tolist()
                df_sales_qty = df_sales_filtered[df_sales_filtered['Qty_Sold_sum_al'] > 0]['Qty_Sold_sum_al'].values.tolist()
                df_sales_day_diff = [x.days for x in df_sales_day_diff]
                df_diff_qty_ratios = [x / y for x, y in zip(df_sales_day_diff[1::], df_sales_qty[1::])]
                df_diff_qty_ratios_mean = np.mean(df_diff_qty_ratios)
                df_diff_qty_ratios_median = np.median(df_diff_qty_ratios)

                df_solve.loc[0, ['Part_Ref', 'Cost', 'PVP', 'Margin', 'Last Stock', 'Last Stock Value', 'Last Year Sales', 'Last Year Sales Mean', 'Last Year COGS', 'DII Year', 'DII Year weekdays', 'DII Year weekdays v2', 'DaysToSell_1_Part', 'DTS_Total_Qty_Sold', 'DTS_Min_Total_Qty_Sold', 'DTS_Max_Total_Qty_Sold', 'DaysToSell_1_Part_v2_mean', 'DaysToSell_1_Part_v2_median']] = \
                    [part_ref, last_cost, last_pvp, margin, last_stock, last_stock_value, last_year_sales, last_year_sales_avg, last_year_cogs, dii_year, avg_sales_per_day, avg_sales_per_day_v2, days_to_sell_1_part, dts_total_qty_sold, dts_min_total_qty_sold, dts_max_total_qty_sol, df_diff_qty_ratios_mean, df_diff_qty_ratios_median]

            else:
                return None
        else:
            return None

    return df_solve

