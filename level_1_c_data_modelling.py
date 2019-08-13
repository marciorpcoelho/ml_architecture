import pandas as pd
import numpy as np
import datetime
import pickle
import time
import nltk
from os import listdir
import operator
from dateutil.relativedelta import relativedelta
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from scipy.spatial.distance import cdist, pdist
from sklearn.model_selection import GridSearchCV
from gap_statistic import OptimalK
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from level_0_performance_report import log_record, pool_workers_count
from level_1_a_data_acquisition import sql_mapping_retrieval
from level_1_b_data_processing import remove_punctuation_and_digits
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
    keyword_dict, ranking_dict = sql_mapping_retrieval(options_file.DSN_MLG, options_file.sql_info['database_final'], ['SDK_Setup_Keywords'], 'Keyword_Group', options_file, multiple_columns=1)
    keyword_dict = keyword_dict[0]

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
                if label in user_dict.keys():
                    user_dict[label].append(user_id)
                else:
                    user_dict[label] = [user_id]
                continue

            if ' ' in keywords:
                rank = 1 + consecutive_flag
                keywords = remove_punctuation_and_digits(keywords)
                tokenized_key_word = nltk.tokenize.word_tokenize(keywords)
                try:
                    tokenized_key_word = [stemmer_pt.stem(x) for x in tokenized_key_word]
                    selected_cols = df_top_words[tokenized_key_word]
                except KeyError:
                    log_record('Palavras chave {} que foram reduzidas a {} não foram encontradas de forma consecutiva.'.format(keywords, tokenized_key_word), options_file.project_id, flag=1)
                    continue

                matched_index = selected_cols[selected_cols == 1].dropna(axis=0).index.values  # returns the requests with the keyword present
                if consecutive_flag:
                    matched_index = consecutive_keyword_testing(df, matched_index, tokenized_key_word)  # out of all the requests with the keywords present, searches them for those where they keywords are consecutive

                if matched_index is not None:
                    requests_dict_2 = request_matches(label, keywords, rank, matched_index, requests_dict_2)

                    if len(matched_index):
                        df_top_words.loc[df_top_words.index.isin(matched_index), 'Label'] = label
                else:
                    log_record('Palavras chave {} que foram reduzidas a {} não foram encontradas de forma consecutiva.'.format(keywords, tokenized_key_word), options_file.project_id, flag=1)

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

    df = requests_draw_handling(df, requests_dict_2, ranking_dict)

    df, _ = user_label_assignment(df, df_top_words, user_dict)

    df.sort_values(by='Request_Num', inplace=True)
    df_top_words.sort_index(inplace=True)

    if [(x, y) for (x, y) in zip(df['Request_Num'].values, df_top_words.index.values) if x != y]:
        unique_requests_df = df['Request_Num'].unique()
        unique_requests_df_top_words = df_top_words.index.values
        log_record('Existem pedidos sem classificação!', options_file.project_id, flag=1)
        log_record('Pedidos não classificados no dataset original: {}'.format([x for x in unique_requests_df if x not in unique_requests_df_top_words]), options_file.project_id, flag=1)
        log_record('Pedidos não classificados no dataset Top_Words: {}'.format([x for x in unique_requests_df_top_words if x not in unique_requests_df]), options_file.project_id, flag=1)
        raise ValueError('Existem pedidos sem classificação!')

    log_record('{:.2f}% de pedidos Não Definidos'.format((df[df['Label'] == 'Não Definido'].shape[0] / df['Request_Num'].nunique()) * 100), options_file.project_id)

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


def user_label_assignment(df, df_top_words, user_dict):
    for key in user_dict.keys():
        matched_requests = df[df['Contact_Customer_Id'].isin([int(user) for user in user_dict[key]])]['Request_Num']
        df_top_words.loc[df_top_words.index.isin(matched_requests), 'Label'] = key
        df.loc[df['Request_Num'].isin(matched_requests), 'Label'] = key
    return df, df_top_words


# Old Version, without Priority
# def requests_draw_handling(df, requests_dict):
#
#     df['Label'] = 'Não Definido'
#     for request in requests_dict.keys():
#         matches_count = len(requests_dict[request])
#         unique_labels_count = len(set([x[0] for x in requests_dict[request]]))
#         labels = [x[0] for x in requests_dict[request]]
#         unique_labels = set(labels)
#         unique_ranks_count = len(set([x[1] for x in requests_dict[request]]))
#         highest_rank_label = max(requests_dict[request], key=operator.itemgetter(1))[0]
#
#         if matches_count > 1 and unique_labels_count > 1:
#             if 'Workspace' in unique_labels:  # Workspace has priority over any other label
#                 df.loc[df['Request_Num'] == request, 'Label'] = 'Workspace'
#             else:
#                 if highest_rank_label == 'Demonstração Resultados' and 'Importador' in unique_labels:  # Importador has priority over any rank of Demonstração Resultados
#                     df.loc[df['Request_Num'] == request, 'Label'] = 'Importador'
#                 else:
#                     if unique_ranks_count == 1 and unique_labels_count != len(labels):
#                         label_counter = Counter(labels)
#                         df.loc[df['Request_Num'] == request, 'Label'] = label_counter.most_common(1)[0][0]
#                     elif unique_ranks_count == 1 and unique_labels_count == len(labels):
#                         # print('DRAW:', request, requests_dict[request])
#                         df.loc[df['Request_Num'] == request, 'Label'] = 'Draw: ' + '+'.join([x[0] for x in requests_dict[request]])
#                     else:
#                         df.loc[df['Request_Num'] == request, 'Label'] = highest_rank_label
#         else:
#             df.loc[df['Request_Num'] == request, 'Label'] = requests_dict[request][0][0]
#
#     return df


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


def part_ref_selection(df_al, current_date):
    # Proj_ID = 2259
    print('Selection of Part Reference')

    last_year_date = pd.to_datetime(current_date) - relativedelta(years=1)

    df_al_filtered = df_al[(df_al['Movement_Date'] > last_year_date) & (df_al['Movement_Date'] <= current_date)]

    all_unique_part_refs = df_al_filtered['Part_Ref'].unique()

    all_unique_part_refs_bm = [x for x in all_unique_part_refs if x.startswith('BM')]
    all_unique_part_refs_mn = [x for x in all_unique_part_refs if x.startswith('MN')]

    all_unique_part_refs = all_unique_part_refs_bm + all_unique_part_refs_mn

    all_unique_part_refs_at = [x for x in all_unique_part_refs if x.endswith('AT')]

    all_unique_part_refs = [x for x in all_unique_part_refs if x not in all_unique_part_refs_at]

    [print('{} has a weird size!'.format(x)) for x in all_unique_part_refs if len(x) > 17 or len(x) < 13]

    print('{} unique part_refs sold between {} and {}.'.format(len(all_unique_part_refs), last_year_date.date(), current_date))

    return all_unique_part_refs


def apv_last_stock_calculation(min_date, max_date, pse_code):
    # Proj_ID = 2259

    results_files = [datetime.datetime.strptime(f[17:25], format('%Y%m%d')) for f in listdir('output/') if f.startswith('results_merge_{}_'.format(pse_code))]
    min_date = datetime.datetime.strptime(min_date, format('%Y%m%d'))
    max_date = datetime.datetime.strptime(max_date, format('%Y%m%d'))

    try:
        max_file_date = np.max(results_files)
        if min_date < max_file_date:
            if max_file_date == max_date:
                raise Exception('All data has been processed already up to date {}.'.format(max_file_date))
            else:
                print('Data already processed from {} to {}. Will adjust accordingly...'.format(min_date, max_file_date))
            return max_file_date
        else:
            return min_date
    except ValueError:
        return min_date


def apv_stock_evolution_calculation(pse_code, selected_parts, df_sales, df_al, df_stock, df_reg_al_clients, df_purchases, min_date, max_date):

    try:
        results = pd.read_csv('output/results_merge_{}_{}.csv'.format(pse_code, max_date), index_col=0)
        print('File results_merge_{}_{} found.'.format(pse_code, max_date))

    except FileNotFoundError:
        print('File results_merge_{} not found. Processing...'.format(pse_code))
        min_date = apv_last_stock_calculation(min_date, max_date, pse_code)

        df_stock.set_index('Record_Date', inplace=True)
        df_purchases.set_index('Movement_Date', inplace=True)
        df_al['Unit'] = df_al['Unit'] * (-1)  # Turn the values to their symmetrical so it matches the other dfs

        i, parts_count = 1, len(selected_parts)
        dataframes_list = [df_sales, df_al, df_stock, df_reg_al_clients, df_purchases]
        datetime_index = pd.date_range(start=min_date, end=max_date)
        results = pd.DataFrame()
        positions = []

        print('PSE_Code = {}'.format(pse_code))
        for part_ref in selected_parts:
            start = time.time()
            result_part_ref, stock_evolution_correct_flag, offset = sql_data([part_ref], pse_code, min_date, max_date, dataframes_list)

            if result_part_ref.shape[0]:
                result_part_ref = result_part_ref.reindex(datetime_index).reset_index().rename(columns={'Unnamed: 0': 'Movement_Date'})

                ffill_and_zero_fill_cols = ['Stock_Qty_al', 'Sales Evolution_al', 'Purchases Evolution', 'Regulated Evolution', 'Purchases Urgent Evolution', 'Purchases Non Urgent Evolution']
                zero_fill_cols = ['Qty_Purchased_sum', 'Qty_Regulated_sum', 'Qty_Sold_sum_al', 'Qty_Purchased_urgent_sum', 'Qty_Purchased_non_urgent_sum', 'Cost_Purchase_avg', 'Cost_Reg_avg', 'Cost_Sale_avg', 'PVP_avg']
                ffill_and_bfill_cols = ['Part_Ref', 'Stock_Qty']

                [result_part_ref[x].fillna(method='ffill', inplace=True) for x in ffill_and_zero_fill_cols + ffill_and_bfill_cols]
                [result_part_ref[x].fillna(0, inplace=True) for x in zero_fill_cols + ffill_and_zero_fill_cols]
                [result_part_ref[x].fillna(method='bfill', inplace=True) for x in ffill_and_bfill_cols]

                if result_part_ref[result_part_ref['Part_Ref'].isnull()].shape[0]:
                    print('null values found for part_ref: \n{}'.format(part_ref))
                    print('Number of null rows: {}'.format(result_part_ref[result_part_ref['Part_Ref'].isnull()].shape))

                result_part_ref.loc[:, 'Stock_Evolution_Correct_Flag'] = stock_evolution_correct_flag
                result_part_ref.loc[:, 'Stock_Evolution_Offset'] = offset

                # Just so the variation matches with the rest: positive regularization means an increase in stock, while negative is a decrease; Equivalent for cost;
                result_part_ref['Qty_Regulated_sum'] = result_part_ref['Qty_Regulated_sum'] * (-1)
                result_part_ref['Cost_Reg_avg'] = result_part_ref['Cost_Reg_avg'] * (-1)

                results = results.append(result_part_ref)
                print('Elapsed time: {:.2f}.'.format(time.time() - start))

            position = int((i / parts_count) * 100)
            if not position % 1:
                if position not in positions:
                    print('{}% completed'.format(position))
                    positions.append(position)

            i += 1
        results.to_csv('output/results_merge_{}_{}.csv'.format(pse_code, max_date))
    return results


def sql_data(selected_part, pse_code, min_date, max_date, dataframes_list):

    df_sales, df_al, df_stock, df_reg_al_clients, df_purchases = dataframes_list[0], dataframes_list[1], dataframes_list[2], dataframes_list[3], dataframes_list[4]
    result, stock_evolution_correct_flag, offset = pd.DataFrame(), 0, 0

    df_sales_filtered = df_sales[(df_sales['Part_Ref'].isin(selected_part)) & (df_sales['Movement_Date'] >= min_date) & (df_sales['Movement_Date'] <= max_date)]
    df_al_filtered = df_al[(df_al['Part_Ref'].isin(selected_part)) & (df_al['Movement_Date'] >= min_date) & (df_al['Movement_Date'] <= max_date)]
    df_purchases_filtered = df_purchases[(df_purchases['Part_Ref'].isin(selected_part)) & (df_purchases.index > min_date) & (df_purchases.index <= max_date)]
    df_stock_filtered = df_stock[(df_stock['Part_Ref'].isin(selected_part)) & (df_stock.index >= min_date) & (df_stock.index <= max_date)]

    df_al_filtered = auto_line_dataset_cleaning(df_sales_filtered, df_al_filtered, df_purchases_filtered, df_reg_al_clients, pse_code)

    df_sales_filtered = df_sales_filtered.drop_duplicates(subset=['Movement_Date', 'Part_Ref']).sort_values(by='Movement_Date')
    df_sales_filtered.set_index('Movement_Date', inplace=True)

    if not df_al_filtered.shape[0]:
        # raise ValueError('No data found for part_ref {} and/or selected period {}/{}'.format(selected_part[0], min_date, max_date))
        # print('\nNo data found for part_ref {} and/or selected period {}/{}.\n'.format(selected_part[0], min_date, max_date))
        # no_data_flag = 1
        return pd.DataFrame(), stock_evolution_correct_flag, offset
    elif df_al_filtered.shape[0] == 1:
        # raise ValueError('Only 1 row found for part_ref {} and/or selected period {}/{}'.format(selected_part[0], min_date, max_date))
        # print('\nOnly 1 row found for part_ref {} and/or selected period {}/{}. Ignored.\n'.format(selected_part[0], min_date, max_date))
        # one_row_only_flag = 1
        return pd.DataFrame(), stock_evolution_correct_flag, offset

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

    if df_al_filtered['Qty_Sold_sum_al'].sum() != 0 and df_al_filtered[df_al_filtered['Qty_Sold_sum_al'] > 0].shape[0] > 1:
        df_al_filtered.drop(['Unit', 'Preço de custo', 'P. V. P', 'regularization_flag'], axis=1, inplace=True)

        df_al_filtered = df_al_filtered.drop_duplicates(subset=['Movement_Date'])
        df_al_filtered.set_index('Movement_Date', inplace=True)

        df_purchases_filtered = df_purchases_filtered.loc[~df_purchases_filtered.index.duplicated(keep='first')]

        qty_sold_al = df_al_filtered[df_al_filtered.index > min_date]['Qty_Sold_sum_al'].sum()
        qty_purchased = df_purchases_filtered['Qty_Purchased_sum'].sum()
        try:
            stock_start = df_stock_filtered[df_stock_filtered.index == min_date]['Stock_Qty'].values[0]
        except IndexError:
            stock_start = 0  # When stock is 0, it is not saved in SQL, hence why the previous line doesn't return any value;
        try:
            stock_end = df_stock_filtered[df_stock_filtered.index == max_date]['Stock_Qty'].values[0]
        except IndexError:
            stock_end = 0  # When stock is 0, it is not saved in SQL, hence why the previous line doesn't return any value;

        reg_value = df_al_filtered['Qty_Regulated_sum'].sum()
        # delta_stock = stock_end - stock_start

        if not reg_value:
            reg_value = 0

        result = pd.concat([df_purchases_filtered[['Qty_Purchased_sum', 'Qty_Purchased_urgent_sum', 'Qty_Purchased_non_urgent_sum', 'Cost_Purchase_avg']], df_al_filtered[['Qty_Regulated_sum', 'Cost_Reg_avg', 'Qty_Sold_sum_al', 'Cost_Sale_avg', 'PVP_avg']]], axis=1, sort=False)
        result['Part_Ref'] = selected_part * result.shape[0]
        try:
            # result['Stock_Qty'] = df_stock_filtered['Stock_Qty'].head(1).values[0]
            result['Stock_Qty'] = stock_start
        except IndexError:
            result['Stock_Qty'] = 0  # Cases when there is no stock information

        result_al = stock_start + qty_purchased - qty_sold_al - reg_value

        if result_al != stock_end:
            offset = stock_end - result_al
            print('Selected Part: {} - Values dont match for AutoLine values - Stock has an offset of {:.2f} \n'.format(selected_part, offset))
        else:
            print('Selected Part: {} - Values for AutoLine are correct :D \n'.format(selected_part))
            stock_evolution_correct_flag = 1

        result['Stock_Qty'].fillna(method='ffill', inplace=True)
        result['Part_Ref'].fillna(method='ffill', inplace=True)
        result.fillna(0, inplace=True)

        # The filters in the evolution columns is to compensate for sales/purchases/etc in the first day, even though the stock for that same day already has those movements into consideration.
        # The stock_qty_al initial value is to compensate for the previous line.
        min_date_next_day = min_date + relativedelta(days=1)
        result['Sales Evolution_al'] = result[min_date_next_day:]['Qty_Sold_sum_al'].cumsum()
        result['Purchases Evolution'] = result[min_date_next_day:]['Qty_Purchased_sum'].cumsum()
        result['Purchases Urgent Evolution'] = result[min_date_next_day:]['Qty_Purchased_urgent_sum'].cumsum()
        result['Purchases Non Urgent Evolution'] = result[min_date_next_day:]['Qty_Purchased_non_urgent_sum'].cumsum()
        result['Regulated Evolution'] = result[min_date_next_day:]['Qty_Regulated_sum'].cumsum()

        result['Stock_Qty_al'] = result['Stock_Qty'] - result['Sales Evolution_al'] + result['Purchases Evolution'] - result['Regulated Evolution']
        result.ix[0, 'Stock_Qty_al'] = stock_start  #
        result.loc[result['Qty_Purchased_sum'] == 0, 'Cost_Purchase_avg'] = 0

        # result.to_csv('output/{}_stock_evolution.csv'.format(selected_part[0]))

    return result, stock_evolution_correct_flag, offset


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

        pool = Pool(processes=int(pool_workers_count))
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
