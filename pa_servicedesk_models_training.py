import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_score, confusion_matrix, make_scorer, recall_score
import level_2_pa_servicedesk_2244_options as options_file
from modules.level_1_b_data_processing import df_join_function
from modules.level_1_a_data_acquisition import sql_retrieve_df
from modules.level_1_c_data_modelling import classification_model_training

pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
LIMIT_DUMMIES = 100


def main():
    # get data
    df_facts = sql_retrieve_df(options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'], options_file.sql_info['initial_table_facts_durations'], options_file, columns=options_file.model_training_fact_cols, query_filters={'Cost_Centre': '6825'}, parse_dates=options_file.date_columns)
    df_labels = sql_retrieve_df(options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'], options_file.sql_info['final_table'], options_file, parse_dates=options_file.date_columns)

    labeled_requests = first_step(df_facts, df_labels)
    labeled_requests = second_step(labeled_requests)
    labeled_requests = third_step(labeled_requests)
    classified_dataset, non_classified_dataset = fourth_step(labeled_requests)
    non_classified_df_scored = fifth_step(classified_dataset, non_classified_dataset)

    return non_classified_df_scored


def first_step(requests, labels):
    # Join requests and labels on Request_Num
    requests['Request_Num'] = requests['Request_Num'].apply(coerce_to_unicode)
    labels['Request_Num'] = labels['Request_Num'].apply(coerce_to_unicode)

    # how=inner because I only want to keep requests with matching labels. The unmatched requests are PBI requests;
    labeled_requests = df_join_function(requests, labels[['Request_Num', 'StemmedDescription', 'Language', 'Label']].set_index('Request_Num'), on='Request_Num', how='inner')

    return labeled_requests


def second_step(df):
    # Replace Requests with Draws in labels
    df['Label'] = df['Label'].str.replace('Draw: .+', 'Não Definido')

    # Normalize text - Lowercase convertion:
    sel_cols_to_normalize = ['Summary', 'Description']
    df = lowercase_column_conversion(df.copy(), sel_cols_to_normalize)

    # Normalize text - trim
    df = trim_columns(df.copy(), sel_cols_to_normalize)

    # Outlier Cleanup:
    filter_sla_resolution_minutes_lower = df['SLA_Resolution_Minutes'] >= 0.0
    filter_sla_resolution_minutes_upper = df['SLA_Resolution_Minutes'] <= 3600.0
    filter_creation_timespent_lower = df['Creation_TimeSpent'] >= 0.0
    filter_creation_timespent_upper = df['Creation_TimeSpent'] <= 40.0
    filter_timespent_minutes_lower = df['TimeSpent_Minutes'] >= 0.0
    filter_timespent_minutes_upper = df['TimeSpent_Minutes'] <= 90.0
    filter_sla_assignee_minutes_lower = df['SLA_Assignee_Minutes'] >= 0.0
    filter_sla_assignee_minutes_upper = df['SLA_Assignee_Minutes'] <= 360.0
    filter_sla_close_minutes_upper = df['SLA_Close_Minutes'] <= 3227.0
    filter_waitingtime_assignee_minutes_upper = df['WaitingTime_Assignee_Minutes'] <= 20.0

    df = df[filter_sla_resolution_minutes_lower & filter_sla_resolution_minutes_upper
            & filter_creation_timespent_lower & filter_creation_timespent_upper
            & filter_timespent_minutes_lower & filter_timespent_minutes_upper
            & filter_sla_assignee_minutes_lower & filter_sla_assignee_minutes_upper
            & filter_sla_close_minutes_upper
            & filter_waitingtime_assignee_minutes_upper]

    return df


def third_step(df):
    # Filter Label with only a single use:
    value_count_series = df['Label'].value_counts()
    sel_classes = value_count_series[value_count_series > 1].index.values
    df_filtered = df[df['Label'].isin(sel_classes)]

    return df_filtered


def fourth_step(df):
    # Separate classified and non-classified requests
    classified_df = df[df['Label'] != 'Não Definido']
    non_classified_df = df[df['Label'] == 'Não Definido']

    return classified_df, non_classified_df


def fifth_step(classified_df, non_classified_df):
    print('classified_df.shape', classified_df.shape)
    print('non_classified_df.shape', non_classified_df.shape)

    dummy_values = select_dummy_values(pd.concat([x for x in [classified_df, non_classified_df] if x is not None]), options_file.categorical_to_dummy_encode)

    # Model Training
    _, clf_model, _, _, metrics_dict = model_training_dataiku(classified_df, dummy_values)

    print('non_classified_df.shape', non_classified_df.shape)

    # Classification
    print('before:', non_classified_df.shape)
    print('before:', non_classified_df.head())

    all_classified_df, _, _, _, _ = model_training_dataiku(pd.concat([classified_df, non_classified_df]), dummy_values, clf=clf_model)
    print('after:', all_classified_df.shape)
    print('after:', all_classified_df.head())

    non_classified_df_scored = all_classified_df[all_classified_df['Label'] == 'Não Definido']
    print(non_classified_df_scored.shape)
    print(non_classified_df_scored.head())

    return non_classified_df_scored[['Request_Num', 'Label', 'prediction']]


def dataset_preparation(ml_dataset, dummy_values):
    ml_dataset = ml_dataset[
        [u'NumReq_ReOpen_Nr', u'SLA_StdTime_Resolution_Hours', u'Resolve_Date_Orig', u'Request_Type_Orig', u'Label', u'Request_Type', u'SLA_Violation_Orig', u'Close_Date', u'SLA_Resolution_Flag', u'Assignee_Date_Orig', u'WaitingTime_Resolution_Minutes', u'Creation_TimeSpent', u'SLA_Assignee_Minutes_Above', u'SLA_StdTime_Assignee_Hours', u'SLA_Close_Minutes', u'Contact_Assignee_Id', u'SLA_Assignee_Minutes', u'SLA_Resolution_Minutes', u'TimeSpent_Minutes', u'SLA_Id', u'Next_3Days_Minutes',
         u'WaitingTime_Resolution_Minutes_Customer', u'SLA_Assignee_Violation', u'Priority_Id', u'Application_Id', u'Assignee_Date', u'Language', u'Status_Id', u'Resolve_Date', u'WaitingTime_Resolution_Minutes_Internal', u'Creation_Date', u'SLA_Resolution_Minutes_Above', u'Next_1Day_Minutes', u'NumReq_ReOpen', u'Summary', u'Description', u'SLA_Violation', u'WaitingTime_Resolution_Minutes_Supplier', u'StemmedDescription', u'Contact_Customer_Id', u'WaitingTime_Assignee_Minutes',
         u'SLA_Resolution_Violation', u'Open_Date', u'Category_Id', u'Next_2Days_Minutes']]

    ml_dataset = def_data_type_convertion(ml_dataset, options_file.categorical_features, options_file.text_features, options_file.numerical_features)

    target_map = target_map_creation(ml_dataset, 'Label')
    inv_target_map = {target_map[label]: label for label in target_map}
    ml_dataset['__target__'] = ml_dataset['Label'].map(str).map(target_map)

    del ml_dataset['Label']

    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]

    # if ml_dataset['__target__'].nunique() > 1:
    train, test = train_test_split(ml_dataset, test_size=0.2, stratify=ml_dataset['__target__'], random_state=42)
    # elif ml_dataset['__target__'].nunique() == 1:
    #     This is for the case when only a single label is present, which happens when I want to classify the Non Classified Requests.
        # train = ml_dataset
        # test = ml_dataset

    print('Train data has %i rows and %i columns' % (train.shape[0], train.shape[1]))
    print('Test data has %i rows and %i columns' % (test.shape[0], test.shape[1]))

    train, test = def_drop_rows_when_missing(train, test, options_file.drop_rows_when_missing_features)
    train, test = def_impute_when_missing(train, test, options_file.impute_when_missing_features)

    print('1', train.shape)
    print('1', test.shape)
    dummy_encode_dataframe(train, dummy_values)
    print('2', train.shape)
    print('2', test.shape)
    # if ml_dataset['__target__'].nunique() > 1:
    dummy_encode_dataframe(test, dummy_values)  # No need for this, as it applies to both datasets simultaneously
    print('3', train.shape)
    print('3', test.shape)

    train, test = def_rescale_feature(train, test, options_file.rescale_features)
    train, test = def_text_feature_handling(train, test, options_file.text_features)

    # #### Modeling
    train_x = train.drop('__target__', axis=1)
    test_x = test.drop('__target__', axis=1)
    train_y = np.array(train['__target__'])
    test_y = np.array(test['__target__'])

    return train, test, train_x, test_x, train_y, test_y, target_map, inv_target_map


def def_data_type_convertion(ml_dataset, categorical_features, text_features, numerical_features):
    for feature in categorical_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
    for feature in numerical_features:
        if ml_dataset[feature].dtype == np.dtype('M8[ns]'):
            ml_dataset[feature].fillna(ml_dataset[feature].mean(), inplace=True)
            ml_dataset[feature] = ml_dataset[feature].astype('int64') // 1e9
        else:
            ml_dataset[feature] = ml_dataset[feature].astype('double')

    return ml_dataset


def def_text_feature_handling(train, test, text_features):
    text_svds = {}
    for text_feature in text_features:
        n_components = 50
        text_svds[text_feature] = TruncatedSVD(n_components=n_components)
        s = HashingVectorizer(n_features=100000).transform(train[text_feature])
        text_svds[text_feature].fit(s)

        train_transformed = text_svds[text_feature].transform(s)
        test_transformed = text_svds[text_feature].transform(HashingVectorizer(n_features=100000).transform(test[text_feature]))

        for i in range(0, n_components):
            train.loc[:, text_feature + ":text:" + str(i)] = train_transformed[:, i]
            test.loc[:, text_feature + ":text:" + str(i)] = test_transformed[:, i]

        train.drop(text_feature, axis=1, inplace=True)
        # try:
        test.drop(text_feature, axis=1, inplace=True)
        # except KeyError:
        #     continue

    return train, test


def def_rescale_feature(train, test, rescale_features):
    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'MINMAX':
            _min = train[feature_name].min()
            _max = train[feature_name].max()
            scale = _max - _min
            shift = _min
        else:
            shift = train[feature_name].mean()
            scale = train[feature_name].std()
        if scale == 0.:
            del train[feature_name]
            # try:
            del test[feature_name]
            # except KeyError:
            #     continue
            # print('Feature %s was dropped because it has no variance' % feature_name)
        else:
            # print('Rescaled %s' % feature_name)
            train.loc[:, feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
            test.loc[:, feature_name] = (test[feature_name] - shift).astype(np.float64) / scale

    return train, test


def def_impute_when_missing(train, test, impute_when_missing_features):
    # Features for which we impute missing values"
    for feature in impute_when_missing_features:
        if feature['impute_with'] == 'MEAN':
            v = train[feature['feature']].mean()
        elif feature['impute_with'] == 'MEDIAN':
            v = train[feature['feature']].median()
        elif feature['impute_with'] == 'CREATE_CATEGORY':
            v = 'NULL_CATEGORY'
        elif feature['impute_with'] == 'MODE':
            v = train[feature['feature']].value_counts().index[0]
        elif feature['impute_with'] == 'CONSTANT':
            v = feature['value']
        train[feature['feature']] = train[feature['feature']].fillna(v)
        test[feature['feature']] = test[feature['feature']].fillna(v)
        # print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))

    return train, test


def def_drop_rows_when_missing(train, test, drop_rows_when_missing_features):
    # Features for which we drop rows with missing values"
    for feature in drop_rows_when_missing_features:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]
        # print('Dropped missing records in %s' % feature)

    return train, test


def target_map_creation(df, target_col_name):
    target_map, i = {}, 0
    for value in df[target_col_name].unique():
        target_map[value] = i
        i += 1

    return target_map


def model_training_dataiku(ml_dataset, dummy_values, clf=None):
    train, test, train_X, test_X, train_Y, test_Y, target_map, inv_target_map = dataset_preparation(ml_dataset, dummy_values)
    cm_flag = 0
    df_cm_train, df_cm_test = pd.DataFrame(), pd.DataFrame()
    metrics_dict = {}

    if not clf:
        print('Classification Model not found. Training a new one...')
        scorer = make_scorer(recall_score, average='weighted')
        classes, best_models, running_times = classification_model_training(['lr'], train_X, train_Y, options_file.gridsearch_parameters, 3, scorer, options_file.project_id)
        clf = best_models['lr']
        cm_flag = 1

    predictions_test, probabilities_test, text_x_scored = model_prediction(clf, test_X, test, target_map, inv_target_map)
    predictions_test_converted_classes = predictions_test
    predictions_test = predictions_test.map(inv_target_map)

    predictions_train, probabilities_train, train_x_scored = model_prediction(clf, train_X, train, target_map, inv_target_map)
    predictions_train_converted_classes = predictions_train
    predictions_train = predictions_train.map(inv_target_map)

    if cm_flag:
        cm = confusion_matrix([inv_target_map[x] for x in test_Y], predictions_test.values, labels=[inv_target_map[x] for x in clf.classes_])
        labels = [inv_target_map[x] for x in clf.classes_]
        df_cm_test = pd.DataFrame(cm, columns=labels, index=labels)
        test_y_ser = pd.Series(test_Y)
        metrics_dict['precision_score_weighted'] = precision_score(test_y_ser, predictions_test_converted_classes, average='weighted', labels=np.unique(predictions_test_converted_classes))
        metrics_dict['precision_score_micro'] = precision_score(test_y_ser, predictions_test_converted_classes, average='micro', labels=np.unique(predictions_test_converted_classes))
        metrics_dict['precision_score_macro'] = precision_score(test_y_ser, predictions_test_converted_classes, average='macro', labels=np.unique(predictions_test_converted_classes))
        metrics_dict['recall_score_weighted'] = recall_score(test_y_ser, predictions_test_converted_classes, average='weighted', labels=np.unique(predictions_test_converted_classes))
        metrics_dict['recall_score_micro'] = recall_score(test_y_ser, predictions_test_converted_classes, average='micro', labels=np.unique(predictions_test_converted_classes))
        metrics_dict['recall_score_macro'] = recall_score(test_y_ser, predictions_test_converted_classes, average='macro', labels=np.unique(predictions_test_converted_classes))

        print('\n### Precision Score (Weighted):', metrics_dict['precision_score_weighted'])
        print('### Precision Score (Micro):', metrics_dict['precision_score_micro'])
        print('### Precision Score (Macro):\n', metrics_dict['precision_score_macro'])
        print('\n### Recall Score (Weighted):', metrics_dict['recall_score_weighted'])
        print('### Recall Score (Micro):', metrics_dict['recall_score_micro'])
        print('### Recall Score (Macro):\n', metrics_dict['recall_score_macro'])

    if cm_flag:
        labels = [inv_target_map[x] for x in clf.classes_]
        cm = confusion_matrix([inv_target_map[x] for x in train_Y], predictions_train.values, labels=[inv_target_map[x] for x in clf.classes_])
        df_cm_train = pd.DataFrame(cm, columns=labels, index=labels)

    predictions = pd.concat([predictions_train, predictions_test])
    probabilities = pd.concat([probabilities_train, probabilities_test])

    ml_dataset_scored = ml_dataset.join(predictions, how='left')
    ml_dataset_scored = ml_dataset_scored.join(probabilities, how='left')

    return ml_dataset_scored, clf, df_cm_train, df_cm_test, metrics_dict


def model_prediction(model, df_to_predict, full_df, target_map, inv_target_map):
    model_classes = model.classes_

    start_time = time.time()
    _predictions = model.predict(df_to_predict)
    print("Predictions: --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    _probas = model.predict_proba(df_to_predict)
    print("Probabilities: --- %s seconds ---" % (time.time() - start_time))

    # print(df_to_predict.shape)
    # print(df_to_predict.head())
    # print(list(df_to_predict))
    #
    # print(full_df.shape)
    # print(full_df.head())
    # print(list(full_df))

    predictions = pd.Series(data=_predictions, index=df_to_predict.index, name='prediction')
    # cols = [
    #     u'probability_of_value_%s' % label
    #     for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
    # ]
    cols = [
        u'probability_of_value_%s' % label
        for (_, label) in sorted([(int(target_map[label]), label) for label in target_map if target_map[label] in model_classes])
    ]
    probabilities = pd.DataFrame(data=_probas, index=df_to_predict.index, columns=cols)

    # Build scored dataset
    df_scored = df_to_predict.join(predictions, how='left')
    df_scored = df_scored.join(probabilities, how='left')
    df_scored = df_scored.join(full_df['__target__'], how='left')
    df_scored = df_scored.rename(columns={'__target__': 'Product_Group_DW'})

    return predictions, probabilities, df_scored


def coerce_to_unicode(x):
    return str(x)


# Only keep the top 100 values
def select_dummy_values(train, features):
    dummy_values = {}
    for feature in features:
        values = [
            value
            for (value, _) in Counter(train[feature]).most_common(LIMIT_DUMMIES)
        ]
        dummy_values[feature] = values

    return dummy_values


def dummy_encode_dataframe(df, dummy_values):
    for (feature, dummy_values) in dummy_values.items():
        for dummy_value in dummy_values:
            dummy_name = u'%s_value_%s' % (feature, coerce_to_unicode(dummy_value))
            df.loc[:, dummy_name] = (df[feature] == dummy_value).astype(float)
        del df[feature]
        print('Dummy-encoded feature %s' % feature)


def lowercase_column_conversion(df, columns):
    df[columns] = df[columns].apply(lambda x: x.str.lower())

    return df


def trim_columns(df, columns):
    df[columns] = df[columns].apply(lambda x: x.str.strip())

    return df


if __name__ == '__main__':
    main()
