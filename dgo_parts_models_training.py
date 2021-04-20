import sys
import re
import gc
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from joblib import dump
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, precision_score, confusion_matrix, make_scorer, recall_score, accuracy_score
from modules.level_1_c_data_modelling import classification_model_training
import level_2_pa_part_reference_options as options_file
# Text Features
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from collections import defaultdict, Counter

pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


def main():
    return


def model_training(df_full, cols, tag, clf=None):
    train_X, test_X, train_Y, test_Y, target_map, inv_target_map = dataset_preparation(df_full[cols], tag)
    cm_flag = 0
    df_cm_train, df_cm_test = pd.DataFrame(), pd.DataFrame()
    metrics_dict = {}

    if not clf:
        print('Classification Model not found. Training a new one...')

        scorer = make_scorer(recall_score, average='weighted')
        classes, best_models, running_times = classification_model_training(options_file.gridsearch_parameters.keys(), train_X, train_Y, options_file.gridsearch_parameters, 3, scorer, 2610)
        metrics_dict['Running_Time'] = running_times['lgb']
        clf = best_models['lgb']
        cm_flag = 1

    predictions_test, probabilities_test = model_prediction(clf, test_X, target_map)
    predictions_test_converted_classes = predictions_test
    predictions_test = predictions_test.map(inv_target_map)

    if cm_flag:
        cm = confusion_matrix([inv_target_map[x] for x in test_Y], predictions_test.values, labels=[inv_target_map[x] for x in clf.classes_])
        labels = [inv_target_map[x] for x in clf.classes_]
        df_cm_test = pd.DataFrame(cm, columns=labels, index=labels)
        test_y_ser = pd.Series(test_Y)
        metrics_dict['Precision_Weighted'] = precision_score(test_y_ser, predictions_test_converted_classes, average='weighted', labels=np.unique(predictions_test_converted_classes))
        metrics_dict['Precision_Micro'] = precision_score(test_y_ser, predictions_test_converted_classes, average='micro', labels=np.unique(predictions_test_converted_classes))
        metrics_dict['Precision_Macro'] = precision_score(test_y_ser, predictions_test_converted_classes, average='macro', labels=np.unique(predictions_test_converted_classes))
        metrics_dict['Recall_Weighted'] = recall_score(test_y_ser, predictions_test_converted_classes, average='weighted', labels=np.unique(predictions_test_converted_classes))
        metrics_dict['Recall_Micro'] = recall_score(test_y_ser, predictions_test_converted_classes, average='micro', labels=np.unique(predictions_test_converted_classes))
        metrics_dict['Recall_Macro'] = recall_score(test_y_ser, predictions_test_converted_classes, average='macro', labels=np.unique(predictions_test_converted_classes))
        metrics_dict['Accuracy'] = accuracy_score(test_y_ser, predictions_test_converted_classes)

        # print('\n### Precision Score (Weighted):', metrics_dict['precision_score_weighted'])
        # print('### Precision Score (Micro):', metrics_dict['precision_score_micro'])
        # print('### Precision Score (Macro):\n', metrics_dict['precision_score_macro'])
        # print('\n### Recall Score (Weighted):', metrics_dict['recall_score_weighted'])
        # print('### Recall Score (Micro):', metrics_dict['recall_score_micro'])
        # print('### Recall Score (Macro):\n', metrics_dict['recall_score_macro'])

    predictions_train, probabilities_train = model_prediction(clf, train_X, target_map)
    # predictions_train_converted_classes = predictions_train
    predictions_train = predictions_train.map(inv_target_map)

    if cm_flag:
        labels = [inv_target_map[x] for x in clf.classes_]
        cm = confusion_matrix([inv_target_map[x] for x in train_Y], predictions_train.values, labels=[inv_target_map[x] for x in clf.classes_])
        df_cm_train = pd.DataFrame(cm, columns=labels, index=labels)

    del train_X, test_X, train_Y, test_Y
    gc.collect()

    predictions = pd.concat([predictions_train, predictions_test])
    del predictions_train, predictions_test
    gc.collect()

    probabilities = pd.concat([probabilities_train, probabilities_test])
    del probabilities_train, probabilities_test
    gc.collect()

    ml_dataset_scored = df_full.join(predictions, how='left')
    del df_full
    gc.collect()

    ml_dataset_scored = ml_dataset_scored.join(probabilities, how='left')
    # ml_dataset_scored['Product_Group_DW'] = ml_dataset_scored['__target__'].replace(inv_target_map)
    # ml_dataset_scored.drop('__target__', axis=1, inplace=True)

    return ml_dataset_scored, clf, df_cm_train, df_cm_test, metrics_dict


def dataset_preparation(ml_dataset, tag):
    print('1:{}'.format(ml_dataset.shape))

    categorical_features = [u'Client_Id', u'brand']
    numerical_features = [u'PVP_1_avg', u'Average_Cost_avg']
    text_features = [u'Part_Desc_PT_concat', u'Part_Desc_concat']
    for feature in categorical_features:
        ml_dataset.loc[:, feature] = ml_dataset[feature].apply(coerce_to_unicode)
    for feature in text_features:
        ml_dataset.loc[:, feature] = ml_dataset[feature].apply(coerce_to_unicode)
    for feature in numerical_features:
        ml_dataset.loc[:, feature] = ml_dataset[feature].astype('double')
    print('2:{}'.format(ml_dataset.shape))

    target_map = target_map_creation(ml_dataset)
    inv_target_map = {target_map[label]: label for label in target_map}

    file_handler = open(options_file.inv_target_map.format(tag), 'wb')
    pickle.dump(inv_target_map, file_handler)
    file_handler.close()

    ml_dataset['__target__'] = ml_dataset['Product_Group_DW'].map(str).map(target_map)
    del ml_dataset['Product_Group_DW']
    print('3:{}'.format(ml_dataset.shape))

    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    print('3.5:{}'.format(ml_dataset.shape))
    train, test = train_test_split(ml_dataset, test_size=0.2, stratify=ml_dataset['__target__'])
    print('4:{}'.format(train.shape))
    print('4:{}'.format(test.shape))

    # print('Train data has %i rows and %i columns' % (train.shape[0], train.shape[1]))
    # print('Test data has %i rows and %i columns' % (test.shape[0], test.shape[1]))

    drop_rows_when_missing = ['Part_Ref']
    impute_when_missing = [{'impute_with': u'MEAN', 'feature': u'PVP_1_avg'}, {'impute_with': u'MEAN', 'feature': u'Average_Cost_avg'}]

    # Features for which we drop rows with missing values
    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]
        # print('Dropped missing records in %s' % feature)
    del train['Part_Ref']
    del test['Part_Ref']
    print('4:{}'.format(train.shape))
    print('4:{}'.format(test.shape))

    # Features for which we impute missing values
    for feature in impute_when_missing:
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

    print('5:{}'.format(train.shape))
    print('5:{}'.format(test.shape))
    dummy_values = select_dummy_values(train, categorical_features, limit_dummies=100)
    dummy_encode_dataframe(train, dummy_values, tag, save_enc=1)
    dummy_encode_dataframe(test, dummy_values, tag)

    print('6:{}'.format(train.shape))
    print('6:{}'.format(test.shape))

    rescale_features = {u'Average_Cost_avg': u'AVGSTD', u'PVP_1_avg': u'AVGSTD'}
    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'MINMAX':
            scaler = MinMaxScaler()
            scaler.fit(train[[feature_name]])
            scale = scaler.scale_
        else:
            scaler = StandardScaler()
            scaler.fit(train[[feature_name]])
            scale = scaler.scale_
        if scale == 0.:
            del train[feature_name]
            del test[feature_name]
            # print('Feature %s was dropped because it has no variance' % feature_name)
        else:
            # print('Rescaled %s' % feature_name)
            dump(scaler, options_file.scaler_path.format(feature_name, tag))
            train.loc[:, feature_name] = scaler.transform(train[[feature_name]])
            test.loc[:, feature_name] = scaler.transform(test[[feature_name]])

    print('7:{}'.format(train.shape))
    print('7:{}'.format(test.shape))

    text_svds = {}
    for text_feature in text_features:
        n_components = 50
        text_svds[text_feature] = TruncatedSVD(n_components=n_components)
        s = HashingVectorizer(n_features=100000).transform(train[text_feature])
        dump(s, options_file.hashing_vectorizer_path.format(text_feature, tag))

        text_svds[text_feature].fit(s)
        svd_truncated = text_svds[text_feature]
        dump(svd_truncated, options_file.svd_truncated_path.format(text_feature, tag))

        train_transformed = text_svds[text_feature].transform(s)
        test_transformed = text_svds[text_feature].transform(HashingVectorizer(n_features=100000).transform(test[text_feature]))

        for i in range(0, n_components):
            train.loc[:, text_feature + ":text:" + str(i)] = train_transformed[:, i]
            test.loc[:, text_feature + ":text:" + str(i)] = test_transformed[:, i]

        train.drop(text_feature, axis=1, inplace=True)
        test.drop(text_feature, axis=1, inplace=True)

    print('8:{}'.format(train.shape))
    print('8:{}'.format(test.shape))

    train_x = train.drop('__target__', axis=1)
    test_x = test.drop('__target__', axis=1)
    train_y = np.array(train['__target__'])
    test_y = np.array(test['__target__'])

    train_x = train_x.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_x = duplicate_column_renaming(train_x)
    test_x = test_x.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    test_x = duplicate_column_renaming(test_x)

    print('9:{}'.format(train_x.shape))
    print('9:{}'.format(test_x.shape))

    return train_x, test_x, train_y, test_y, target_map, inv_target_map


def duplicate_column_renaming(df):
    renamer = defaultdict()

    for column_name in df.columns[df.columns.duplicated(keep=False)].tolist():
        if column_name not in renamer:
            renamer[column_name] = [column_name + '_0']
        else:
            renamer[column_name].append(column_name + '_' + str(len(renamer[column_name])))

    df = df.rename(
        columns=lambda col_name: renamer[col_name].pop(0)
        if col_name in renamer
        else col_name
    )
    return df


def model_prediction(model, df_to_predict, target_map):
    model_classes = model.classes_

    start_time = time.time()
    _predictions = model.predict(df_to_predict)
    print("Predictions: --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    _probas = model.predict_proba(df_to_predict)
    print("Probabilities: --- %s seconds ---" % (time.time() - start_time))

    predictions = pd.Series(data=_predictions, index=df_to_predict.index, name='prediction')
    cols = [
        u'probability_of_value_%s' % label
        for (_, label) in sorted([(int(target_map[label]), label) for label in target_map if target_map[label] in model_classes])
    ]
    probabilities = pd.DataFrame(data=_probas, index=df_to_predict.index, columns=cols)
    probabilities['Max_Prob'] = probabilities[cols].max(axis=1)
    probabilities.drop(cols, axis=1, inplace=True)

    # Build scored dataset
    # df_scored = df_to_predict.join(predictions, how='left')
    # del df_to_predict  # Need to delete this so it doesn't reach Memory limits. This df is no longer used after the first merge
    # print('df_scored: {}'.format(df_scored.shape))
    # df_scored = df_scored.join(full_df['__target__'], how='left')
    # df_scored = df_scored.join(probabilities, how='left')
    # df_scored = df_scored.rename(columns={'__target__': 'Product_Group_DW'})

    return predictions, probabilities


def dummy_encode_dataframe(df, dummy_values, tag, save_enc=0):
    for (feature, dummy_values) in dummy_values.items():
        if save_enc:
            encoder = OneHotEncoder(handle_unknown='ignore')
            encoder.fit(df[df[feature].isin(dummy_values)][[feature]])
            dump(encoder, options_file.encoder_path.format(feature, tag))

        for dummy_value in dummy_values:
            dummy_name = u'%s_value_%s' % (feature, coerce_to_unicode(dummy_value))
            df[dummy_name] = (df[feature] == dummy_value).astype(float)
        del df[feature]
        # print('Dummy-encoded feature %s' % feature)


# Only keep the top 100 values
def select_dummy_values(train, features, limit_dummies):
    dummy_values = {}
    for feature in features:
        values = [
            value
            for (value, _) in Counter(train[feature]).most_common(limit_dummies)
        ]
        dummy_values[feature] = values
    return dummy_values


def coerce_to_unicode(x):
    return str(x)


def target_map_creation(df):
    target_map, i = {}, 0
    for value in df['Product_Group_DW'].unique():
        target_map[value] = i
        i += 1

    return target_map


if __name__ == '__main__':
    main()

