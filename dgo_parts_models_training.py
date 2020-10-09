import sys
import numpy as np
import pandas as pd
import sklearn as sk
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, precision_score, confusion_matrix
# Text Features
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

from collections import defaultdict, Counter

pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


def main():
    return
    # main_families_model_training(df)
    # other_families_model_training(df)


def model_training(ml_dataset, clf=None):
    train, test, train_X, test_X, train_Y, test_Y, target_map, inv_target_map = dataset_preparation(ml_dataset)

    if not clf:
        print('Classification Model not found. Training a new one...')
        clf = LogisticRegression(penalty="l2", random_state=1337, max_iter=500)

        start_time = time.time()
        clf.fit(train_X, train_Y)

        print("Fitting: --- %s seconds ---" % (time.time() - start_time))

    predictions_test, probabilities_test, text_x_scored = model_prediction(clf, test_X, test, target_map, inv_target_map)
    print(test_Y, predictions_test)
    # print(confusion_matrix(test_Y.astype(str), predictions_test.astype(str), labels=[inv_target_map[x] for x in clf.classes_]))
    print(confusion_matrix([inv_target_map[x] for x in test_Y], predictions_test.values, labels=[inv_target_map[x] for x in clf.classes_]))
    # test_y_ser = pd.Series(test_Y)
    # print('Precision Score:', precision_score(test_y_ser, predictions, average='weighted'))
    # print('Precision Score:', precision_score(test_y_ser, predictions, average='micro'))
    # print('Precision Score:', precision_score(test_y_ser, predictions, average='macro'))

    predictions_train, probabilities_train, train_x_scored = model_prediction(clf, train_X, train, target_map, inv_target_map)
    print(train_Y, predictions_train)
    print(confusion_matrix([inv_target_map[x] for x in train_Y], predictions_train.values, labels=[inv_target_map[x] for x in clf.classes_]))
    # print(confusion_matrix(train_Y.astype(str), predictions_train.astype(str), labels=[inv_target_map[x] for x in clf.classes_]))
    # train_y_ser = pd.Series(train_Y)
    # print('Precision Score:', precision_score(train_y_ser, predictions, average='weighted'))
    # print('Precision Score:', precision_score(train_y_ser, predictions, average='micro'))
    # print('Precision Score:', precision_score(train_y_ser, predictions, average='macro'))

    predictions = pd.concat([predictions_train, predictions_test])
    probabilities = pd.concat([probabilities_train, probabilities_test])

    ml_dataset_scored = ml_dataset.join(predictions, how='left')
    ml_dataset_scored = ml_dataset_scored.join(probabilities, how='left')
    # print(ml_dataset_scored.head())
    # ml_dataset_scored['Product_Group_DW'] = ml_dataset_scored['__target__'].replace(inv_target_map)
    # ml_dataset_scored.drop('__target__', axis=1, inplace=True)

    return ml_dataset_scored, clf


def dataset_preparation(ml_dataset):
    print('Base data has %i rows and %i columns' % (ml_dataset.shape[0], ml_dataset.shape[1]))
    # Five first records",
    ml_dataset = ml_dataset[[u'PLR_Account_first', u'Product_Group_DW', u'PVP_1_avg', u'Part_Desc_PT_concat', u'Client_Id', u'Part_Desc_concat', u'Part_Ref', u'Average_Cost_avg']]

    categorical_features = [u'PLR_Account_first', u'Client_Id', u'Part_Ref']
    numerical_features = [u'PVP_1_avg', u'Average_Cost_avg']
    text_features = [u'Part_Desc_PT_concat', u'Part_Desc_concat']
    # from dataiku.doctor.utils import datetime_to_epoch
    for feature in categorical_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
    for feature in numerical_features:
        # if ml_dataset[feature].dtype == np.dtype('M8[ns]'):
        #     ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
        # else:
        ml_dataset[feature] = ml_dataset[feature].astype('double')

    target_map = target_map_creation(ml_dataset)
    inv_target_map = {target_map[label]: label for label in target_map}
    ml_dataset['__target__'] = ml_dataset['Product_Group_DW'].map(str).map(target_map)
    del ml_dataset['Product_Group_DW']

    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    train, test = train_test_split(ml_dataset, test_size=0.2, stratify=ml_dataset['__target__'])

    print('Train data has %i rows and %i columns' % (train.shape[0], train.shape[1]))
    print('Test data has %i rows and %i columns' % (test.shape[0], test.shape[1]))

    drop_rows_when_missing = [u'Part_Ref']
    impute_when_missing = [{'impute_with': u'MEAN', 'feature': u'PVP_1_avg'}, {'impute_with': u'MEAN', 'feature': u'Average_Cost_avg'}]

    # Features for which we drop rows with missing values"
    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]
        print('Dropped missing records in %s' % feature)

    # Features for which we impute missing values"
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
        print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))

    categorical_to_dummy_encode = [u'PLR_Account_first', u'Client_Id', u'Part_Ref']
    dummy_values = select_dummy_values(train, categorical_to_dummy_encode, limit_dummies=100)
    dummy_encode_dataframe(train, dummy_values)
    dummy_encode_dataframe(test, dummy_values)

    rescale_features = {u'Average_Cost_avg': u'AVGSTD', u'PVP_1_avg': u'AVGSTD'}
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
            del test[feature_name]
            print('Feature %s was dropped because it has no variance' % feature_name)
        else:
            print('Rescaled %s' % feature_name)
            train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
            test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale

    text_svds = {}
    for text_feature in text_features:
        n_components = 50
        text_svds[text_feature] = TruncatedSVD(n_components=n_components)
        s = HashingVectorizer(n_features=100000).transform(train[text_feature])
        text_svds[text_feature].fit(s)
        train_transformed = text_svds[text_feature].transform(s)

        test_transformed = text_svds[text_feature].transform(HashingVectorizer(n_features=100000).transform(test[text_feature]))

        for i in range(0, n_components):
            train[text_feature + ":text:" + str(i)] = train_transformed[:,i]

            test[text_feature + ":text:" + str(i)] = test_transformed[:,i]

        train.drop(text_feature, axis=1, inplace=True)
        test.drop(text_feature, axis=1, inplace=True)

    train_x = train.drop('__target__', axis=1)
    test_x = test.drop('__target__', axis=1)
    train_y = np.array(train['__target__'])
    test_y = np.array(test['__target__'])

    return train, test, train_x, test_x, train_y, test_y, target_map, inv_target_map


def other_families_model_training(ml_dataset):
    print('Base data has %i rows and %i columns' % (ml_dataset.shape[0], ml_dataset.shape[1]))
    # Five first records",
    ml_dataset.head(5)
    ml_dataset = ml_dataset[[u'PLR_Account_first', u'Product_Group_DW', u'PVP_1_avg', u'Part_Desc_PT_concat', u'Client_Id', u'Part_Desc_concat', u'Part_Ref', u'Average_Cost_avg']]

    categorical_features = [u'PLR_Account_first', u'Client_Id', u'Part_Ref']
    numerical_features = [u'PVP_1_avg', u'Average_Cost_avg']
    text_features = [u'Part_Desc_PT_concat', u'Part_Desc_concat']
    for feature in categorical_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
    for feature in numerical_features:
        # if ml_dataset[feature].dtype == np.dtype('M8[ns]'):
        #     ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
        # else:
        ml_dataset[feature] = ml_dataset[feature].astype('double')

    target_map = target_map_creation(ml_dataset)
    inv_target_map = {target_map[label]: label for label in target_map}

    ml_dataset['__target__'] = ml_dataset['Product_Group_DW'].map(str).map(target_map)
    del ml_dataset['Product_Group_DW']

    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    train, test = train_test_split(ml_dataset, test_size=0.2, stratify=ml_dataset['__target__'])

    print('Train data has %i rows and %i columns' % (train.shape[0], train.shape[1]))
    print('Test data has %i rows and %i columns' % (test.shape[0], test.shape[1]))

    drop_rows_when_missing = [u'Part_Ref']
    impute_when_missing = [{'impute_with': u'MEAN', 'feature': u'PVP_1_avg'}, {'impute_with': u'MEAN', 'feature': u'Average_Cost_avg'}]

    # Features for which we drop rows with missing values"
    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]
        print('Dropped missing records in %s' % feature)

    # Features for which we impute missing values"
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
        print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))

    categorical_to_dummy_encode = [u'PLR_Account_first', u'Client_Id', u'Part_Ref']
    dummy_values = select_dummy_values(train, categorical_to_dummy_encode, limit_dummies=100)
    dummy_encode_dataframe(train, dummy_values)
    dummy_encode_dataframe(test, dummy_values)

    rescale_features = {u'Average_Cost_avg': u'AVGSTD', u'PVP_1_avg': u'AVGSTD'}
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
            del test[feature_name]
            print('Feature %s was dropped because it has no variance' % feature_name)
        else:
            print('Rescaled %s' % feature_name)
            train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
            test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale

    text_svds = {}
    for text_feature in text_features:
        n_components = 50
        text_svds[text_feature] = TruncatedSVD(n_components=n_components)
        s = HashingVectorizer(n_features=100000).transform(train[text_feature])
        text_svds[text_feature].fit(s)
        train_transformed = text_svds[text_feature].transform(s)

        test_transformed = text_svds[text_feature].transform(HashingVectorizer(n_features=100000).transform(test[text_feature]))

        for i in range(0, n_components):
            train[text_feature + ":text:" + str(i)] = train_transformed[:, i]

            test[text_feature + ":text:" + str(i)] = test_transformed[:, i]

        train.drop(text_feature, axis=1, inplace=True)

        test.drop(text_feature, axis=1, inplace=True)

    train_X = train.drop('__target__', axis=1)
    test_X = test.drop('__target__', axis=1)
    train_Y = np.array(train['__target__'])
    test_Y = np.array(test['__target__'])

    clf = LogisticRegression(penalty="l2", random_state=1337, max_iter=500)

    start_time = time.time()
    clf.fit(train_X, train_Y)
    print("Fitting: --- %s seconds ---" % (time.time() - start_time))

    return clf
    # predictions_test, probabilities_test, text_x_scored = model_prediction(clf, test_X, test, target_map, inv_target_map)
    # # test_y_ser = pd.Series(test_Y)
    # # print('Precision Score:', precision_score(test_y_ser, predictions, average='weighted'))
    # # print('Precision Score:', precision_score(test_y_ser, predictions, average='micro'))
    # # print('Precision Score:', precision_score(test_y_ser, predictions, average='macro'))
    #
    # predictions_train, probabilities_train, train_x_scored = model_prediction(clf, train_X, train, target_map, inv_target_map)
    # # train_y_ser = pd.Series(train_Y)
    # # print('Precision Score:', precision_score(train_y_ser, predictions, average='weighted'))
    # # print('Precision Score:', precision_score(train_y_ser, predictions, average='micro'))
    # # print('Precision Score:', precision_score(train_y_ser, predictions, average='macro'))
    #
    # predictions = pd.concat([predictions_train, predictions_test])
    # probabilities = pd.concat([probabilities_train, probabilities_test])
    #
    # ml_dataset_scored = ml_dataset.join(predictions, how='left')
    # ml_dataset_scored = ml_dataset_scored.join(probabilities, how='left')
    # ml_dataset_scored['Product_Group_DW'] = ml_dataset_scored['__target__'].replace(inv_target_map)
    #
    # ml_dataset_scored.drop('__target__', axis=1, inplace=True)
    # print(ml_dataset_scored.head())


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

    predictions = predictions.map(inv_target_map)

    return predictions, probabilities, df_scored


def dummy_encode_dataframe(df, dummy_values):
    for (feature, dummy_values) in dummy_values.items():
        for dummy_value in dummy_values:
            dummy_name = u'%s_value_%s' % (feature, coerce_to_unicode(dummy_value))
            df[dummy_name] = (df[feature] == dummy_value).astype(float)
        del df[feature]
        print('Dummy-encoded feature %s' % feature)


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
    # target_map = {u'O. Mota': 6, u'O. Merchandising': 7, u'O. Colis\xe3o': 1, u'O. Manuten\xe7\xe3o': 2, u'O. Consum\xedveis': 3, u'O. Diversos': 5, u'O. Repara\xe7\xe3o': 0, u'O. Acess\xf3rios': 4}
    # target_map = {u'75/77': 16, u'61': 40, u'178': 26, u'82': 29, u'139': 33, u'52': 14, u'24': 34, u'81': 46, u'49': 1, u'46': 22, u'47': 42, u'43': 30, u'41': 45, u'3': 15, u'5': 8, u'4': 24, u'7': 4, u'6': 27, u'9': 12, u'8': 7, u'99': 10, u'76': 13, u'38': 25, u'73': 0, u'72': 18, u'102': 41, u'100': 43, u'92': 21, u'95': 37, u'94': 47, u'97': 23, u'11': 17, u'10': 9, u'13': 20, u'12': 6, u'15': 19, u'14': 11, u'17': 39, u'98': 2, u'33': 5, u'32': 38, u'30': 32, u'51': 28, u'35': 35, u'34': 3, u'19': 31, u'74': 44, u'162': 36}

    target_map, i = {}, 0
    for value in df['Product_Group_DW'].unique():
        target_map[value] = i
        i += 1

    return target_map


if __name__ == '__main__':
    main()

