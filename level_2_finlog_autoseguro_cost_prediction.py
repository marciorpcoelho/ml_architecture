import os
import sys
import re
import time
import pickle
import pyodbc
import logging
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import level_2_finlog_autoseguro_cost_prediction_options as options_file
from level_2_finlog_autoseguro_cost_prediction_options import project_id
from modules.level_1_e_deployment import odbc_connection_creation
from modules.level_1_a_data_acquisition import sql_retrieve_df_specified_query
from modules.level_1_c_data_modelling import ClassificationTraining
from modules.level_0_performance_report import log_record, project_dict, performance_info_append

import sklearn as sk
from collections import defaultdict, Counter

import lightgbm as lgb
from joblib import dump

pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S @ %d/%m/%y', filename=options_file.log_files['full_log'], filemode='a')
logging.Logger('errors')
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # Allows the stdout to be seen in the console
logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))  # Allows the stderr to be seen in the console


def main():
    log_record('Projeto: {}'.format(project_dict[options_file.project_id]), options_file.project_id)

    df = data_acquisition()
    # df = pd.read_csv('dbs/dataset_train_20200817_v5.csv')

    df_prob_cost = data_modelling(df)

    deployment(df_prob_cost, options_file.DATA_PROB_PATH_ALL_COST)
    return


def data_acquisition():
    performance_info_append(time.time(), 'Section_A_Start')
    log_record('Início Secção A...', project_id)

    df = sql_retrieve_df_specified_query(options_file.DSN_MLG_PRD, options_file.sql_info['database_mlg'], options_file, options_file.get_train_dataset_query)

    # df.to_csv('dbs/dataset_train_20200817_v6.csv', index=False)

    log_record('Fim Secção A.', project_id)
    performance_info_append(time.time(), 'Section_A_End')
    return df


def data_modelling(df):
    performance_info_append(time.time(), 'Section_D_Start')
    log_record('Início Secção D...', project_id)

    start_date_3_years_ago = date.today() + relativedelta(months=-37)  # This split date is to ensure around 20% of test dataset size
    split_date = date(start_date_3_years_ago.year, start_date_3_years_ago.month, 1)

    df = feat_eng(df)
    pickle_dict('Customer_Group', 'Customer_Name', df)  # Creates a dictionary with all Customer Groups/Companies to be used in the streamlit app;
    enc_LL, df = custom_ohenc_v2('LL', df)
    enc_AR, df = custom_ohenc_v2('AR', df)
    enc_FI, df = custom_ohenc_v2('FI', df)
    enc_Make, df = custom_ohenc_v2('Make', df)
    enc_Fuel, df = custom_ohenc_v2('Fuel', df)
    enc_Vehicle_Tipology, df = custom_ohenc_v2('Vehicle_Tipology', df)
    # enc_Client_type, df = custom_ohenc_v2('Client_type', df)
    enc_Num_Vehicles_Total, df = custom_ohenc_v2('Num_Vehicles_Total', df)
    enc_Num_Vehicles_Finlog, df = custom_ohenc_v2('Num_Vehicles_Finlog', df)
    enc_Customer_Group, df = custom_ohenc_v2('Customer_Group', df)
    # enc_Customer_Name = create_encoder('Customer_Name', df)
    # enc_Customer_Name = custom_ohenc_v2('Customer_Name', df)

    dump(enc_LL, 'models/enc_LL.joblib')
    dump(enc_AR, 'models/enc_AR.joblib')
    dump(enc_FI, 'models/enc_FI.joblib')
    dump(enc_Make, 'models/enc_Make.joblib')
    dump(enc_Fuel, 'models/enc_Fuel.joblib')
    dump(enc_Vehicle_Tipology, 'models/enc_Vehicle_Tipology.joblib')
    # dump(enc_Client_type, 'models/enc_Client_type.joblib')
    dump(enc_Num_Vehicles_Total, 'models/enc_Num_Vehicles_Total.joblib')
    dump(enc_Num_Vehicles_Finlog, 'models/enc_Num_Vehicles_Finlog.joblib')
    dump(enc_Customer_Group, 'models/enc_Customer_Group.joblib')
    # dump(enc_Customer_Name, 'models/enc_Customer_Name.joblib')

    columns_to_drop = [
        # 'contract_duration',
        # 'Contract_km',
        # 'Color Coach-work',
        # 'Description',
        # 'Type',
        # 'Model',
        # 'Registration Date',
        # 'FI_Code',
        # 'LA_Code',
        # 'LL_Code',
        # 'PI_Code',
        # 'AR_Code',
        # 'AR_Franchise',
        'contract_customer',
        # 'Customer_Group',
        'contract_contract',
        'Vehicle_No',
        'Accident_No',
        # 'target_accident',
        # 'target_accident',
        # 'LL_Description',
        # 'AR_Description',
        # 'PI_Description',
        # 'LA_Description',
        # 'FI_Description',
        'contract_start_date',
        'contract_end_date',
        'Customer_Name'
    ]

    df_prob_target_cost = model_training(df, 'target_cost', split_date, columns_to_drop + ['target_qiv', 'target_dp'], options_file.MODEL_PATH)
    df_prob_target_qiv = model_training(df, 'target_qiv', split_date, columns_to_drop + ['target_cost', 'target_dp'], options_file.MODEL_PATH_QIV)
    df_prob_target_dp = model_training(df, 'target_dp', split_date, columns_to_drop + ['target_qiv', 'target_cost'], options_file.MODEL_PATH_DP)
    df_prob_target_cost['target_qiv'] = df_prob_target_qiv['target_qiv']
    df_prob_target_cost['pred_prob_qiv'] = df_prob_target_qiv['pred_prob']
    df_prob_target_cost['target_dp'] = df_prob_target_dp['target_dp']
    df_prob_target_cost['pred_prob_dp'] = df_prob_target_dp['pred_prob']

    log_record('Fim Secção D.', project_id)
    performance_info_append(time.time(), 'Section_D_End')
    return df_prob_target_cost


def model_training(df, target_col, split_date, columns_to_drop, model_path):

    # Cell 1
    train_X = df[df['contract_start_date'] < str(split_date)].reset_index(drop=True)
    train_X = train_X.drop(columns_to_drop, axis=1)
    test_X = df[df['contract_start_date'] > str(split_date)].reset_index(drop=True)
    test_X = test_X.drop(columns_to_drop, axis=1)

    train_X_original = train_X.copy()
    test_X_original = test_X.copy()

    train_X[target_col + '_accident'] = 0
    test_X[target_col + '_accident'] = 0
    train_X.loc[~train_X[target_col].isna(), target_col + '_accident'] = 1
    test_X.loc[~test_X[target_col].isna(), target_col + '_accident'] = 1

    train_y_cost = train_X.pop(target_col).reset_index(drop=True)
    test_y_cost = test_X.pop(target_col).reset_index(drop=True)
    # train_y_accident = train_X.pop('target_accident').reset_index(drop=True)
    # test_y_accident = test_X.pop('target_accident').reset_index(drop=True)
    train_y_accident = train_X.pop(target_col + '_accident').reset_index(drop=True)
    test_y_accident = test_X.pop(target_col + '_accident').reset_index(drop=True)

    # Cells 3 and 4
    train_X.columns = train_X.columns.str.encode('ascii', errors='ignore')
    test_X.columns = test_X.columns.str.encode('ascii', errors='ignore')
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')

    # Cell 5
    gridsearch_flag = 0
    if not gridsearch_flag:
        clf_best = lgb.LGBMClassifier()
        clf_best.fit(train_X, train_y_accident)
    if gridsearch_flag:
        all_clf = ClassificationTraining(clf=lgb.LGBMClassifier)
        all_clf.grid_search(parameters=options_file.gridsearch_parameters['lgb'][1], k=10, score='roc_auc')
        all_clf.clf_grid_fit(x=train_X, y=train_y_accident)
        clf_best = all_clf.grid.best_estimator_
        clf_best.fit(train_X, train_y_accident)
    dump(clf_best, model_path)

    # Cell 6
    # predict the results
    y_pred_clf = clf_best.predict(test_X)

    # Cell 7
    # predict the probabilities
    pred_prob = clf_best.predict_proba(test_X)

    # Cells 8+
    y_pred_prob = clf_best.predict_proba(test_X)[:, 1]
    fpr, tpr, _ = sk.metrics.roc_curve(test_y_accident, y_pred_prob)
    auc = sk.metrics.roc_auc_score(test_y_accident, y_pred_prob)
    print('AUC Score is {:.3f}'.format(auc))

    df_test_prob = pd.concat([
        test_X_original,
        pd.DataFrame(
            y_pred_clf,
            columns=['pred'])
    ], axis=1)

    df_test_prob = pd.concat([
        df_test_prob,
        pd.DataFrame(
            y_pred_prob,
            columns=['pred_prob'])
    ], axis=1)

    y_pred_clf_train = clf_best.predict(train_X)
    y_pred_prob_train = clf_best.predict_proba(train_X)

    df_train_prob = pd.concat([
        train_X_original,
        pd.DataFrame(
            y_pred_clf_train,
            columns=['pred'])
    ], axis=1)

    df_train_prob = pd.concat([
        df_train_prob,
        pd.DataFrame(
            y_pred_prob,
            columns=['pred_prob'])
    ], axis=1)

    df_train_test_prob = pd.concat([df_test_prob, df_train_prob], axis=0)

    print("Test dataset is ", 100 * round(test_X.shape[0] / df.shape[0], 3), "% of the total")

    return df_train_test_prob


def deployment(df, file_path):

    df.to_csv(file_path)  # ToDo: Upload to SQL?

    return


def pickle_dict(key_col, value_col, df):

    df_in = df[[key_col, value_col]].copy()
    df_in.drop_duplicates(inplace=True)

    from collections import defaultdict
    d = defaultdict(list)
    for i, j in zip(df[key_col], df[value_col]):
        if j in d[i]:
            pass
        else:
            d[i].append(j)

    file_name = options_file.Customer_Group_dict_path

    file_handler = open(file_name, 'wb')
    pickle.dump(d, file_handler)
    file_handler.close()

    return


def feat_eng(df_in):
    df = df_in.copy()

    df['Power_Weight_Ratio'] = df['Power_kW'] / df['Weight_Empty']
    df['Contract_km'] = df['Contract_km'] / 1000
    # df['Num_Vehicles_Ratio'] = df['Num_Vehicles_Finlog'] / df['Num_Vehicles_Total']

    # contract start month
    df['contract_start_month'] = df['contract_start_date'].dt.month
    # df['Fuel'] = df['Fuel'].astype(str)
    # df['Traction'] = df['Traction'].astype(str)
    # df['km_per_month'] = df['Contract_km'] / df['contract_duration']

    # create additional column, representing accident vs no accident
    # df['target_accident'] = 0
    # df.loc[~df.target.isna(), 'target_accident'] = 1

    # change target column name, representing the cost
    df['target_cost'] = df.target
    df['target_qiv'] = df.target_QIV
    df['target_dp'] = df.target_DP

    # df['target_accident_qiv'] = 0
    # df.loc[~df.target_qiv.isna(), 'target_accident_qiv'] = 1
    # df['target_accident_dp'] = 0
    # df.loc[~df.target_dp.isna(), 'target_accident_dp'] = 1

    df = df.drop(['target', 'target_QIV', 'target_DP'], axis=1)
    # df['target_cost'] = df['target_cost'].fillna(0)
    # df['target_qiv'] = df['target_qiv'].fillna(0)
    # df['target_dp'] = df['target_dp'].fillna(0)

    values = {
        'Mean_repair_value_cust_full': 0,
        'Sum_contrat_km_full': 0,
        'Sum_repair_value_full': 0,
        'Mean_accident_rel_date_cust_full': 0,
        'Mean_contract_duration_cust_full': 0,
        'Mean_monthly_repair_cost_cust_full': 0,
        'Mean_repair_value_cust_full.1': 0,
        'Mean_repair_value_cust_1year': 0,
        'Mean_accident_rel_date_cust_1year': 0,
        'Mean_contract_duration_cust_1year': 0,
        'Mean_monthly_repair_cost_cust_1year': 0,
        'Mean_repair_value_cust_1year.1': 0,
        'Mean_repair_value_cust_5year': 0,
        'Mean_accident_rel_date_cust_5year': 0,
        'Mean_contract_duration_cust_5year': 0,
        'Mean_monthly_repair_cost_cust_5year': 0,
        'Mean_repair_value_cust_5year.1': 0,
        # 'LL_Description': '0',
        # 'AR_Description': '0',
        # 'PI_Description': '0',
        # 'LA_Description': '0',
        # 'FI_Description': '0'
    }
    df = df.fillna(value=values)

    df.loc[df['LL'].str.startswith('€50.000.000'), 'LL'] = '€50.000.000'
    df['AR'] = df['AR'].str.extract(r'^(.+%)')
    df.loc[df['FI'].str.startswith('Até €1.000/Ano'), 'FI'] = 'Até €1.000/Ano'

    values = {
        'LL': '0',
        'AR': '0',
        'FI': '0',
        'Fuel': '0'
    }
    df = df.fillna(value=values)

    return df


def custom_ohenc(col, df_train_in, df_test_in):

    df_train = df_train_in.copy()
    df_test = df_test_in.copy()

    enc = sk.preprocessing.OneHotEncoder(handle_unknown='ignore')

    # fit the encoder
    enc.fit(df_train_in[[col]])

    # process train df
    df_train = pd.concat([
        df_train,
        pd.DataFrame(
            enc.transform(df_train[[col]]).toarray(),
            columns=col + '_' + enc.get_feature_names()
        )
    ], axis=1).drop([col], axis=1)

    # process test df
    df_test = pd.concat([
        df_test,
        pd.DataFrame(
            enc.transform(df_test[[col]]).toarray(),
            columns=col + '_' + enc.get_feature_names())
    ], axis=1).drop([col], axis=1)

    return enc, df_train, df_test


def custom_ohenc_v2(col, df_in):

    df = df_in.copy()

    enc = create_encoder(col, df)

    # process train df
    df = pd.concat([
        df,
        pd.DataFrame(
            enc.transform(df[[col]]).toarray(),
            columns=col + '_' + enc.get_feature_names()
        )
    ], axis=1).drop([col], axis=1)

    return enc, df


def create_encoder(col, df_in):

    encoder = sk.preprocessing.OneHotEncoder()

    # fit the encoder
    encoder.fit(df_in[[col]])

    return encoder


def apply_ohenc(col, df_apply_in, enc):

    df_apply = df_apply_in.copy()

    # process test df
    df_apply = pd.concat([
        df_apply,
        pd.DataFrame(
            enc.transform(df_apply[[col]]).toarray(),
            columns=col + '_' + enc.get_feature_names())
    ], axis=1).drop([col], axis=1)

    return df_apply


if __name__ == '__main__':
    main()
