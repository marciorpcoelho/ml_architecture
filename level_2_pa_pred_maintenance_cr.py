import sys
import time
import logging
import numpy as np
import pandas as pd
import time
from pyod.models.lof import LOF
from pyod.models.lmdd import LMDD
from pyod.models.iforest import IsolationForest
from pyod.models.knn import KNN
from traceback import format_exc
import level_2_pa_pred_maintenance_cr_options as options_file
from modules.level_1_a_data_acquisition import read_csv, sql_retrieve_df, sql_mapping_retrieval
from modules.level_1_b_data_processing import null_analysis


def main():
    df_initial = data_acquisition(options_file)
    data_processing(df_initial)

    # print(df_initial[df_initial['Chassis_Number'] == 'WVWZZZ1KZBW021958'])


def data_acquisition(options_file_in):

    try:
        df = pd.read_csv(options_file_in.temp_file_loc)
    except FileNotFoundError:
        print('File not found, retrieving from SQL...')
        df = sql_retrieve_df(options_file.DSN_MLG_DEV, options_file_in.sql_info['database_source'], options_file_in.sql_info['source_table'], options_file_in)
        df.to_csv(options_file_in.temp_file_loc, index=False)

    return df.sort_values(by=['Movement_Date', 'WIP_Number'], na_position='first')


def data_processing(df):
    # print(df.loc[(df['Chassis_Number'] == 'WBA2B31000V238871') & (df['Owner_Account'] == 'C256174'), :])
    # null_analysis(df)

    # test_df = df.loc[(df['Chassis_Number'] == 'WBA2B31000V238871') & (df['Owner_Account'] == 'C256174'), :]

    # Negative KM Evolution Handling
    keys = ['Chassis_Number', 'Owner_Account']

    try:
        df = pd.read_csv(options_file.temp_file_grouped_loc).sort_values(by=['Movement_Date', 'WIP_Number'], na_position='first')
    except FileNotFoundError:
        print('File not found, calculating differences between kms...')
        # df = df[df['Chassis_Number'] == 'KL1TF4839CB007249']
        start = time.time()

        # Calculating the KMs evolution.
        start_kms_evo = time.time()
        df['Kms_diff'] = df.groupby(keys)['Kms'].diff().fillna(0)
        print('Kms Diff creation time: {:.2f}s'.format(time.time() - start_kms_evo))

        negative_values_evolution_control(df, 'Initial Step')

        start_scenario_1_fix = time.time()
        # Scenario 1 - Same day WIP_Numbers with different KMs. Currently choosing the highest value.  # ToDo Add verbose to validate the removed rows and/or the chosen chassis_numbers and owner_accounts
        idx = df.groupby(keys + ['Movement_Date'])['Kms'].transform(max) == df['Kms']
        df = df[idx]
        print('Scenario 1 fix time: {:.2f}s'.format(time.time() - start_scenario_1_fix))
        negative_values_evolution_control(df, 'After Scenario 1')

        # Scenario 2 - Lower Kms between two identical Km measures.
        chassis_numbers_w_negative_evolution = df[df['Kms_diff'] < 0]['Chassis_Number']
        owner_accounts_w_negative_evolution = df[df['Kms_diff'] < 0]['Owner_Account']
        df_with_negative_evolutions = df.loc[(df['Chassis_Number'].isin(chassis_numbers_w_negative_evolution)) & (df['Owner_Account'].isin(owner_accounts_w_negative_evolution))]
        df_without_negative_evolutions = df.drop(df_with_negative_evolutions.index)

        print('Starting scenario 2, solution 1...')
        start_scenario_2_fix_solution_1 = time.time()
        df_with_negative_evolutions_cleaned_1 = df_with_negative_evolutions.groupby(keys).apply(lower_between_same_kms_solution_1, ).reset_index(level=[0, 1], drop=True)
        df_with_negative_evolutions_cleaned_1.to_csv('dbs/df_cleaned_1.csv', index=False)
        df_1 = pd.concat([df_without_negative_evolutions, df_with_negative_evolutions_cleaned_1])
        df_1['Kms_diff'] = df.groupby(keys)['Kms'].diff().fillna(0)
        print('Scenario 2, Solution 1 fix time: {:.2f}s'.format(time.time() - start_scenario_2_fix_solution_1))
        negative_values_evolution_control(df_1, 'After Scenario 1')

        print('Starting scenario 2, solution 2...')
        start_scenario_2_fix_solution_2 = time.time()
        df_with_negative_evolutions_cleaned_2 = df_with_negative_evolutions.groupby(keys).apply(lower_between_same_kms_solution_2, ).reset_index(level=[0, 1], drop=True)
        df_with_negative_evolutions_cleaned_2.to_csv('dbs/df_cleaned_2.csv', index=False)
        df_2 = pd.concat([df_without_negative_evolutions, df_with_negative_evolutions_cleaned_2])
        df_2['Kms_diff'] = df.groupby(keys)['Kms'].diff().fillna(0)
        print('Scenario 2, Solution 2 fix time: {:.2f}s'.format(time.time() - start_scenario_2_fix_solution_2))
        negative_values_evolution_control(df_2, 'After Scenario 1')

        df.sort_values(by=['Movement_Date', 'WIP_Number'], na_position='first')
        print('Total Time: {:.2f}s'.format(time.time() - start))
        # df.to_csv(options_file.temp_file_grouped_loc, index=False)

    negative_values_evolution_control(df, 'Ending')

    # print(df[(df['Owner_Account'] == 'C215126') & (df['Chassis_Number'] == 'KL1TF4839CB007249')])
    # print(df[(df['Owner_Account'] == '140958')])
    # print(df[(df['Owner_Account'] == '140958') & (df['Chassis_Number'] == 'VF3LCYHZPJS332137')])

    # outlier_detection(df)

    return


def lower_between_same_kms_solution_1(x):
    # ToDo: merge with lower_between_same_kms_solution_2
    rows_to_remove_df = pd.DataFrame()

    # Finds the index of rows with the same values of WIP_Kms. reset_index resets the index to test for consecutive and non-consecutive same-kms rows.
    x.reset_index(inplace=True)
    duplicated_kms_rows_flag = x['Kms'].duplicated(keep=False)
    duplicated_kms_rows = x[duplicated_kms_rows_flag]
    # duplicated_kms_rows_index = duplicated_kms_rows.index

    rows_with_same_kms_count = sum(duplicated_kms_rows_flag)  # Python can sum Trues the same way it counts 1s
    if rows_with_same_kms_count > 1:
        print('repeated kms')
        # Check if all are consecutive:

        kms_groups = duplicated_kms_rows.groupby('Kms')
        for km_key, km_group in kms_groups:
            if sorted(km_group.index.values) == list(range(min(km_group.index.values), max(km_group.index.values) + 1)):
                print('all consecutive rows, nothing to do here')
            else:
                print('not all are consecutive, lets fix this! - {}, {}'.format(x['Chassis_Number'].head(1).values[0], x['Owner_Account'].head(1).values[0]))
                min_date_duplicated_kms_rows = min(km_group['Movement_Date'])
                max_date_duplicated_kms_rows = max(km_group['Movement_Date'])
                kms_duplicated_kms_rows = max(km_group['Kms'])

                rows_to_remove_df = pd.concat([rows_to_remove_df, x.loc[(x['Movement_Date'] > min_date_duplicated_kms_rows) & (x['Movement_Date'] < max_date_duplicated_kms_rows) & (x['Kms'] < kms_duplicated_kms_rows)]])

        x = x.drop(index=rows_to_remove_df.index)  # There can't multiple kms value for the same day, this should be fixed in previous steps.
        x.set_index('index', inplace=True, drop=True)

    else:
        print('no repeated kms')

    x.index.name = None
    return x


def lower_between_same_kms_solution_2(x):
    # ToDo: merge with lower_between_same_kms_solution_1
    rows_to_remove_df = pd.DataFrame()

    x.reset_index(inplace=True)
    duplicated_kms_rows_flag = x['Kms'].duplicated(keep=False)
    duplicated_kms_rows = x[duplicated_kms_rows_flag]
    # duplicated_kms_rows_index = duplicated_kms_rows.index

    rows_with_same_kms_count = sum(duplicated_kms_rows_flag)  # Python can sum Trues the same way it counts 1s
    if rows_with_same_kms_count > 1:
        print('repeated kms')
        # Check if all are consecutive:
        kms_groups = duplicated_kms_rows.groupby('Kms')
        for km_key, km_group in kms_groups:
            if sorted(km_group.index.values) == list(range(min(km_group.index.values), max(km_group.index.values) + 1)):
                print('all consecutive rows, nothing to do here')
            else:
                print('not all are consecutive, lets fix this! - {}, {}'.format(x['Chassis_Number'].head(1).values[0], x['Owner_Account'].head(1).values[0]))
                min_km_between_last_duplicated_rows = km_key  # Initial Value
                i = 0

                while min_km_between_last_duplicated_rows == km_key:
                    # For this approach, we only need the last values between the last two same-kms rows.
                    # But it might happen a case when there are no values between them, other the same-km values. Hence why the while, until we find something between the same-kms rows.
                    # And we know there has to be something between the same-kms rows, because otherwise they'd be consecutive and that is tested in the steps before
                    last_same_km_rows = km_group.tail(2)  # Assumes the group and respective dataframe to be ordered by Movement_Date

                    min_date_duplicated_kms_rows = min(km_group['Movement_Date'])
                    # max_date_duplicated_kms_rows = max(km_group['Movement_Date'])
                    min_date_last_duplicated_rows = min(last_same_km_rows['Movement_Date'])
                    max_date_last_duplicated_rows = max(last_same_km_rows['Movement_Date'])  # Should be equal to max_date_duplicated_kms_rows

                    # print('values_between dates\n', x.loc[(x['Movement_Date'] > min_date_duplicated_kms_rows) & (x['Movement_Date'] < max_date_duplicated_kms_rows), 'Movement_Date'])

                    min_km_between_last_duplicated_rows = min(x.loc[(x['Movement_Date'] > min_date_last_duplicated_rows) & (x['Movement_Date'] < max_date_last_duplicated_rows), 'Kms'])
                    min_date_between_last_duplicated_rows = min(x.loc[(x['Movement_Date'] > min_date_last_duplicated_rows) & (x['Movement_Date'] < max_date_last_duplicated_rows), 'Movement_Date'])

                    rows_to_remove = x.loc[(x['Movement_Date'] >= min_date_duplicated_kms_rows) & (x['Movement_Date'] < min_date_between_last_duplicated_rows)]
                    i += 1

                rows_to_remove_df = pd.concat([rows_to_remove_df, rows_to_remove])

        x = x.drop(index=rows_to_remove_df.index)  # There can't multiple kms value for the same day, this should be fixed in previous steps.
        x.set_index('index', inplace=True, drop=True)

    else:
        print('no repeated kms')

    return x


def negative_values_evolution_control(df, tag):

    owner_account_w_negative_kms_evo = df[df['Kms_diff'] < 0]['Owner_Account'].unique()
    print('{} - #Owner Accounts com evolução de Kms negativa: {}'.format(tag, len(owner_account_w_negative_kms_evo)))
    print('{} - Owner Accounts com evolução de Kms negativa: {}'.format(tag, owner_account_w_negative_kms_evo))

    return


def lower_between_same_kms(x):

    # Find counts with the same Kms
    idx = 1

    return


def outlier_detection(df):

    testing_df = df[(df['Chassis_Number'] == 'WBA1C11080J829552')]
    # testing_df = df[(df['Chassis_Number'] == 'VF3LCYHZPJS332137')]

    clf = LOF(
        n_neighbors=10,
        contamination=0.1
    )
    data_reshaped = np.array(testing_df['Kms'].values).reshape(-1, 1)
    data_reshaped = np.round(data_reshaped, 0)
    clf.fit(data_reshaped)
    y_pred = clf.predict(np.array(data_reshaped).reshape(-1, 1))
    # y_pred[y_pred < 0] = 0.0
    testing_df['outlier_score_lof'] = y_pred

    clf = LMDD(
        n_iter=100,
        contamination=0.1
    )
    data_reshaped = np.array(testing_df['Kms'].values).reshape(-1, 1)
    data_reshaped = np.round(data_reshaped, 0)
    clf.fit(data_reshaped)
    y_pred = clf.predict(np.array(data_reshaped).reshape(-1, 1))
    # y_pred[y_pred < 0] = 0.0
    testing_df['outlier_score_lmdd'] = y_pred

    clf = IsolationForest(
        n_estimators=100,
        contamination=0.1
    )
    data_reshaped = np.array(testing_df['Kms'].values).reshape(-1, 1)
    data_reshaped = np.round(data_reshaped, 0)
    clf.fit(data_reshaped)
    y_pred = clf.predict(np.array(data_reshaped).reshape(-1, 1))
    # y_pred[y_pred < 0] = 0.0
    testing_df['outlier_score_isolation_forest'] = y_pred

    clf = KNN(
        method='mean',
        n_neighbors=3,
        contamination=0.1
    )
    data_reshaped = np.array(testing_df['Kms'].values).reshape(-1, 1)
    data_reshaped = np.round(data_reshaped, 0)
    clf.fit(data_reshaped)
    y_pred = clf.predict(np.array(data_reshaped).reshape(-1, 1))
    # y_pred[y_pred < 0] = 0.0
    testing_df['outlier_score_knn_mean'] = y_pred

    clf = KNN(
        method='median',
        n_neighbors=3,
        contamination=0.1
    )
    data_reshaped = np.array(testing_df['Kms'].values).reshape(-1, 1)
    data_reshaped = np.round(data_reshaped, 0)
    clf.fit(data_reshaped)
    y_pred = clf.predict(np.array(data_reshaped).reshape(-1, 1))
    # y_pred[y_pred < 0] = 0.0
    testing_df['outlier_score_knn_median'] = y_pred

    print(testing_df[['Movement_Date', 'Kms', 'Kms_diff', 'outlier_score_lof', 'outlier_score_lmdd', 'outlier_score_isolation_forest', 'outlier_score_knn_mean', 'outlier_score_knn_median']])

    return


if __name__ == '__main__':
    main()
