import pandas as pd
import numpy as np
import nltk
import csv
from pylab import *
from scipy import stats as st
import sys
from ast import literal_eval
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import level_0_performance_report
from multiprocessing import Pool
from py_dotenv import read_dotenv
from datetime import datetime
from level_1_a_data_acquisition import sql_retrieve_df_specified_query
import apv_sales_options as options_file
from level_1_b_data_processing import null_analysis
pd.set_option('display.expand_frame_repr', False)
dotenv_path = 'info.env'
read_dotenv(dotenv_path)
my_dpi = 96
lognormal_fit = 0
slr_date = 0
wip_date = 0


def main():
    df_sales, df_purchases, df_stock, df_reg, df_reg_al_clients = data_acquisition(options_file)
    weather_df = weather_treatment()
    df_sales, df_purchases = dataset_treatment(weather_df, df_sales, df_purchases, df_stock)
    sys.exit()

    # df_sales = pd.read_csv('dbs/df_sales_w_flag_v1.csv', parse_dates=['SLR_Document_Date'], usecols=['SLR_Document_Date', 'Part_Ref', 'PVP_1', 'Cost_Sale_1', 'Qty_Sold_sum', 'T', 'U', 'Ff', 'weekday'])
    df_sales = pd.read_csv('dbs/df_sales_cleaned.csv', parse_dates=['SLR_Document_Date', 'WIP_Date_Created'], usecols=['SLR_Document_Date', 'WIP_Date_Created', 'Part_Ref', 'PVP_1', 'Cost_Sale_1', 'Qty_Sold_sum_wip', 'Qty_Sold_sum_slr', 'T', 'U', 'Ff', 'weekday'])
    df_purchases = pd.read_csv('dbs/df_purchases_cleaned.csv', parse_dates=['Movement_Date'])

    # sys.exit()

    # selected_parts = ['BM11.42.8.507.683', 'BM83.19.2.158.851', 'BM51.91.0.008.617', 'BM51.91.1.052.469', 'BM51.91.9.057.943']
    selected_parts = ['BM83.21.2.405.675']
    # selected_parts = df_sales[df_sales['SLR_Document_Date'] > '2018-05-21']['Part_Ref'].unique()

    selected_parts_count = len(selected_parts)
    print('Total Number of Parts: {}'.format(selected_parts_count))

    # part_ref_price_cost_dict = read_dict('dbs/part_ref_price_cost.csv')

    days = [2, 3]
    temps = ['cold', 'hot']
    for (week_day, temperature_range) in zip(days, temps):
        # for week_day in range(2, 3):
        #     for temperature_range in ['hot', 'cold']:
        # for cloud_cover in ['very cloudy', 'clear', 'cloudy']:
        print('Creating order for day \'{}\', temperature range \'{}\'...'.format(week_day, temperature_range))

        # week_day = 2  # 0 = Monday, 1 = Tuesday, etc...
        # temperature_range = 'cold'  # hot, warm, cold
        # cloud_cover = 'very cloudy'  # clear, cloudy, very cloudy

        seo_df = pd.DataFrame(columns=['Part_Ref', 'Optimized Stock', 'Stock Flag', 'Weekday', 'Temperature Range'])
        # selected_parts_count = len(selected_parts)
        # print('Total Number of Parts: {}'.format(selected_parts_count))

        start = time.time()
        time_tag = time.strftime("%d/%m/%y")
        current_date = datetime.strptime(time_tag, '%d/%m/%y')

        pool = Pool(processes=int(level_0_performance_report.pool_workers_count))
        results = pool.map(seo, [(part_ref, df_sales, week_day, temperature_range, current_date) for part_ref in selected_parts])
        pool.close()
        print('Elapsed time: {:.2f} seconds.'.format(time.time() - start))

        optimized_order = [result[0] for result in results if result is not None]
        stock_flags = [result[1] for result in results if result is not None]

        seo_df['Part_Ref'] = selected_parts
        seo_df['Optimized Stock'] = optimized_order
        seo_df['Stock Flag'] = stock_flags
        seo_df['Weekday'] = [week_day] * selected_parts_count
        seo_df['Temperature Range'] = [temperature_range] * selected_parts_count
        # print(seo_df)
        # seo_df['Cloud Cover'] = [cloud_cover] * selected_parts_count
        seo_df.to_csv('output/seo_df_weekday_{}_temperaturerange_{}_v7.csv'.format(week_day, temperature_range))


def data_acquisition(options_info):
    print('Starting section A...')
    start = time.time()

    sales_info = ['dbs/df_sales_03_06_19', options_file.sales_query]  # ToDo will need to update and let this run today
    purchases_info = ['dbs/df_purchases_03_06_19', options_file.purchases_query]
    stock_info = ['dbs/df_stock_03_06_19', options_file.stock_query]
    reg_info = ['dbs/df_reg_03_06_19', options_file.reg_query]
    reg_al_info = ['dbs/df_reg_al_client_03_06_19', options_file.reg_autoline_clients]

    dfs = []

    for dimension in [sales_info, purchases_info, stock_info, reg_info, reg_al_info]:
        try:
            df = pd.read_csv(dimension[0] + '.csv', index_col=0)
            print('{} file found.'.format(dimension[0]))
        except FileNotFoundError:
            print('{} file not found. Retrieving data from SQL...'.format(dimension[0]))
            df = sql_retrieve_df_specified_query(options_info.DSN_PRD, options_info.sql_info['database'], options_info, dimension[1])
            df.to_csv(dimension[0] + '.csv')

        dfs.append(df)

    df_sales = dfs[0]
    df_purchases = dfs[1]
    df_stock = dfs[2]
    df_reg = dfs[3]
    df_reg_al_clients = dfs[4]

    df_purchases['Movement_Date'] = pd.to_datetime(df_purchases['Movement_Date'], format='%Y%m%d')
    df_purchases['WIP_Date_Created'] = pd.to_datetime(df_purchases['WIP_Date_Created'], format='%Y%m%d')

    df_sales['SLR_Document_Date'] = pd.to_datetime(df_sales['SLR_Document_Date'], format='%Y%m%d')
    df_sales['WIP_Date_Created'] = pd.to_datetime(df_sales['WIP_Date_Created'], format='%Y%m%d')
    df_sales['Movement_Date'] = pd.to_datetime(df_sales['Movement_Date'], format='%Y%m%d')

    print('Ended section A - Elapsed time: {:.2f}'.format(time.time() - start))

    return df_sales, df_purchases, df_stock, df_reg, df_reg_al_clients


def seo(args):
    stock, stock_flag = 0, 0
    part_ref, df, week_day, temperature_range, current_date = args
    # stock_flag, part_mean, part_sigma, price_sale, cost_buy = estimation(df, part_ref, week_day, temperature_range, current_date)

    best_fit_name, best_fit_params, stock_flag, price_sale, cost_buy = estimation(df, part_ref, week_day, temperature_range, current_date)
    # print('part_ref {} has flag: {}'.format(part_ref, stock_flag))

    if not stock_flag:
        # print('part_ref {} has no stock flag!'.format(part_ref))
        best_dist = getattr(st, best_fit_name)
        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fit_params)])
        # dist_str = '{}({})'.format(best_fit_name, param_str)

        stock, stock_flag = optimization(best_dist, best_fit_params, part_ref, price_sale, cost_buy)
        # print('part_ref {} has stock {} and stock flag {}'.format(part_ref, stock, stock_flag))
        # print(part_ref, stock, stock_flag)

    # print('part_ref: {}, stock: {}, stock_flag: {}'.format(part_ref, stock, stock_flag))
    return stock, stock_flag


def estimation(df, part_ref, day, temp, current_date):
    stock_flag, best_fit_name, best_fit_params = 0, '', ()

    all_sel_df = df[(df['Part_Ref'] == part_ref) & (df['T'] == temp) & (df['weekday'] == day)]
    price_sale, cost_buy = all_sel_df['PVP_1'].mean(), all_sel_df['Cost_Sale_1'].mean()

    if slr_date:
        sel_df = all_sel_df.drop_duplicates(subset=['SLR_Document_Date', 'Part_Ref']).sort_values(by='SLR_Document_Date')

        if sel_df.shape[0]:
            last_sale_date = sel_df['SLR_Document_Date'].max()
            # last_sale_date = sel_df['WIP_Date_Created'].max()
            if (current_date - last_sale_date).days < 365 and sum(sel_df['Qty_Sold_sum_slr'].unique() != [0]):
                # print('part_ref {} is in!'.format(part_ref))
                x = np.array(sel_df['Qty_Sold_sum_slr'].values).ravel()

                best_dist, best_fit_name, best_fit_params = best_fit_distribution(x)

            else:
                stock_flag = 3
        else:
            stock_flag = 1

    if wip_date:
        sel_df = all_sel_df.drop_duplicates(subset=['WIP_Date_Created', 'Part_Ref']).sort_values(by='WIP_Date_Created')

        if sel_df.shape[0]:
            last_sale_date = sel_df['WIP_Date_Created'].max()
            # last_sale_date = sel_df['WIP_Date_Created'].max()
            if (current_date - last_sale_date).days < 365 and sum(sel_df['Qty_Sold_sum_wip'].unique() != [0]):
                # print('part_ref {} is in!'.format(part_ref))
                x = np.array(sel_df['Qty_Sold_sum_wip'].values).ravel()

                best_dist, best_fit_name, best_fit_params = best_fit_distribution(x)

            else:
                stock_flag = 3
        else:
            stock_flag = 1

    # print(part_ref, stock_flag)
    return best_fit_name, best_fit_params, stock_flag, price_sale, cost_buy


def read_dict(dict_loc):
    with open(dict_loc, mode='r') as infile:
        reader = csv.reader(infile)
        mydict = {row[0]: row[1] for row in reader}

    return mydict


def optimization(best_dist, best_fit_params, part_ref, price_sale, cost_buy):

    if (price_sale - cost_buy) < 0 or price_sale == cost_buy or price_sale == 0 or cost_buy == 0:
        q = 0
        stock_flag = 2
    else:
        critical_fractile = ((price_sale - cost_buy) / price_sale)
        # print('critical fractile: {:.3f}'.format(critical_fractile))
        try:
            inv_cdf = calculate_ppf(critical_fractile, best_dist, best_fit_params)
            # norm_inv = st.norm.ppf(critical_fractile)
            # print(norm_inv)
            # q = int(part_mean * np.exp(part_sigma * norm_inv))
            # q = int(part_mean + part_sigma * norm_inv)
            q = int(inv_cdf)
            # print('q: {}, Part_Ref: {}, Price: {:.2f}, Cost: {:.2f}, Critical Fractile: {}, Inv_CDF: {}'.format(q, part_ref, price_sale, cost_buy, critical_fractile, inv_cdf))
            if q < 0:
                # print('q: {}, Part_Ref: {}, Price: {:.2f}, Cost: {:.2f}, Critical Fractile: {}, Inv_CDF: {}'.format(q, part_ref, price_sale, cost_buy, critical_fractile, inv_cdf))
                q = 0

        except ValueError:
            q = 0
            stock_flag = 4  # Problems calculating the inv_cdf
            return q, stock_flag
        stock_flag = 0

    print(q, stock_flag)
    return q, stock_flag


def weather_treatment():
    print('Weather Data Treatment started.')
    start = time.time()
    df_weather = pd.read_csv('dbs/weather_data_lisbon.csv', skiprows=6, delimiter=';', usecols=['Local time in Lisbon / Portela (airport)', 'T', 'U', 'Ff', 'c'], parse_dates=['Local time in Lisbon / Portela (airport)'])

    df_weather['Ff'] = df_weather['Ff'] * 3.6

    df_weather['T'] = df_weather['T'].fillna(method='bfill')
    df_weather['U'] = df_weather['U'].fillna(method='bfill')

    # Cloud Cover isn't used atm
    # df_weather['c'] = df_weather['c'].fillna(method='bfill')
    #
    # unique_c = df_weather['c'].unique()
    # for c in unique_c:
    #     tokenized_c = nltk.tokenize.word_tokenize(c)
    #     try:
    #         cover_limits = tokenized_c[3]
    #         if 'vertical' in tokenized_c and 'visibility' in tokenized_c:
    #             df_weather.loc[df_weather['c'] == c, 'cloud_cover'] = 100
    #         else:
    #             # print(tokenized_c, cover_limits, int(cover_limits[:2]), int(cover_limits[3:5]), (int(cover_limits[:1]) + int(cover_limits[3:5])), (int(cover_limits[:1]) + int(cover_limits[3:5])) / 2)
    #             df_weather.loc[df_weather['c'] == c, 'cloud_cover'] = (int(cover_limits[:2]) + int(cover_limits[3:5])) / 2
    #             df_weather.loc[df_weather['c'] == c, 'c'] = tokenized_c[3]
    #     except (ValueError, IndexError):
    #         df_weather.loc[df_weather['c'] == c, 'cloud_cover'] = 100

    df_weather.drop('c', axis=1, inplace=True)

    # Daily Resample
    df_weather.index = df_weather['Local time in Lisbon / Portela (airport)']
    df_weather_daily = df_weather.resample('d').mean()

    df_weather_daily.dropna(inplace=True)

    # Weather Data categories
    # df_weather_daily['T'] = ['cold' if x < 15 else 'hot' if x > 25 else 'warm' for x in df_weather_daily['T']]
    df_weather_daily['T'] = ['hot' if x > 15 else 'cold' for x in df_weather_daily['T']]
    df_weather_daily['U'] = ['dry' if x < 70 else 'moist' if x > 90 else 'normal' for x in df_weather_daily['U']]
    df_weather_daily['Ff'] = ['windy' if x > 18 else 'not windy' for x in df_weather_daily['Ff']]
    # df_weather_daily['cloud_cover'] = ['clear' if x < 35 else 'very cloudy' if x > 80 else 'cloudy' for x in df_weather_daily['cloud_cover']]

    print('Weather Data Treatment finished. Elapsed time: {:.2f}.'.format(time.time() - start))
    return df_weather_daily


def dataset_treatment(df_weather_daily, df_sales, df_purchases, df_stock):
    print('Dataset processing started.')
    start = time.time()
    dictionary_prices_cost = {}

    try:
        df_sales = pd.read_csv('dbs/df_sales_processed.csv', index_col=0, parse_dates=['SLR_Document_Date', 'WIP_Date_Created', 'Movement_Date'])
    except FileNotFoundError:
        df_sales = df_sales[df_sales['Qty_Sold'] != 0]

        df_sales = df_sales[(df_sales['WIP_Number'] == 38381) & (df_sales['Part_Ref'] == 'BM83.19.2.158.851')]

        # print(df_sales)

        df_sales = data_processing_negative_values(df_sales, sales_flag=1)
        df_sales.to_csv('dbs/df_sales_processed.csv')
        # df_sales['SLR_Document_Date'] = pd.to_datetime(df_sales['SLR_Document_Date'], format='%Y%m%d')
        # df_sales['WIP_Date_Created'] = pd.to_datetime(df_sales['WIP_Date_Created'], format='%Y%m%d')
        # df_sales['Movement_Date'] = pd.to_datetime(df_sales['Movement_Date'], format='%Y%m%d')

    try:
        df_purchases = pd.read_csv('dbs/df_purchases_processed.csv', index_col=0, parse_dates=['Movement_Date', 'WIP_Date_Created'])
    except FileNotFoundError:
        # df_purchases = data_processing_negative_values(df_purchases, purchases_flag=1)
        df_purchases.to_csv('dbs/df_purchases_processed.csv')
        # df_purchases['Movement_Date'] = pd.to_datetime(df_purchases['Movement_Date'], format='%Y%m%d')
        # df_purchases['WIP_Date_Created'] = pd.to_datetime(df_purchases['WIP_Date_Created'], format='%Y%m%d')

    df_sales['PVP_1'] = df_sales['PVP'] / df_sales['Qty_Sold']
    df_sales['Cost_Sale_1'] = df_sales['Cost_Sale'] / df_sales['Qty_Sold']
    df_purchases['Cost_Value_1'] = df_purchases['Cost_Value'] / df_purchases['Quantity']

    df_sales.drop(['PVP', 'Sale_Value', 'Gross_Margin', 'Cost_Sale'], axis=1, inplace=True)
    df_purchases.drop(['Cost_Value'], axis=1, inplace=True)

    # df_sales_grouped = df_sales.groupby(['SLR_Document_Date', 'Part_Ref'])
    df_sales_grouped_wip = df_sales.groupby(['WIP_Date_Created', 'Part_Ref'])  # New approach, where WIP_Date_Created is used instead of the SLR_Document_Date
    df_sales_grouped_slr = df_sales.groupby(['SLR_Document_Date', 'Part_Ref'])  # Old Approach, using SLR_Document_Date
    df_sales_grouped_mov = df_sales.groupby(['Movement_Date', 'Part_Ref'])  # Old Approach, using SLR_Document_Date
    df_purchases_grouped = df_purchases.groupby(['Movement_Date', 'Part_Ref'])

    df_sales['Qty_Sold_sum_wip'] = df_sales_grouped_wip['Qty_Sold'].transform('sum')
    df_sales['Qty_Sold_sum_slr'] = df_sales_grouped_slr['Qty_Sold'].transform('sum')
    df_sales['Qty_Sold_sum_mov'] = df_sales_grouped_mov['Qty_Sold'].transform('sum')
    df_purchases['Qty_Purchased_sum'] = df_purchases_grouped['Quantity'].transform('sum')
    df_sales.drop('Qty_Sold', axis=1, inplace=True)
    df_purchases.drop('Quantity', axis=1, inplace=True)
    # print(df.head(10))

    # df_sales = df_sales.join(df_weather_daily, on='SLR_Document_Date')
    # df_sales = df_sales.join(df_weather_daily, on='WIP_Date_Created')  # New approach, where WIP_Date_Created is used instead of the SLR_Document_Date
    df_sales = df_sales.join(df_weather_daily, on='Movement_Date')
    df_sales.sort_index(inplace=True)
    df_sales.fillna(method='bfill', inplace=True)

    # df_sales['weekday'] = df_sales['SLR_Document_Date'].dt.dayofweek
    # df_sales['weekday'] = df_sales['WIP_Date_Created'].dt.dayofweek  # New approach, where WIP_Date_Created is used instead of the SLR_Document_Date
    df_sales['weekday'] = df_sales['Movement_Date'].dt.dayofweek

    df_sales.to_csv('dbs/df_sales_cleaned.csv')
    df_purchases.to_csv('dbs/df_purchases_cleaned.csv')

    print('Dataset processing finished. Elapsed time: {:.2f}'.format(time.time() - start))
    return df_sales, df_purchases


def data_processing_negative_values(df, sales_flag=0, purchases_flag=0):
    start = time.time()

    print('number of wips', len(df.groupby('WIP_Number')))
    pool = Pool(processes=int(level_0_performance_report.pool_workers_count))
    results = pool.map(matching_negative_row_removal_2, [(y[0], y[1], sales_flag, purchases_flag) for y in df.groupby('WIP_Number')])
    pool.close()
    gt_treated = pd.concat([result for result in results if result is not None])

    print(gt_treated)

    print('Elapsed Time: {:.2f}'.format(time.time() - start))
    return gt_treated


def matching_negative_row_removal_2(args):
    # print(args)
    key, group, sales_flag, purchases_flag = args
    # print('full wip: \n{}'.format(group))
    negative_rows = pd.DataFrame()

    # print(key, group, sales_flag, purchases_flag)

    # matching_positive_rows = pd.DataFrame()
    # # negative_rows = group[group['Sale_Value'] < 0]
    # negative_rows = group[group['Qty_Sold'] < 0]
    # if negative_rows.shape[0]:
    #     if sales_flag:
    #         for key, row in negative_rows.iterrows():
    #             matching_positive_row = group[(group['Qty_Sold'] == abs(row['Qty_Sold'])) & (group['PVP'] == abs(row['PVP'])) & (group['Sale_Value'] == abs(row['Sale_Value'])) & (group['Cost_Sale'] == abs(row['Cost_Sale'])) & (group['Gross_Margin'] == abs(row['Gross_Margin']))]
    #             # matching_positive_row = group[(group['Sale_Value'] == abs(row['Sale_Value'])) & (group['Cost_Sale'] == abs(row['Cost_Sale'])) & (group['Gross_Margin'] == abs(row['Gross_Margin']))]
    #             matching_positive_rows = pd.concat([matching_positive_rows, matching_positive_row.head(1)])
    #     elif purchases_flag:
    #         if negative_rows.shape[0]:
    #             for key, row in negative_rows.iterrows():
    #                 matching_positive_row = group[(group['Quantity'] == abs(row['Quantity'])) & (group['Cost_Sale'] == row['Cost_Sale']) & (group['Part_Ref'] == row['Part_Ref']) & (group['WIP_Number'] == row['WIP_Number'])]
    #                 # matching_positive_row = group[(group['Sale_Value'] == abs(row['Sale_Value'])) & (group['Cost_Sale'] == abs(row['Cost_Sale'])) & (group['Gross_Margin'] == abs(row['Gross_Margin']))]
    #                 matching_positive_rows = pd.concat([matching_positive_rows, matching_positive_row.head(1)])
    #
    #     group.drop(negative_rows.index, axis=0, inplace=True)
    #     group.drop(matching_positive_rows.index, axis=0, inplace=True)
    #     # Note: Sometimes, identical negative rows with only Part_Ref different will match with the same row with positive values. This is okay as when I remove the matched rows from the
    #     # original group I remove by index, so double matched rows make no problem whatsoever

    matching_positive_rows = pd.DataFrame()

    if sales_flag:
        print(group)
        negative_rows = group[group['Qty_Sold'] < 0]
        print('negative_rows: \n{}'.format(negative_rows))

        if negative_rows.shape[0]:
            for key, row in negative_rows.iterrows():
                matching_positive_row = group[(group['Movement_Date'] == row['Movement_Date']) & (group['Qty_Sold'] == row['Qty_Sold'] * -1) & (group['Sale_Value'] == row['Sale_Value'] * -1) & (group['Cost_Sale'] == row['Cost_Sale'] * -1) & (group['Gross_Margin'] == row['Gross_Margin'] * -1)]
                # matching_positive_row = group[(group['Movement_Date'] == row['Movement_Date']) & (group['Qty_Sold'] == row['Qty_Sold'] * -1) & (group['PVP'] == row['PVP'] * -1) & (group['Sale_Value'] == row['Sale_Value'] * -1) & (group['Cost_Sale'] == row['Cost_Sale'] * -1) & (group['Gross_Margin'] == row['Gross_Margin'] * -1)]

                # matching_positive_row = group[(group['Movement_Date'] == row['Movement_Date']) & (group['Qty_Sold'] == abs(row['Qty_Sold'])) & (group['PVP'] == abs(row['PVP'])) & (group['Sale_Value'] == abs(row['Sale_Value'])) & (group['Cost_Sale'] == abs(row['Cost_Sale'])) & (group['Gross_Margin'] == abs(row['Gross_Margin']))]
                print('matching_positive_row: \n{}'.format(matching_positive_row))

                # if matching_positive_row.shape[0]:
                #     if group['WIP_Number'].unique() == 28883:
                #         if negative_rows[negative_rows['Part_Ref'] == 'BM07.14.9.213.164'].shape[0]:
                #             print('negative rows: \n {}'.format(negative_rows[negative_rows['Part_Ref'] == 'BM07.14.9.213.164']))
                #             print('matching_positive_row: \n {}'.format(matching_positive_row[matching_positive_row['Part_Ref'] == 'BM07.14.9.213.164']))

                if matching_positive_row.shape[0] > 1:
                    matching_positive_rows = pd.concat([matching_positive_rows, matching_positive_row[matching_positive_row.index == matching_positive_row['Movement_Date'].idxmax()]])
                    print('matching_positive_rows that will be removed \n{}'.format(matching_positive_rows))
                else:
                    matching_positive_rows = pd.concat([matching_positive_rows, matching_positive_row.head(1)])
                    print('matching_positive_rows that will be removed \n{}'.format(matching_positive_rows))

    elif purchases_flag:
        negative_rows = group[group['Quantity'] < 0]
        if negative_rows.shape[0]:
            for key, row in negative_rows.iterrows():
                matching_positive_row = group[(group['Quantity'] == abs(row['Quantity'])) & (group['Cost_Value'] == abs(row['Cost_Value'])) & (group['Part_Ref'] == row['Part_Ref']) & (group['WIP_Number'] == row['WIP_Number'])]

                if matching_positive_row.shape[0] > 1:
                    matching_positive_rows = pd.concat([matching_positive_rows, matching_positive_row[matching_positive_row.index == matching_positive_row['Movement_Date'].idxmax()]])
                else:
                    matching_positive_rows = pd.concat([matching_positive_rows, matching_positive_row.head(1)])

    if negative_rows.shape[0]:
        group.drop(negative_rows.index, axis=0, inplace=True)
        group.drop(matching_positive_rows.index, axis=0, inplace=True)
        # Note: Sometimes, identical negative rows with only Part_Ref different will match with the same row with positive values. This is okay as when I remove the matched rows from the
        # original group I remove by index, so double matched rows make no problem whatsoever

    print(group)
    return group


def calculate_ppf(critical_fractile, dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get same start and end points of distribution
    inv_cdf = dist.ppf(critical_fractile, *arg, loc=loc, scale=scale)  # if arg else dist.ppf(0.01, loc=loc, scale=scale)

    return inv_cdf


def best_fit_distribution(data, bins=200, ax=None):

    # Distributions to check
    if not lognormal_fit:
        print('Searching for the best fit to the distribution...')
        # Get histogram of original data
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        distributions = [
            st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2, st.cosine,
            st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f, st.fatiguelife, st.fisk,
            st.foldcauchy, st.foldnorm, st.frechet_r, st.frechet_l, st.genlogistic, st.genpareto, st.gennorm, st.genexpon,
            st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz, st.gumbel_r,
            st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant, st.invgamma, st.invgauss,
            st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign, st.laplace, st.levy, st.levy_l,
            st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke, st.nakagami, st.ncx2, st.ncf,
            st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm, st.rdist, st.reciprocal,
            st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t, st.triang, st.truncexpon, st.truncnorm, st.tukeylambda,
            st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy
        ]

        # Best holders
        best_distribution = st.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf

        # Estimate distribution parameters from data
        for distribution in distributions:

            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    # fit dist to data
                    params = distribution.fit(data)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    # if axis pass in add to plot
                    try:
                        if ax:
                            pd.Series(pdf, x).plot(ax=ax)
                    except Exception:
                        pass

                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse

            except Exception:
                pass
    else:
        with np.errstate(invalid='ignore'):  # Just to ignore invalid values when calculating the log of data
            data = np.log(data)

        best_distribution = st.norm
        best_params = best_distribution.fit(data)

    return best_distribution, best_distribution.name, best_params


if __name__ == '__main__':
    main()
