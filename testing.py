import os
import sys
import time
import pandas as pd
import numpy as np
from multiprocessing import Pool
from dateutil.relativedelta import relativedelta
import level_0_performance_report
import warnings
pd.set_option('display.expand_frame_repr', False)
warnings.filterwarnings('ignore')

DSN = os.getenv('DSN_MLG')
DSN_PRD = os.getenv('DSN_Prd')
UID = os.getenv('UID')
PWD = os.getenv('PWD')
EMAIL = os.getenv('EMAIL')
EMAIL_PASS = os.getenv('EMAIL_PASS')


class FakeOptionsFile(object):
    def __init__(self):
        self.UID = UID
        self.PWD = PWD
        self.project_id = '2162'


def main():

    # pse_code = '0I'  # Expo
    pse_code = '0B'  # Gaia
    selected_parts = []

    if pse_code == '0I':
        selected_parts = ['BM83.21.2.405.675', 'BM07.12.9.952.104', 'BM07.14.9.213.164', 'BM83.19.2.158.851', 'BM64.11.9.237.555']  # PSE_Code = 0I
    if pse_code == '0B':
        # selected_parts = ['BM51.16.7.363.919']  # Need to check this
        # selected_parts = ['BM61.31.9.217.643']  # Need to check this
        # selected_parts = ['BM83.21.0.406.573', 'BM83.13.9.415.965', 'BM51.18.1.813.017', 'BM11.42.8.507.683', 'BM64.11.9.237.555']  # PSE_Code = 0B
        selected_parts = ['BM34.33.6.796.853', 'BM13.62.7.804.742', 'BM32.10.6.884.404AT', 'BM51.16.8.159.698', 'BM34.21.1.161.806', 'BM34.33.6.857.405', 'BM83.21.0.406.573', 'BM11.12.7.799.225', 'BM32.30.6.854.768', 'BM61.21.6.924.023']

    # min_date = '2017-12-31'
    # max_date = '2019-06-28'  # Why are the stock levels in 28/06 instead of  30/06 ? - Caused by Migration
    # print('full data')
    # min_date = '2018-07-31'
    # max_date = '2019-05-31'
    # print('full year')
    min_date = '2018-01-31'
    max_date = '2019-06-28'
    print('single month')

    version = 'v13'

    # weather_data_daily = weather_treatment()
    selected_parts = part_ref_selection(pse_code, max_date)
    # selected_parts = selected_parts[0:3]
    selected_parts = ['BM80.14.2.454.715', 'BM80.14.2.454.716', 'BM11.61.8.575.534', 'BM80.14.2.298.174', 'BM80.28.2.411.529', 'BM72.11.7.321.413', 'BM51.12.8.408.392', 'BM80.14.2.454.605', 'BM61.34.9.350.797', 'BM61.34.9.302.183',
                      'BM34.40.6.857.640', 'BM67.13.7.232.743', 'BM80.14.2.454.595', 'BM80.42.2.351.056', 'BM51.77.7.157.105', 'BM51.16.2.993.420', 'BM52.10.7.374.875', 'BM64.11.1.394.286', 'BM63.12.8.375.303', 'BM51.16.6.954.945',
                      'BM83.13.0.443.029', 'BM01.40.2.969.874', 'BM51.71.8.204.894', 'BM61.31.9.225.710', 'BM51.45.6.997.929', 'BM51.75.7.125.441', 'BM41.61.8.238.461', 'BM64.12.6.939.511', 'BM51.41.8.224.595', 'BM66.53.9.291.386',
                      'BM84.21.2.289.717', 'BM12.31.7.525.376', 'BM51.71.0.443.130', 'BM61.31.9.225.940', 'BM36.12.0.396.391', 'BM32.30.6.782.596', 'BM32.30.6.783.829', 'BM33.31.1.504.023', 'BM51.31.7.285.936', 'BM51.47.9.200.682',
                      'BM31.11.6.796.693', 'BM36.11.6.783.631', 'BM41.00.7.203.980', 'BM51.16.7.363.919', 'BM36.11.6.772.249', 'BM18.30.3.449.081', 'BM51.45.9.123.695', 'BM83.12.2.285.678', 'BM51.13.1.934.178', 'BM07.14.6.985.596']

    df_sales, df_al, df_stock, df_reg_al_clients, df_purchases = data_retrieval(pse_code)
    stock_evolution_calculation(pse_code, selected_parts, df_sales, df_al, df_stock, df_reg_al_clients, df_purchases, min_date, max_date)


def data_retrieval(pse_code):
    df_stock, df_reg_al_clients, df_purchases = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # df_al = pd.read_excel('dbs/{}_{}.xlsx'.format(selected_part[0], pse_code), usecols=['Data Mov', 'Refª da peça', 'Descrição', 'Unit', 'Nº de factura', 'WIP nº', 'Sugestão nº  (Enc)', 'Conta', 'Nº auditoria stock', 'Preço de custo', 'P. V. P'])
    df_al = pd.read_csv('dbs/auto_line_part_ref_history_{}.csv'.format(pse_code), usecols=['Data Mov', 'Refª da peça', 'Descrição', 'Unit', 'Nº de factura', 'WIP nº', 'Sugestão nº  (Enc)', 'Conta', 'Nº auditoria stock', 'Preço de custo', 'P. V. P'])
    df_sales = pd.read_csv('dbs/df_sales_cleaned_{}.csv'.format(pse_code), parse_dates=['Movement_Date', 'WIP_Date_Created', 'SLR_Document_Date'], usecols=['Movement_Date', 'WIP_Number', 'SLR_Document', 'WIP_Date_Created', 'SLR_Document_Date', 'Part_Ref', 'PVP_1', 'Cost_Sale_1', 'Qty_Sold_sum_wip', 'Qty_Sold_sum_slr', 'Qty_Sold_sum_mov']).sort_values(by='WIP_Date_Created')
    df_al.rename(index=str, columns={'Data Mov': 'Movement_Date', 'Refª da peça': 'Part_Ref', 'Descrição': 'Part_Desc', 'Nº de factura': 'SLR_Document_Number', 'WIP nº': 'WIP_Number', 'Sugestão nº  (Enc)': 'Encomenda', 'Conta': 'SLR_Document_Account', 'Nº auditoria stock': 'Audit_Number'}, inplace=True)
    df_al['Movement_Date'] = pd.to_datetime(df_al['Movement_Date'], format='%d/%m/%Y')
    df_al.sort_values(by='Movement_Date', inplace=True)

    if pse_code == '0I':
        df_purchases = pd.read_csv('dbs/df_purchases_cleaned_{}.csv'.format(pse_code), index_col=0, parse_dates=['Movement_Date']).sort_values(by='Movement_Date')
        df_purchases.rename(index=str, columns={'Qty_Sold_sum': 'Qty_Purchased_sum'}, inplace=True)  # Will be removed next time i run the data_processement
        df_stock = pd.read_csv('dbs/df_stock_' + str(pse_code) + '_01_07_19.csv', parse_dates=['Record_Date'], usecols=['Part_Ref', 'Quantity', 'Record_Date']).sort_values(by='Record_Date')
        df_stock.rename(index=str, columns={'Quantity': 'Stock_Qty'}, inplace=True)
        # df_reg = pd.read_csv('dbs/df_reg_' + str(pse_code) + '_01_07_19.csv', parse_dates=['Movement_Date'], usecols=['Movement_Date', 'Part_Ref', 'Quantity', 'SLR_Document', 'Cost_Value']).sort_values(by='Movement_Date')
        # df_reg.rename(index=str, columns={'Quantity': 'Qty_Regulated'}, inplace=True)
        df_reg_al_clients = pd.read_csv('dbs/df_reg_al_client_' + str(pse_code) + '_01_07_19.csv', index_col=0)
    elif pse_code == '0B':
        df_purchases = pd.read_csv('dbs/df_purchases_cleaned_{}.csv'.format(pse_code), index_col=0, parse_dates=['Movement_Date']).sort_values(by='Movement_Date')
        df_purchases.rename(index=str, columns={'Qty_Sold_sum': 'Qty_Purchased_sum'}, inplace=True)  # Will be removed next time i run the data_processement
        df_stock = pd.read_csv('dbs/df_stock_' + str(pse_code) + '_15_07_19.csv', parse_dates=['Record_Date'], usecols=['Part_Ref', 'Quantity', 'Record_Date']).sort_values(by='Record_Date')
        df_stock.rename(index=str, columns={'Quantity': 'Stock_Qty'}, inplace=True)
        # df_reg = pd.read_csv('dbs/df_reg_' + str(pse_code) + '_15_07_19.csv', parse_dates=['Movement_Date'], usecols=['Movement_Date', 'Part_Ref', 'Quantity', 'SLR_Document', 'Cost_Value']).sort_values(by='Movement_Date')
        # df_reg.rename(index=str, columns={'Quantity': 'Qty_Regulated'}, inplace=True)
        df_reg_al_clients = pd.read_csv('dbs/df_reg_al_client_' + str(pse_code) + '_15_07_19.csv', index_col=0)

    return df_sales, df_al, df_stock, df_reg_al_clients, df_purchases


def stock_evolution_calculation(pse_code, selected_parts, df_sales, df_al, df_stock, df_reg_al_clients, df_purchases, min_date, max_date):

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
        # result_part_ref.to_csv('output/results_merge_MN51712285495_first_step_{}.csv'.format(pse_code))

        if result_part_ref.shape[0]:
            print(part_ref)
            print(result_part_ref)
            result_part_ref = result_part_ref.reindex(datetime_index).reset_index().rename(columns={'Unnamed: 0': 'Movement_Date'})
            # result_part_ref.to_csv('output/results_merge_MN51712285495_second_step_{}.csv'.format(pse_code))

            # ffill_cols = ['Part_Ref', 'Stock_Qty', 'Stock_Qty_al', 'Sales Evolution_al', 'Purchases Evolution', 'Regulated Evolution', 'Purchases Urgent Evolution', 'Purchases Non Urgent Evolution', 'Cost_Sale_avg', 'PVP_avg', 'Stock_Evolution_Correct_Flag', 'Stock_Evolution_Offset']
            # zero_fill_cols = ['Qty_Purchased_sum', 'Qty_Regulated_sum', 'Qty_Sold_sum_al', 'Qty_Purchased_urgent_sum', 'Qty_Purchased_non_urgent_sum', 'Cost_Purchase_avg', 'Cost_Reg_avg']
            ffill_and_zero_fill_cols = ['Stock_Qty_al', 'Sales Evolution_al', 'Purchases Evolution', 'Regulated Evolution', 'Purchases Urgent Evolution', 'Purchases Non Urgent Evolution']
            zero_fill_cols = ['Qty_Purchased_sum', 'Qty_Regulated_sum', 'Qty_Sold_sum_al', 'Qty_Purchased_urgent_sum', 'Qty_Purchased_non_urgent_sum', 'Cost_Purchase_avg', 'Cost_Reg_avg', 'Cost_Sale_avg', 'PVP_avg']
            ffill_and_bfill_cols = ['Part_Ref', 'Stock_Qty']

            [result_part_ref[x].fillna(method='ffill', inplace=True) for x in ffill_and_zero_fill_cols + ffill_and_bfill_cols]
            [result_part_ref[x].fillna(0, inplace=True) for x in zero_fill_cols + ffill_and_zero_fill_cols]
            [result_part_ref[x].fillna(method='bfill', inplace=True) for x in ffill_and_bfill_cols]

            if result_part_ref[result_part_ref['Part_Ref'].isnull()].shape[0]:
                print('null values found for part_ref: \n{}'.format(part_ref))
                print('Number of null rows: {}'.format(result_part_ref[result_part_ref['Part_Ref'].isnull()].shape))
                # print(result_part_ref.head())
                # print(result_part_ref.tail())
                # print(result_part_ref[result_part_ref['Part_Ref'].isnull()])

            result_part_ref.loc[:, 'Stock_Evolution_Correct_Flag'] = stock_evolution_correct_flag
            result_part_ref.loc[:, 'Stock_Evolution_Offset'] = offset

            # result_part_ref['weekday'] = result_part_ref['index'].dt.dayofweek
            # result_part_ref['weekofyear'] = result_part_ref['index'].dt.weekofyear  # Careful with the end of year dates and respective weeks
            # result_part_ref['year'] = result_part_ref['index'].dt.year
            # result_part_ref['unique_weeks'] = result_part_ref['year'].apply(str) + '_' + result_part_ref['weekofyear'].apply(str)
            # End of year handling:
            # result_part_ref.loc[result_part_ref['index'] == '2018-12-31', 'unique_weeks'] = '2019_1'

            # Just so the variation matches with the rest: positive regularization means an increase in stock, while negative is a decrease; Equivalent for cost;
            result_part_ref['Qty_Regulated_sum'] = result_part_ref['Qty_Regulated_sum'] * (-1)
            result_part_ref['Cost_Reg_avg'] = result_part_ref['Cost_Reg_avg'] * (-1)

            # result_part_ref = cost_calculation_function(result_part_ref, part_ref, min_date, max_date, version)
            # print(result_part_ref)
            results = results.append(result_part_ref)
            print('Elapsed time: {:.2f}.'.format(time.time() - start))

        position = int((i / parts_count) * 100)
        if not position % 1:
            if position not in positions:
                print('{}% completed'.format(position))
                positions.append(position)

        i += 1
    # print(results)
    # results.to_csv('output/results_merge_{}.csv'.format(pse_code))
    return results


def part_ref_selection(pse_code, current_date):
    print('Selection of Part Reference')

    df_al = pd.read_csv('dbs/auto_line_part_ref_history_{}.csv'.format(pse_code), index_col=0)
    df_al.rename(index=str, columns={'Data Mov': 'Movement_Date', 'Refª da peça': 'Part_Ref', 'Descrição': 'Part_Desc', 'Nº de factura': 'SLR_Document_Number', 'WIP nº': 'WIP_Number', 'Sugestão nº  (Enc)': 'Encomenda', 'Conta': 'SLR_Document_Account', 'Nº auditoria stock': 'Audit_Number'}, inplace=True)
    df_al['Movement_Date'] = pd.to_datetime(df_al['Movement_Date'], format='%d/%m/%Y')
    df_al.sort_values(by='Movement_Date', inplace=True)

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


def cost_calculation_function(results_merge, part_ref, min_date, max_date, version):
    ### Note: I changed the way Cost_Sale_avg and PVP_avg are filled from ffill to fillna(0). If this function does not work or provide erroneous results, consider this change - 30/07/19

    emergency_cost = 5
    emergency_cost_per_purchase = 1
    emergency_cost_per_part = 0
    hot_start = 1

    stock_levels = pd.read_csv('output/seo_{}_top5_part_refs.csv'.format(version), index_col=0)
    sel_weeks = stock_levels['Week_Start_Day'].unique()
    # results_merge = pd.read_csv('output/results_merge.csv', index_col=0, parse_dates=['index'])

    print('\n##### Part Ref: {} '.format(part_ref))

    ### Current Costs
    # Cold Start - Cost of buying a stock from 0 up to selected day stock
    initial_cost = results_merge[(results_merge['index'] >= min_date) & (results_merge['Part_Ref'] == part_ref) & (results_merge['Cost_Sale_avg'] > 0)]['Cost_Sale_avg'].head(1).values[0]  # In order to get the earliest price for this part_ref, after the min_date
    initial_stock = results_merge[(results_merge['index'] == min_date) & (results_merge['Part_Ref'] == part_ref)]['Stock_Qty_al'].values[0]
    cold_start_cost = initial_cost * initial_stock

    # Dataset Selection
    sel_results_merge = results_merge.loc[(results_merge['Part_Ref'] == part_ref) & (results_merge['index'] > min_date) & (results_merge['index'] <= max_date), :].copy()
    sel_stock_levels = stock_levels[stock_levels['Part_Ref'] == part_ref]

    # Current Purchases Cost - Non Urgent Purchases -> Cost = Cost Purchase (just copies some values from this column to a new one)
    sel_results_merge.loc[:, 'Purchases_Cost_non_urgent'] = sel_results_merge.loc[sel_results_merge['Qty_Purchased_non_urgent_sum'] > 0]['Cost_Purchase_avg']  # This operations is merely copying values from one column to another. It serves a purpose of consistency.

    # Current Purchases Cost - Urgent Purchases - If Order_Type_DW is 4 or 5
    if emergency_cost_per_part:  # If considering an emergency cost per part -> Cost = Cost Purchases Urgent + Qty Purchase Urgent * Emergency Cost
        sel_results_merge.loc[:, 'Purchases_Cost_urgent'] = sel_results_merge[sel_results_merge['Qty_Purchased_urgent_sum'] > 0]['Cost_Purchase_avg'] + sel_results_merge[sel_results_merge['Qty_Purchased_urgent_sum'] > 0]['Qty_Purchased_urgent_sum'] * emergency_cost
    elif emergency_cost_per_purchase:  # If considering an emergency cost per purchase/order -> Cost = Cost Purchases Urgent + Emergency Cost
        sel_results_merge.loc[:, 'Purchases_Cost_urgent'] = sel_results_merge[sel_results_merge['Qty_Purchased_urgent_sum'] > 0]['Cost_Purchase_avg'] + emergency_cost

    # Total Current Purchases Cost -> Sum of each respective column (urgent/non urgent)
    current_purchases_cost_non_urgent = sel_results_merge.loc[:, 'Purchases_Cost_non_urgent'].sum()
    current_purchases_cost_urgent = sel_results_merge.loc[:, 'Purchases_Cost_urgent'].sum()
    current_purchases_cost = current_purchases_cost_non_urgent + current_purchases_cost_urgent

    # Current Regularizations Cost - Cost = Cost Regularization (just copies some values from this column to a new one and sums the new one)
    sel_results_merge.loc[:, 'Regularizations_Cost'] = sel_results_merge.loc[sel_results_merge['Qty_Regulated_sum'] != 0]['Cost_Reg_avg']  # This operations is merely copying values from one column to another. It serves a purpose of consistency.
    current_regularizations_cost = sel_results_merge.loc[:, 'Regularizations_Cost'].sum()

    # Total Costs -> Total Cost = Urgent Purchase Cost + Non Urgent Cost + Regularization Cost (sums the three columns into a new one, counting NaN as 0)
    sel_results_merge['current_total_cost'] = sel_results_merge[['Purchases_Cost_non_urgent', 'Purchases_Cost_urgent', 'Regularizations_Cost']].sum(axis=1, skipna=True)
    current_total_cost = sel_results_merge['current_total_cost'].sum()

    # Total Cost Cumulative Sum
    sel_results_merge['total_cost evolution'] = sel_results_merge['current_total_cost'].cumsum()

    # print('Cold Start Cost for date {}: {:.3f}'.format(min_date, cold_start_cost))
    print('Current Cost: {:.3f}'.format(current_total_cost))
    # print('Cold Start Cost + Current Cost: {:.3f}'.format(cold_start_cost + current_total_cost))

    # Quantities:
    # print('\nTotal Purchase Qty: {} \nUrgent Purchase Qty: {} \nNon Urgent Purchase Qty: {} \nRegularizations Qty: {}'
    #       .format(sel_results_merge['Qty_Purchased_sum'].sum(), sel_results_merge['Qty_Purchased_urgent_sum'].sum(), sel_results_merge['Qty_Purchased_non_urgent_sum'].sum(), sel_results_merge['Qty_Regulated_sum'].sum()))
    # Costs
    # print('\nUrgent Purchase Cost: {:.3f} \nNon Urgent Purchase Cost: {:.3f} \nTotal Purchase Cost: {:.3f} \nRegularizations Cost: {:.3f} \nTotal Cost: {:.3f}'
    #       .format(current_purchases_cost_urgent, current_purchases_cost_non_urgent, current_purchases_cost, current_regularizations_cost, current_total_cost))

    # print(sel_results_merge[sel_results_merge['Qty_Purchased_non_urgent_sum'] > 0][['index', 'Qty_Purchased_sum', 'Qty_Purchased_non_urgent_sum', 'Qty_Purchased_urgent_sum', 'Cost_Purchase_avg', 'Purchases_Cost_urgent', 'Purchases_Cost_non_urgent']])
    # print(sel_results_merge[sel_results_merge['Qty_Purchased_urgent_sum'] > 0][['index', 'Qty_Purchased_sum', 'Qty_Purchased_non_urgent_sum', 'Qty_Purchased_urgent_sum', 'Cost_Purchase_avg', 'Purchases_Cost_urgent', 'Purchases_Cost_non_urgent']])
    # print(sel_results_merge[sel_results_merge['Qty_Regulated_sum'] > 0][['index', 'Qty_Regulated_sum', 'Cost_Reg_avg', 'Regularizations_Cost']])

    ### SEO Purchases Cost
    # Cold Start - Cost of buying a stock from 0 up to selected stock level
    cold_start_seo_min_cost = sel_stock_levels['SEO_Week_Min'].values[0] * results_merge[(results_merge['index'] >= min_date) & (results_merge['Part_Ref'] == part_ref) & (results_merge['Cost_Sale_avg'] > 0)]['Cost_Sale_avg'].head(1).values[0]
    cold_start_seo_max_cost = sel_stock_levels['SEO_Week_Max'].values[0] * results_merge[(results_merge['index'] >= min_date) & (results_merge['Part_Ref'] == part_ref) & (results_merge['Cost_Sale_avg'] > 0)]['Cost_Sale_avg'].head(1).values[0]
    cold_start_seo_avg_cost = sel_stock_levels['SEO_Week_Avg'].values[0] * results_merge[(results_merge['index'] >= min_date) & (results_merge['Part_Ref'] == part_ref) & (results_merge['Cost_Sale_avg'] > 0)]['Cost_Sale_avg'].head(1).values[0]

    # Hot start - When there is already a surplus of paid stock
    sel_results_merge = sel_results_merge.assign(initial_stock=initial_stock)
    sel_results_merge = sel_results_merge.assign(sales_evolution_local=sel_results_merge['Qty_Sold_sum_al'].cumsum())
    sel_results_merge.loc[:, 'stock_evolution_only_sales'] = sel_results_merge['initial_stock'] - sel_results_merge['sales_evolution_local']

    # Setting the Stock Levels
    # sel_results_merge = sel_results_merge.assign(seo_week_min=sel_stock_levels['SEO_Week_Min'].values[0])
    # sel_results_merge = sel_results_merge.assign(seo_week_max=sel_stock_levels['SEO_Week_Max'].values[0])
    # sel_results_merge = sel_results_merge.assign(seo_week_avg=sel_stock_levels['SEO_Week_Avg'].values[0])

    # Setting the Stock Levels
    for first_week_initial_week_day, second_week_initial_day in zip(sel_weeks, sel_weeks[1:]):
        sel_results_merge.loc[(sel_results_merge['index'] >= first_week_initial_week_day) & (sel_results_merge['index'] < second_week_initial_day), 'seo_week_min'] = sel_stock_levels[sel_stock_levels['Week_Start_Day'] == first_week_initial_week_day]['SEO_Week_Min'].values[0]
        sel_results_merge.loc[(sel_results_merge['index'] >= first_week_initial_week_day) & (sel_results_merge['index'] < second_week_initial_day), 'seo_week_max'] = sel_stock_levels[sel_stock_levels['Week_Start_Day'] == first_week_initial_week_day]['SEO_Week_Max'].values[0]
        sel_results_merge.loc[(sel_results_merge['index'] >= first_week_initial_week_day) & (sel_results_merge['index'] < second_week_initial_day), 'seo_week_avg'] = sel_stock_levels[sel_stock_levels['Week_Start_Day'] == first_week_initial_week_day]['SEO_Week_Avg'].values[0]

    last_date = sel_weeks[-1]
    [sel_results_merge[x].fillna(method='bfill', inplace=True) for x in ['seo_week_avg', 'seo_week_max', 'seo_week_min']]  # Handling the first days before any SEO stock
    # Handling the last week
    sel_results_merge.loc[(sel_results_merge['index'] >= last_date), 'seo_week_min'] = sel_stock_levels[sel_stock_levels['Week_Start_Day'] == last_date]['SEO_Week_Min'].values[0]
    sel_results_merge.loc[(sel_results_merge['index'] >= last_date), 'seo_week_max'] = sel_stock_levels[sel_stock_levels['Week_Start_Day'] == last_date]['SEO_Week_Max'].values[0]
    sel_results_merge.loc[(sel_results_merge['index'] >= last_date), 'seo_week_avg'] = sel_stock_levels[sel_stock_levels['Week_Start_Day'] == last_date]['SEO_Week_Avg'].values[0]

    # Calculating Differences between Stock level and remaining stock after sales -> Stock Left = Qty Sold - Stock Level
    sel_results_merge.loc[:, 'stock_diff_seo_min'] = sel_results_merge['Qty_Sold_sum_al'] - sel_results_merge['seo_week_min']
    sel_results_merge.loc[:, 'stock_diff_seo_max'] = sel_results_merge['Qty_Sold_sum_al'] - sel_results_merge['seo_week_max']
    sel_results_merge.loc[:, 'stock_diff_seo_avg'] = sel_results_merge['Qty_Sold_sum_al'] - sel_results_merge['seo_week_avg']

    # Sales <= Optimized Stock - Cost of restocking when the amount of sales is less than the stock level -> Cost = Qty Sold * Cost
    sel_results_merge.loc[sel_results_merge['Qty_Sold_sum_al'] <= sel_results_merge['seo_week_min'], 'seo_stock_week_min_cost'] = sel_results_merge['Qty_Sold_sum_al'] * sel_results_merge['Cost_Sale_avg']
    sel_results_merge.loc[sel_results_merge['Qty_Sold_sum_al'] <= sel_results_merge['seo_week_max'], 'seo_stock_week_max_cost'] = sel_results_merge['Qty_Sold_sum_al'] * sel_results_merge['Cost_Sale_avg']
    sel_results_merge.loc[sel_results_merge['Qty_Sold_sum_al'] <= sel_results_merge['seo_week_avg'], 'seo_stock_week_avg_cost'] = sel_results_merge['Qty_Sold_sum_al'] * sel_results_merge['Cost_Sale_avg']

    # Sales > Optimized Stock - Cost of restocking when the amount of sales is higher than the stock level
    if emergency_cost_per_part:  # If considering an emergency cost per part -> Cost = Cost * Stock Level + Stock Left * (Cost + 5)
        sel_results_merge.loc[sel_results_merge['Qty_Sold_sum_al'] > sel_results_merge['seo_week_min'], 'seo_stock_week_min_cost'] = sel_results_merge['Cost_Sale_avg'] * sel_results_merge['seo_week_min'] + sel_results_merge['stock_diff_seo_min'] * (sel_results_merge['Cost_Sale_avg'] + emergency_cost)
        sel_results_merge.loc[sel_results_merge['Qty_Sold_sum_al'] > sel_results_merge['seo_week_max'], 'seo_stock_week_max_cost'] = sel_results_merge['Cost_Sale_avg'] * sel_results_merge['seo_week_max'] + sel_results_merge['stock_diff_seo_max'] * (sel_results_merge['Cost_Sale_avg'] + emergency_cost)
        sel_results_merge.loc[sel_results_merge['Qty_Sold_sum_al'] > sel_results_merge['seo_week_avg'], 'seo_stock_week_avg_cost'] = sel_results_merge['Cost_Sale_avg'] * sel_results_merge['seo_week_avg'] + sel_results_merge['stock_diff_seo_avg'] * (sel_results_merge['Cost_Sale_avg'] + emergency_cost)

    elif emergency_cost_per_purchase:  # If considering an emergency cost per purchase/order -> Cost = Cost * Stock Level + Stock Left * Cost + 5
        sel_results_merge.loc[sel_results_merge['Qty_Sold_sum_al'] > sel_results_merge['seo_week_min'], 'seo_stock_week_min_cost'] = sel_results_merge['Cost_Sale_avg'] * sel_results_merge['seo_week_min'] + sel_results_merge['stock_diff_seo_min'] * sel_results_merge['Cost_Sale_avg'] + emergency_cost
        sel_results_merge.loc[sel_results_merge['Qty_Sold_sum_al'] > sel_results_merge['seo_week_max'], 'seo_stock_week_max_cost'] = sel_results_merge['Cost_Sale_avg'] * sel_results_merge['seo_week_max'] + sel_results_merge['stock_diff_seo_max'] * sel_results_merge['Cost_Sale_avg'] + emergency_cost
        sel_results_merge.loc[sel_results_merge['Qty_Sold_sum_al'] > sel_results_merge['seo_week_avg'], 'seo_stock_week_avg_cost'] = sel_results_merge['Cost_Sale_avg'] * sel_results_merge['seo_week_avg'] + sel_results_merge['stock_diff_seo_avg'] * sel_results_merge['Cost_Sale_avg'] + emergency_cost

    if hot_start:
        # Hot Start Cost Correction
        # Stock > Stock Level
        sel_results_merge['seo_stock_week_min_cost_backup'] = sel_results_merge['seo_stock_week_min_cost']
        sel_results_merge['seo_stock_week_max_cost_backup'] = sel_results_merge['seo_stock_week_max_cost']
        sel_results_merge['seo_stock_week_avg_cost_backup'] = sel_results_merge['seo_stock_week_avg_cost']
        sel_results_merge.loc[sel_results_merge['stock_evolution_only_sales'] > sel_results_merge['seo_week_min'], 'seo_stock_week_min_cost'] = 0
        sel_results_merge.loc[sel_results_merge['stock_evolution_only_sales'] > sel_results_merge['seo_week_max'], 'seo_stock_week_avg_cost'] = 0
        sel_results_merge.loc[sel_results_merge['stock_evolution_only_sales'] > sel_results_merge['seo_week_avg'], 'seo_stock_week_max_cost'] = 0

        try:  # This is for cases where the stock doesn't reach its minimum level
            index_min = sel_results_merge.loc[sel_results_merge['stock_evolution_only_sales'] <= sel_results_merge['seo_week_min'], 'seo_stock_week_min_cost'].index[0]

            # First row where Stock < Stock Level
            sel_results_merge.loc[index_min, 'seo_stock_week_min_cost'] = (sel_results_merge.loc[index_min, 'seo_week_min'] - sel_results_merge.loc[index_min, 'stock_evolution_only_sales']) * sel_results_merge.loc[index_min, 'Cost_Sale_avg']
            if emergency_cost_per_purchase:
                if sel_results_merge.loc[index_min, 'stock_evolution_only_sales'] < 0:
                    sel_results_merge.loc[index_min, 'seo_stock_week_min_cost'] = sel_results_merge.loc[index_min, 'seo_stock_week_min_cost'] + emergency_cost
            elif emergency_cost_per_part:
                if sel_results_merge.loc[index_min, 'stock_evolution_only_sales'] < 0:
                    sel_results_merge.loc[index_min, 'seo_stock_week_min_cost'] = (sel_results_merge.loc[index_min, 'seo_week_min'] - sel_results_merge.loc[index_min, 'stock_evolution_only_sales']) * (sel_results_merge.loc[index_min, 'Cost_Sale_avg'] + emergency_cost)
        except IndexError:
            pass

        try:
            index_avg = sel_results_merge.loc[sel_results_merge['stock_evolution_only_sales'] <= sel_results_merge['seo_week_avg'], 'seo_stock_week_avg_cost'].index[0]

            # First row where Stock < Stock Level
            sel_results_merge.loc[index_avg, 'seo_stock_week_avg_cost'] = (sel_results_merge.loc[index_avg, 'seo_week_avg'] - sel_results_merge.loc[index_avg, 'stock_evolution_only_sales']) * sel_results_merge.loc[index_avg, 'Cost_Sale_avg']
            if emergency_cost_per_purchase:
                if sel_results_merge.loc[index_avg, 'stock_evolution_only_sales'] < 0:
                    sel_results_merge.loc[index_avg, 'seo_stock_week_avg_cost'] = sel_results_merge.loc[index_avg, 'seo_stock_week_avg_cost'] + emergency_cost
            elif emergency_cost_per_part:
                if sel_results_merge.loc[index_avg, 'stock_evolution_only_sales'] < 0:
                    sel_results_merge.loc[index_avg, 'seo_stock_week_avg_cost'] = (sel_results_merge.loc[index_avg, 'seo_week_avg'] - sel_results_merge.loc[index_avg, 'stock_evolution_only_sales']) * (sel_results_merge.loc[index_avg, 'Cost_Sale_avg'] + emergency_cost)
        except IndexError:
            pass

        try:
            index_max = sel_results_merge.loc[sel_results_merge['stock_evolution_only_sales'] <= sel_results_merge['seo_week_max'], 'seo_stock_week_max_cost'].index[0]

            # First row where Stock < Stock Level
            sel_results_merge.loc[index_max, 'seo_stock_week_max_cost'] = (sel_results_merge.loc[index_max, 'seo_week_max'] - sel_results_merge.loc[index_max, 'stock_evolution_only_sales']) * sel_results_merge.loc[index_max, 'Cost_Sale_avg']
            if emergency_cost_per_purchase:
                if sel_results_merge.loc[index_max, 'stock_evolution_only_sales'] < 0:
                    sel_results_merge.loc[index_max, 'seo_stock_week_max_cost'] = sel_results_merge.loc[index_max, 'seo_stock_week_max_cost'] + emergency_cost
            elif emergency_cost_per_part:
                if sel_results_merge.loc[index_max, 'stock_evolution_only_sales'] < 0:
                    sel_results_merge.loc[index_max, 'seo_stock_week_max_cost'] = (sel_results_merge.loc[index_max, 'seo_week_max'] - sel_results_merge.loc[index_max, 'stock_evolution_only_sales']) * (sel_results_merge.loc[index_max, 'Cost_Sale_avg'] + emergency_cost)
        except IndexError:
            pass

        # if emergency_cost_per_purchase:
        #     if sel_results_merge.loc[index_min, 'stock_evolution_only_sales'] < 0:
        #         sel_results_merge.loc[index_min, 'seo_stock_week_min_cost'] = sel_results_merge.loc[index_min, 'seo_stock_week_min_cost'] + 5
        #     if sel_results_merge.loc[index_avg, 'stock_evolution_only_sales'] < 0:
        #         sel_results_merge.loc[index_avg, 'seo_stock_week_avg_cost'] = sel_results_merge.loc[index_avg, 'seo_stock_week_avg_cost'] + 5
        #     if sel_results_merge.loc[index_max, 'stock_evolution_only_sales'] < 0:
        #         sel_results_merge.loc[index_max, 'seo_stock_week_max_cost'] = sel_results_merge.loc[index_max, 'seo_stock_week_max_cost'] + 5
        # elif emergency_cost_per_part:
        #     if sel_results_merge.loc[index_min, 'stock_evolution_only_sales'] < 0:
        #         sel_results_merge.loc[index_min, 'seo_stock_week_min_cost'] = (sel_results_merge.loc[index_min, 'seo_week_min'] - sel_results_merge.loc[index_min, 'stock_evolution_only_sales']) * (sel_results_merge.loc[index_min, 'Cost_Sale_avg'] + 5)
        #     if sel_results_merge.loc[index_avg, 'stock_evolution_only_sales'] < 0:
        #         sel_results_merge.loc[index_avg, 'seo_stock_week_avg_cost'] = (sel_results_merge.loc[index_avg, 'seo_week_avg'] - sel_results_merge.loc[index_avg, 'stock_evolution_only_sales']) * (sel_results_merge.loc[index_avg, 'Cost_Sale_avg'] + 5)
        #     if sel_results_merge.loc[index_max, 'stock_evolution_only_sales'] < 0:
        #         sel_results_merge.loc[index_max, 'seo_stock_week_max_cost'] = (sel_results_merge.loc[index_max, 'seo_week_max'] - sel_results_merge.loc[index_max, 'stock_evolution_only_sales']) * (sel_results_merge.loc[index_max, 'Cost_Sale_avg'] + 5)

        # Total Costs Calculations:
        # Calculation of total cost
        seo_purchase_cost_min_stock = sel_results_merge['seo_stock_week_min_cost'].sum()
        seo_purchase_cost_max_stock = sel_results_merge['seo_stock_week_max_cost'].sum()
        seo_purchase_cost_avg_stock = sel_results_merge['seo_stock_week_avg_cost'].sum()

        # Calculation of total cumulative cost
        sel_results_merge['seo_stock_week_min_cost evolution'] = sel_results_merge['seo_stock_week_min_cost'].cumsum()
        sel_results_merge['seo_stock_week_max_cost evolution'] = sel_results_merge['seo_stock_week_max_cost'].cumsum()
        sel_results_merge['seo_stock_week_avg_cost evolution'] = sel_results_merge['seo_stock_week_avg_cost'].cumsum()

        # print('\nSEO Costs: \nWeekly Min Value: {:.3f} + Cold Start Cost: {:.3f} = {:.3f} \nWeekly Max Value: {:.3f} + Cold Start Cost: {:.3f} = {:.3f} \nWeekly Avg Value: {:.3f} + Cold Start Cost: {:.3f} = {:.3f} \n'
        #       .format(seo_purchase_cost_min_stock, cold_start_seo_min_cost, seo_purchase_cost_min_stock + cold_start_seo_min_cost, seo_purchase_cost_max_stock, cold_start_seo_max_cost, seo_purchase_cost_max_stock + cold_start_seo_max_cost, seo_purchase_cost_avg_stock, cold_start_seo_avg_cost, seo_purchase_cost_avg_stock + cold_start_seo_avg_cost))
        print('\nSEO Costs: \nWeekly Min Value: {:.3f} \nWeekly Max Value: {:.3f} \nWeekly Avg Value: {:.3f} \n'
              .format(seo_purchase_cost_min_stock, seo_purchase_cost_max_stock, seo_purchase_cost_avg_stock))

        # print(sel_results_merge[['index', 'Cost_Sale_avg', 'Stock_Qty_al', 'Qty_Sold_sum_al', 'initial_stock', 'sales_evolution_local', 'stock_evolution_only_sales', 'seo_stock_week_min_cost', 'seo_week_min', 'seo_stock_week_avg_cost', 'seo_week_avg', 'seo_stock_week_max_cost', 'seo_week_max']].head(55))

        # print(sel_results_merge[sel_results_merge['Qty_Regulated_sum'] != 0][['index', 'Qty_Regulated_sum', 'Cost_Reg_avg', 'Regularizations_Cost', 'Qty_Purchased_non_urgent_sum', 'Purchases_Cost_non_urgent', 'Qty_Purchased_urgent_sum', 'Purchases_Cost_urgent', 'Cost_Purchase_avg']])
        # print(sel_results_merge[sel_results_merge['Qty_Purchased_urgent_sum'] != 0][['index', 'Qty_Purchased_urgent_sum', 'Purchases_Cost_urgent', 'Cost_Purchase_avg']])
        # print(sel_results_merge[sel_results_merge['Qty_Purchased_non_urgent_sum'] != 0][['index', 'Qty_Purchased_non_urgent_sum', 'Purchases_Cost_non_urgent', 'Cost_Purchase_avg']])

    # print(sel_results_merge[['index', 'Part_Ref', 'Qty_Purchased_sum', 'Qty_Regulated_sum', 'Qty_Sold_sum_al', 'Stock_Qty_al', 'current_total_cost', 'total_cost evolution', seo_week_min  seo_week_max  seo_week_avg]].head(30))
    return sel_results_merge


def weather_treatment():
    print('Weather Data Treatment started...')
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


def sql_data(selected_part, pse_code, min_date, max_date, dataframes_list):

    df_sales, df_al, df_stock, df_reg_al_clients, df_purchases = dataframes_list[0], dataframes_list[1], dataframes_list[2], dataframes_list[3], dataframes_list[4]
    result, stock_evolution_correct_flag, offset = pd.DataFrame(), 0, 0

    df_sales_filtered = df_sales[(df_sales['Part_Ref'].isin(selected_part)) & (df_sales['Movement_Date'] > min_date) & (df_sales['Movement_Date'] <= max_date)]
    df_al_filtered = df_al[(df_al['Part_Ref'].isin(selected_part)) & (df_al['Movement_Date'] > min_date) & (df_al['Movement_Date'] <= max_date)]
    df_purchases_filtered = df_purchases[(df_purchases['Part_Ref'].isin(selected_part)) & (df_purchases.index > min_date) & (df_purchases.index <= max_date)]
    df_stock_filtered = df_stock[(df_stock['Part_Ref'].isin(selected_part)) & (df_stock.index >= min_date) & (df_stock.index <= max_date)]

    df_al_filtered = auto_line_dataset_cleaning(df_sales_filtered, df_al_filtered, df_purchases_filtered, df_reg_al_clients, pse_code)

    df_sales_filtered = df_sales_filtered.drop_duplicates(subset=['Movement_Date', 'Part_Ref']).sort_values(by='Movement_Date')
    df_sales_filtered.set_index('Movement_Date', inplace=True)

    if not df_al_filtered.shape[0]:
        # raise ValueError('No data found for part_ref {} and/or selected period {}/{}'.format(selected_part[0], min_date, max_date))
        # print('\nNo data found for part_ref {} and/or selected period {}/{}.\n'.format(selected_part[0], min_date, max_date))
        no_data_flag = 1
        return pd.DataFrame(), stock_evolution_correct_flag, offset
    elif df_al_filtered.shape[0] == 1:
        # raise ValueError('Only 1 row found for part_ref {} and/or selected period {}/{}'.format(selected_part[0], min_date, max_date))
        # print('\nOnly 1 row found for part_ref {} and/or selected period {}/{}. Ignored.\n'.format(selected_part[0], min_date, max_date))
        one_row_only_flag = 1
        return pd.DataFrame(), stock_evolution_correct_flag, offset

    df_al_filtered['Qty_Sold_sum_al'], df_al_filtered['Cost_Sale_avg'], df_al_filtered['PVP_avg'] = 0, 0, 0  # Placeholder for cases without sales
    df_al_grouped = df_al_filtered[df_al_filtered['regularization_flag'] == 0].groupby(['Movement_Date', 'Part_Ref'])
    for key, row in df_al_grouped:
        rows_selection = (df_al_filtered['Movement_Date'] == key[0]) & (df_al_filtered['Part_Ref'] == key[1])
        df_al_filtered.loc[rows_selection, 'Qty_Sold_sum_al'] = row['Unit'].sum()
        df_al_filtered.loc[rows_selection, 'Cost_Sale_avg'] = row['Preço de custo'].mean()
        df_al_filtered.loc[rows_selection, 'PVP_avg'] = row['P. V. P'].mean()

    df_al_filtered['Qty_Regulated_sum'], df_al_filtered['Cost_Reg_avg'] = 0, 0  # Placeholder for cases without regularizations
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

        qty_sold_mov = df_sales_filtered['Qty_Sold_sum_mov'].sum()
        qty_sold_al = df_al_filtered['Qty_Sold_sum_al'].sum()
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

        # test_dfs = [df_stock_filtered['Stock_Qty'].head(1), df_purchases_filtered[['Qty_Purchased_sum', 'Qty_Purchased_urgent_sum', 'Qty_Purchased_non_urgent_sum', 'Cost_Purchase_avg']], df_al_filtered[['Qty_Regulated_sum', 'Cost_Reg_avg', 'Qty_Sold_sum_al', 'Cost_Sale_avg', 'PVP_avg']]]
        # for df in test_dfs:
        #     print(df.head())
        #     print(df.tail())

        # result = pd.concat([df_stock_filtered.head(1), df_purchases_filtered[['Qty_Purchased_sum', 'Qty_Purchased_urgent_sum', 'Qty_Purchased_non_urgent_sum', 'Cost_Purchase_avg']], df_sales_filtered['Qty_Sold_sum_mov'], df_al_filtered[['Qty_Regulated_sum', 'Cost_Reg_avg', 'Qty_Sold_sum_al', 'Cost_Sale_avg', 'PVP_avg']]], axis=1, sort=False)
        # if not df_stock_filtered.shape[0]:
        #     df_stock_filtered['Part_Ref'] = selected_part

        result = pd.concat([df_purchases_filtered[['Qty_Purchased_sum', 'Qty_Purchased_urgent_sum', 'Qty_Purchased_non_urgent_sum', 'Cost_Purchase_avg']], df_al_filtered[['Qty_Regulated_sum', 'Cost_Reg_avg', 'Qty_Sold_sum_al', 'Cost_Sale_avg', 'PVP_avg']]], axis=1, sort=False)
        result['Part_Ref'] = selected_part * result.shape[0]
        try:
            result['Stock_Qty'] = df_stock_filtered['Stock_Qty'].head(1).values[0]
        except IndexError:
            result['Stock_Qty'] = 0  # Cases when there is no stock information

        # print('\nStock at Start: {} \nSum Purchases: {} \nSum Sales SQL: {} \nSum Sales AutoLine: {} \nSum Regularizations: {} \nStock at End: {} \nStock Variance: {}'.format(stock_start, qty_purchased, qty_sold_mov, qty_sold_al, reg_value, stock_end, delta_stock))
        # result_mov = stock_start + qty_purchased - qty_sold_mov - reg_value
        result_al = stock_start + qty_purchased - qty_sold_al - reg_value

        # if result_mov != stock_end:
        #     print('\nValues dont match for SQL values - Stock has an offset of {:.2f}'.format(stock_end - result_mov))
        # else:
        #     print('\nValues for SQL are correct :D')

        if result_al != stock_end:
            offset = stock_end - result_al
            # print('Selected Part: {} - Values dont match for AutoLine values - Stock has an offset of {:.2f} \n'.format(selected_part, offset))
        else:
            # print('Selected Part: {} - Values for AutoLine are correct :D \n'.format(selected_part))
            stock_evolution_correct_flag = 1

        result['Stock_Qty'].fillna(method='ffill', inplace=True)
        result['Part_Ref'].fillna(method='ffill', inplace=True)
        result.fillna(0, inplace=True)

        result['Sales Evolution_al'] = result['Qty_Sold_sum_al'].cumsum()
        # result['Sales Evolution_mov'] = result['Qty_Sold_sum_mov'].cumsum()
        result['Purchases Evolution'] = result['Qty_Purchased_sum'].cumsum()
        result['Purchases Urgent Evolution'] = result['Qty_Purchased_urgent_sum'].cumsum()
        result['Purchases Non Urgent Evolution'] = result['Qty_Purchased_non_urgent_sum'].cumsum()
        result['Regulated Evolution'] = result['Qty_Regulated_sum'].cumsum()
        result['Stock_Qty_al'] = result['Stock_Qty'] - result['Sales Evolution_al'] + result['Purchases Evolution'] - result['Regulated Evolution']
        # result['Stock_Qty_mov'] = result['Stock_Qty'] - result['Sales Evolution_mov'] + result['Purchases Evolution'] - result['Regulated Evolution']
        result.loc[result['Qty_Purchased_sum'] == 0, 'Cost_Purchase_avg'] = 0

        # print(result[result.index <= '2018-01-31'][['Stock_Qty', 'Qty_Purchased_sum', 'Qty_Regulated_sum', 'Qty_Sold_sum_al', 'Stock_Qty_al']])
        # sys.exit()
        # print(df_stock_filtered)
        # print(result[result.index.isin(df_stock_filtered.index)]['Stock_Qty_al'])

        # result = result.join(weather_data_daily)  ### Note: Right now, no weather data is needed.

        # print(result.shape)
        # print(result.head())
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


def purchases_reg_cleaning(df_al, purchases_unique_plr, reg_unique_slr):

    matched_rows_purchases = df_al[df_al['SLR_Document_Number'].isin(purchases_unique_plr)]
    if matched_rows_purchases.shape[0]:
        df_al = df_al.drop(matched_rows_purchases.index, axis=0)

    matched_rows_reg = df_al[df_al['SLR_Document_Account'].isin(reg_unique_slr)].index

    df_al['regularization_flag'] = 0
    df_al.loc[df_al.index.isin(matched_rows_reg), 'regularization_flag'] = 1
    df_al.loc[df_al['regularization_flag'] == 1, 'Cost_Reg'] = df_al['Unit'] * df_al['Preço de custo']

    return df_al


def sales_cleaning(args):
    key, group, df_sales, pse_code = args
    group_size = group.shape[0]

    # print('initial group: \n', group)

    # if group['WIP_Number'].unique() == 23468:
    #     print('initial group: \n', group)
    # Note: There might not be a match!
    if group_size > 1 and group['Audit_Number'].nunique() < group_size:
        # print(group)

        # if group['WIP_Number'].unique() == 23468:
        #     print('group to clean: \n', group)

        matching_sales = df_sales[(df_sales['Movement_Date'] == key[0]) & (df_sales['Part_Ref'] == key[1]) & (df_sales['WIP_Number'] == key[2])]

        # if group['WIP_Number'].unique() == 23468:
        #     print('matched sales: \n', matching_sales)

        number_matched_lines, group_size = matching_sales.shape[0], group.shape[0]
        if 0 < number_matched_lines <= group_size:

            # if group['WIP_Number'].unique() == 23468:
            #     print('matched lines under group size')

            group = group[group['SLR_Document_Number'].isin(matching_sales['SLR_Document'].unique())]

            # if group['WIP_Number'].unique() == 23468:
            #     print('group after cleaning by sales: \n', group)

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


if __name__ == '__main__':
    main()
