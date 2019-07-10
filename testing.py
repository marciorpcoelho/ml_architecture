import os
import sys
import time
import pandas as pd
import numpy as np
from multiprocessing import Pool
pd.set_option('display.expand_frame_repr', False)
from level_1_a_data_acquisition import sql_retrieve_df_specified_query
import apv_sales_options as options_file
from level_1_b_data_processing import null_analysis
import level_0_performance_report
import warnings
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

    # selected_parts = ['BM83.21.2.405.675']
    selected_parts = ['BM83.21.2.405.675', 'BM07.12.9.952.104', 'BM07.14.9.213.164', 'BM83.19.2.158.851', 'BM64.11.9.237.555']

    min_date = '2018-06-30'
    max_date = '2019-05-31'
    print('full year')
    # min_date = '2018-12-03'
    # max_date = '2019-04-30'
    # print('single month')

    version = 'v13'

    datetime_index = pd.date_range(start=min_date, end=max_date)
    results = pd.DataFrame()

    weather_data_daily = weather_treatment()

    for part_ref in selected_parts:
        result_part_ref = sql_data([part_ref], min_date, max_date, weather_data_daily)

        result_part_ref = result_part_ref.reindex(datetime_index).reset_index().rename(columns={'Unnamed: 0': 'Movement_Date'})
        ffill_cols = ['Part_Ref', 'Stock_Qty', 'Stock_Qty_al', 'Stock_Qty_mov', 'Sales Evolution_al', 'Sales Evolution_mov', 'Purchases Evolution', 'Regulated Evolution', 'Purchases Urgent Evolution', 'Purchases Non Urgent Evolution', 'T', 'U', 'Ff', 'Cost_Sale_avg', 'PVP_avg']
        zero_fill_cols = ['Qty_Purchased_sum', 'Qty_Regulated_sum', 'Qty_Sold_sum_mov', 'Qty_Sold_sum_al', 'Qty_Purchased_urgent_sum', 'Qty_Purchased_non_urgent_sum', 'Cost_Purchase_avg', 'Cost_Reg_avg']

        [result_part_ref[x].fillna(0, inplace=True) for x in zero_fill_cols]
        [result_part_ref[x].fillna(method='ffill', inplace=True) for x in ffill_cols]
        result_part_ref['weekday'] = result_part_ref['index'].dt.dayofweek
        result_part_ref['weekofyear'] = result_part_ref['index'].dt.weekofyear  # Careful with the end of year dates and respective weeks
        result_part_ref['year'] = result_part_ref['index'].dt.year
        result_part_ref['unique_weeks'] = result_part_ref['year'].apply(str) + '_' + result_part_ref['weekofyear'].apply(str)
        # End of year handling:
        result_part_ref.loc[result_part_ref['index'] == '2018-12-31', 'unique_weeks'] = '2019_1'

        # Just so the variation matches with rest: positive regularization means an increase in stock, while negative is a decrease; Equivalent for cost;
        result_part_ref['Qty_Regulated_sum'] = result_part_ref['Qty_Regulated_sum'] * (-1)
        result_part_ref['Cost_Reg_avg'] = result_part_ref['Cost_Reg_avg'] * (-1)

        result_part_ref = cost_calculation_function(result_part_ref, part_ref, min_date, max_date, version)

        results = results.append(result_part_ref)

    results.to_csv('output/results_merge.csv')


def cost_calculation_function(results_merge, part_ref, min_date, max_date, version):
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


def sql_data(selected_part, min_date, max_date, weather_data_daily):
    wip_date = 0
    slr_date = 0
    mov_date = 1

    print('Selected Part: {}'.format(selected_part))

    df_sales = pd.read_csv('dbs/df_sales_cleaned.csv', parse_dates=['Movement_Date', 'WIP_Date_Created', 'SLR_Document_Date'], usecols=['Movement_Date', 'WIP_Number', 'SLR_Document', 'WIP_Date_Created', 'SLR_Document_Date', 'Part_Ref', 'PVP_1', 'Cost_Sale_1', 'Qty_Sold_sum_wip', 'Qty_Sold_sum_slr', 'Qty_Sold_sum_mov']).sort_values(by='WIP_Date_Created')
    df_al = pd.read_excel('dbs/{}.xlsx'.format(selected_part[0]), usecols=['Data Mov', 'Refª da peça', 'Descrição', 'Unit', 'Nº de factura', 'WIP nº', 'Sugestão nº  (Enc)', 'Conta', 'Nº auditoria stock', 'Preço de custo', 'P. V. P'])
    df_al.rename(index=str, columns={'Data Mov': 'Movement_Date', 'Refª da peça': 'Part_Ref', 'Descrição': 'Part_Desc', 'Nº de factura': 'SLR_Document_Number', 'WIP nº': 'WIP_Number', 'Sugestão nº  (Enc)': 'Encomenda', 'Conta': 'SLR_Document_Account', 'Nº auditoria stock': 'Audit_Number'}, inplace=True)
    df_al['Movement_Date'] = pd.to_datetime(df_al['Movement_Date'], format='%d/%m/%Y')
    df_al.sort_values(by='Movement_Date', inplace=True)

    df_purchases = pd.read_csv('dbs/df_purchases_cleaned.csv', index_col=0, parse_dates=['Movement_Date']).sort_values(by='Movement_Date')
    df_purchases.rename(index=str, columns={'Qty_Sold_sum': 'Qty_Purchased_sum'}, inplace=True)  # Will be removed next time i run the data_processement
    df_stock = pd.read_csv('dbs/df_stock_01_07_19.csv', parse_dates=['Record_Date'], usecols=['Part_Ref', 'Quantity', 'Record_Date']).sort_values(by='Record_Date')
    df_stock.rename(index=str, columns={'Quantity': 'Stock_Qty'}, inplace=True)
    df_reg = pd.read_csv('dbs/df_reg_01_07_19.csv', parse_dates=['Movement_Date'], usecols=['Movement_Date', 'Part_Ref', 'Quantity', 'SLR_Document', 'Cost_Value']).sort_values(by='Movement_Date')
    df_reg.rename(index=str, columns={'Quantity': 'Qty_Regulated'}, inplace=True)
    df_reg_al_clients = pd.read_csv('dbs/df_reg_al_client_03_06_19.csv', index_col=0)

    # WIP Date
    # if wip_date:
    #     print('Using WIP_Date_Created')
    #     df_sales_filtered = df_sales[df_sales['Part_Ref'].isin(selected_part)]
    #     df_sales_filtered = df_sales_filtered.drop_duplicates(subset=['WIP_Date_Created', 'Part_Ref']).sort_values(by='WIP_Date_Created')
    #
    #     df_purchases_filtered = df_purchases[df_purchases['Part_Ref'].isin(selected_part)]
    #     # df_purchases_filtered['Purchases_Flag'] = 1
    #     df_stock_filtered = df_stock[df_stock['Part_Ref'].isin(selected_part)]
    #     # df_stock_filtered['Stock_Flag'] = 1
    #
    #     # dfs_filtered = [df_sales_filtered, df_purchases_filtered, df_stock_filtered]
    #
    #     df_stock_filtered.set_index('Record_Date', inplace=True)
    #     df_purchases_filtered.set_index('Movement_Date', inplace=True)
    #     df_sales_filtered.set_index('WIP_Date_Created', inplace=True)
    #
    #     print(df_stock_filtered[(df_stock_filtered.index >= min_date) & (df_stock_filtered.index <= max_date)])
    #     print(df_purchases_filtered[(df_purchases_filtered.index >= min_date) & (df_purchases_filtered.index <= max_date)])
    #     print(df_sales_filtered[(df_stock_filtered.index > min_date) & (df_sales_filtered.index <= max_date)])
    #
    #     sys.exit()
    #
    #     result = pd.concat([df_stock_filtered[['Part_Ref', 'Stock_Qty']], df_purchases_filtered[['Qty_Purchased_sum']], df_sales_filtered['Qty_Sold_sum_wip']], axis=1, sort=False)
    #
    #     # result_2 = pd.concat([result, df_sales_filtered['Qty_Sold_sum']], axis=1, sort=False)
    #
    #     print('Qty_Sold_sum_WIP: {}'.format(result[(result.index >= '2019-03-31') & (result.index <= '2019-04-30')]['Qty_Sold_sum_wip'].sum()))
    #
    #     print(result[(result.index >= '2019-03-31') & (result.index <= '2019-04-30')])

    # SLR Date
    # if slr_date:
    #     print('Using SLR_Document_Date:')
    #     df_sales_filtered = df_sales[df_sales['Part_Ref'].isin(selected_parts)]
    #     df_sales_filtered = df_sales_filtered.drop_duplicates(subset=['SLR_Document_Date', 'Part_Ref']).sort_values(by='SLR_Document_Date')
    #
    #     df_purchases_filtered = df_purchases[df_purchases['Part_Ref'].isin(selected_parts)]
    #     # df_purchases_filtered['Purchases_Flag'] = 1
    #     df_stock_filtered = df_stock[df_stock['Part_Ref'].isin(selected_parts)]
    #     # df_stock_filtered['Stock_Flag'] = 1
    #     df_reg_filtered = df_reg[df_reg['Part_Ref'].isin(selected_parts)]
    #
    #     # dfs_filtered = [df_sales_filtered, df_purchases_filtered, df_stock_filtered]
    #
    #     df_stock_filtered.set_index('Record_Date', inplace=True)
    #     df_purchases_filtered.set_index('Movement_Date', inplace=True)
    #     df_sales_filtered.set_index('SLR_Document_Date', inplace=True)
    #     df_reg_filtered.set_index('Movement_Date', inplace=True)
    #
    #     print(df_stock_filtered[(df_stock_filtered.index >= min_date) & (df_stock_filtered.index <= max_date)])
    #     print(df_purchases_filtered[(df_purchases_filtered.index >= min_date) & (df_purchases_filtered.index <= max_date)])
    #     print(df_sales_filtered[(df_sales_filtered.index > min_date) & (df_sales_filtered.index <= max_date)])
    #     print(df_reg_filtered[(df_reg_filtered.index > min_date) & (df_reg_filtered.index <= max_date)])
    #
    #     qty_sold = df_sales_filtered[(df_sales_filtered.index > '2019-01-31') & (df_sales_filtered.index <= '2019-02-28')]['Qty_Sold_sum_slr'].sum()
    #     qty_purchased = df_purchases_filtered[(df_purchases_filtered.index > min_date) & (df_purchases_filtered.index <= max_date)]['Qty_Purchased_sum'].sum()
    #     stock_start = df_stock_filtered[df_stock_filtered.index == min_date]['Stock_Qty'].values[0]
    #     stock_end = df_stock_filtered[df_stock_filtered.index == max_date]['Stock_Qty'].values[0]
    #     reg_value = df_reg_filtered[(df_reg_filtered.index > min_date) & (df_reg_filtered.index <= max_date)]['Quantity'].sum()
    #
    #     if not reg_value:
    #         reg_value = 0
    #
    #     # print('Here: Qty_Sold_sum: {}'.format(df_sales_filtered[(df_sales_filtered.index > '2019-01-31') & (df_sales_filtered.index <= '2019-02-28')]['Qty_Sold_sum_slr'].sum()))
    #     print('\nStock at Start: {} \nSum Purchases: {} \nSum Sales: {} \nSum Regularizations: {} \nStock at End: {}'.format(stock_start, qty_purchased, qty_sold, reg_value, stock_end))
    #     result = stock_start + qty_purchased - qty_sold + reg_value
    #     if result != stock_end:
    #         print('\nValues dont match!')
    #         print('Stock has an offset of {}'.format(stock_end - result))
    #     else:
    #         print('\nValues are correct :D')
    #     sys.exit()
    #
    #     result = pd.concat([df_stock_filtered[['Part_Ref', 'Stock_Qty']], df_purchases_filtered[['Qty_Purchased_sum']], df_sales_filtered['Qty_Sold_sum_slr']], axis=1, sort=False)
    #
    #     # result_2 = pd.concat([result, df_sales_filtered['Qty_Sold_sum']], axis=1, sort=False)
    #
    #     print('Qty_Sold_sum_SLR: {}'.format(result[(result.index > '2019-03-31') & (result.index <= '2019-04-30')]['Qty_Sold_sum_slr'].sum()))
    #
    #     print(result[(result.index >= '2019-03-31') & (result.index <= '2019-04-30')])

    # Movement_Date
    if mov_date:
        print('Using Movement_Date:')
        df_sales_filtered = df_sales[(df_sales['Part_Ref'].isin(selected_part)) & (df_sales['Movement_Date'] > min_date) & (df_sales['Movement_Date'] <= max_date)]

        df_al['Unit'] = df_al['Unit'] * (-1)  # Turn the values to their symmetrical so it matches the other dfs
        df_al_filtered = df_al[(df_al['Part_Ref'].isin(selected_part)) & (df_al['Movement_Date'] > min_date) & (df_al['Movement_Date'] <= max_date)]

        df_stock.set_index('Record_Date', inplace=True)
        df_purchases.set_index('Movement_Date', inplace=True)
        df_reg.set_index('Movement_Date', inplace=True)

        df_purchases_filtered = df_purchases[(df_purchases['Part_Ref'].isin(selected_part)) & (df_purchases.index > min_date) & (df_purchases.index <= max_date)]
        df_stock_filtered = df_stock[(df_stock['Part_Ref'].isin(selected_part)) & (df_stock.index >= min_date) & (df_stock.index <= max_date)]
        df_reg_filtered = df_reg[(df_reg['Part_Ref'].isin(selected_part)) & (df_reg.index > min_date) & (df_reg.index <= max_date)]

        df_al_filtered = auto_line_dataset_cleaning(df_sales_filtered, df_al_filtered, df_purchases_filtered, df_reg_al_clients)

        df_sales_filtered = df_sales_filtered.drop_duplicates(subset=['Movement_Date', 'Part_Ref']).sort_values(by='Movement_Date')
        df_sales_filtered.set_index('Movement_Date', inplace=True)

        df_al_grouped = df_al_filtered.groupby(['Movement_Date', 'Part_Ref'])
        df_al_filtered['Qty_Sold_sum_al'] = df_al_grouped['Unit'].transform('sum')
        df_al_filtered['Cost_Sale_avg'] = df_al_grouped['Preço de custo'].transform('mean')
        df_al_filtered['PVP_avg'] = df_al_grouped['P. V. P'].transform('mean')
        df_al_filtered.drop(['Unit', 'Preço de custo', 'P. V. P'], axis=1, inplace=True)

        df_reg_grouped = df_reg_filtered.groupby(df_reg_filtered.index)
        df_reg_filtered['Qty_Regulated_sum'] = df_reg_grouped['Qty_Regulated'].transform('sum')
        df_reg_filtered['Cost_Reg_avg'] = df_reg_grouped['Cost_Value'].transform('mean')
        df_reg_filtered = df_reg_filtered.loc[~df_reg_filtered.index.duplicated(keep='first')]

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

        reg_value = df_reg_filtered['Qty_Regulated_sum'].sum()
        delta_stock = stock_end - stock_start

        if not reg_value:
            reg_value = 0

        result = pd.concat([df_stock_filtered.head(1), df_purchases_filtered[['Qty_Purchased_sum', 'Qty_Purchased_urgent_sum', 'Qty_Purchased_non_urgent_sum', 'Cost_Purchase_avg']], df_reg_filtered[['Qty_Regulated_sum', 'Cost_Reg_avg']], df_sales_filtered['Qty_Sold_sum_mov'], df_al_filtered[['Qty_Sold_sum_al', 'Cost_Sale_avg', 'PVP_avg']]], axis=1, sort=False)

        print('\nStock at Start: {} \nSum Purchases: {} \nSum Sales SQL: {} \nSum Sales AutoLine: {} \nSum Regularizations: {} \nStock at End: {} \nStock Variance: {}'.format(stock_start, qty_purchased, qty_sold_mov, qty_sold_al, reg_value, stock_end, delta_stock))
        result_mov = stock_start + qty_purchased - qty_sold_mov - reg_value
        result_al = stock_start + qty_purchased - qty_sold_al - reg_value

        # if result_mov != stock_end:
        #     print('\nValues dont match for SQL values - Stock has an offset of {:.2f}'.format(stock_end - result_mov))
        # else:
        #     print('\nValues for SQL are correct :D')

        if result_al != stock_end:
            print('\nValues dont match for AutoLine values - Stock has an offset of {:.2f}'.format(stock_end - result_al))
        else:
            print('\nValues for AutoLine are correct :D \n')

        result['Stock_Qty'].fillna(method='ffill', inplace=True)
        result['Part_Ref'].fillna(method='ffill', inplace=True)
        result.fillna(0, inplace=True)

        result['Sales Evolution_al'] = result['Qty_Sold_sum_al'].cumsum()
        result['Sales Evolution_mov'] = result['Qty_Sold_sum_mov'].cumsum()
        result['Purchases Evolution'] = result['Qty_Purchased_sum'].cumsum()
        result['Purchases Urgent Evolution'] = result['Qty_Purchased_urgent_sum'].cumsum()
        result['Purchases Non Urgent Evolution'] = result['Qty_Purchased_non_urgent_sum'].cumsum()
        result['Regulated Evolution'] = result['Qty_Regulated_sum'].cumsum()
        result['Stock_Qty_al'] = result['Stock_Qty'] - result['Sales Evolution_al'] + result['Purchases Evolution'] - result['Regulated Evolution']
        result['Stock_Qty_mov'] = result['Stock_Qty'] - result['Sales Evolution_mov'] + result['Purchases Evolution'] - result['Regulated Evolution']
        result.loc[result['Qty_Purchased_sum'] == 0, 'Cost_Purchase_avg'] = 0

        result = result.join(weather_data_daily)

        result.to_csv('output/{}_stock_evolution.csv'.format(selected_part[0]))

        return result


def auto_line_dataset_cleaning(df_sales, df_al, df_purchases, df_reg_al_clients):
    start = time.time()
    print('AutoLine and PSE_Sales Lines comparison started')

    purchases_unique_plr = df_purchases['PLR_Document'].unique().tolist()
    reg_unique_slr = df_reg_al_clients['SLR_Account'].unique().tolist() + ['@Check']

    duplicated_rows = df_al[df_al.duplicated(subset='Encomenda', keep=False)]
    # print('duplicated_rows: \n{}'.format(duplicated_rows))
    if duplicated_rows.shape[0]:
        duplicated_rows_grouped = duplicated_rows.groupby(['Movement_Date', 'Part_Ref', 'WIP_Number'])

        df_al = df_al.drop(duplicated_rows.index, axis=0)

        pool = Pool(processes=int(level_0_performance_report.pool_workers_count))
        results = pool.map(sales_cleaning, [(key, group, df_sales) for (key, group) in duplicated_rows_grouped])
        pool.close()
        df_al_merged = pd.concat([df_al, pd.concat([result for result in results if result is not None])], axis=0)

        df_al_cleaned = purchases_reg_cleaning(df_al_merged, purchases_unique_plr, reg_unique_slr)

        print('AutoLine and PSE_Sales Lines comparison ended. Elapsed time: {:.2f}'.format(time.time() - start))
    else:
        df_al_cleaned = purchases_reg_cleaning(df_al, purchases_unique_plr, reg_unique_slr)

    return df_al_cleaned


def purchases_reg_cleaning(df_al, purchases_unique_plr, reg_unique_slr):

    matched_rows_purchases = df_al[df_al['SLR_Document_Number'].isin(purchases_unique_plr)]
    if matched_rows_purchases.shape[0]:
        df_al = df_al.drop(matched_rows_purchases.index, axis=0)

    matched_rows_reg = df_al[df_al['SLR_Document_Account'].isin(reg_unique_slr)]
    if matched_rows_reg.shape[0]:
        df_al = df_al.drop(matched_rows_reg.index, axis=0)

    return df_al


def sales_cleaning(args):
    key, group, df_sales = args
    group_size = group.shape[0]

    # print('initial group: \n', group)
    # Note: There might not be a match!
    if group_size > 1 and group['Audit_Number'].nunique() < group_size:
        # print(group)

        # if group['WIP_Number'].unique() == 63960:
        #     print('group to clean: \n', group)

        matching_sales = df_sales[(df_sales['Movement_Date'] == key[0]) & (df_sales['Part_Ref'] == key[1]) & (df_sales['WIP_Number'] == key[2])]

        # if group['WIP_Number'].unique() == 63960:
        #     print('matched sales: \n', matching_sales)

        number_matched_lines, group_size = matching_sales.shape[0], group.shape[0]
        if 0 < number_matched_lines < group_size:

            # if group['WIP_Number'].unique() == 63960:
                # print('matched lines under group size')

            group = group[group['SLR_Document_Number'].isin(matching_sales['SLR_Document'].unique())]

            # if group['WIP_Number'].unique() == 63960:
            #     print('group after cleaning by sales: \n', group)

        elif number_matched_lines > 0 and number_matched_lines == group_size:
            # print('matched lines equal to group size')
            # print('sales = autoline: no rows to remove')
            pass  # ToDo will need to handle these exceptions better
        elif number_matched_lines > 0 and number_matched_lines > group_size:
            # print('matched lines over group size')
            # print('sales > autoline - weird case?')
            pass  # ToDo will need to handle these exceptions better
        elif number_matched_lines == 0:
            # print('NO MATCHED ROWS?!?')
            # print(group, '\n', matching_sales)
            group = group.tail(1)  # ToDo Needs to be confirmed

        # ToDo Martelanço:
        if group['Part_Ref'].unique() == 'BM83.19.2.158.851' and key[2] == 38381:
            group = group[group['SLR_Document_Number'] != 44446226]
        if group['Part_Ref'].unique() == 'BM83.21.2.405.675' and key[2] == 63960:
            group = group[group['SLR_Document_Number'] != 44462775]

    return group


if __name__ == '__main__':
    main()
