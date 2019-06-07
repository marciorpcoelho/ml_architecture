import os
import sys
import time
import pandas as pd
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

    # min_date = '2019-03-31'
    # max_date = '2019-04-30'
    # print('single month')

    for part_ref in selected_parts:
        sql_data([part_ref], min_date, max_date)
    # auto_line_data(selected_parts, min_date, max_date)


def sql_data(selected_part, min_date, max_date):
    wip_date = 0
    slr_date = 0
    mov_date = 1

    print('Selected Part: {}'.format(selected_part))

    df_sales = pd.read_csv('dbs/df_sales_cleaned.csv', parse_dates=['Movement_Date', 'WIP_Date_Created', 'SLR_Document_Date'], usecols=['Movement_Date', 'WIP_Number', 'SLR_Document', 'WIP_Date_Created', 'SLR_Document_Date', 'Part_Ref', 'PVP_1', 'Cost_Sale_1', 'Qty_Sold_sum_wip', 'Qty_Sold_sum_slr', 'Qty_Sold_sum_mov', 'T', 'weekday']).sort_values(by='WIP_Date_Created')
    df_al = pd.read_excel('dbs/{}.xlsx'.format(selected_part[0]), usecols=['Data Mov', 'Refª da peça', 'Descrição', 'Unit', 'Nº de factura', 'WIP nº', 'Sugestão nº  (Enc)', 'Conta', 'Nº auditoria stock'])
    df_al.rename(index=str, columns={'Data Mov': 'Movement_Date', 'Refª da peça': 'Part_Ref', 'Descrição': 'Part_Desc', 'Nº de factura': 'SLR_Document_Number', 'WIP nº': 'WIP_Number', 'Sugestão nº  (Enc)': 'Encomenda', 'Conta': 'SLR_Document_Account', 'Nº auditoria stock': 'Audit_Number'}, inplace=True)
    df_al['Movement_Date'] = pd.to_datetime(df_al['Movement_Date'], format='%d/%m/%Y')
    df_al.sort_values(by='Movement_Date', inplace=True)

    df_purchases = pd.read_csv('dbs/df_purchases_cleaned.csv', index_col=0, parse_dates=['Movement_Date']).sort_values(by='Movement_Date')
    df_purchases.rename(index=str, columns={'Qty_Sold_sum': 'Qty_Purchased_sum'}, inplace=True)  # Will be removed next time i run the data_processement
    df_stock = pd.read_csv('dbs/df_stock_03_06_19.csv', parse_dates=['Record_Date'], usecols=['Part_Ref', 'Quantity', 'Record_Date']).sort_values(by='Record_Date')
    df_stock.rename(index=str, columns={'Quantity': 'Stock_Qty'}, inplace=True)
    df_reg = pd.read_csv('dbs/df_reg_03_06_19.csv', parse_dates=['Movement_Date'], usecols=['Movement_Date', 'Part_Ref', 'Quantity', 'SLR_Document']).sort_values(by='Movement_Date')
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
        # df_sales_filtered = df_sales_filtered.drop_duplicates(subset=['Movement_Date', 'Part_Ref']).sort_values(by='Movement_Date')

        df_al['Unit'] = df_al['Unit'] * (-1)  # Turn the values to their symmetrical so it matches the other dfs
        df_al_filtered = df_al[(df_al['Part_Ref'].isin(selected_part)) & (df_al['Movement_Date'] > min_date) & (df_al['Movement_Date'] <= max_date)]

        # if selected_part == ['BM07.12.9.952.104']:
        #     print('a \n', df_al_filtered[df_al_filtered['Movement_Date'] == '2018-08-22'])

        df_stock.set_index('Record_Date', inplace=True)
        df_purchases.set_index('Movement_Date', inplace=True)
        df_reg.set_index('Movement_Date', inplace=True)

        df_purchases_filtered = df_purchases[(df_purchases['Part_Ref'].isin(selected_part)) & (df_purchases.index > min_date) & (df_purchases.index <= max_date)]
        df_stock_filtered = df_stock[(df_stock['Part_Ref'].isin(selected_part)) & (df_stock.index >= min_date) & (df_stock.index <= max_date)]
        df_reg_filtered = df_reg[(df_reg['Part_Ref'].isin(selected_part)) & (df_reg.index > min_date) & (df_reg.index <= max_date)]

        # print(df_purchases_filtered)
        # print(df_reg_filtered)

        # if selected_part == ['BM07.12.9.952.104']:
            # print('a.1 \n', df_al_filtered[df_al_filtered['Movement_Date'] == '2018-08-22'])
        df_al_filtered = auto_line_dataset_cleaning(df_sales_filtered, df_al_filtered, df_purchases_filtered, df_reg_al_clients)
        # if selected_part == ['BM07.12.9.952.104']:
            # print('a.2 \n', df_al_filtered[df_al_filtered['Movement_Date'] == '2018-08-22'])

        df_sales_filtered = df_sales_filtered.drop_duplicates(subset=['Movement_Date', 'Part_Ref']).sort_values(by='Movement_Date')
        df_sales_filtered.set_index('Movement_Date', inplace=True)

        df_al_grouped = df_al_filtered.groupby(['Movement_Date', 'Part_Ref'])
        df_al_filtered['Qty_Sold_sum_al'] = df_al_grouped['Unit'].transform('sum')
        df_al_filtered.drop('Unit', axis=1, inplace=True)

        # if selected_part == ['BM07.12.9.952.104']:
        #     print('b \n', df_al_filtered[df_al_filtered['Movement_Date'] == '2018-08-22'])
        #     print('b \n', df_sales_filtered[df_sales_filtered.index == '2018-08-22'])

        df_reg_grouped = df_reg_filtered.groupby(df_reg_filtered.index)
        df_reg_filtered['Qty_Regulated_sum'] = df_reg_grouped['Qty_Regulated'].transform('sum')
        # df_reg_filtered = df_reg_filtered.drop_duplicates(subset=df_reg_filtered)
        df_reg_filtered = df_reg_filtered.loc[~df_reg_filtered.index.duplicated(keep='first')]

        df_al_filtered = df_al_filtered.drop_duplicates(subset=['Movement_Date'])
        df_al_filtered.set_index('Movement_Date', inplace=True)

        # if selected_part == ['BM07.12.9.952.104']:
        #     print('c \n', df_al_filtered[df_al_filtered.index == '2018-08-22'])

        # df_purchases_grouped = df_purchases_filtered.groupby(df_purchases_filtered.index)
        # df_purchases_filtered['Qty_Purchased_sum_2'] = df_purchases_grouped['Qty_Purchased_sum'].transform('sum')
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

        # for value in [df_stock_filtered, df_purchases_filtered['Qty_Purchased_sum'], df_reg_filtered['Qty_Regulated'], df_sales_filtered['Qty_Sold_sum_mov'], df_al_filtered['Qty_Sold_sum_al']]:
        #     print(value)

        # if selected_part == ['BM07.12.9.952.104']:
        #     print('d \n', df_al_filtered[df_al_filtered.index == '2018-08-22']['Qty_Sold_sum_al'])
        # result = pd.concat([df_stock_filtered.head(1), df_purchases_filtered['Qty_Purchased_sum_2'], df_reg_filtered['Qty_Regulated'], df_sales_filtered['Qty_Sold_sum_mov'], df_al_filtered['Qty_Sold_sum_al']], axis=1, sort=False)
        result = pd.concat([df_stock_filtered.head(1), df_purchases_filtered['Qty_Purchased_sum'], df_reg_filtered['Qty_Regulated_sum'], df_sales_filtered['Qty_Sold_sum_mov'], df_al_filtered['Qty_Sold_sum_al']], axis=1, sort=False)

        # if selected_part == ['BM07.12.9.952.104']:
        #     print('e \n', result[result.index == '2018-08-22'])

        print('\nStock at Start: {} \nSum Purchases: {} \nSum Sales SQL: {} \nSum Sales AutoLine: {} \nSum Regularizations: {} \nStock at End: {} \nStock Variance: {}'.format(stock_start, qty_purchased, qty_sold_mov, qty_sold_al, reg_value, stock_end, delta_stock))
        result_mov = stock_start + qty_purchased - qty_sold_mov - reg_value
        result_al = stock_start + qty_purchased - qty_sold_al - reg_value

        if result_mov != stock_end:
            print('\nValues dont match for SQL values - Stock has an offset of {:.2f}'.format(stock_end - result_mov))
        else:
            print('\nValues for SQL are correct :D')

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
        result['Regulated Evolution'] = result['Qty_Regulated_sum'].cumsum()
        result['Stock_Qty_al'] = result['Stock_Qty'] - result['Sales Evolution_al'] + result['Purchases Evolution'] - result['Regulated Evolution']
        result['Stock_Qty_mov'] = result['Stock_Qty'] - result['Sales Evolution_mov'] + result['Purchases Evolution'] - result['Regulated Evolution']

        # if selected_part == ['BM07.12.9.952.104']:
        #     print('f \n', result[result.index == '2018-08-22'])

        # print(result)
        result.to_csv('output/{}_stock_evolution.csv'.format(selected_part[0]))

        # result['teste'] = result['Qty_Sold_sum_mov'] - result['Qty_Sold_sum_al']
        # print(result[result['teste'] != 0])


def auto_line_dataset_cleaning(df_sales, df_al, df_purchases, df_reg_al_clients):
    start = time.time()
    print('AutoLine and PSE_Sales Lines comparison started')

    purchases_unique_plr = df_purchases['PLR_Document'].unique().tolist()
    reg_unique_slr = df_reg_al_clients['SLR_Account'].unique().tolist() + ['@Check']

    duplicated_rows = df_al[df_al.duplicated(subset='Encomenda', keep=False)]
    # print('duplicated_rows: \n{}'.format(duplicated_rows))
    if duplicated_rows.shape[0]:
        duplicated_rows_grouped = duplicated_rows.groupby(['Movement_Date', 'Part_Ref', 'WIP_Number'])

        # print('Duplicated groups:')
        # for key, group in duplicated_rows_grouped:
        #     print(key, '\n', group)

        df_al = df_al.drop(duplicated_rows.index, axis=0)

        pool = Pool(processes=int(level_0_performance_report.pool_workers_count))
        results = pool.map(sales_cleaning, [(key, group, df_sales) for (key, group) in duplicated_rows_grouped])
        pool.close()
        df_al_merged = pd.concat([df_al, pd.concat([result for result in results if result is not None])], axis=0)

        # print('df_al_merged: \n', df_al_merged)
        df_al_cleaned = purchases_reg_cleaning(df_al_merged, purchases_unique_plr, reg_unique_slr)

        print('AutoLine and PSE_Sales Lines comparison ended. Elapsed time: {:.2f}'.format(time.time() - start))
    else:
        df_al_cleaned = purchases_reg_cleaning(df_al, purchases_unique_plr, reg_unique_slr)

    return df_al_cleaned


def purchases_reg_cleaning(df_al, purchases_unique_plr, reg_unique_slr):

    matched_rows_purchases = df_al[df_al['SLR_Document_Number'].isin(purchases_unique_plr)]
    # print('matched_rows_purchase', matched_rows_purchases)
    # print('purchase plrs:', purchases_unique_plr)
    if matched_rows_purchases.shape[0]:
        df_al = df_al.drop(matched_rows_purchases.index, axis=0)

    matched_rows_reg = df_al[df_al['SLR_Document_Account'].isin(reg_unique_slr)]
    # print('matched_rows_reg', matched_rows_reg)
    # print('reg slrs:', reg_unique_slr)
    if matched_rows_reg.shape[0]:
        # print('matched reg lines: \n{}'.format(matched_rows_reg[matched_rows_reg['SLR_Document_Number'] == 0]))
        df_al = df_al.drop(matched_rows_reg.index, axis=0)
        # print('returned results: \n{}'.format(df_al[df_al['Movement_Date'] == '2018-09-20']))

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
            print('NO MATCHED ROWS?!?')
            print(group, '\n', matching_sales)
            group = group.tail(1)  # ToDo Needs to be confirmed

        # ToDo Martelanço:
        if group['Part_Ref'].unique() == 'BM83.19.2.158.851' and key[2] == 38381:
            group = group[group['SLR_Document_Number'] != 44446226]
        if group['Part_Ref'].unique() == 'BM83.21.2.405.675' and key[2] == 63960:
            group = group[group['SLR_Document_Number'] != 44462775]

    return group


if __name__ == '__main__':
    main()
