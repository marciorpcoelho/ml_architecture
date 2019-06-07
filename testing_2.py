import sys
import pandas as pd
pd.set_option('display.expand_frame_repr', False)


def main():

    selected_parts = ['BM83.21.2.405.675', 'BM07.12.9.952.104', 'BM07.14.9.213.164', 'BM83.19.2.158.851', 'BM64.11.9.237.555']
    # selected_parts = ['BM07.12.9.952.104']

    datetime_index = pd.date_range(start='2018-06-30', end='2019-05-31')
    results = pd.DataFrame()

    for part_ref in selected_parts:
        result = pd.read_csv('output/{}_stock_evolution.csv'.format(part_ref), parse_dates=['Unnamed: 0'], dtype={'Part_Ref': str})

        result = result.set_index('Unnamed: 0').reindex(datetime_index).reset_index().rename(columns={'Unnamed: 0': 'Movement_Date'})
        ffill_cols = ['Part_Ref', 'Stock_Qty_al', 'Stock_Qty_mov', 'Sales Evolution_al', 'Sales Evolution_mov', 'Purchases Evolution', 'Regulated Evolution']
        zero_fill_cols = ['Qty_Purchased_sum', 'Qty_Regulated_sum', 'Qty_Sold_sum_mov', 'Qty_Sold_sum_al']

        [result[x].fillna(0, inplace=True) for x in zero_fill_cols]
        [result[x].fillna(method='ffill', inplace=True) for x in ffill_cols]
        result['Weekday'] = result['index'].dt.dayofweek
        # result = result[result['Weekday'] < 5]

        print(result.shape)

        results = results.append(result)

    # print(results)
    results.to_csv('output/results_merge.csv')


if __name__ == '__main__':
    main()
