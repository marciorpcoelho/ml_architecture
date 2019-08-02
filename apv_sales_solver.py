import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds
from dateutil.relativedelta import relativedelta
pd.set_option('display.expand_frame_repr', False)
my_dpi = 96

cost_evolution = []
sale_evolution = []
profit_evolution = []
dtss_evolution = []
scale_argument = 1000


def main():
    # df_sales = pd.read_csv('output/results_merge_case_study_0B.csv', parse_dates=['index'], index_col=0)
    # df_sales = pd.read_csv('output/results_merge_0B.csv', parse_dates=['index'], index_col=0)
    # current_date = '2019-06-28'
    # df_solve = solver_dataset_preparation(df_sales, current_date)

    df_solve = pd.read_csv('output/df_solve_0B.csv', index_col=0)

    unique_parts = df_solve['Part_Ref'].unique()
    df_solve = df_solve[df_solve['Part_Ref'].isin(unique_parts)]

    start = time.time()
    n_size = df_solve['Part_Ref'].nunique()

    costs = df_solve['Cost'].values.tolist()
    pvps = df_solve['PVP'].values.tolist()
    margins = df_solve['Margin'].values.tolist()
    dtss = df_solve['DaysToSell_1_Part']

    cost_goal = 252051
    sales_goal = 300000
    dtss_goal = 15  # Weekdays only!

    constraint_1 = [
        {'type': 'ineq', 'fun': lambda n: cost_calculation(n, costs, cost_goal)},
        {'type': 'ineq', 'fun': lambda n: sales_calculation(n, pvps, sales_goal)},
        {'type': 'ineq', 'fun': lambda n: days_to_sell_calculation(n, dtss, dtss_goal)},
    ]

    bnds = [(0, 10000) for i in range(n_size)]
    n_init = np.array([1] * n_size)

    # res = minimize(profit_function, x0=n_init, args=(costs, pvps), method='trust-constr', hess=zero_hess, bounds=bnds, constraints=constraint_1, options={'xtol': 1e-8, 'gtol': 1e-5, 'maxiter': 100000})
    res = minimize(profit_function, x0=n_init, args=(costs, pvps), method='SLSQP', bounds=bnds, constraints=constraint_1, options={'maxiter': 100000})

    # Dummy Example
    # A = np.array([[1, -2], [-1, -2], [-1, 2]])
    # b = np.array([-2, -6, -2])
    # bnds = [(0, None) for i in range(A.shape[1])]  # x_1 >= 0, x_2 >= 0
    # xinit = np.array([0, 0])
    # cons = [{'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2},
    #         {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
    #         {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2}]
    #
    # res = minimize(test_function, x0=xinit, bounds=bnds, constraints=cons)

    # print(res)
    print('Termination Message: {}'.format(res.message))
    print('Number of iterations: {}'.format(res.nit))
    [print(part + ': ' + str(qty)) for part, qty in zip(unique_parts, res.x)]

    costs = [n*cost for n, cost in zip(res.x, costs)]
    sales = [n*pvp for n, pvp in zip(res.x, pvps)]
    dtss = [n*dts for n, dts in zip(res.x, dtss)]
    # profits = [n*margin for n, margin in zip(res.x, margins)]
    # print(costs, '\n', profits)
    print('Total cost of: {:.2f} / {:.2f} ({:.2f}%) \nTotal Sales of: {:.2f} / {:.2f} ({:.2f}%) \nDays to Sell: {:.2f} / {:.2f} ({:.2f}%) \nProfit of {:.2f}'
          .format(np.sum(costs), cost_goal, (np.sum(costs) / cost_goal) * 100, np.sum(sales), sales_goal, (np.sum(sales) / sales_goal) * 100, np.max(dtss), dtss_goal, (np.max(dtss) / dtss_goal) * 100, -res.fun * scale_argument))

    print('Elapsed time: {:.2f} seconds.'.format(time.time() - start))

    f, ax = plt.subplots(2, 2, figsize=(1400 / my_dpi, 1000 / my_dpi), dpi=my_dpi)

    ax[0, 0].plot(range(len(cost_evolution)), [x + cost_goal for x in cost_evolution], label='Cost Evolution', c='blue')
    ax[0, 0].axhline(y=cost_goal, label='Cost Goal')
    ax[0, 0].grid()
    ax[0, 0].legend()
    ax[0, 0].set_title('Cost Value Evolution')

    ax[0, 1].plot(range(len(sale_evolution)), [x + sales_goal for x in sale_evolution], label='Sale Evolution', c='red')
    ax[0, 1].axhline(y=sales_goal, label='Sales Goal')
    ax[0, 1].grid()
    ax[0, 1].legend()
    ax[0, 1].set_title('Sale Value Evolution')

    ax[1, 0].plot(range(len(dtss_evolution)), [x - dtss_goal for x in dtss_evolution], label='DTS Evolution', c='cyan')
    ax[1, 0].axhline(y=dtss_goal, label='DTS Goal')
    ax[1, 0].grid()
    ax[1, 0].legend()
    ax[1, 0].set_title('Sale Value Evolution')

    ax[1, 1].plot(range(len(profit_evolution)), [-x for x in profit_evolution], label='Profit Evolution', c='green')
    ax[1, 1].grid()
    ax[1, 1].legend()
    ax[1, 1].set_title('Profit Value Evolution')

    plt.tight_layout()
    plt.title('N={}'.format(n_size))
    plt.show()


def cost_calculation(ns, costs, cost_goal):

    total_cost = np.sum([n * cost for n, cost in zip(ns, costs)]) - cost_goal

    cost_evolution.append(total_cost)
    # print('cost', total_cost)
    return total_cost


def sales_calculation(ns, pvps, sales_goal):

    total_sales = np.sum([n * pvp for n, pvp in zip(ns, pvps)]) - sales_goal

    sale_evolution.append(total_sales)
    # print('sales', total_sales)
    return total_sales


def days_to_sell_calculation(ns, dtss, dtss_goal):

    days_to_sell = [n * dts for n, dts in zip(ns, dtss)]
    max_days_to_sell = dtss_goal - np.max(days_to_sell)

    dtss_evolution.append(max_days_to_sell)
    return max_days_to_sell


def zero_hess(*args):

    list_of_zeros = [0] * len(args[0])
    return np.array(list_of_zeros)


def profit_function(n, costs, pvps):

    pvp_sum = np.sum([n_single * pvp for n_single, pvp in zip(n, pvps)])
    cost_sum = np.sum([n_single * cost for n_single, cost in zip(n, costs)])
    profit = -(pvp_sum - cost_sum)
    profit_evolution.append(profit)

    return profit / scale_argument  # I divide by 1000 in order to scale down on values in order to help with converge (tries to deal with Termination Message: Positive directional derivative for linesearch)


def test_function(x):
    return (x[0] - 1)**2 + (x[1] - 2.5)**2


def solver_dataset_preparation(df_sales, current_date):
    df_solve = pd.DataFrame(columns=['Part_Ref', 'Cost', 'PVP', 'Margin', 'DII Year', 'DII Year weekdays', 'DaysToSell_1_Part'])
    unique_part_refs = df_sales['Part_Ref'].unique()

    df_sales['weekday'] = df_sales['index'].dt.dayofweek

    last_year_date = pd.to_datetime(current_date) - relativedelta(years=1)

    # print(df_sales[(df_sales['weekday'] == 5) & (df_sales['Qty_Sold_sum_al'] > 0)])
    # print(df_sales[(df_sales['weekday'] == 5) & (df_sales['Qty_Sold_sum_al'] > 0)].shape)  # 13 rows of sales at saturdays
    # print(df_sales[(df_sales['weekday'] == 6) & (df_sales['Qty_Sold_sum_al'] > 0)])
    # print(df_sales[(df_sales['weekday'] == 6) & (df_sales['Qty_Sold_sum_al'] > 0)].shape)  # 0 rows of sales at sundays

    df_sales = df_sales[(df_sales['weekday'] >= 0) & (df_sales['weekday'] < 5)]  # Removal of weekend data
    df_sales.set_index('Part_Ref', inplace=True)
    i, parts_count, positions = 1, len(unique_part_refs), []
    k = 0

    for part_ref in unique_part_refs:
        # print('Part_Ref: {}'.format(part_ref))

        # start = time.time()
        df_sales_filtered = df_sales.loc[part_ref, :]
        if df_sales_filtered['Qty_Sold_sum_al'].sum() != 0 and df_sales_filtered[df_sales_filtered['Qty_Sold_sum_al'] > 0].shape[0] > 1:  # ToDo: This checkup will be removed tomorrow (02/08/19) as it will be enforced in a previous step

            df_sales_filtered.set_index('index', inplace=True)
            # I should use last purchase date/cost, but first: not all parts have purchases (e.g. BM83.13.9.415.965) and second, they should give the same value.

            # last_sale_date = df_sales_filtered[(df_sales_filtered['Qty_Sold_sum_al'] > 0)]['index'].max()
            last_sale_date = df_sales_filtered[(df_sales_filtered['Qty_Sold_sum_al'] > 0)].index.max()

            # last_cost = df_sales_filtered[(df_sales_filtered['index'] == last_sale_date)]['Cost_Sale_avg'].values[0]
            # last_pvp = df_sales_filtered[(df_sales_filtered['index'] == last_sale_date)]['PVP_avg'].values[0]
            last_cost = df_sales_filtered.loc[last_sale_date, 'Cost_Sale_avg']
            last_pvp = df_sales_filtered.loc[last_sale_date, 'PVP_avg']

            last_stock = df_sales_filtered['Stock_Qty_al'].tail(1).values[0]
            if last_stock:
                last_stock_value = last_stock * last_cost

                # last_year_sales = df_sales_filtered[df_sales_filtered['index'] >= last_year_date]['Qty_Sold_sum_al'].sum()
                # last_year_sales_avg = df_sales_filtered[df_sales_filtered['index'] >= last_year_date]['Qty_Sold_sum_al'].mean()
                last_year_sales = df_sales_filtered.loc[last_year_date::, 'Qty_Sold_sum_al'].sum()
                last_year_sales_avg = df_sales_filtered.loc[last_year_date::, 'Qty_Sold_sum_al'].mean()

                if last_year_sales:
                    last_year_cogs = last_year_sales * last_cost
                    margin = (last_pvp - last_cost)

                    dii = last_stock_value / last_year_cogs
                    dii_year = dii * 365

                    avg_sales_per_day = last_stock / last_year_sales_avg  # The diff between dii_year and avg_sales_per_day is caused by the number of non-weekdays;

                    days_to_sell_1_part = 1 / last_year_sales_avg

                    if margin <= 0:
                        pass

                    # print('Part_Ref: {} \n Cost: {:.3f} \n Margin: {:.3f} \n Last Stock: {:.3f} \n Last Stock Value: {:.3f} \n Last Year Sales: {:.3f}'
                    #       .format(part_ref, last_cost, margin, last_stock, last_stock_value, last_year_sales))
                    df_solve.loc[k, ['Part_Ref', 'Cost', 'PVP', 'Margin', 'DII Year', 'DII Year weekdays', 'DaysToSell_1_Part']] = \
                                                                            [part_ref, last_cost, last_pvp, margin, dii_year, avg_sales_per_day, days_to_sell_1_part]
                    k += 1

                    # print('Part_Ref: {}, Cost: {:.2f}, PVP: {:.2f}, Margin: {:.2f}, Last Stock Value: {}, Last Year COGS: {}, DII Year: {:.2f}, Last Stock: {}, Last Year Sales Avg: {}, DII Year weekdays: {:.2f}, DaysToSell_1_Part: {:.2f},'
                    #       .format(part_ref, last_cost, last_pvp, margin, last_stock_value, last_year_cogs, dii_year, last_stock, last_year_sales_avg, avg_sales_per_day, days_to_sell_1_part))

        position = int((i / parts_count) * 100)
        if not position % 1:
            if position not in positions:
                print('{}% completed'.format(position))
                positions.append(position)

        i += 1
        # except IndexError:
        #     print(part_ref)
        # print('{} - Elapsed Time: {:.2f} seconds.'.format(part_ref, time.time() - start))
    # print(df_solve)
    return df_solve


if __name__ == '__main__':
    main()
