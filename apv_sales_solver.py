import sys
import time
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import level_0_performance_report
from level_1_d_model_evaluation import save_fig
from level_1_b_data_processing import df_join_function
from level_2_order_optimization_apv_baviera_options import group_goals, group_goals_type
from scipy.optimize import minimize, LinearConstraint, Bounds
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.expand_frame_repr', False)
my_dpi = 96

value_evolution = []
number_of_unique_parts_evolution = []
number_of_parts_evolution = []
cost_evolution = []
sale_evolution = []
profit_evolution = []
dtss_evolution = []
scale_argument = 10000


def main():
    current_date = '20190831'
    df_part_refs_ta = pd.read_csv('output/part_ref_ta_{}.csv'.format(current_date), index_col=0)

    current_date = '20190816'
    try:
        # df_solve = pd.read_csv('output/df_solve_0B_filtered.csv', index_col=0)
        df_solve = pd.read_csv('output/df_solve_0B_backup.csv', index_col=0)
        print('df_solve found...')
    except FileNotFoundError:
        print('df_solve file not found, processing a new one...')
        # df_sales = pd.read_csv('dbs/results_merge_case_study_0B.csv', parse_dates=['index'], index_col=0)
        df_sales = pd.read_csv('output/results_merge_0B_{}.csv'.format(current_date), parse_dates=['index'], index_col=0)
        df_solve = solver_dataset_preparation(df_sales, df_part_refs_ta, group_goals['dtss_goal'], current_date)

    dtss_goal = group_goals['dtss_goal']
    number_of_parts_goal = group_goals['number_of_unique_parts']

    # df_solve = df_solve[df_solve['DaysToSell_1_Part'] <= dtss_goal]  # 6110 to 650
    df_solve = df_solve[df_solve['DaysToSell_1_Part'] > 0]
    df_solve = df_solve[df_solve['Group'] != 'NO_TA']
    for key, group in df_solve.groupby('Group'):
        # if key in ['Outros', 'MINI_Bonus_Group_1', 'MINI_Bonus_Group_2', 'MINI_Bonus_Group_3', 'MINI_Bonus_Group_4', 'BMW_Bonus_Group_1', 'BMW_Bonus_Group_3', 'BMW_Bonus_Group_4']:
        if key in ['Outros', 'MINI_Bonus_Group_1', 'MINI_Bonus_Group_2', 'MINI_Bonus_Group_3', 'MINI_Bonus_Group_4']:
            continue
        else:
            goal_value = group_goals[key][0]
            goal_value_limit = group_goals[key + '_limit'][0]
            goal_type = group_goals_type[key]
            total_number_of_parts_goal = group_goals['number_of_total_parts']
            # cost_goal = group_goals[key][0]
            # sales_goal = group_goals[key][1]
            print('There are {} unique part_refs for {}'.format(len(group), key))
            # print('Cost goal: {}, Sale Goal: {}, DTSS Goal: {}'.format(cost_goal, sales_goal, dtss_goal))
            print('{} goal: {:.1f}, {} goal limit: {:.1f}, DTSS Goal: {}'.format(goal_type, goal_value, goal_type, goal_value_limit, dtss_goal))
            # solver_lp(df_solve[df_solve['Part_Ref'].isin(group['Part_Ref'].unique())], key, goal_value, goal_type, dtss_goal, number_of_parts_goal, total_number_of_parts_goal)
            # solver_ip_example()
            solver_ip(df_solve[df_solve['Part_Ref'].isin(group['Part_Ref'].unique())], key, goal_value, goal_value_limit, goal_type, dtss_goal, number_of_parts_goal, total_number_of_parts_goal)


def solver_ip_example():
    P = 165
    weights = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82])
    utilities = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72])

    # The variable we are solving for
    selection = cp.Variable(len(weights), boolean=True)
    print('a', selection)

    # The sum of the weights should be less than or equal to P
    weight_constraint = weights * selection <= P
    print('a', weight_constraint)

    # Our total utility is the sum of the item utilities
    total_utility = utilities * selection

    # We tell cvxpy that we want to maximize total utility
    # subject to weight_constraint. All constraints in
    # cvxpy must be passed as a list
    knapsack_problem = cp.Problem(cp.Maximize(total_utility), [weight_constraint])

    # Solving the problem
    result = knapsack_problem.solve(solver=cp.GLPK_MI)
    print('Result of optimization:', result)

    print('Result:', selection.value)


def solver_ip(df_solve, group, goal_value, goal_value_limit, goal_type, dtss_goal, number_of_unique_parts_goal, total_number_of_parts_goal):
    print('Solving for {}...'.format(group))

    df_solve = df_solve[df_solve['DaysToSell_1_Part'] <= dtss_goal]

    unique_parts = df_solve['Part_Ref'].unique()
    df_solve = df_solve[df_solve['Part_Ref'].isin(unique_parts)]
    start = time.time()

    n_size = df_solve['Part_Ref'].nunique()  # Number of different parts
    print('Number of parts inside initial conditions:', n_size)
    # dtss_goals = np.array([dtss_goal] * n_size)

    values = np.array(df_solve[goal_type].values.tolist())  # Costs/Sale prices for each reference, info#1
    dtss = np.array(df_solve['DaysToSell_1_Part'].values.tolist())  # Days to Sell of each reference, info#2

    selection = cp.Variable(n_size, integer=True)

    dtss_constraint = cp.max(cp.diag(cp.multiply(selection.T, dtss)))

    total_value = selection * values

    problem_testing_2 = cp.Problem(cp.Maximize(total_value), [dtss_constraint <= dtss_goal, selection >= 0, selection <= 100])

    result = problem_testing_2.solve(solver=cp.GLPK_MI, verbose=False)

    print('Status of optimization:', problem_testing_2.status)
    print('Result of optimization:', result)
    print('Part Order:\n', selection.value)
    if selection.value is not None:
        print('Solution Found :)')
        if result >= goal_value:
            print('Solution found and above set goal :D')
        else:
            print('Solution does not reach the goal :(')

        # [print(part + ': ' + str(qty) + ' and cost/sell value of {}, DTSS: {:.2f}'.format(value, dts)) for part, value, qty, dts in zip(unique_parts, values, selection.value, dtss)]
        # print(np.matmul(selection.value, values))

        print('Total {} of: {:.2f} / {:.2f} ({:.2f}%) \nDays to Sell: {:.2f} / {:.2f} ({:.2f}%)'
              .format(goal_type, result, goal_value, (result / goal_value) * 100, np.max([dts * qty for dts, qty in zip(dtss, selection.value)]), dtss_goal, (np.max([dts * qty for dts, qty in zip(dtss, selection.value)]) / dtss_goal) * 100))
    else:
        print('No Solution found :(')

    print('Number of Inequations: {}'.format(problem_testing_2.size_metrics.num_scalar_leq_constr))
    print(problem_testing_2.size_metrics.max_data_dimension)

    print('Elapsed time: {:.2f} seconds.\n'.format(time.time() - start))


def solver_lp(df_solve, group, goal_value, goal_type, dtss_goal, number_of_unique_parts_goal, total_number_of_parts_goal):
    print('Solving for {}...'.format(group))
    unique_parts = df_solve['Part_Ref'].unique()
    df_solve = df_solve[df_solve['Part_Ref'].isin(unique_parts)]

    start = time.time()
    n_size = df_solve['Part_Ref'].nunique()

    # costs = df_solve['Cost'].values.tolist()
    # pvps = df_solve['PVP'].values.tolist()
    values = df_solve[goal_type].values.tolist()
    margins = df_solve['Margin'].values.tolist()
    dtss = df_solve['DaysToSell_1_Part']

    # [print(part + ' - ' + str(dts)) for part, dts in zip(unique_parts, dtss)]

    # constraint_1 = [
    #     {'type': 'ineq', 'fun': lambda n: cost_calculation(n, costs, cost_goal)},
    #     {'type': 'ineq', 'fun': lambda n: sales_calculation(n, pvps, sales_goal)},
    #     {'type': 'ineq', 'fun': lambda n: days_to_sell_calculation(n, dtss, dtss_goal)},
    # ]

    constraint_1 = [
        {'type': 'ineq', 'fun': lambda n: value_calculation(n, values, goal_value)},
        # {'type': 'ineq', 'fun': lambda n: number_of_unique_parts_calculation(n, number_of_unique_parts_goal)},
        {'type': 'ineq', 'fun': lambda n: total_number_parts_calculation(n, total_number_of_parts_goal)},
        {'type': 'ineq', 'fun': lambda n: days_to_sell_calculation(n, dtss, dtss_goal)},
    ]

    constraint_2 = [
        {'type': 'ineq', 'fun': lambda n: value_calculation(n, values, goal_value)},
        {'type': 'ineq', 'fun': lambda n: days_to_sell_calculation(n, dtss, dtss_goal)},
    ]

    bnds = [(0, 100) for i in range(n_size)]
    n_init = np.array([2] * n_size)

    res = minimize(profit_function, x0=n_init, args=values, method='trust-constr', hess=zero_hess, bounds=bnds, constraints=constraint_2, options={'xtol': 1e-8, 'gtol': 1e-5, 'maxiter': 100000})
    # res = minimize(profit_function, x0=n_init, args=values, method='SLSQP', bounds=bnds, constraints=constraint_2, options={'maxiter': 100000})
    # res = minimize(profit_function, x0=n_init, args=(costs, pvps), method='Nelder-Mead', bounds=bnds, constraints=constraint_1, options={'maxiter': 100000})  # RuntimeWarning: Method Nelder-Mead cannot handle constraints nor bounds.
    # res = minimize(profit_function, x0=n_init, args=(costs, pvps), method='Powell', bounds=bnds, constraints=constraint_1, options={'maxiter': 100000})  # Method Powell cannot handle constraints nor bounds.
    # res = minimize(profit_function, x0=n_init, args=(costs, pvps), method='CG', bounds=bnds, constraints=constraint_1, options={'maxiter': 100000})  # RuntimeWarning: Method CG cannot handle constraints nor bounds.
    # res = minimize(profit_function, x0=n_init, args=(costs, pvps), method='BFGS', bounds=bnds, constraints=constraint_1, options={'maxiter': 100000})  # RuntimeWarning: Method CG cannot handle constraints nor bounds.
    # res = minimize(profit_function, x0=n_init, args=(costs, pvps), method='Newton-CG', jac=None, bounds=bnds, constraints=constraint_1, options={'maxiter': 100000})  # RuntimeWarning: Method Newton-CG cannot handle constraints nor bounds., ValueError: Jacobian is required for Newton-CG method
    # res = minimize(profit_function, x0=n_init, args=(costs, pvps), method='L-BFGS-B', jac=None, bounds=bnds, constraints=constraint_1, options={'maxiter': 100000})  # RuntimeWarning: Method L-BFGS-B cannot handle constraints.
    # res = minimize(profit_function, x0=n_init, args=(costs, pvps), method='TNC', jac=None, bounds=bnds, constraints=constraint_1, options={'maxiter': 100000})  # RuntimeWarning: Method TNC cannot handle constraints.
    # res = minimize(profit_function, x0=n_init, args=(costs, pvps), method='COBYLA', bounds=bnds, constraints=constraint_1, options={'maxiter': 100000})  # RuntimeWarning: Method COBYLA cannot handle bounds.
    # res = minimize(profit_function, x0=n_init, args=(costs, pvps), method='dogleg', jac=None, bounds=bnds, constraints=constraint_1, options={'maxiter': 100000})  # RuntimeWarning: Method dogleg cannot handle constraints nor bounds., ValueError: Jacobian is required for dogleg minimization
    # res = minimize(profit_function, x0=n_init, args=(costs, pvps), method='trust-ncg', jac=None, bounds=bnds, constraints=constraint_1, options={'maxiter': 100000})  # RuntimeWarning: Method trust-ncg cannot handle constraints nor bounds., ValueError: Jacobian is required for Newton-CG trust-region minimization
    # res = minimize(profit_function, x0=n_init, args=(costs, pvps), method='trust-exact', jac=zero_hess, hess=zero_hess, bounds=bnds, constraints=constraint_1, options={'maxiter': 100000})  # numpy.AxisError: axis 1 is out of bounds for array of dimension 1
    # res = minimize(profit_function, x0=n_init, args=(costs, pvps), method='trust-krylov', jac=zero_hess, hess=zero_hess, bounds=bnds, constraints=constraint_1, options={'maxiter': 100000, 'inexact': True, 'tol': 1e-4, 'xtol': 1e-8, 'gtol': 1e-5})

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

    print('Termination Success? {}'.format(res.success))
    print('Termination Message: {}'.format(res.message))
    print('Number of iterations: {}'.format(res.nit))

    # costs = [n * cost for n, cost in zip(res.x, costs)]
    # sales = [n * pvp for n, pvp in zip(res.x, pvps)]
    values = [n * value for n, value in zip(res.x, values)]
    dtss = [n * dts for n, dts in zip(res.x, dtss)]

    [print(part + ': ' + str(qty) + ', DTSS: {:.2f}'.format(dts)) for part, qty, dts in zip(unique_parts, res.x, dtss) if qty >= 1]

    # profits = [n*margin for n, margin in zip(res.x, margins)]
    # print(costs, '\n', profits)
    # print('Total cost of: {:.2f} / {:.2f} ({:.2f}%) \nTotal Sales of: {:.2f} / {:.2f} ({:.2f}%) \nDays to Sell: {:.2f} / {:.2f} ({:.2f}%) \nProfit of {:.2f}'
    #       .format(np.sum(costs), cost_goal, (np.sum(costs) / cost_goal) * 100, np.sum(sales), sales_goal, (np.sum(sales) / sales_goal) * 100, np.max(dtss), dtss_goal, (np.max(dtss) / dtss_goal) * 100, -res.fun * scale_argument))

    print('Total {} of: {:.2f} / {:.2f} ({:.2f}%) \nDays to Sell: {:.2f} / {:.2f} ({:.2f}%)'
          .format(goal_type, np.sum(values), goal_value, (np.sum(values) / goal_value) * 100, np.max(dtss), dtss_goal, (np.max(dtss) / dtss_goal) * 100))
    print('Total Number of Parts: {:.2f} for number of unique parts: {}'.format(np.sum(res.x), len(np.unique(res.x))))

    print('Elapsed time: {:.2f} seconds.\n'.format(time.time() - start))

    # evolution_plots(n_size, group, goal_value, goal_type, dtss_goal, total_number_of_parts_goal, number_of_unique_parts_goal)


def evolution_plots(n_size, group, goal_value, goal_type, dtss_goal, total_number_of_parts_goal, number_of_unique_parts_goal):
    f, ax = plt.subplots(2, 3, figsize=(1400 / my_dpi, 1000 / my_dpi), dpi=my_dpi)

    ax[0, 0].plot(range(len(value_evolution)), [x + goal_value for x in value_evolution], label='Cost Evolution', c='blue')
    ax[0, 0].axhline(y=goal_value, label='{} Goal'.format(goal_type))
    ax[0, 0].grid()
    ax[0, 0].legend()
    ax[0, 0].set_title('{} Value Evolution'.format(goal_type))

    ax[0, 1].plot(range(len(number_of_unique_parts_evolution)), [x + number_of_unique_parts_goal for x in number_of_unique_parts_evolution], label='Unique Parts Evolution', c='red')
    ax[0, 1].axhline(y=number_of_unique_parts_goal, label='Unique Parts Goal')
    ax[0, 1].grid()
    ax[0, 1].legend()
    ax[0, 1].set_title('Unique Parts Evolution')

    ax[1, 0].plot(range(len(number_of_parts_evolution)), [x + number_of_unique_parts_goal for x in number_of_parts_evolution], label='Total Parts Evolution', c='red')
    ax[1, 0].axhline(y=total_number_of_parts_goal, label='Total Parts Goal')
    ax[1, 0].grid()
    ax[1, 0].legend()
    ax[1, 0].set_title('Total Parts Evolution')

    ax[1, 1].plot(range(len(dtss_evolution)), [x - dtss_goal for x in dtss_evolution], label='DTS Evolution', c='cyan')
    ax[1, 1].axhline(y=dtss_goal, label='DTS Goal')
    ax[1, 1].grid()
    ax[1, 1].legend()
    ax[1, 1].set_title('DTS Value Evolution')

    ax[0, 2].plot(range(len(profit_evolution)), [-x for x in profit_evolution], label='Profit Evolution', c='green')
    ax[0, 2].grid()
    ax[0, 2].legend()
    ax[0, 2].set_title('Profit Value Evolution')

    plt.tight_layout()
    plt.title('N={}'.format(n_size))
    save_fig('apv_solver_metrics_evolution_{}'.format(group))
    # plt.show()


# This is a generic cost/pvp calculation;
def value_calculation(ns, values, goal_value):
    # total_values = np.sum([n * value for n, value in zip(ns, values)]) - goal_value
    # print(total_values)

    total_values = np.matmul(ns, values)
    # current_value = total_values - goal_value
    current_value = goal_value - total_values

    value_evolution.append(-current_value)
    # print('sales', total_sales)
    return -current_value


def number_of_unique_parts_calculation(ns, number_of_parts_goal):
    current_number_of_unique_parts = len(ns) - number_of_parts_goal
    # print(current_number_of_parts)

    number_of_unique_parts_evolution.append(current_number_of_unique_parts)
    return current_number_of_unique_parts


def total_number_parts_calculation(ns, total_number_of_parts_goal):
    current_number_total_parts = total_number_of_parts_goal - sum(ns)

    number_of_parts_evolution.append(current_number_total_parts)
    return current_number_total_parts


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
    # days_to_sell = [n * dts for n, dts in zip(ns, dtss)]
    # max_days_to_sell = dtss_goal - np.max(days_to_sell)

    days_to_sell = np.matmul(ns, dtss)  # This is wrong. It returns a single value, which is the sum of all the multiplications, which is not the value we want, is it? 02/10/19
    current_days_to_sell = dtss_goal - days_to_sell  # Goal - DTS >= 0

    dtss_evolution.append(current_days_to_sell)
    return current_days_to_sell


def zero_hess(*args):
    # list_of_zeros = [0] * len(args[0])
    # return np.array(list_of_zeros)
    return np.zeros((len(args[0]), len(args[1])))


def profit_function(n, values):
    # pvp_sum = np.sum([n_single * pvp for n_single, pvp in zip(n, pvps)])
    # cost_sum = np.sum([n_single * cost for n_single, cost in zip(n, costs)])

    # value_sum = np.sum([n_single * value for n_single, value in zip(n, values)])
    value = np.matmul(n, values)

    # profit = -value
    profit_evolution.append(value)

    return value / scale_argument  # I divide by 1000 in order to scale down on values in order to help with converge (tries to deal with Termination Message: Positive directional derivative for linesearch)


def test_function(x):
    return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2


def solver_dataset_preparation(df_sales, df_part_refs_ta, dtss_goal, current_date):
    start = time.time()

    unique_part_refs = df_sales['Part_Ref'].unique()
    unique_part_refs_count = df_sales['Part_Ref'].nunique()

    df_sales['weekday'] = df_sales['index'].dt.dayofweek

    last_year_date = pd.to_datetime(current_date) - relativedelta(years=1)
    dts_interval_sales = df_sales.tail(dtss_goal)['index'].min()

    # print(df_sales[(df_sales['weekday'] == 5) & (df_sales['Qty_Sold_sum_al'] > 0)])
    # print(df_sales[(df_sales['weekday'] == 5) & (df_sales['Qty_Sold_sum_al'] > 0)].shape)  # 13 rows of sales at saturdays
    # print(df_sales[(df_sales['weekday'] == 6) & (df_sales['Qty_Sold_sum_al'] > 0)])
    # print(df_sales[(df_sales['weekday'] == 6) & (df_sales['Qty_Sold_sum_al'] > 0)].shape)  # 0 rows of sales at sundays

    df_sales = df_sales[(df_sales['weekday'] >= 0) & (df_sales['weekday'] < 5)]  # Removal of weekend data
    df_sales.set_index('Part_Ref', inplace=True)
    i, parts_count, positions = 1, len(unique_part_refs), []
    # k = 0

    df_sales_grouped = df_sales.groupby('Part_Ref')
    pool = Pool(processes=level_0_performance_report.pool_workers_count)
    results = pool.map(solver_metrics_per_part_ref, [(part_ref, group, last_year_date, dts_interval_sales) for (part_ref, group) in df_sales_grouped])
    pool.close()
    df_solve = pd.concat([result for result in results if result is not None])

    df_solve.to_csv('output/df_solve_0B.csv')

    after_solve_preparation = df_solve['Part_Ref'].nunique()
    # Filtered by DTS Goal
    # df_solve_filtered = df_solve[df_solve['DaysToSell_1_Part'] <= dtss_goal]  # 6110 to 650
    # after_dtss_filter = df_solve_filtered['Part_Ref'].nunique()
    # df_solve_filtered.reset_index(inplace=True)
    #
    # print('From the initial value of {} \nit was reduced to {} \nand then filtered down to {}'.format(unique_part_refs_count, after_solve_preparation, after_dtss_filter))
    # # # df_solve_filtered_2 = df_solve_filtered[df_solve_filtered['DTS_Max_Total_Qty_Sold'] > 0]  # 650 to 318  # I can't do this, but maybe i can add a flag to signal these part_refs
    #
    # df_solve_filtered = df_join_function(df_solve_filtered, df_part_refs_ta[['Part_Ref', 'Group']].set_index('Part_Ref'), on='Part_Ref')
    #
    # df_solve_filtered.to_csv('output/df_solve_0B_filtered.csv')

    # Non-filtered by DTS Goal
    df_solve_filtered = df_join_function(df_solve, df_part_refs_ta[['Part_Ref', 'Group']].set_index('Part_Ref'), on='Part_Ref')

    df_solve_filtered.to_csv('output/df_solve_0B.csv')

    print('Elapsed Time: {:.2f}'.format(time.time() - start))
    return df_solve_filtered


def solver_metrics_per_part_ref(args):
    part_ref, df_sales_filtered, last_year_date, dts_interval_sales = args
    df_solve = pd.DataFrame(columns=['Part_Ref', 'Cost', 'PVP', 'Margin', 'DII Year', 'DII Year weekdays', 'DaysToSell_1_Part', 'DTS_Total_Qty_Sold', 'DTS_Min_Total_Qty_Sold', 'DTS_Max_Total_Qty_Sold'])
    # final_df = pd.DataFrame()

    if df_sales_filtered['Qty_Sold_sum_al'].sum() >= 0 and df_sales_filtered[df_sales_filtered['Qty_Sold_sum_al'] > 0].shape[0] > 1:  # This checkup is needed as I can not enforce it before when processing a time interval, only when processing all data
        df_sales_filtered.set_index('index', inplace=True)
        # I should use last purchase date/cost, but first: not all parts have purchases (e.g. BM83.13.9.415.965) and second, they should give the same value.

        last_sale_date = df_sales_filtered[(df_sales_filtered['Qty_Sold_sum_al'] > 0)].index.max()

        if last_sale_date < last_year_date:
            return None

        df_last_sale = df_sales_filtered.loc[last_sale_date, ['Cost_Sale_avg', 'PVP_avg']].values
        # last_cost = df_sales_filtered.loc[last_sale_date, 'Cost_Sale_avg']
        # last_pvp = df_sales_filtered.loc[last_sale_date, 'PVP_avg']
        last_cost = df_last_sale[0]
        last_pvp = df_last_sale[1]

        dts_sales = df_sales_filtered.loc[dts_interval_sales::, 'Qty_Sold_sum_al']
        dts_total_qty_sold = dts_sales.sum()
        dts_min_total_qty_sold = dts_sales[dts_sales >= 0].min()
        dts_max_total_qty_sol = dts_sales.max()

        # print(part_ref, dts_total_qty_sold, dts_min_total_qty_sold, dts_max_total_qty_sol)
        # print('last_cost', last_cost, '\n', 'last_pvp', last_pvp)

        last_stock = df_sales_filtered['Stock_Qty_al'].tail(1).values[0]
        # print('last_stock', last_stock)
        if last_stock >= 0:
            last_stock_value = last_stock * last_cost
            # print('last_stock_value', last_stock_value)

            df_last_year = df_sales_filtered.loc[last_year_date::, 'Qty_Sold_sum_al']
            # last_year_sales = df_sales_filtered.loc[last_year_date::, 'Qty_Sold_sum_al'].sum()
            last_year_sales = df_last_year.sum()
            # print('last_year_sales', last_year_sales)

            # last_year_sales_avg = df_sales_filtered.loc[last_year_date::, 'Qty_Sold_sum_al'].mean()
            last_year_sales_avg = df_last_year.mean()
            # print('last_year_sales_avg', last_year_sales_avg)

            if last_year_sales:
                last_year_cogs = last_year_sales * last_cost
                # print('last_year_cogs', last_year_cogs)
                margin = (last_pvp - last_cost)

                dii = last_stock_value / last_year_cogs
                dii_year = dii * 365

                avg_sales_per_day = last_stock / last_year_sales_avg  # The diff between dii_year and avg_sales_per_day is caused by the number of non-weekdays;

                days_to_sell_1_part = 1 / last_year_sales_avg

                if margin <= 0:
                    return None

                # print('Part_Ref: {} \n Cost: {:.3f} \n Margin: {:.3f} \n Last Stock: {:.3f} \n Last Stock Value: {:.3f} \n Last Year Sales: {:.3f}'
                #       .format(part_ref, last_cost, margin, last_stock, last_stock_value, last_year_sales))
                df_solve.loc[0, ['Part_Ref', 'Cost', 'PVP', 'Margin', 'DII Year', 'DII Year weekdays', 'DaysToSell_1_Part', 'DTS_Total_Qty_Sold', 'DTS_Min_Total_Qty_Sold', 'DTS_Max_Total_Qty_Sold']] = \
                    [part_ref, last_cost, last_pvp, margin, dii_year, avg_sales_per_day, days_to_sell_1_part, dts_total_qty_sold, dts_min_total_qty_sold, dts_max_total_qty_sol]

                # print('Part_Ref: {}, Cost: {:.2f}, PVP: {:.2f}, Margin: {:.2f}, Last Stock Value: {}, Last Year COGS: {}, DII Year: {:.2f}, Last Stock: {}, Last Year Sales Avg: {}, DII Year weekdays: {:.2f}, DaysToSell_1_Part: {:.2f},'
                #       .format(part_ref, last_cost, last_pvp, margin, last_stock_value, last_year_cogs, dii_year, last_stock, last_year_sales_avg, avg_sales_per_day, days_to_sell_1_part))

    elif df_sales_filtered['Qty_Sold_sum_al'].sum() >= 0 and df_sales_filtered[df_sales_filtered['Qty_Sold_sum_al'] > 0].shape[0] == 1:  # For these cases ill try to fetch the last purchase for this part;
        df_sales_filtered.set_index('index', inplace=True)

        df_sales_filtered_positive_sales_mask = df_sales_filtered[df_sales_filtered['Qty_Sold_sum_al'] > 0]

        unique_sale_date = df_sales_filtered_positive_sales_mask.index.values[0]
        quantity_sold = df_sales_filtered_positive_sales_mask['Qty_Sold_sum_al'].sum()
        purchases = df_sales_filtered[(df_sales_filtered['Qty_Purchased_sum'] > 0) & (df_sales_filtered.index < unique_sale_date)].index.values  # Select the purchases with date inferior to the sale date. Less than and not less or equal, as I will ignore the (weird) same day-purchase-sales

        df_last_sale = df_sales_filtered.loc[unique_sale_date, ['Cost_Sale_avg', 'PVP_avg']].values
        last_cost = df_last_sale[0]
        last_pvp = df_last_sale[1]
        margin = (last_pvp - last_cost)

        if not len(purchases):
            return None

        last_purchase_date = max(purchases)
        days_to_sell_1_part_unique_sale = pd.to_timedelta(unique_sale_date - last_purchase_date, unit='D').days / quantity_sold

        if margin < 0:
            return None

        df_solve.loc[0, ['Part_Ref', 'Cost', 'PVP', 'Margin', 'DII Year', 'DII Year weekdays', 'DaysToSell_1_Part', 'DTS_Total_Qty_Sold', 'DTS_Min_Total_Qty_Sold', 'DTS_Max_Total_Qty_Sold']] = \
            [part_ref, last_cost, last_pvp, margin, np.NaN, np.NaN, days_to_sell_1_part_unique_sale, np.NaN, quantity_sold, quantity_sold]

    return df_solve


if __name__ == '__main__':
    main()
