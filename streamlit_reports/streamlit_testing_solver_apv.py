import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import os
import sys
import pyodbc
import time

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')) + '\\'
sys.path.insert(1, base_path)
import level_2_order_optimization_apv_baviera_options as options_file
import modules.level_1_a_data_acquisition as level_1_a_data_acquisition
import modules.level_1_e_deployment as level_1_e_deployment


"""
# APV Parts Suggestion
Solver Optimization for APV Parts
"""


def main():
    current_date = '20191031'
    # solve_dataset_name = base_path + 'output/df_solve_0B_{}.csv'.format(current_date)
    solve_data = get_data(options_file)

    # sel_metric = st.sidebar.selectbox('Please select a metric:', ['None'] + ['DaysToSell_1_Part', 'DaysToSell_1_Part_v2_mean', 'DaysToSell_1_Part_v2_median'], index=0)
    sel_metric = 'Days_To_Sell_Median'

    if sel_metric != 'None':
        solve_data = solve_data[solve_data[sel_metric] > 0]
        solve_data = solve_data[(solve_data['Part_Ref_Group'] != 'NO_TA') & (solve_data['Part_Ref_Group'] != 'Outros')]

    solve_data = solve_data[solve_data['Part_Ref_Group'] != 'BMW_Bonus_Group_1']

    sel_group = st.sidebar.selectbox('Please select a Parts Group:', ['None'] + list(solve_data['Part_Ref_Group'].dropna().unique()), index=0)
    dtss_goal = st.sidebar.number_input('Please select a limit of days to sell', 0, 30, value=15)
    max_part_number = st.sidebar.number_input('Please select a max number of parts (0 = No Limit):', 0, 5000, value=0)
    minimum_cost_or_pvp = st.sidebar.number_input('Please select a minimum cost/pvp for each part (0 = No Limit):', 0.0, max([max(solve_data['PVP']), max(solve_data['Cost'])]), value=0.0)

    sel_values_filters = [sel_group]
    sel_values_col_filters = ['Part_Ref_Group']

    if 'None' not in sel_values_filters:

        goal_value = options_file.group_goals[sel_group][0]

        goal_type = options_file.group_goals_type[sel_group]
        non_goal_type = [x for x in options_file.goal_types if x not in goal_type][0]

        data_filtered = filter_data(solve_data, sel_values_filters, sel_values_col_filters)

        st.write('Goal for selected parts group: {}'.format(goal_type))

        # st.write('Dataset:', data_filtered)
        # st.write('Dataset Size:', data_filtered.shape)

        status, total_value_optimized, selection, above_goal_flag, df_solution = solver(data_filtered, sel_group, goal_value, goal_type, non_goal_type, dtss_goal, max_part_number, minimum_cost_or_pvp, sel_metric)

        st.write('Optimization Status: {}'.format(status))
        st.write('Max solution achieved: {:.2f} €'.format(total_value_optimized))

        try:
            if above_goal_flag:
                st.write('Goal Achieved: {:.2f}/{:.2f} ({:.2f}%)'.format(total_value_optimized, goal_value, total_value_optimized / goal_value * 100))
            else:
                st.write('Goal ' + '$\\bold{not}$' + ' Achieved: {:.2f}/{:.2f} ({:.2f}%)'.format(total_value_optimized, goal_value, total_value_optimized / goal_value * 100))
        except ZeroDivisionError:
            pass

        if df_solution.shape[0]:
            df_solution_filtered = df_solution[df_solution['Qty'] > 0]
            st.write("Total Parts: {:.0f}".format(df_solution_filtered['Qty'].sum()))
            st.write("#Different Parts: {}".format(df_solution_filtered['Part_Ref'].nunique()))
            st.write("Days to Sell: {:.2f}".format(df_solution_filtered['DtS_Per_Qty'].max()))
            st.write("Solution:", df_solution_filtered[['Part_Ref', 'Qty', goal_type, 'DtS', 'DtS_Per_Qty']])

            solution_saving(df_solution_filtered, sel_group, current_date)


def solver(df_solve, group, goal_value, goal_type, non_goal_type, dtss_goal, max_part_number, minimum_cost_or_pvp, sel_metric):
    print('Solving for {}...'.format(group))

    above_goal_flag = 0
    df_solution = pd.DataFrame()
    df_solve = df_solve[df_solve[sel_metric] <= dtss_goal]

    if minimum_cost_or_pvp:
        df_solve = df_solve[df_solve[goal_type] >= minimum_cost_or_pvp]

    unique_parts = df_solve['Part_Ref'].unique()
    df_solve = df_solve[df_solve['Part_Ref'].isin(unique_parts)]

    n_size = df_solve['Part_Ref'].nunique()  # Number of different parts
    print('Number of parts inside initial conditions:', n_size)

    values = np.array(df_solve[goal_type].values.tolist())  # Costs/Sale prices for each reference, info#1
    other_values = df_solve[non_goal_type].values.tolist()
    dtss = np.array(df_solve[sel_metric].values.tolist())  # Days to Sell of each reference, info#2

    selection = cp.Variable(n_size, integer=True)

    dtss_constraint = cp.max(cp.diag(cp.multiply(selection.T, dtss)))

    total_value = selection * values

    if max_part_number:
        problem_testing_2 = cp.Problem(cp.Maximize(total_value), [dtss_constraint <= dtss_goal, selection >= 0, selection <= 100, cp.sum(selection) <= max_part_number])
    else:
        problem_testing_2 = cp.Problem(cp.Maximize(total_value), [dtss_constraint <= dtss_goal, selection >= 0, selection <= 100])

    result = problem_testing_2.solve(solver=cp.GLPK_MI, verbose=False)

    if selection.value is not None:
        if result >= goal_value:
            above_goal_flag = 1
            # st.write('Solution found and above set goal :D')
        else:
            above_goal_flag = 0
            # st.write('Solution does not reach the goal :(')

        df_solution = solution_dataframe_creation(goal_type, non_goal_type, selection, unique_parts, values, other_values, dtss, above_goal_flag, group)

    return problem_testing_2.status, result, selection.value, above_goal_flag, df_solution


def solution_saving(df_solution, group_name, current_date):

    df_solution.to_csv(base_path + '/output/solver_solution_{}_{}.csv'.format(group_name, current_date))
    level_1_e_deployment.sql_inject(df_solution, options_file.DSN_MLG, options_file.sql_info['database_final'], options_file.sql_info['optimization_solution_table'], options_file, list(df_solution[options_file.columns_sql_solver_solution]), truncate=1)


def solution_dataframe_creation(goal_type, non_goal_type, selection, unique_parts, values, other_values, dtss, above_goal_flag, group_name):
    df_solution = pd.DataFrame(columns={'Part_Ref', 'Qty', goal_type, 'DtS', 'DtS_Per_Qty'})

    df_solution['Part_Ref'] = [part for part in unique_parts]
    df_solution['Qty'] = [qty for qty in selection.value]
    df_solution[goal_type] = [value for value in values]
    df_solution[non_goal_type] = [value for value in other_values]
    df_solution['DtS'] = [dts for dts in dtss]
    df_solution['DtS_Per_Qty'] = [qty * dts for qty, dts in zip(selection.value, dtss)]
    df_solution['Above_Goal_Flag'] = [above_goal_flag] * len(unique_parts)
    df_solution['Part_Ref_Group'] = [group_name] * len(unique_parts)

    return df_solution


@st.cache
def get_data(options_file_in):
    df = level_1_a_data_acquisition.sql_retrieve_df(options_file_in.DSN_MLG, options_file_in.sql_info['database_final'], options_file_in.sql_info['final_table'], options_file_in)

    return df


def filter_data(dataset, filters_list, col_filters_list):
    data_filtered = dataset.copy()

    for col_filter, filter_value in zip(col_filters_list, filters_list):
        data_filtered = data_filtered[data_filtered[col_filter] == filter_value]

    return data_filtered


if __name__ == '__main__':
    main()
