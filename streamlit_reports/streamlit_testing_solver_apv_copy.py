import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import os
import pyodbc
import time
from py_dotenv import read_dotenv
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')) + '\\'
dotenv_path = base_path + 'info.env'
read_dotenv(dotenv_path)

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')) + '\\'

"""
# APV Parts Suggestion
Solver Optimization for APV Parts
"""

performance_sql_info = {'DSN': os.getenv('DSN_MLG'),
                        'UID': os.getenv('UID'),
                        'PWD': os.getenv('PWD'),
                        'DB': 'BI_MLG',
                        'test_view': 'streamlit_test_apv'
                        }

configuration_parameters_full = ['Motor_Desc', 'Alarm', 'AC_Auto', 'Open_Roof', 'Auto_Trans', 'Colour_Ext', 'Colour_Int', 'LED_Lights', 'Xenon_Lights', 'Rims_Size', 'Model_Code', 'Navigation', 'Park_Front_Sens', 'Roof_Bars', 'Interior_Type', 'Version']
extra_parameters = ['Average_Score_Euros', 'Number_Cars_Sold', 'Average_Score_Euros_Local', 'Number_Cars_Sold_Local', 'Sales_Place']

# 'MINI_Bonus_Group_1': ['1', '2'],  # Peças + Óleos
# 'MINI_Bonus_Group_3': ['3', '5', '7'],  # Acessórios + Jantes + Lifestyle
# 'MINI_Bonus_Group_4': ['8'],  # Pneus

group_goals = {
    'dtss_goal': 15,  # Weekdays only!
    'number_of_unique_parts': 50,
    'number_of_total_parts': 50,

    # 'BMW_Bonus_Group_1': [223713 - 208239],  # Purchase - September Goal
    # 'BMW_Bonus_Group_3': [4504 - 2226],  # Purchase
    # 'BMW_Bonus_Group_4': [8890 - 8250],  # Sales
    # 'BMW_Bonus_Group_5': [8085 - 3320],  # Sales
    # 'MINI_Bonus_Group_1': [0],
    # 'MINI_Bonus_Group_3': [0],
    # 'MINI_Bonus_Group_4': [0],

    'BMW_Bonus_Group_1': [220170 - 51416],  # Purchase - November Goal
    'BMW_Bonus_Group_3': [4433 - 749],  # Purchase
    'BMW_Bonus_Group_4': [8749 - 347],  # Sales
    'BMW_Bonus_Group_5': [7957 - 127],  # Sales
    'MINI_Bonus_Group_1': [29382 - 7843],  # Purchase
    'MINI_Bonus_Group_3': [2069 - 220],  # Sales
    'MINI_Bonus_Group_4': [723 - 466],  # Sales
}

group_goals_type = {
    'BMW_Bonus_Group_1': 'Cost',  # Purchase
    'BMW_Bonus_Group_3': 'Cost',  # Purchase
    'BMW_Bonus_Group_4': 'PVP',  # Sales
    'BMW_Bonus_Group_5': 'PVP',  # Sales
    'MINI_Bonus_Group_1': 'Cost',  # Sales
    'MINI_Bonus_Group_3': 'PVP',  # Sales
    'MINI_Bonus_Group_4': 'PVP',  # Sales
}

goal_types = ['Cost', 'PVP']


def main():
    current_date = '20191031'
    solve_dataset_name = base_path + 'output/df_solve_0B_{}.csv'.format(current_date)
    solve_data = get_data(solve_dataset_name)
    # ta_dataset_name = base_path + 'output/part_ref_ta_{}.csv'.format(current_date)
    # ta_data = get_data(ta_dataset_name)

    # sel_metric = st.sidebar.selectbox('Please select a metric:', ['None'] + ['DaysToSell_1_Part', 'DaysToSell_1_Part_v2_mean', 'DaysToSell_1_Part_v2_median'], index=0)
    sel_metric = 'DaysToSell_1_Part_v2_median'

    if sel_metric != 'None':
        solve_data = solve_data[solve_data[sel_metric] > 0]
        solve_data = solve_data[(solve_data['Group'] != 'NO_TA') & (solve_data['Group'] != 'Outros')]

    solve_data = solve_data[solve_data['Group'] != 'BMW_Bonus_Group_1']

    sel_group = st.sidebar.selectbox('Please select a Parts Group:', ['None'] + list(solve_data['Group'].dropna().unique()), index=0)
    dtss_goal = st.sidebar.number_input('Please select a limit of days to sell', 0, 30, value=15)
    max_part_number = st.sidebar.number_input('Please select a max number of parts (0 = No Limit):', 0, 5000, value=0)
    minimum_cost_or_pvp = st.sidebar.number_input('Please select a minimum cost/pvp for each part (0 = No Limit):', 0.0, max([max(solve_data['PVP']), max(solve_data['Cost'])]), value=0.0)

    sel_values_filters = [sel_group]
    sel_values_col_filters = ['Group']

    if 'None' not in sel_values_filters:

        goal_value = group_goals[sel_group][0]

        goal_type = group_goals_type[sel_group]
        non_goal_type = [x for x in goal_types if x not in goal_type][0]

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
            st.write("Days to Sell: {:.2f}".format(df_solution_filtered['Qty x DtS'].max()))
            st.write("Solution:", df_solution_filtered[['Part_Ref', 'Qty', goal_type, 'DtS', 'Qty x DtS']])

            cols_name = ['Part_Ref', 'Qty']
            sql_inject(df_solution_filtered[cols_name])


def sql_inject(df_solution):
    df_solution.rename(index=str, columns={'Qty': 'Quantity'}, inplace=True)
    df_solution['Quantity'] = pd.to_numeric(df_solution['Quantity'], downcast='integer')

    print(df_solution.dtypes)

    columns = list(df_solution)

    sql_truncate(performance_sql_info['DSN'], performance_sql_info['UID'], performance_sql_info['PWD'], performance_sql_info['DB'], performance_sql_info['test_view'])

    columns_string, values_string = sql_string_preparation_v1(columns)

    cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'
                          .format(performance_sql_info['DSN'], performance_sql_info['UID'], performance_sql_info['PWD'], performance_sql_info['DB']), searchescape='\\')

    cursor = cnxn.cursor()
    for index, row in df_solution.iterrows():
        cursor.execute("INSERT INTO " + performance_sql_info['test_view'] + "(" + columns_string + ') VALUES ( ' + '\'{}\', \'{}\''.format(row['Part_Ref'], row['Quantity']) + ' )')

    cursor.commit()
    cursor.close()


def sql_truncate(dsn, uid, pwd, database, view):
    cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'.format(dsn, uid, pwd, database), searchescape='\\')
    query = "TRUNCATE TABLE " + view
    cursor = cnxn.cursor()
    cursor.execute(query)

    cnxn.commit()
    cursor.close()
    cnxn.close()


def sql_string_preparation_v1(values_list):
    columns_string = '[%s]' % "], [".join(values_list)

    values_string = ['\'?\''] * len(values_list)
    values_string = 'values (%s)' % ', '.join(values_string)

    return columns_string, values_string


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

        df_solution = solution_saving_csv(goal_type, non_goal_type, selection, unique_parts, values, other_values, dtss)

    return problem_testing_2.status, result, selection.value, above_goal_flag, df_solution


def solution_saving_csv(goal_type, non_goal_type, selection, unique_parts, values, other_values, dtss):
    df_solution = pd.DataFrame(columns={'Part_Ref', 'Qty', goal_type, 'DtS', 'Qty x DtS'})

    df_solution['Part_Ref'] = [part for part in unique_parts]
    df_solution['Qty'] = [qty for qty in selection.value]
    df_solution[goal_type] = [value for value in values]
    df_solution[non_goal_type] = [value for value in other_values]
    df_solution['DtS'] = [dts for dts in dtss]
    df_solution['Qty x DtS'] = [qty * dts for qty, dts in zip(selection.value, dtss)]
    # df_solution['AboveGoal?'] = [above_goal_flag] * len(unique_parts)
    # df_solution['Group_Name'] = [group_name] * len(unique_parts)

    # df_solution.to_csv('output/solver_solution_{}_goal_{}_{}.csv'.format(group_name, goal_type, current_date))

    return df_solution


@st.cache
def get_data(dataset_name):
    df = pd.read_csv(dataset_name, index_col=0)

    return df


def filter_data(dataset, filters_list, col_filters_list):
    data_filtered = dataset.copy()

    for col_filter, filter_value in zip(col_filters_list, filters_list):
        data_filtered = data_filtered[data_filtered[col_filter] == filter_value]

    return data_filtered


if __name__ == '__main__':
    main()
