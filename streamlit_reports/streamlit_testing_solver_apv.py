import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import os
import time

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')) + '\\'

"""
# Streamlit second test
Here's my first attempt at deploying a solver
"""

configuration_parameters_full = ['Motor_Desc', 'Alarm', 'AC_Auto', 'Open_Roof', 'Auto_Trans', 'Colour_Ext', 'Colour_Int', 'LED_Lights', 'Xenon_Lights', 'Rims_Size', 'Model_Code', 'Navigation', 'Park_Front_Sens', 'Roof_Bars', 'Interior_Type', 'Version']
extra_parameters = ['Average_Score_Euros', 'Number_Cars_Sold', 'Average_Score_Euros_Local', 'Number_Cars_Sold_Local', 'Sales_Place']

group_goals = {
    'dtss_goal': 15,  # Weekdays only!
    'number_of_unique_parts': 50,
    'number_of_total_parts': 50,
    # 'BMW_Bonus_Group_1': [640345 - 399127, 253279],  # Cost, Sales (Values are for a single month, using the goal for the 3-month period minus the already sold in the first two months
    # 'BMW_Bonus_Group_2': [155713 - 93374, 3655],  # Cost, Sales
    # 'BMW_Bonus_Group_3': [12893 - 10808, 2189],  # Cost, Sales
    # 'BMW_Bonus_Group_4': [25445 - 16152, 9758],  # Cost, Sales
    # 'BMW_Bonus_Group_5': [23143 - 14146, 9447],  # Cost, Sales
    # 'BMW_Bonus_Group_1': [229979 * 1.05],  # Purchase - September Goal
    # 'BMW_Bonus_Group_3': [4630 * 1.05],  # Purchase
    # 'BMW_Bonus_Group_4': [9139 * 1.05],  # Sales
    # 'BMW_Bonus_Group_5': [8132 * 1.05],  # Sales
    'BMW_Bonus_Group_1': [223713 - 208239],  # Purchase - September Goal
    'BMW_Bonus_Group_3': [4504 - 2226],  # Purchase
    'BMW_Bonus_Group_4': [8890 - 8250],  # Sales
    'BMW_Bonus_Group_5': [8085 - 3320],  # Sales
    'BMW_Bonus_Group_1_limit': [223713 * 1.05 - 208239],  # Purchase - September Goal
    'BMW_Bonus_Group_3_limit': [4504 * 1.05 - 2226],  # Purchase
    'BMW_Bonus_Group_4_limit': [8890 * 1.05 - 8250],  # Sales
    'BMW_Bonus_Group_5_limit': [8085 * 1.05 - 3320],  # Sales
    'MINI_Bonus_Group_1': [0],
    'MINI_Bonus_Group_3': [0],
    'MINI_Bonus_Group_4': [0],
    'MINI_Bonus_Group_1_limit': [0],
    'MINI_Bonus_Group_3_limit': [0],
    'MINI_Bonus_Group_4_limit': [0],
}

group_goals_type = {
    'BMW_Bonus_Group_1': 'Cost',  # Purchase
    'BMW_Bonus_Group_3': 'Cost',  # Purchase
    'BMW_Bonus_Group_4': 'PVP',  # Sales
    'BMW_Bonus_Group_5': 'PVP',  # Sales
    'MINI_Bonus_Group_1': 'PVP',  # Sales
    'MINI_Bonus_Group_3': 'PVP',  # Sales
    'MINI_Bonus_Group_4': 'PVP',  # Sales
}

goal_types = ['Cost', 'PVP']


def main():
    current_date = '20190831'
    # ta_dataset_name = base_path + 'output/part_ref_ta_{}.csv'.format(current_date)
    solve_dataset_name = base_path + 'output/df_solve_0B_backup.csv'
    # ta_data = get_data(ta_dataset_name)
    solve_data = get_data(solve_dataset_name)

    dtss_goal = group_goals['dtss_goal']
    number_of_parts_goal = group_goals['number_of_unique_parts']

    solve_data = solve_data[solve_data['DaysToSell_1_Part'] > 0]
    solve_data = solve_data[(solve_data['Group'] != 'NO_TA') & (solve_data['Group'] != 'Outros')]

    sel_group = st.sidebar.selectbox('Please select a Parts Group', ['None'] + list(solve_data['Group'].dropna().unique()), index=0)

    sel_values_filters = [sel_group]
    sel_values_col_filters = ['Group']

    if 'None' not in sel_values_filters:

        if sel_group[0:4] == 'MINI':
            goal_value = 0  # Currently have no goals for MINI groups
        else:
            goal_value = group_goals[sel_group][0]

        goal_value_limit = group_goals[sel_group + '_limit'][0]
        goal_type = group_goals_type[sel_group]
        non_goal_type = [x for x in goal_types if x not in goal_type][0]
        total_number_of_parts_goal = group_goals['number_of_total_parts']

        data_filtered = filter_data(solve_data, sel_values_filters, sel_values_col_filters)

        st.write('Goal for selected parts group: {}'.format(goal_type))

        # st.write('Dataset:', data_filtered)
        # st.write('Dataset Size:', data_filtered.shape)

        status, total_value_optimized, selection, above_goal_flag, df_solution = solver(data_filtered, sel_group, goal_value, goal_value_limit, goal_type, non_goal_type, dtss_goal, number_of_parts_goal, total_number_of_parts_goal)

        st.write('Optimization Status: {}'.format(status))
        st.write('Max solution achieved: {:.2f} €'.format(total_value_optimized))

        try:
            if above_goal_flag:
                st.write('Goal Achieved: {:.2f}/{:.2f} ({:.2f}%)'.format(total_value_optimized, goal_value, total_value_optimized / goal_value * 100))
            else:
                st.write('Goal Not Achieved: {:.2f}/{:.2f} ({:.2f}%)'.format(total_value_optimized, goal_value, total_value_optimized / goal_value * 100))
        except ZeroDivisionError:
            pass

        # st.write('Solution: {}'.format(selection))

        if df_solution.shape[0]:
            st.write("Solution:", df_solution)


def solver(df_solve, group, goal_value, goal_value_limit, goal_type, non_goal_type, dtss_goal, number_of_parts_goal, total_number_of_parts_goal):
    print('Solving for {}...'.format(group))

    above_goal_flag = 0
    df_solution = pd.DataFrame()
    df_solve = df_solve[df_solve['DaysToSell_1_Part'] <= dtss_goal]

    unique_parts = df_solve['Part_Ref'].unique()
    df_solve = df_solve[df_solve['Part_Ref'].isin(unique_parts)]
    start = time.time()

    n_size = df_solve['Part_Ref'].nunique()  # Number of different parts
    print('Number of parts inside initial conditions:', n_size)
    # dtss_goals = np.array([dtss_goal] * n_size)

    values = np.array(df_solve[goal_type].values.tolist())  # Costs/Sale prices for each reference, info#1
    other_values = df_solve[non_goal_type].values.tolist()
    dtss = np.array(df_solve['DaysToSell_1_Part'].values.tolist())  # Days to Sell of each reference, info#2

    selection = cp.Variable(n_size, integer=True)

    dtss_constraint = cp.max(cp.diag(cp.multiply(selection.T, dtss)))

    total_value = selection * values

    problem_testing_2 = cp.Problem(cp.Maximize(total_value), [dtss_constraint <= dtss_goal, selection >= 0, selection <= 100])

    result = problem_testing_2.solve(solver=cp.GLPK_MI, verbose=False)

    # print('Status of optimization:', problem_testing_2.status)
    # print('Result of optimization:', result)
    # print('Part Order:\n', selection.value)
    if selection.value is not None:
        # print('Solution Found :)')
        if result >= goal_value:
            above_goal_flag = 1
            print('Solution found and above set goal :D')
        else:
            above_goal_flag = 0
            print('Solution does not reach the goal :(')

        # [print(part + ': ' + str(qty) + ' and cost/sell value of {}, DTSS: {:.2f}'.format(value, dts)) for part, value, qty, dts in zip(unique_parts, values, selection.value, dtss)]
        # print(np.matmul(selection.value, values))

        # print('Total {} of: {:.2f} / {:.2f} ({:.2f}%) \nDays to Sell: {:.2f} / {:.2f} ({:.2f}%)'
        #       .format(goal_type, result, goal_value, (result / goal_value) * 100, np.max([dts * qty for dts, qty in zip(dtss, selection.value)]), dtss_goal, (np.max([dts * qty for dts, qty in zip(dtss, selection.value)]) / dtss_goal) * 100))

        df_solution = solution_saving_csv(group, goal_type, non_goal_type, above_goal_flag, selection, unique_parts, values, other_values, dtss)

    # else:
    #     print('No Solution found :(')

    # print('Number of Inequations: {}'.format(problem_testing_2.size_metrics.num_scalar_leq_constr))
    # print(problem_testing_2.size_metrics.max_data_dimension)

    # print('Elapsed time: {:.2f} seconds.\n'.format(time.time() - start))

    return problem_testing_2.status, result, selection.value, above_goal_flag, df_solution


def solution_saving_csv(group_name, goal_type, non_goal_type, above_goal_flag, selection, unique_parts, values, other_values, dtss):
    # print(goal_type)
    # print(non_goal_type)

    df_solution = pd.DataFrame(columns={'Part_Ref', 'Qty', goal_type, 'DtS', 'AboveGoal?'})

    df_solution['Part_Ref'] = [part for part in unique_parts]
    df_solution['Qty'] = [qty for qty in selection.value]
    df_solution[goal_type] = [value for value in values]
    df_solution[non_goal_type] = [value for value in other_values]
    df_solution['DtS'] = [dts for dts in dtss]
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
