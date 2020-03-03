import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import os
import sys

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')) + '\\'
sys.path.insert(1, base_path)
import level_2_order_optimization_apv_baviera_options as options_file
import modules.level_1_a_data_acquisition as level_1_a_data_acquisition
import modules.level_1_e_deployment as level_1_e_deployment
import modules.SessionState as SessionState
from plotly import graph_objs as go

saved_solutions_pairs_query = ''' SELECT DISTINCT Part_Ref_Group_Desc, [Date]
      FROM [BI_MLG].[dbo].[{}]
      GROUP BY Part_Ref_Group_Desc, [Date]'''.format(options_file.sql_info['optimization_solution_table'])

truncate_query = '''DELETE FROM [BI_MLG].[dbo].[{}]
    WHERE Part_Ref_Group_Desc = '{}' '''

"""
# Sugestão de Encomenda de Peças - Após-Venda Baviera
Otimização de Encomenda de Peças
"""

# Hides the menu's hamburguer
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

session_state = SessionState.get(overwrite_button_pressed=0, save_button_pressed_flag=0, part_ref_group='')

column_translate_dict = {
    'Part_Ref': 'Referência',
    'Part_Ref_Group_Desc': 'Grupo de Peças',
    'Quantity': 'Quantidade',
    'Cost': 'Custo',
    'Days_To_Sell': 'Dias de Venda',
    'Date': 'Data',
}

goal_type_translation = {
    'PVP': 'Vendas',
    'Cost': 'Compras',
}

reb_subs = {v: k for v, k in options_file.part_groups_desc_mapping.items()}


def main():
    current_date = '20191031'
    # solve_dataset_name = base_path + 'output/df_solve_0B_{}.csv'.format(current_date)

    solve_data = get_data(options_file)
    saved_suggestions_dict, saved_suggestions_df = get_suggestions_dict(options_file)

    # sel_metric = st.sidebar.selectbox('Please select a metric:', ['-'] + ['DaysToSell_1_Part', 'DaysToSell_1_Part_v2_mean', 'DaysToSell_1_Part_v2_median'], index=0)
    sel_metric = 'Days_To_Sell_Median'

    if saved_suggestions_df.shape[0]:
        saved_suggestions_df_display = saved_suggestions_df.rename(columns=column_translate_dict).replace(options_file.part_groups_desc_mapping)
        st.write('Sugestões gravadas:')

        fig = go.Figure(data=[go.Table(
            columnwidth=[250, 120],
            header=dict(
                values=[['Grupo de Peças'], ['Data']],
                align=['center', 'center'],
            ),
            cells=dict(
                values=[saved_suggestions_df_display['Grupo de Peças'], saved_suggestions_df_display['Data']],
                align=['center', 'center'],
            )
            )
        ])
        fig.update_layout(width=600, height=240)
        st.write(fig)

    if sel_metric != '-':
        solve_data = solve_data[solve_data[sel_metric] > 0]
        solve_data = solve_data[(solve_data['Part_Ref_Group_Desc'] != 'NO_TA') & (solve_data['Part_Ref_Group_Desc'] != 'Outros')]

    solve_data = solve_data[solve_data['Part_Ref_Group_Desc'] != 'BMW_Bonus_Group_1']
    sel_group_original = st.sidebar.selectbox('Por favor escolha um grupo de peças:', ['-'] + options_file.part_groups_desc, index=0)

    try:
        sel_group = [x for x in options_file.part_groups_desc_mapping.keys() if options_file.part_groups_desc_mapping[x] == sel_group_original][0]
    except IndexError:
        return '-'

    dtss_goal = st.sidebar.number_input('Por favor escolha um limite de dias de venda:', 0, 30, value=15)
    max_part_number = st.sidebar.number_input('Por favor escolha um valor máximo de peças (0=Sem Limite):', 0, 5000, value=0)
    minimum_cost_or_pvp = st.sidebar.number_input('Por favor escolha um valor mínimo de Custo/PVP para cada peça (0=Sem Limite):', 0.0, max([max(solve_data['PVP']), max(solve_data['Cost'])]), value=0.0)

    sel_values_filters = [sel_group]
    sel_values_col_filters = ['Part_Ref_Group_Desc']

    if '-' not in sel_values_filters:

        goal_value = options_file.group_goals[sel_group][0]

        goal_type = options_file.group_goals_type[sel_group]
        non_goal_type = [x for x in options_file.goal_types if x not in goal_type][0]

        data_filtered = filter_data(solve_data, sel_values_filters, sel_values_col_filters)

        st.write('Objetivo para o grupo de peças escolhido: {}'.format(goal_type_translation[goal_type]))

        status, total_value_optimized, selection, above_goal_flag, df_solution = solver(data_filtered, sel_group, goal_value, goal_type, non_goal_type, dtss_goal, max_part_number, minimum_cost_or_pvp, sel_metric)

        st.write('Máxima solução atingida: {:.2f} €'.format(total_value_optimized))

        try:
            if above_goal_flag:
                st.write('Objetivo atingido: {:.2f}/{:.2f} ({:.2f}%)'.format(total_value_optimized, goal_value, total_value_optimized / goal_value * 100))
            else:
                st.write('Objetivo ' + '$\\bold{não}$' + ' atingido: {:.2f}/{:.2f} ({:.2f}%)'.format(total_value_optimized, goal_value, total_value_optimized / goal_value * 100))
        except ZeroDivisionError:
            pass

        if df_solution.shape[0]:
            df_solution_filtered = df_solution[df_solution['Quantity'] > 0]
            st.write("Quantidade total de Peças: {:.0f}".format(df_solution_filtered['Quantity'].sum()))
            st.write("Número de peças diferentes: {}".format(df_solution_filtered['Part_Ref'].nunique()))
            # st.write("Dias estimados para venda da encomenda: {:.2f}".format(df_solution_filtered['DtS_Per_Qty'].max()))

            df_display = df_solution_filtered[['Part_Ref', 'Quantity', goal_type, 'Days_To_Sell']]
            st.write(df_display.rename(columns=column_translate_dict).style.format({'Quantidade': '{:.1f}', 'PVP': '{:.2f}', 'Dias de Venda': '{:.2f}'}))

            if st.button('Gravar Sugestão') or session_state.save_button_pressed_flag == 1:
                session_state.save_button_pressed_flag = 1

                if sel_group in saved_suggestions_dict.keys() or session_state.overwrite_button_pressed == 1:
                    st.write('Já existe Sugestão de Encomenda para o Grupo de Peças {}. Pretende substituir pela atual sugestão?'.format(sel_group_original))
                    session_state.overwrite_button_pressed = 1
                    if st.button('Sim'):
                        solution_saving(df_solution_filtered, sel_group, sel_group_original)
                        session_state.save_button_pressed_flag = 0
                        session_state.overwrite_button_pressed = 0
                else:
                    solution_saving(df_solution_filtered, sel_group, sel_group_original)
                    session_state.save_button_pressed_flag = 0
                    session_state.overwrite_button_pressed = 0


def solver(df_solve, group, goal_value, goal_type, non_goal_type, dtss_goal, max_part_number, minimum_cost_or_pvp, sel_metric):
    above_goal_flag = 0
    df_solution = pd.DataFrame()
    df_solve = df_solve[df_solve[sel_metric] <= dtss_goal]

    if minimum_cost_or_pvp:
        df_solve = df_solve[df_solve[goal_type] >= minimum_cost_or_pvp]

    unique_parts = df_solve['Part_Ref'].unique()
    df_solve = df_solve[df_solve['Part_Ref'].isin(unique_parts)]

    n_size = df_solve['Part_Ref'].nunique()  # Number of different parts

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
        else:
            above_goal_flag = 0

        df_solution = solution_dataframe_creation(goal_type, non_goal_type, selection, unique_parts, values, other_values, dtss, above_goal_flag, group)

    return problem_testing_2.status, result, selection.value, above_goal_flag, df_solution


def solution_saving(df_solution, group_name, group_name_original):

    level_1_e_deployment.sql_truncate(options_file.DSN_MLG, options_file, options_file.sql_info['database_final'], options_file.sql_info['optimization_solution_table'], query=truncate_query.format(options_file.sql_info['optimization_solution_table'], group_name))

    level_1_e_deployment.sql_inject(df_solution,
                                    options_file.DSN_MLG,
                                    options_file.sql_info['database_final'],
                                    options_file.sql_info['optimization_solution_table'],
                                    options_file,
                                    list(df_solution[options_file.columns_sql_solver_solution]),
                                    check_date=1)

    st.write('Sugestão gravada com sucesso - {}'.format(group_name_original))
    return


def solution_dataframe_creation(goal_type, non_goal_type, selection, unique_parts, values, other_values, dtss, above_goal_flag, group_name):
    df_solution = pd.DataFrame(columns={'Part_Ref', 'Quantity', goal_type, 'Days_To_Sell', 'DtS_Per_Qty'})

    df_solution['Part_Ref'] = [part for part in unique_parts]
    df_solution['Quantity'] = [qty for qty in selection.value]
    df_solution[goal_type] = [value for value in values]
    df_solution[non_goal_type] = [value for value in other_values]
    df_solution['Days_To_Sell'] = [dts for dts in dtss]
    df_solution['DtS_Per_Qty'] = [qty * dts for qty, dts in zip(selection.value, dtss)]
    df_solution['Above_Goal_Flag'] = [above_goal_flag] * len(unique_parts)
    df_solution['Part_Ref_Group_Desc'] = [group_name] * len(unique_parts)

    return df_solution


@st.cache
def get_data(options_file_in):
    df = level_1_a_data_acquisition.sql_retrieve_df(options_file_in.DSN_MLG, options_file_in.sql_info['database_final'], options_file_in.sql_info['final_table'], options_file_in)

    return df


def get_suggestions_dict(options_file_in):
    saved_suggestions_dict = {}
    saved_suggestions_df = level_1_a_data_acquisition.sql_retrieve_df_specified_query(options_file_in.DSN_MLG, options_file_in.sql_info['database_final'], options_file_in, query=saved_solutions_pairs_query)

    saved_suggestions_df_grouped = saved_suggestions_df.groupby('Part_Ref_Group_Desc')
    for key, group in saved_suggestions_df_grouped:
        saved_suggestions_dict[key] = list(group.values)

    return saved_suggestions_dict, saved_suggestions_df


def filter_data(dataset, filters_list, col_filters_list):
    data_filtered = dataset.copy()

    for col_filter, filter_value in zip(col_filters_list, filters_list):
        data_filtered = data_filtered[data_filtered[col_filter] == filter_value]

    return data_filtered


if __name__ == '__main__':
    main()
