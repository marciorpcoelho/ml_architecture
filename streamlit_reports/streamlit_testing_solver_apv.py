import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import os
import json
import time
import base64
import sys
import datetime
from traceback import format_exc
import requests
from streamlit.script_runner import RerunException
from streamlit.script_request_queue import RerunData

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
sys.path.insert(1, base_path)
import level_2_order_optimization_apv_baviera_options as options_file
from modules.level_0_performance_report import log_record, error_upload
from modules.level_0_api_endpoint import api_endpoint_ip
import modules.level_1_a_data_acquisition as level_1_a_data_acquisition
import modules.level_1_e_deployment as level_1_e_deployment
import modules.SessionState as SessionState

st.set_page_config(page_title='Sugestão Encomenda de Peças - Baviera - APV')

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

api_backend = api_endpoint_ip + options_file.api_backend_loc

url_hyperlink = '''
    <a href= "{}" > <p style="text-align:right"> Documentação </p></a>
'''.format(options_file.documentation_url_solver_apv)
st.markdown(url_hyperlink, unsafe_allow_html=True)

session_state = SessionState.get(run_id=0, overwrite_button_pressed=0, save_button_pressed_flag=0, part_ref_group='', total_value_optimized=0, df_solution=pd.DataFrame(),
                                 dtss_goal=0, max_part_number=9999, minimum_cost_or_pvp=0, sel_group='', sel_local='')

sel_parameters = [session_state.sel_local, session_state.sel_group, session_state.dtss_goal, session_state.max_part_number, session_state.minimum_cost_or_pvp]
sel_parameters_desc = ['sel_local', 'sel_group', 'dtss_goal', 'max_part_number', 'minimum_cost_or_pvp']

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
    current_date, _ = level_1_e_deployment.time_tags(format_date='%Y%m%d')
    current_month = datetime.date(1900, int(current_date[4:6]), 1).strftime('%B')

    # solve_dataset_name = base_path + 'output/df_solve_0B_{}.csv'.format(current_date)

    solve_data, df_goals = get_data(options_file)
    # saved_suggestions_dict, saved_suggestions_df = get_suggestions_dict(options_file)

    # sel_metric = st.sidebar.selectbox('Please select a metric:', ['-'] + ['DaysToSell_1_Part', 'DaysToSell_1_Part_v2_mean', 'DaysToSell_1_Part_v2_median'], index=0)
    sel_metric = 'Days_To_Sell_Median'

    # if saved_suggestions_df.shape[0]:
    #     saved_suggestions_df_display = saved_suggestions_df.rename(columns=column_translate_dict).replace(options_file.part_groups_desc_mapping)
    #     st.write('Sugestões gravadas:')
    #
    #     fig = go.Figure(data=[go.Table(
    #         columnwidth=[250, 120],
    #         header=dict(
    #             values=[['Grupo de Peças'], ['Data']],
    #             align=['center', 'center'],
    #         ),
    #         cells=dict(
    #             values=[saved_suggestions_df_display['Grupo de Peças'], saved_suggestions_df_display['Data']],
    #             align=['center', 'center'],
    #         )
    #         )
    #     ])
    #     fig.update_layout(width=600, height=240)
    #     st.write(fig)

    if sel_metric != '-':
        solve_data = solve_data[solve_data[sel_metric] > 0]
        solve_data = solve_data[(solve_data['Part_Ref_Group_Desc'] != 'NO_TA') & (solve_data['Part_Ref_Group_Desc'] != 'Outros')]

    solve_data = solve_data[solve_data['Part_Ref_Group_Desc'] != 'MINI_Bonus_Group_2']
    sel_local_original = st.sidebar.selectbox('Por favor escolha uma Concessão:', ['-'] + list(options_file.pse_code_desc_mapping.values()), index=0, key=session_state.run_id)
    sel_group_original = st.sidebar.selectbox('Por favor escolha um grupo de peças:', ['-'] + options_file.part_groups_desc, index=0, key=session_state.run_id)
    try:
        sel_group = [x for x in options_file.part_groups_desc_mapping.keys() if options_file.part_groups_desc_mapping[x] == sel_group_original][0]
        sel_local = [x for x in options_file.pse_code_desc_mapping.keys() if options_file.pse_code_desc_mapping[x] == sel_local_original][0]
    except IndexError:
        return '-'

    dtss_goal = st.sidebar.number_input('Por favor escolha um limite de dias de venda:', 0, 30, value=15)
    max_part_number = st.sidebar.number_input('Por favor escolha um valor máximo de peças (0=Sem Limite):', 0, 5000, value=0)
    minimum_cost_or_pvp = st.sidebar.number_input('Por favor escolha um valor mínimo de Custo/PVP para cada peça (0=Sem Limite):', 0.0, max([max(solve_data['PVP']), max(solve_data['Cost'])]), value=0.0)

    sel_values_filters = [sel_local, sel_group]
    sel_values_col_filters = ['PSE_Code', 'Part_Ref_Group_Desc']

    if '-' not in sel_values_filters:

        try:
            goal_value = int(df_goals.loc[(df_goals['PSE_Code'] == sel_local) & (df_goals['Parts_Group'] == sel_group) & (df_goals['Month'] == current_month), 'Parts_Group_Goal'].values[0])
        except IndexError:
            goal_value = 1
            st.write('Não existe objetivo definido para a Concessão: {}'.format(sel_local_original))
        goal_type = options_file.group_goals_type[sel_group]

        non_goal_type = [x for x in options_file.goal_types if x not in goal_type][0]

        data_filtered = filter_data(solve_data, sel_values_filters, sel_values_col_filters)
        if data_filtered.shape[0] == 0:
            st.error('AVISO: Não existem dados para a concessão e/ou grupo de peças escolhido. Por favor escolha outra combinação de parâmetros.')
            return

        st.write('Objetivo para o grupo de peças escolhido: {}'.format(goal_type_translation[goal_type]))

        if session_state.dtss_goal != dtss_goal or session_state.max_part_number != max_part_number or session_state.minimum_cost_or_pvp != minimum_cost_or_pvp or session_state.sel_group != sel_group or session_state.sel_local != sel_local:
            try:
                # session_state.total_value_optimized, session_state.df_solution = solver(data_filtered, sel_local, sel_group, goal_value, goal_type, non_goal_type, dtss_goal, max_part_number, minimum_cost_or_pvp, sel_metric)
                session_state.total_value_optimized, session_state.df_solution = solver(data_filtered, sel_local, sel_group, goal_value, goal_type, non_goal_type, dtss_goal, max_part_number, minimum_cost_or_pvp, sel_metric)
                session_state.dtss_goal, session_state.max_part_number, session_state.minimum_cost_or_pvp, session_state.sel_group, session_state.sel_local = dtss_goal, max_part_number, minimum_cost_or_pvp, sel_group, sel_local

                try:
                    goal_completed_percentage = session_state.total_value_optimized / goal_value * 100
                    if session_state.total_value_optimized > goal_value:
                        st.write('Objetivo atingido: {:.2f}/{:.2f} ({:.2f}%)'.format(session_state.total_value_optimized, goal_value, goal_completed_percentage))
                    else:
                        st.write('Objetivo ' + '$\\bold{não}$' + ' atingido: {:.2f}/{:.2f} ({:.2f}%)'.format(session_state.total_value_optimized, goal_value, goal_completed_percentage))
                except ZeroDivisionError:
                    pass

                if session_state.df_solution.shape[0]:
                    df_solution_filtered = session_state.df_solution[session_state.df_solution['Quantity'] > 0]
                    st.markdown(
                        '''
                        - Quantidade total de Peças: {:.0f}
                        - Número de peças diferentes: {}
                        - Intervalo de Preços: [{:.2f}, {:.2f}] €
                        '''.format(df_solution_filtered['Quantity'].sum(),
                                   df_solution_filtered['Part_Ref'].nunique(),
                                   df_solution_filtered[goal_type].min(),
                                   df_solution_filtered[goal_type].max())
                    )

                    df_display = df_solution_filtered[['Part_Ref', 'Part_Desc', 'Quantity', goal_type, 'Days_To_Sell']]
                    st.write(df_display.rename(columns=column_translate_dict).style.format({'Quantidade': '{:.1f}', 'PVP': '{:.2f}', 'Dias de Venda': '{:.2f}'}))

                    # if st.button('Gravar Sugestão') or session_state.save_button_pressed_flag == 1:
                    #     session_state.save_button_pressed_flag = 1

                    # file_export(df_display[['Part_Ref', 'Quantity']].rename(columns=column_translate_dict), file_name='Otimização_{}_{}'.format(sel_group_original, current_date), file_extension='.xlsx')
                    file_export_2(df_display[['Part_Ref', 'Part_Desc', 'Quantity']].rename(columns=column_translate_dict), 'Otimização_{}_{}'.format(sel_group_original, current_date))

                    # if sel_group in saved_suggestions_dict.keys() or session_state.overwrite_button_pressed == 1:
                    #     st.write('Já existe Sugestão de Encomenda para o Grupo de Peças {}. Pretende substituir pela atual sugestão?'.format(sel_group_original))
                    #     session_state.overwrite_button_pressed = 1
                    #     if st.button('Sim'):
                    #         solution_saving(df_solution_filtered, sel_group, sel_group_original)
                    #         session_state.save_button_pressed_flag = 0
                    #         session_state.overwrite_button_pressed = 0
                    # else:
                    #     solution_saving(df_solution_filtered, sel_group, sel_group_original)
                    #     session_state.save_button_pressed_flag = 0
                    #     session_state.overwrite_button_pressed = 0

                    # session_state.save_button_pressed_flag = 0
                    # session_state.overwrite_button_pressed = 0
                    # session_state.total_value_optimized = 0
                    # session_state.df_solution = pd.DataFrame()

            except TypeError:
                st.error('AVISO: Não existem peças disponíveis para sugestão para os parâmetros que escolheu. Por favor escolha outra combinação de parâmetros.')
                return

# def file_export(df, file_name, file_extension):
#
#     root = tk.Tk()
#     export_file_path = filedialog.asksaveasfilename(defaultextension=file_extension, initialfile=file_name, master=root)
#     if not export_file_path:
#         return
#
#     df.to_excel(export_file_path, index=False, header=True)
#     # session_state.total_value_optimized = 0
#     # session_state.df_solution = pd.DataFrame()
#
#     return


def file_export_2(df, file_name):

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Gravar Otimização</a> (carregar botão direito e Guardar Link como: {file_name}.csv)'
    st.markdown(href, unsafe_allow_html=True)


def solver(df_solve, sel_local, group, goal_value, goal_type, non_goal_type, dtss_goal, max_part_number, minimum_cost_or_pvp, sel_metric):

    data = {
        'name': 'apv_optimization_test',
        'df_solve': df_solve.to_json(orient='records'),
        'sel_local': sel_local,
        'group': group,
        'goal_value': goal_value,
        'goal_type': goal_type,
        'non_goal_type': non_goal_type,
        'dtss_goal': dtss_goal,
        'max_part_number': max_part_number,
        'minimum_cost_or_pvp': minimum_cost_or_pvp,
        'sel_metric': sel_metric,
    }

    result = requests.get(api_backend, data=json.dumps(data))

    result_dict = json.loads(result.text)

    df_solution = solution_dataframe_creation(data['goal_type'], data['non_goal_type'], result_dict['selection'], result_dict['unique_parts'], result_dict['descriptions'], result_dict['values'], result_dict['other_values'], result_dict['dtss'], result_dict['above_goal_flag'], data['group'], data['sel_local'])

    return result_dict['optimization_total_sum'], df_solution


# def solver(df_solve, sel_local, group, goal_value, goal_type, non_goal_type, dtss_goal, max_part_number, minimum_cost_or_pvp, sel_metric):
#     df_solution = pd.DataFrame()
#     df_solve = df_solve[df_solve[sel_metric] <= dtss_goal]
#
#     if minimum_cost_or_pvp:
#         df_solve = df_solve[df_solve[goal_type] >= minimum_cost_or_pvp]
#
#     unique_parts = df_solve['Part_Ref'].unique()
#     descriptions = [x for x in df_solve['Part_Desc']]
#     df_solve = df_solve[df_solve['Part_Ref'].isin(unique_parts)]
#
#     n_size = df_solve['Part_Ref'].nunique()  # Number of different parts
#     if not n_size:
#         return None
#
#     values = np.array(df_solve[goal_type].values.tolist())  # Costs/Sale prices for each reference, info#1
#     other_values = df_solve[non_goal_type].values.tolist()
#     dtss = np.array(df_solve[sel_metric].values.tolist())  # Days to Sell of each reference, info#2
#
#     selection = cp.Variable(n_size, integer=True)
#
#     dtss_constraint = cp.multiply(selection.T, dtss)
#
#     total_value = selection * values
#
#     if max_part_number:
#         problem_testing_2 = cp.Problem(cp.Maximize(total_value), [dtss_constraint <= dtss_goal, selection >= 0, selection <= 100, cp.sum(selection) <= max_part_number])
#     else:
#         problem_testing_2 = cp.Problem(cp.Maximize(total_value), [dtss_constraint <= dtss_goal, selection >= 0, selection <= 100])
#
#     result = problem_testing_2.solve(solver=cp.GLPK_MI, verbose=False, parallel=True)
#
#     if selection.value is not None:
#         if result >= goal_value:
#             above_goal_flag = 1
#         else:
#             above_goal_flag = 0
#
#         df_solution = solution_dataframe_creation(goal_type, non_goal_type, selection, unique_parts, descriptions, values, other_values, dtss, above_goal_flag, group, sel_local)
#
#     return result, df_solution


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


def solution_dataframe_creation(goal_type, non_goal_type, selection, unique_parts, descriptions, values, other_values, dtss, above_goal_flag, group_name, sel_local):
    df_solution = pd.DataFrame(columns={'Part_Ref', 'Quantity', 'Part_Desc', goal_type, 'Days_To_Sell', 'DtS_Per_Qty'})

    df_solution['Part_Ref'] = [part for part in unique_parts]
    df_solution['Part_Desc'] = [desc for desc in descriptions]
    df_solution['Quantity'] = [qty for qty in selection]
    df_solution[goal_type] = [value for value in values]
    df_solution[non_goal_type] = [value for value in other_values]
    df_solution['Days_To_Sell'] = [dts for dts in dtss]
    df_solution['DtS_Per_Qty'] = [qty * dts for qty, dts in zip(selection, dtss)]
    df_solution['Above_Goal_Flag'] = [above_goal_flag] * len(unique_parts)
    df_solution['Part_Ref_Group_Desc'] = [group_name] * len(unique_parts)
    df_solution['PSE_Code'] = sel_local

    return df_solution


@st.cache(show_spinner=False, ttl=60*60*12)
def get_data(options_file_in):
    df = level_1_a_data_acquisition.sql_retrieve_df(options_file_in.DSN_MLG, options_file_in.sql_info['database_final'], options_file_in.sql_info['final_table'], options_file_in)
    df_goals = level_1_a_data_acquisition.sql_retrieve_df(options_file_in.DSN_MLG, options_file_in.sql_info['database_final'], options_file_in.sql_info['goals_table'], options_file_in)

    return df, df_goals


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
    try:
        main()
    except Exception as exception:
        project_identifier, exception_desc = options_file.project_id, str(sys.exc_info()[1])
        log_record('OPR Error - ' + exception_desc, project_identifier, flag=2, solution_type='OPR')
        error_upload(options_file, project_identifier, format_exc(), exception_desc, error_flag=1, solution_type='OPR')
        session_state.run_id += 1
        st.error('AVISO: Ocorreu um erro. Os administradores desta página foram notificados com informação do erro e este será corrigido assim que possível. Entretanto, esta aplicação será reiniciada. Obrigado pela sua compreensão.')
        time.sleep(10)
        raise RerunException(RerunData())
