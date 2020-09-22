import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import sys
import os
import base64
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
sys.path.insert(1, base_path)
import modules.level_1_a_data_acquisition as level_1_a_data_acquisition
import modules.level_1_b_data_processing as level_1_b_data_processing
import modules.level_1_e_deployment as level_1_e_deployment
import level_2_optionals_baviera_options as options_file
import modules.SessionState as sessionstate

st.beta_set_page_config(page_title='Sugestão de Encomenda - Baviera')

"""
# Sugestão de Encomenda Baviera - DEMO
Sugestão de Configurações para a encomenda mensal de viaturas BMW
"""

configuration_parameters_full = ['Motor_Desc', 'Alarm', 'AC_Auto', 'Open_Roof', 'Auto_Trans', 'Colour_Ext', 'Colour_Int', 'LED_Lights', 'Rims_Size', 'Model_Code', 'Navigation', 'Park_Front_Sens', 'Roof_Bars', 'Interior_Type', 'Version']
configuration_parameters_full_rename = ['Motorização', 'Alarme', 'AC Auto', 'Teto Abrir', 'Caixa Auto.', 'Cor Exterior', 'Cor Interior', 'Faróis LED', 'Tam. Jantes', 'Modelo', 'Navegação', 'Sens. Diant.', 'Barras Tej.', 'Tipo Interior', 'Versão']
extra_parameters = ['Average_Score_Euros_Local_Fase2_Level_1', 'Number_Cars_Sold', 'Number_Cars_Sold_Local_Fase2_Level_1', 'Sales_Place_Fase2_Level_1']
extra_parameters_rename = ['Score (€)', '#Vendas Global', '#Vendas Local', 'Concessão']
boolean_columns = ['Alarme', 'AC Auto', 'Teto Abrir', 'Caixa Auto.', 'Faróis LED', 'Navegação', 'Sens. Diant.', 'Barras Tej.']
# min_number_of_configuration = 5

saved_solutions_pairs_query = ''' SELECT DISTINCT Sales_Place_Fase2_Level_1, Model_Code, [Date]
  FROM [BI_MLG].[dbo].[VHE_Fact_BI_OrderOptimization_Solver_Optimization]
  GROUP BY Sales_Place_Fase2_Level_1, Model_Code, [Date]'''

truncate_query = '''DELETE 
FROM [BI_MLG].[dbo].[VHE_Fact_BI_OrderOptimization_Solver_Optimization]
WHERE Sales_Place_Fase2_Level_1 = '{}' and Model_Code = '{}' '''

column_translate = {
    'Average_Score_Euros_Local_Fase2_Level_1': 'Score (€)',
    'Number_Cars_Sold_Local_Fase2_Level_1': '#Vendas Local',
    'Number_Cars_Sold': '#Vendas Global',
    'Sales_Place_Fase2_Level_1': 'Concessão',
    'Model_Code': 'Modelo',
    'Colour_Ext': 'Cor Exterior',
    'Colour_Int': 'Cor Interior',
    'Motor_Desc': 'Motorização',
    'Quantity': 'Quantidade',
}

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


session_state = sessionstate.get(overwrite_button_pressed=0, save_button_pressed_flag=0, locals='-', model='')


def main():
    current_date, _ = level_1_e_deployment.time_tags(format_date='%Y%m%d')
    data = get_data(options_file)

    # saved_suggestions_dict, saved_suggestions_df = get_suggestions_dict(options_file)
    parameters_values, parameter_restriction_vectors = [], []

    max_number_of_cars_sold = max(data[column_translate['Number_Cars_Sold_Local_Fase2_Level_1']])

    sel_locals = st.sidebar.multiselect('Concessões:', list(data[column_translate['Sales_Place_Fase2_Level_1']].unique()))
    sel_model = st.sidebar.selectbox('Modelo:', ['-'] + [x for x in data[column_translate['Model_Code']].unique()], index=0)

    if sel_locals != session_state.locals or sel_model != session_state.model:
        session_state.local = sel_locals
        session_state.model = sel_model
        session_state.overwrite_button_pressed, session_state.save_button_pressed_flag = 0, 0

    if sel_model != '-' and sel_locals != '':
        if data[(data[column_translate['Sales_Place_Fase2_Level_1']].isin(sel_locals)) & (data[column_translate['Model_Code']] == sel_model)].shape[0]:
            max_number_of_cars_sold = max(data[(data[column_translate['Sales_Place_Fase2_Level_1']].isin(sel_locals)) & (data[column_translate['Model_Code']] == sel_model)][column_translate['Number_Cars_Sold_Local_Fase2_Level_1']])
        else:
            st.error('Não foram encontrados registos para a concessão e modelo selecionados.')
            return

    sel_min_sold_cars = st.sidebar.number_input('Por favor escolha um valor mínimo de viaturas vendidas localmente por configuração (valor máximo é de {}):'.format(max_number_of_cars_sold), 1, max_number_of_cars_sold, value=1)
    sel_number_of_configuration = st.sidebar.number_input('Por favor escolha o número de configurações a apresentar:', value=5)
    sel_values_filters = [sel_locals, sel_min_sold_cars, sel_model]
    sel_values_col_filters = [column_translate['Sales_Place_Fase2_Level_1'], column_translate['Number_Cars_Sold_Local_Fase2_Level_1'], column_translate['Model_Code']]

    if '-' not in sel_values_filters and max_number_of_cars_sold:
        data_filtered = filter_data(data, sel_values_filters, sel_values_col_filters, ['in', 'ge', None])
        data_filtered = configuration_id_preparation(data_filtered)

        if data_filtered.shape[0]:
            st.write('Número de Configurações:', data_filtered.shape[0])
            sel_order_size = st.sidebar.number_input('Por favor escolha o número de viaturas a encomendar:', 1, 1000, value=50)

            for parameter in [x for x in configuration_parameters_full_rename if x != column_translate['Model_Code']]:
                sel_parameter_values = st.sidebar.multiselect('Escolha os valores para {}:'.format(parameter), [x for x in data_filtered[parameter].unique()])
                parameters_values.append(sel_parameter_values)

            data_filtered = filter_data(data_filtered, parameters_values, [x for x in configuration_parameters_full_rename if x != column_translate['Model_Code']], ['in' for x in configuration_parameters_full if x != column_translate['Model_Code']])
            status, total_value_optimized, selection = solver(data_filtered, parameter_restriction_vectors, sel_order_size)

            if status == 'optimal':
                data_filtered['Quantity'] = selection
                data_filtered.sort_values(by=column_translate['Average_Score_Euros_Local_Fase2_Level_1'], inplace=True, ascending=False)

                current_solution_size = len([x for x in selection if x > 0])
                if current_solution_size < sel_number_of_configuration:  # Checks if the optimization results is a single configuration or too few (< min_number_of_configuration)
                    complementary_configurations, complementary_configurations_index = complementary_configurations_function(data_filtered.copy(), current_solution_size, sel_number_of_configuration)
                    data_filtered.loc[data_filtered.index.isin(complementary_configurations_index), 'Quantity'] = 1

                # if saved_suggestions_df.shape[0]:
                #     st.write('Sugestões gravadas:', saved_suggestions_df)

                sel_configurations = quantity_processing(data_filtered.copy(deep=True), sel_order_size)
                df_display = sel_configurations[['Quantity'] + [x for x in configuration_parameters_full_rename if x not in column_translate['Model_Code']] + [column_translate['Number_Cars_Sold_Local_Fase2_Level_1']] + [column_translate['Number_Cars_Sold']] + [column_translate['Average_Score_Euros_Local_Fase2_Level_1']]].reset_index(drop=True).rename(columns={'Quantity': 'Quantidade'})
                st.write('Sugestão Encomenda:', df_display)

                file_export(df_display, 'Sugestão_Encomenda_{}_{}_{}'.format('_'.join(sel_locals), sel_model, current_date))
                # if st.button('Gravar Sugestão') or session_state.save_button_pressed_flag == 1:
                #     session_state.save_button_pressed_flag = 1
                #
                #     if sel_local in saved_suggestions_dict.keys() and sel_model in saved_suggestions_dict[sel_local] or session_state.overwrite_button_pressed == 1:
                #         st.write('Já existe Sugestão de Encomenda para a Concessão {} e Modelo {}. Pretende substituir pela atual sugestão?'.format(sel_local, sel_model))
                #         session_state.overwrite_button_pressed = 1
                #         if st.button('Sim'):
                #             solution_saving(sel_configurations, sel_local, sel_model)
                #             session_state.save_button_pressed_flag = 0
                #             session_state.overwrite_button_pressed = 0
                #     else:
                #         st.write('Sugestão Gravada')
                #         solution_saving(sel_configurations, sel_local, sel_model)
                #         session_state.save_button_pressed_flag = 0
                #         session_state.overwrite_button_pressed = 0

            elif status == 'infeasible':
                st.error('Não foi possível gerar uma sugestão de encomenda.')
                st.error('Por favor aumente o número de viaturas da encomenda ou reduza o valor para o(s) parâmetro(s).')

        else:
            st.error('Não foram encontrados registos para a Concessão e modelo selecionados.')

    else:
        st.write('Por favor escolha uma Concessão e Modelo.')


def solution_saving(df, sel_local, sel_model):
    # df = level_1_b_data_processing.boolean_replacement(df, boolean_columns)
    # df = level_1_b_data_processing.column_rename(df, configuration_parameters_full_rename + extra_parameters_rename, configuration_parameters_full + extra_parameters)
    #
    # level_1_e_deployment.sql_truncate(options_file.DSN_MLG, options_file, options_file.sql_info['database_final'], options_file.sql_info['optimization_solution_table'], query=truncate_query.format(sel_local, sel_model))
    #
    # level_1_e_deployment.sql_inject(df, options_file.DSN_MLG, options_file.sql_info['database_final'], options_file.sql_info['optimization_solution_table'], options_file,
    #                                 configuration_parameters_full + ['Quantity', 'Average_Score_Euros_Local_Fase2_Level_1', 'Sales_Place_Fase2_Level_1'], check_date=1)

    st.write('Sugestão gravada com sucesso - {} & {}'.format(sel_local, sel_model))
    return


def file_export(df, file_name):

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Gravar Classificações</a> (carregar botão direito e Guardar Link como: {file_name}.csv)'
    st.markdown(href, unsafe_allow_html=True)


def complementary_configurations_function(df, current_solution_size, min_number_of_configuration):
    df = df.loc[df['Quantity'] == 0, :]
    missing_number_of_configuration = min_number_of_configuration - current_solution_size
    sel_complementary_configurations = df.head(missing_number_of_configuration)
    sel_complementary_configurations_index = df.head(missing_number_of_configuration).index

    return sel_complementary_configurations, sel_complementary_configurations_index


def quantity_processing(df, sel_order_size):
    df = df.loc[df['Quantity'] > 0, :]
    total_score = df[column_translate['Average_Score_Euros_Local_Fase2_Level_1']].sum()

    df.loc[:, 'Score Weight'] = df.loc[:, column_translate['Average_Score_Euros_Local_Fase2_Level_1']] / total_score
    df.loc[:, 'Weighted Order'] = df.loc[:, 'Score Weight'] * sel_order_size
    df.loc[:, 'Quantity'] = df.loc[:, 'Weighted Order'].round()

    # What if the total value of the suggested order is not according to sel_order_size due to roundings?
    current_order_total = df['Quantity'].sum()

    if current_order_total < sel_order_size:
        first_row_index = df.head(1).index
        order_diff = sel_order_size - current_order_total
        df.loc[first_row_index, 'Quantity'] = df.loc[first_row_index, 'Quantity'] + order_diff
    elif current_order_total > sel_order_size:
        last_row_index = df[df['Quantity'] > 0].tail(1).index
        order_diff = current_order_total - sel_order_size
        df.loc[last_row_index, 'Quantity'] = df.loc[last_row_index, 'Quantity'] - order_diff

    return df.loc[df['Quantity'] > 0, :]


def get_parameter_positions(df, parameter_name, parameter_values_limit):

    color_index = df.sort_values(by=column_translate['Average_Score_Euros_Local_Fase2_Level_1'], ascending=False).reset_index().drop_duplicates(subset=parameter_name, keep='first').index.values[0:parameter_values_limit]
    color_restriction_vector = np.zeros(df.shape[0])
    color_restriction_vector = [1 if index in color_index else 0 for index, value in enumerate(color_restriction_vector)]

    return color_restriction_vector


def solver(dataset, parameter_restriction_vectors, sel_order_size):
    parameter_restriction = []

    unique_ids_count = dataset['Configuration_ID'].nunique()
    scores = dataset[column_translate['Average_Score_Euros_Local_Fase2_Level_1']].values.tolist()

    selection = cp.Variable(unique_ids_count, integer=True)
    for parameter_vector in parameter_restriction_vectors:
        parameter_restriction.append(selection >= parameter_vector)

    order_size_restriction = cp.sum(selection) <= sel_order_size
    total_value = selection * scores

    problem = cp.Problem(cp.Maximize(total_value), [selection >= 0, selection <= 100,
                                                    order_size_restriction,
                                                    ] + parameter_restriction)

    result = problem.solve(solver=cp.GLPK_MI, verbose=False, qcp=True)

    return problem.status, result, selection.value


@st.cache(show_spinner=False)
def get_data(options_file_in):
    df = level_1_a_data_acquisition.sql_retrieve_df(options_file_in.DSN_MLG, options_file_in.sql_info['database_final'], options_file_in.sql_info['final_table'], options_file_in)
    df = level_1_b_data_processing.column_rename(df, configuration_parameters_full + extra_parameters, configuration_parameters_full_rename + extra_parameters_rename)
    df = df.loc[df[column_translate['Model_Code']] != '', :]

    df = level_1_b_data_processing.boolean_replacement(df, boolean_columns)

    df = df[df[column_translate['Colour_Ext']] != 'undefined']
    df = df[df[column_translate['Colour_Int']] != '0']

    return df[configuration_parameters_full_rename + extra_parameters_rename]


def get_suggestions_dict(options_file_in):
    saved_suggestions_dict = {}
    saved_suggestions_df = level_1_a_data_acquisition.sql_retrieve_df_specified_query(options_file_in.DSN_MLG, options_file_in.sql_info['database_final'], options_file_in, query=saved_solutions_pairs_query)
    saved_suggestions_df = level_1_b_data_processing.column_rename(saved_suggestions_df, ['Model_Code', 'Sales_Place_Fase2_Level_1'], [column_translate['Model_Code'], column_translate['Sales_Place_Fase2_Level_1']])

    saved_suggestions_df_grouped = saved_suggestions_df[[column_translate['Sales_Place_Fase2_Level_1'], column_translate['Model_Code']]].groupby(column_translate['Sales_Place_Fase2_Level_1'])
    for key, group in saved_suggestions_df_grouped:
        saved_suggestions_dict[key] = list(group[column_translate['Model_Code']].values)

    return saved_suggestions_dict, saved_suggestions_df


@st.cache(show_spinner=False, allow_output_mutation=True)
def filter_data(dataset, value_filters_list, col_filters_list, operations_list):
    data_filtered = dataset.copy()

    for col_filter, filter_value, operation_value in zip(col_filters_list, value_filters_list, operations_list):
        if not operation_value:
            data_filtered = data_filtered.loc[data_filtered[col_filter] == filter_value, :]
        elif operation_value == 'gt':
            data_filtered = data_filtered.loc[data_filtered[col_filter] > filter_value, :]
        elif operation_value == 'ge':
            data_filtered = data_filtered.loc[data_filtered[col_filter] >= filter_value, :]
        elif operation_value == 'lt':
            data_filtered = data_filtered.loc[data_filtered[col_filter] < filter_value, :]
        elif operation_value == 'le':
            data_filtered = data_filtered.loc[data_filtered[col_filter] <= filter_value, :]
        elif operation_value == 'in' and len(filter_value) > 0:
            data_filtered = data_filtered.loc[data_filtered[col_filter].isin(filter_value), :]

    return data_filtered


@st.cache(show_spinner=False)
def configuration_id_preparation(df):
    df['Configuration_ID'] = df.groupby(configuration_parameters_full_rename + extra_parameters_rename).ngroup()
    df.drop_duplicates(subset='Configuration_ID', inplace=True)
    df.sort_values(by=column_translate['Average_Score_Euros_Local_Fase2_Level_1'], ascending=False, inplace=True)

    df = df[df[column_translate['Average_Score_Euros_Local_Fase2_Level_1']] > 0]

    return df


if __name__ == '__main__':
    main()
