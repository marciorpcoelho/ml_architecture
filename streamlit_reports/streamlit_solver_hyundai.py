import streamlit as st
# import pandas as pd
import numpy as np
import cvxpy as cp
import os
import sys
import time
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')) + '\\'
sys.path.insert(1, base_path)
import modules.level_1_a_data_acquisition as level_1_a_data_acquisition
import modules.level_1_b_data_processing as level_1_b_data_processing
import modules.level_1_e_deployment as level_1_e_deployment
import level_2_order_optimization_hyundai_options as options_file
import modules.SessionState as sessionstate

configuration_parameters = ['PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc', 'PT_PDB_Exterior_Color_Desc', 'PT_PDB_Interior_Color_Desc']
min_number_of_configuration = 5
saved_solutions_pairs_query = ''' SELECT DISTINCT Customer_Group_Desc, PT_PDB_Model_Desc, [Date]
  FROM [BI_MLG].[dbo].[VHE_Fact_BI_OrderOptimization_Solver_Optimization_DTR]
  GROUP BY Customer_Group_Desc, PT_PDB_Model_Desc, [Date]'''
truncate_query = '''DELETE 
FROM [BI_MLG].[dbo].[VHE_Fact_BI_OrderOptimization_Solver_Optimization_DTR]
WHERE Customer_Group_Desc = '{}' and PT_PDB_Model_Desc = '{}' '''

"""
# Sugestão de Encomenda - Importador
Sugestão de Configurações para a encomenda mensal de viaturas Hyundai/Honda
"""

session_state = sessionstate.get(overwrite_button_pressed=0, save_button_pressed_flag=0, client_lvl_1='', model='')


def main():
    data = get_data(options_file)

    st.write(data.head())

    saved_suggestions_dict, saved_suggestions_df = get_suggestions_dict(options_file)
    parameters_values, parameter_restriction_vectors = [], []
    max_number_of_cars_sold = max(data['Quantity_Sold'])

    sel_model = st.sidebar.selectbox('Modelo:', ['-'] + list(data['PT_PDB_Model_Desc'].unique()), index=0)
    sel_client = st.sidebar.selectbox('Cliente:', ['-'] + list(data['Customer_Group_Desc'].unique()), index=0)
    sel_order_size = st.sidebar.number_input('Por favor escolha o número de viaturas a encomendar:', 1, 1000, value=100)
    sel_min_sold_cars = st.sidebar.number_input('Por favor escolha um valor mínimo de viaturas vendidas por configuração (valor máximo é de {}):'.format(max_number_of_cars_sold), 1, max_number_of_cars_sold, value=1)

    if sel_client != session_state.client_lvl_1 or sel_model != session_state.model:
        session_state.client_lvl_1 = sel_client
        session_state.model = sel_model
        session_state.overwrite_button_pressed, session_state.save_button_pressed_flag = 0, 0

    # if '-' not in [sel_client, sel_model]:
    #     if data[(data['Customer_Group_Desc'] == sel_client) & (data['PT_PDB_Model_Desc'] == sel_model)].shape[0]:
    #         max_number_of_cars_sold = max(data[(data['Customer_Group_Desc'] == sel_client) & (data['PT_PDB_Model_desc'] == sel_model)]['Quantity_Sold'])
    #     else:
    #         st.write('Não foram encontrados registos para a Concessão e modelo selecionados.')
    #         return

    if '-' not in [sel_model, sel_client]:
        data_filtered = filter_data(data, [sel_model, sel_min_sold_cars], ['PT_PDB_Model_Desc', 'Quantity_Sold'])
        st.write('Número de Configurações:', data_filtered.shape[0])

        parameters = st.multiselect('Escolha os parâmetros da configuração que pretende configurar:', [x for x in configuration_parameters if x not in 'PT_PDB_Model_Desc'] + ['Customer_Group_Desc'])

        for parameter in parameters:
            sel_parameter_max_number = data_filtered[parameter].nunique()
            try:
                sel_parameter_value = st.sidebar.number_input('Por favor escolha o número mínimo de diferentes {} a escolher (valor mínimo é {} e o valor máximo é {})'.format(parameter, 1, sel_parameter_max_number), 1, sel_parameter_max_number, value=sel_parameter_max_number - 1)
            except ValueError:
                sel_parameter_value = st.sidebar.number_input('O número de {} disponíveis values é de apenas {}'.format(parameter, sel_parameter_max_number), 1, sel_parameter_max_number, value=sel_parameter_max_number)

            parameters_values.append(sel_parameter_value)

        for parameter, parameter_value in zip(parameters, parameters_values):
            st.write(parameter, parameter_value)
            parameter_restriction_vectors.append(get_parameter_positions(data_filtered.copy(), parameter, parameter_value))

        status, total_value_optimized, selection, selection_configuration_ids = solver(data_filtered, parameter_restriction_vectors, sel_order_size)

        data_filtered['Quantity'] = selection

        if status == 'optimal':
            current_solution_size = len([x for x in selection if x > 0])

            if current_solution_size < min_number_of_configuration:
                complementary_configurations, complementary_configurations_index = complementary_configurations_function(data_filtered.copy(), current_solution_size)
                data_filtered.loc[data_filtered.index.isin(complementary_configurations_index), 'Quantity'] = 1

            if saved_suggestions_df.shape[0]:
                st.write('Sugestões gravadas:', saved_suggestions_df)

            sel_configurations = quantity_processing(data_filtered.copy(deep=True), sel_order_size)
            st.write('Sugestão Encomenda:', sel_configurations[['Quantity'] + [x for x in configuration_parameters if x not in 'PT_PDB_Model_Desc'] + ['Quantity_Sold'] + ['Average_Score_Euros']])

            if st.button('Gravar Sugestão') or session_state.save_button_pressed_flag == 1:
                session_state.save_button_pressed_flag = 1

                if sel_client in saved_suggestions_dict.keys() and sel_model in saved_suggestions_dict[sel_client] or session_state.overwrite_button_pressed == 1:
                    st.write('Já existe Sugestão de Encomenda para a Concessão {} e Modelo {}. Pretende substituir pela atual sugestão?'.format(sel_client, sel_model))
                    session_state.overwrite_button_pressed = 1
                    if st.button('Sim'):
                        solution_saving(sel_configurations, sel_client, sel_model)
                        session_state.save_button_pressed_flag = 0
                        session_state.overwrite_button_pressed = 0
                else:
                    st.write('Sugestão Gravada')
                    solution_saving(sel_configurations, sel_client, sel_model)
                    session_state.save_button_pressed_flag = 0
                    session_state.overwrite_button_pressed = 0

        elif status == 'infeasible':
            st.write('Não foi possível gerar uma sugestão de encomenda.')


@st.cache
def get_data(options_file_in):
    start_get_data_function = time.time()

    df = level_1_a_data_acquisition.sql_retrieve_df(options_file_in.DSN_MLG, options_file_in.sql_info['database_final'], options_file_in.sql_info['final_table'], options_file_in)
    df['Custo_Stock_Dist'] = df['Measure_9'] * 0.015 / 365 * df['DaysInStock_Distributor'] * (-1)
    df['Score_Euros'] = df['Fixed_Margin_II'] - df['Custo_Stock_Dist']

    df_grouped = df.groupby('ML_VehicleData_Code')
    df['Average_DaysInStock_Global'] = df_grouped['DaysInStock_Global'].transform('mean')
    df['Quantity_Sold'] = df_grouped['Chassis_Number'].transform('nunique')
    df['Average_Score_Euros'] = df_grouped['Score_Euros'].transform('mean')

    for model in df['PT_PDB_Model_Desc'].unique():
        if model == 'cr-v':
            df.loc[df['PT_PDB_Model_Desc'] == model, 'PT_PDB_Model_Desc'] = 'CR-V'
        elif model == 'hr-v':
            df.loc[df['PT_PDB_Model_Desc'] == model, 'PT_PDB_Model_Desc'] = 'HR-V'
        elif model == 'ioniq':
            df.loc[df['PT_PDB_Model_Desc'] == model, 'PT_PDB_Model_Desc'] = 'Ioniq'
        elif model[0] != 'i':
            df.loc[df['PT_PDB_Model_Desc'] == model, 'PT_PDB_Model_Desc'] = model.capitalize()

    print('Elapsed time for the get data: {:.2f} seconds.'.format(time.time() - start_get_data_function))
    return df


def filter_data(dataset, value_filters_list, col_filters_list):
    data_filtered = dataset.copy()

    for col_filter, filter_value in zip(col_filters_list, value_filters_list):
        if col_filter == 'Quantity_Sold':
            data_filtered = data_filtered.loc[data_filtered[col_filter].ge(filter_value), :]
        else:
            data_filtered = data_filtered.loc[data_filtered[col_filter] == filter_value, :]

    data_filtered.drop_duplicates(subset='ML_VehicleData_Code', inplace=True)
    data_filtered.sort_values(by='Average_Score_Euros', ascending=False, inplace=True)
    return data_filtered


def solver(dataset, parameter_restriction_vectors, sel_order_size):
    start_solver = time.time()
    parameter_restriction = []

    unique_ids_count = dataset['ML_VehicleData_Code'].nunique()
    unique_ids = dataset['ML_VehicleData_Code'].unique()
    scores = dataset['Average_Score_Euros'].unique()

    selection = cp.Variable(unique_ids_count, integer=True)
    for parameter_vector in parameter_restriction_vectors:
        parameter_restriction.append(selection >= parameter_vector)

    order_size_restriction = cp.sum(selection) <= sel_order_size
    total_value = selection * scores

    problem = cp.Problem(cp.Maximize(total_value), [selection >= 0, selection <= 100,
                                                    order_size_restriction,
                                                    ] + parameter_restriction)

    result = problem.solve(solver=cp.GLPK_MI, verbose=False, qcp=True)

    # selection_configuration_ids = [x for (x, y) in zip(dataset['ML_VehicleData_Code'].unique(), selection.value) if y > 0]

    print('Elapsed time for the solver: {:.2f} seconds.'.format(time.time() - start_solver))

    return problem.status, result, selection.value, unique_ids


def get_parameter_positions(df, parameter_name, parameter_values_limit):

    color_index = df.sort_values(by='Average_Score_Euros', ascending=False).reset_index().drop_duplicates(subset=parameter_name, keep='first').index.values[0:parameter_values_limit]
    color_restriction_vector = np.zeros(df.shape[0])
    color_restriction_vector = [1 if index in color_index else 0 for index, value in enumerate(color_restriction_vector)]

    return color_restriction_vector


def complementary_configurations_function(df, current_solution_size):
    df = df.loc[df['Quantity'] == 0, :]
    missing_number_of_configuration = min_number_of_configuration - current_solution_size
    sel_complementary_configurations = df.head(missing_number_of_configuration)
    sel_complementary_configurations_index = df.head(missing_number_of_configuration).index

    return sel_complementary_configurations, sel_complementary_configurations_index


def quantity_processing(df, sel_order_size):
    df = df.loc[df['Quantity'] > 0, :]
    total_score = df['Average_Score_Euros'].sum()

    df.loc[:, 'Score Weight'] = df.loc[:, 'Average_Score_Euros'] / total_score
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


def get_suggestions_dict(options_file_in):
    saved_suggestions_dict = {}
    saved_suggestions_df = level_1_a_data_acquisition.sql_retrieve_df_specified_query(options_file_in.DSN_MLG, options_file_in.sql_info['database_final'], options_file_in, query=saved_solutions_pairs_query)
    # saved_suggestions_df = level_1_b_data_processing.column_rename(saved_suggestions_df, ['PT_PDB_Model_Desc', 'Customer_Group_Desc'], ['PT_PDB_Model_Desc', 'Customer_Group_Desc'])

    saved_suggestions_df_grouped = saved_suggestions_df[['Customer_Group_Desc', 'PT_PDB_Model_Desc']].groupby('Customer_Group_Desc')
    for key, group in saved_suggestions_df_grouped:
        saved_suggestions_dict[key] = list(group['PT_PDB_Model_Desc'].values)

    return saved_suggestions_dict, saved_suggestions_df


def solution_saving(df, sel_local, sel_model):
    # df = level_1_b_data_processing.boolean_replacement(df, boolean_columns)
    # df = level_1_b_data_processing.column_rename(df, configuration_parameters_full_rename + extra_parameters_rename, configuration_parameters_full + extra_parameters)

    level_1_e_deployment.sql_truncate(options_file.DSN_MLG, options_file, options_file.sql_info['database_final'], options_file.sql_info['optimization_solution_table'], query=truncate_query.format(sel_local, sel_model))

    level_1_e_deployment.sql_inject(df, options_file.DSN_MLG, options_file.sql_info['database_final'], options_file.sql_info['optimization_solution_table'], options_file,
                                    configuration_parameters + ['Quantity', 'Average_Score_Euros', 'Customer_Group_Desc'], check_date=1)

    st.write('Sugestão gravada com sucesso - {} & {}'.format(sel_local, sel_model))
    return


if __name__ == '__main__':
    main()

