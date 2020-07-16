import streamlit as st
import logging
import pandas as pd
import numpy as np
import cvxpy as cp
import os
import sys
import time
from traceback import format_exc
from streamlit.ScriptRunner import RerunException
from streamlit.ScriptRequestQueue import RerunData
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
sys.path.insert(1, base_path)
import modules.level_1_a_data_acquisition as level_1_a_data_acquisition
import modules.level_1_e_deployment as level_1_e_deployment
from modules.level_0_performance_report import log_record, error_upload
import level_2_order_optimization_hyundai_options as options_file
import modules.SessionState as SessionState
from level_2_order_optimization_hyundai_options import configuration_parameters, client_lvl_cols, client_lvl_cols_renamed

# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S @ %d/%m/%y', filename=options_file.log_files['full_log'], filemode='a')
# logging.Logger('errors')
# logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))

min_number_of_configuration = 5

saved_solutions_pairs_query = ' SELECT DISTINCT PT_PDB_Model_Desc, ' + ', '.join(client_lvl_cols) + ', [Date] ' \
                                'FROM [BI_DTR].[dbo].[VHE_Fact_MLG_OrderOptimization_Solver_Optimization_DTR] ' \
                                'GROUP BY PT_PDB_Model_Desc, ' + ', '.join(client_lvl_cols) + ', [Date]'

truncate_query = ''' DELETE 
FROM [BI_DTR].[dbo].[VHE_Fact_MLG_OrderOptimization_Solver_Optimization_DTR]
WHERE PT_PDB_Model_Desc = '{}'  '''

"""
# Sugestão de Encomenda - Importador
Sugestão de Configurações para a encomenda mensal de viaturas Hyundai/Honda
"""

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

url_hyperlink = '''
    <a href= "{}" > <p style="text-align:right"> Manual de Utilizador </p></a>
'''.format(options_file.documentation_url_solver_app)
st.markdown(url_hyperlink, unsafe_allow_html=True)

session_state = SessionState.get(run_id=0, overwrite_button_pressed_flag=0, save_button_pressed_flag=0, order_suggestion_button_pressed_flag=0, client_lvl_1='', client_lvl_2='', client_lvl_3='', client_lvl_4='', client_lvl_5='', client_lvl_6='', client_lvl_7='', model='', brand='')


def main():
    data = get_data(options_file)

    saved_suggestions_dict, saved_suggestions_df = get_suggestions_dict(options_file, client_lvl_cols)
    parameters_values, parameter_restriction_vectors = [], []
    max_number_of_cars_sold = max(data['Quantity_Sold'])

    sel_range = st.sidebar.radio('Apenas Gama Viva?', ['Não', 'Sim'], index=1)
    sel_brand = st.sidebar.selectbox('Marca:', ['-', 'Hyundai', 'Honda'], index=0, key=session_state.run_id)

    if '-' not in sel_brand:
        sel_model = st.sidebar.selectbox('Modelo:', ['-'] + list(data.loc[data['NLR_Code'] == options_file.nlr_code_desc[sel_brand], 'PT_PDB_Model_Desc'].unique()), index=0, key=session_state.run_id)
    else:
        sel_model = ''

    sel_order_size = st.sidebar.number_input('Por favor escolha o número de viaturas a encomendar:', 1, 1000, value=100)
    sel_min_number_of_configuration = st.sidebar.number_input('Por favor escolha o número mínimo de configurações (default={}):'.format(min_number_of_configuration), 1, 100, value=min_number_of_configuration)
    sel_min_sold_cars = st.sidebar.number_input('Por favor escolha um valor mínimo de viaturas vendidas por configuração (valor máximo é de {}):'.format(max_number_of_cars_sold), 1, max_number_of_cars_sold, value=1)

    sel_client_lvl_1 = st.sidebar.selectbox('Tipo Cliente:', ['-'] + list(data['Customer_Group_Desc'].unique()), index=0)
    sel_client_lvl_2 = st.sidebar.selectbox('Agrupamento NIF:', ['-'] + list(data['NDB_VATGroup_Desc'].unique()), index=0)
    sel_client_lvl_3 = st.sidebar.selectbox('NIF - Nome:', ['-'] + list(data['VAT_Number_Display'].unique()), index=0)
    sel_client_lvl_4 = st.sidebar.selectbox('Contrato Concessionário:', ['-'] + list(data['NDB_Contract_Dealer_Desc'].unique()), index=0)
    sel_client_lvl_5 = st.sidebar.selectbox('Agrupamento Performance:', ['-'] + list(data['NDB_VHE_PerformGroup_Desc'].unique()), index=0)
    sel_client_lvl_6 = st.sidebar.selectbox('Equipa de Vendas:', ['-'] + list(data['NDB_VHE_Team_Desc'].unique()), index=0)
    sel_client_lvl_7 = st.sidebar.selectbox('Cliente Morada:', ['-'] + list(data['Customer_Display'].unique()), index=0)

    if sel_model != session_state.model or sel_client_lvl_1 != session_state.client_lvl_1 or sel_client_lvl_2 != session_state.client_lvl_2 or sel_client_lvl_3 != session_state.client_lvl_3 or sel_client_lvl_4 != session_state.client_lvl_4 or sel_client_lvl_5 != session_state.client_lvl_5 or sel_client_lvl_6 != session_state.client_lvl_6 or sel_client_lvl_7 != session_state.client_lvl_7:
        session_state.client_lvl_1 = sel_client_lvl_1
        session_state.client_lvl_2 = sel_client_lvl_2
        session_state.client_lvl_3 = sel_client_lvl_3
        session_state.client_lvl_4 = sel_client_lvl_4
        session_state.client_lvl_5 = sel_client_lvl_5
        session_state.client_lvl_6 = sel_client_lvl_6
        session_state.client_lvl_7 = sel_client_lvl_7
        session_state.model = sel_model
        session_state.overwrite_button_pressed, session_state.save_button_pressed_flag, session_state.order_suggestion_button_pressed_flag = 0, 0, 0

    client_lvl_values = [sel_client_lvl_1, sel_client_lvl_2, sel_client_lvl_3, sel_client_lvl_4, sel_client_lvl_5, sel_client_lvl_6, sel_client_lvl_7]

    if '-' not in [sel_model] and '-' not in [sel_brand]:
        data_filtered = filter_data(data, [sel_range, sel_model, sel_min_sold_cars] + client_lvl_values, ['Gama_Viva_Flag', 'PT_PDB_Model_Desc', 'Quantity_Sold'] + client_lvl_cols)

        if not data_filtered.shape[0]:
            st.write('Não foram encontrados registos para as presentes escolhas - Por favor altere o modelo/cliente/valor mínimo de viaturas por configuração.')
            return
        st.write('Número de Configurações:', data_filtered['ML_VehicleData_Code'].nunique())

        for parameter in [x for x in configuration_parameters if x not in 'PT_PDB_Model_Desc']:
            sel_parameter_values = st.sidebar.multiselect('Escolha os valores para {}:'.format(options_file.column_translate_dict[parameter]), [x for x in data_filtered[parameter].unique()])
            parameters_values.append(sel_parameter_values)

        # for parameter, parameter_value in zip([x for x in configuration_parameters if x not in 'PT_PDB_Model_Desc'], parameters_values):
        #     if parameter != '-':
        data_filtered = filter_data(data_filtered, parameters_values, [x for x in configuration_parameters if x not in 'PT_PDB_Model_Desc'])

        # parameter_restriction_vectors.append(get_parameter_positions(data_filtered.copy(), parameter, parameter_value))

        if st.button('Criar Sugestão') or session_state.order_suggestion_button_pressed_flag == 1:
            if any(x != '-' for x in client_lvl_values):
                proposals_col = 'Proposals_Count'
                stock_col = 'Stock_Count'
            else:
                proposals_col = 'Proposals_Count_VDC'
                stock_col = 'Stock_Count_VDC'

            session_state.order_suggestion_button_pressed_flag = 1
            status, total_value_optimized, selection, selection_configuration_ids = solver(data_filtered, parameter_restriction_vectors, sel_order_size)
            data_filtered['Quantity'] = selection

            if status == 'optimal':
                sel_configurations = quantity_processing(data_filtered.copy(deep=True), sel_order_size, proposals_col, stock_col, sel_min_number_of_configuration)
                if sel_configurations.shape[0]:
                    sel_configurations.rename(index=str, columns={'Quantity': 'Sug.Encomenda'}, inplace=True)  # ToDo: For some reason this column in particular is not changing its name by way of the renaming argument in the previous st.write. This is a temporary solution
                    st.write('Sugestão Encomenda:', sel_configurations[['Sug.Encomenda'] + [proposals_col] + [stock_col] + [x for x in configuration_parameters if x not in 'PT_PDB_Model_Desc'] + ['Quantity_Sold'] + ['Average_Score_Euros']]
                             .rename(columns=options_file.column_translate_dict).reset_index(drop=True)
                             .style.format({'Score (€)': '{:.2f}', 'Sug.Encomenda': '{:.0f}', 'Propostas Entregues': '{:.0f}', 'Em Stock': '{:.0f}'})
                             )

                    st.write('Propostas Entregues para esta sugestão: ', int(sel_configurations[proposals_col].sum()))
                    st.write('Viaturas em Stock para esta sugestão: ', int(sel_configurations[stock_col].sum()))
                    # st.write('CONTROL VALUE - Qty Total', int(sel_configurations['Sug.Encomenda'].sum()))
                    # st.write('CONTROL VALUE - QTY TOTAL + PROPOSTAS: {}'.format(int(sel_configurations['Sug.Encomenda'].sum()) + int(sel_configurations[proposals_col].sum())))
                    sel_configurations.rename(index=str, columns={'Sug.Encomenda': 'Quantity'}, inplace=True)  # ToDo: For some reason this column in particular is not changing its name by way of the renaming argument in the previous st.write. This is a temporary solution
                else:
                    return

                if st.button('Gravar Sugestão') or session_state.save_button_pressed_flag == 1:
                    session_state.save_button_pressed_flag = 1

                    if tuple(client_lvl_values) in saved_suggestions_dict.keys() and sel_model in saved_suggestions_dict[tuple(client_lvl_values)] or session_state.overwrite_button_pressed == 1:
                        st.write('Já existe Sugestão de Encomenda para o Modelo {}'.format(sel_model) + ' '.join([' e {} - {}'.format(x, y) for x, y in zip(client_lvl_cols_renamed, client_lvl_values) if y != '-']) + '.')
                        st.write('Pretende substituir pela atual sugestão?')
                        session_state.overwrite_button_pressed = 1
                        if st.button('Sim'):
                            solution_saving(sel_configurations, sel_model, client_lvl_cols, client_lvl_values)
                            session_state.save_button_pressed_flag = 0
                            session_state.overwrite_button_pressed = 0
                    else:
                        solution_saving(sel_configurations, sel_model, client_lvl_cols, client_lvl_values)
                        session_state.save_button_pressed_flag = 0
                        session_state.overwrite_button_pressed = 0

            elif status == 'infeasible':
                st.write('Não foi possível gerar uma sugestão de encomenda.')

    else:
        st.write('Por favor escolha uma marca e modelo para sugerir a respetiva encomenda.')


@st.cache
def get_data(options_file_in):
    df_cols = ['NLR_Code', 'Chassis_Number', 'Registration_Number', 'PDB_Start_Order_Date', 'PDB_End_Order_Date', 'VehicleData_Code', 'DaysInStock_Distributor', 'DaysInStock_Global', 'Measure_9', 'Fixed_Margin_II', 'NDB_VATGroup_Desc', 'VAT_Number_Display', 'NDB_Contract_Dealer_Desc', 'NDB_VHE_PerformGroup_Desc', 'NDB_VHE_Team_Desc', 'Customer_Display', 'Customer_Group_Desc', 'NDB_Dealer_Code', 'Quantity_Sold', 'Average_DaysInStock_Global', 'PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc', 'PT_PDB_Exterior_Color_Desc', 'PT_PDB_Interior_Color_Desc', 'ML_VehicleData_Code']
    df = level_1_a_data_acquisition.sql_retrieve_df(options_file_in.DSN, options_file_in.sql_info['database_source'], options_file_in.sql_info['final_table'], options_file_in, columns=df_cols, query_filters={'Customer_Group_Desc': ['Direct', 'Dealers', 'Not Defined']})

    df_pdb_cols = ['VehicleData_Code', 'PT_PDB_Model_Desc', 'PT_PDB_Serie_Desc', 'PT_PDB_Bodywork_Desc', 'PT_PDB_Version_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Exterior_Color_Desc', 'PT_PDB_Interior_Color_Desc', 'PT_PDB_Painting_Type_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Fuel_Type_Desc', 'PT_PDB_Vehicle_Type_Desc', 'PT_PDB_Commercial_Version_Desc', 'PDB_Start_Order_Date', 'PDB_End_Order_Date', 'PT_PDB_Version_Desc_New', 'PT_PDB_Engine_Desc_New', 'PT_PDB_Commercial_Version_Desc_New']
    df_pdb = level_1_a_data_acquisition.sql_retrieve_df(options_file_in.DSN, options_file_in.sql_info['database_source'], options_file_in.sql_info['product_db'], options_file_in, columns=df_pdb_cols)
    df_proposals = level_1_a_data_acquisition.sql_retrieve_df(options_file_in.DSN, options_file_in.sql_info['database_source'], options_file_in.sql_info['proposals_view'], options_file_in)
    df_stock = level_1_a_data_acquisition.sql_retrieve_df(options_file_in.DSN, options_file_in.sql_info['database_source'], options_file_in.sql_info['stock_view'], options_file_in)

    proposals_grouped = df_proposals[['VehicleData_Code', 'Proposals_Count']].copy(deep=True)
    proposals_grouped['Proposals_Count_VDC'] = proposals_grouped.groupby('VehicleData_Code')['Proposals_Count'].transform('sum')
    proposals_grouped = proposals_grouped.drop(['Proposals_Count'], axis=1).drop_duplicates()

    stock_grouped = df_stock[['VehicleData_Code', 'Stock_Count']].copy(deep=True)
    stock_grouped['Stock_Count_VDC'] = stock_grouped.groupby('VehicleData_Code')['Stock_Count'].transform('sum')
    stock_grouped = stock_grouped.drop(['Stock_Count'], axis=1).drop_duplicates()

    df_pdb['PDB_End_Order_Date'] = pd.to_datetime(df_pdb['PDB_End_Order_Date'], format='%Y-%m-%d', errors='ignore')
    current_date, _ = level_1_e_deployment.time_tags(format_date='%Y-%m-%d')

    df, sel_vehicledata_codes = gamas_selection(df, df_pdb, current_date)
    gama_viva_mask = df['VehicleData_Code'].isin(sel_vehicledata_codes)

    df['Custo_Stock_Dist'] = df['Measure_9'] * 0.015 / 365 * df['DaysInStock_Distributor'] * (-1)
    df['Score_Euros'] = df['Fixed_Margin_II'] - df['Custo_Stock_Dist']

    df_grouped = df.groupby('ML_VehicleData_Code')
    df['Average_DaysInStock_Global'] = df_grouped['DaysInStock_Global'].transform('mean')
    df['Quantity_Sold'] = df_grouped['Chassis_Number'].transform('nunique')
    df['Average_Score_Euros'] = df_grouped['Score_Euros'].transform('mean')
    df['Gama_Viva_Flag'] = np.where(gama_viva_mask, "Sim", "Não")

    df['NDB_Dealer_Code_alt'] = df['NDB_Dealer_Code'].str.replace(r'[A-Z]$', '')
    df = pd.merge(df, proposals_grouped, left_on=['VehicleData_Code'], right_on=['VehicleData_Code'], how='left').fillna(0)
    df = pd.merge(df, df_proposals, left_on=['VehicleData_Code', 'NDB_Dealer_Code_alt'], right_on=['VehicleData_Code', 'NDB_Installation_Code'], how='left').fillna(0)
    df = pd.merge(df, stock_grouped, left_on=['VehicleData_Code'], right_on=['VehicleData_Code'], how='left').fillna(0)
    df = pd.merge(df, df_stock, left_on=['VehicleData_Code', 'NDB_Dealer_Code'], right_on=['VehicleData_Code', 'NDB_Dealer_Code'], how='left').fillna(0)

    for model in df['PT_PDB_Model_Desc'].unique():
        if model == 'cr-v':
            df.loc[df['PT_PDB_Model_Desc'] == model, 'PT_PDB_Model_Desc'] = 'CR-V'
        elif model == 'hr-v':
            df.loc[df['PT_PDB_Model_Desc'] == model, 'PT_PDB_Model_Desc'] = 'HR-V'
        elif model == 'ioniq':
            df.loc[df['PT_PDB_Model_Desc'] == model, 'PT_PDB_Model_Desc'] = 'Ioniq'
        elif model[0] != 'i':
            df.loc[df['PT_PDB_Model_Desc'] == model, 'PT_PDB_Model_Desc'] = model.capitalize()

    df = df[~df['PT_PDB_Model_Desc'].isin(['i20 van', 'i20 coupe', 'i30 fastback n', 'i40'])]
    df = df[df['Average_Score_Euros'] > 0]
    return df


@st.cache
def gamas_selection(df, df_pdb, current_date):

    # Client Criteria
    gama_viva_mask = df_pdb['PDB_End_Order_Date'] >= current_date
    gama_viva_mask_2 = df_pdb['PDB_End_Order_Date'].isnull()
    gama_viva_mask_3 = df_pdb['PDB_Start_Order_Date'].notnull()
    sel_vehicledata_codes = df_pdb.loc[gama_viva_mask_3 & gama_viva_mask | gama_viva_mask_3 & gama_viva_mask_2]['VehicleData_Code'].unique()

    # Matchup Criteria - Search for the old gamas which have already been matched
    df_pdb = df_pdb.loc[df_pdb['PT_PDB_Commercial_Version_Desc_New'].notnull()]
    df_pdb = df_pdb.loc[df_pdb['PT_PDB_Version_Desc_New'].notnull()]
    sel_vehicledata_codes_matchup = df_pdb['VehicleData_Code'].unique()

    df_updated = update_new_gamas(df, df_pdb)

    return df_updated, list(sel_vehicledata_codes) + list(sel_vehicledata_codes_matchup)


def update_new_gamas(df, df_pdb):
    start = time.time()
    # When a gama is replaced, sometimes it changes its characteristics. To handle such cases, they will be updated in the original df
    df_pdb[configuration_parameters + ['PT_PDB_Engine_Desc_New', 'PT_PDB_Version_Desc_New']] = df_pdb[configuration_parameters + ['PT_PDB_Engine_Desc_New', 'PT_PDB_Version_Desc_New']].apply(lambda x: x.astype(str).str.lower())

    df_pdb.drop(['PT_PDB_Version_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Exterior_Color_Desc', 'PT_PDB_Interior_Color_Desc'], axis=1, inplace=True)
    df_pdb.rename(columns={'PT_PDB_Version_Desc_New': 'PT_PDB_Version_Desc', 'PT_PDB_Engine_Desc_New': 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc_New': 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Exterior_Color_Desc_New': 'PT_PDB_Exterior_Color_Desc', 'PT_PDB_Interior_Color_Desc_New': 'PT_PDB_Interior_Color_Desc'}, inplace=True)
    df_pdb.set_index('VehicleData_Code', inplace=True)
    df_pdb_version = df_pdb[['PT_PDB_Version_Desc', 'PT_PDB_Engine_Desc']]

    df.set_index('VehicleData_Code', inplace=True)
    df.update(df_pdb_version)
    df.reset_index(inplace=True)

    print('Elapsed time update new gamas {:.2f}'.format(time.time() - start))
    return df


def filter_data(dataset, value_filters_list, col_filters_list):
    data_filtered = dataset.copy()

    for col_filter, filter_value in zip(col_filters_list, value_filters_list):
        if filter_value != '-' and type(filter_value) == list and len(filter_value) > 0:
            data_filtered = data_filtered.loc[data_filtered[col_filter].isin(filter_value), :]

        if filter_value != '-' and type(filter_value) != list:
            if col_filter == 'Quantity_Sold':
                data_filtered = data_filtered.loc[data_filtered[col_filter].ge(filter_value), :]
            elif col_filter == 'Gama_Viva_Flag' and filter_value == 'Não':
                continue
            elif filter_value != '-':
                data_filtered = data_filtered.loc[data_filtered[col_filter] == filter_value, :]

    # st.write('Número de viaturas vendidas:', data_filtered['Registration_Number'].nunique())
    data_filtered.drop_duplicates(subset='ML_VehicleData_Code', inplace=True)
    data_filtered.sort_values(by='Average_Score_Euros', ascending=False, inplace=True)
    return data_filtered


def solver(dataset, parameter_restriction_vectors, sel_order_size):
    parameter_restriction = []

    unique_ids_count = dataset['ML_VehicleData_Code'].nunique()
    unique_ids = dataset['ML_VehicleData_Code'].unique()

    scores_values = [dataset[dataset['ML_VehicleData_Code'] == x]['Average_Score_Euros'].head(1).values[0] for x in unique_ids]  # uniques() command doesn't work as intended because there are configurations (Configuration IDs) with repeated average score

    selection = cp.Variable(unique_ids_count, integer=True)

    order_size_restriction = cp.sum(selection) <= sel_order_size
    total_value = selection * scores_values

    problem = cp.Problem(cp.Maximize(total_value), [selection >= 0, selection <= 100,
                                                    order_size_restriction,
                                                    ] + parameter_restriction)

    result = problem.solve(solver=cp.GLPK_MI, verbose=False, qcp=True)

    return problem.status, result, selection.value, unique_ids


def get_parameter_positions(df, parameter_name, parameter_values_limit):

    color_index = df.sort_values(by='Average_Score_Euros', ascending=False).reset_index().drop_duplicates(subset=parameter_name, keep='first').index.values[0:parameter_values_limit]
    color_restriction_vector = np.zeros(df.shape[0])
    color_restriction_vector = [1 if index in color_index else 0 for index, value in enumerate(color_restriction_vector)]

    return color_restriction_vector


def complementary_configurations_function(df, current_solution_size, sel_min_number_of_configuration):
    df = df.loc[df['Quantity'] == 0, :]
    missing_number_of_configuration = sel_min_number_of_configuration - current_solution_size
    sel_complementary_configurations = df.head(missing_number_of_configuration)
    sel_complementary_configurations_index = df.head(missing_number_of_configuration).index

    return sel_complementary_configurations, sel_complementary_configurations_index


def quantity_processing(df, sel_order_size, proposal_col, stock_col, sel_min_number_of_configuration):
    df = df.head(sel_min_number_of_configuration)
    proposals_total = df[proposal_col].sum()
    stock_total = df[stock_col].sum()

    if proposals_total >= sel_order_size:
        st.error('Aviso: Existem mais ou igual número de propostas ({}) do que viaturas a encomendar ({}). Por favor alterar alguma dos parâmetros como nº mínimo de configurações a mostrar, valor mínimo de viaturas vendidas ou cliente.'.format(int(proposals_total), int(sel_order_size)))
        return pd.DataFrame()

    sel_order_size = sel_order_size - proposals_total
    total_score = df['Average_Score_Euros'].sum()

    df.loc[:, 'Score Weight'] = df.loc[:, 'Average_Score_Euros'] / total_score
    df.loc[:, 'Weighted Order'] = df.loc[:, 'Score Weight'] * (sel_order_size - stock_total)
    df.loc[:, 'Quantity'] = df.loc[:, 'Weighted Order'].round()

    # Stock Handling
    df_wo_stock_values = df.loc[df[stock_col] == 0, :]
    df_wo_stock_values_index = df.loc[df[stock_col] == 0, :].index
    df_wo_stock_values.loc[:, 'Score Weight Stock'] = df_wo_stock_values.loc[:, 'Average_Score_Euros'] / total_score
    df_wo_stock_values.loc[:, 'Weighted Order Stock'] = df_wo_stock_values.loc[:, 'Score Weight Stock'] * stock_total
    df_wo_stock_values.loc[:, 'Quantity'] = df_wo_stock_values.loc[:, 'Weighted Order Stock'].round()
    df.loc[df.index.isin(df_wo_stock_values_index), 'Quantity'] = df.loc[df.index.isin(df_wo_stock_values_index), 'Quantity'] + df_wo_stock_values['Quantity']

    # What if the total value of the suggested order is not according to sel_order_size due to roundings?
    current_order_total = df['Quantity'].sum()

    if current_order_total < sel_order_size:
        order_diff = sel_order_size - current_order_total

        df.loc[:, 'Score Weight Tunning'] = df.loc[:, 'Average_Score_Euros'] / total_score
        df.loc[:, 'Weighted Order Tunning'] = df.loc[:, 'Score Weight Tunning'] * order_diff
        df.loc[:, 'Quantity Tunning'] = df.loc[:, 'Weighted Order Tunning'].round()
        df.loc[:, 'Quantity'] = df.loc[:, 'Quantity'] + df.loc[:, 'Quantity Tunning']

        # first_row_index = df.head(1).index
        # df.loc[first_row_index, 'Quantity'] = df.loc[first_row_index, 'Quantity'] + order_diff
    elif current_order_total > sel_order_size:
        order_diff = sel_order_size - current_order_total

        df.loc[:, 'Score Weight Tunning'] = df.loc[:, 'Average_Score_Euros'] / total_score
        df.loc[:, 'Weighted Order Tunning'] = df.loc[:, 'Score Weight Tunning'] * order_diff
        df.loc[:, 'Quantity Tunning'] = df.loc[:, 'Weighted Order Tunning'].round()
        df.loc[:, 'Quantity'] = df.loc[:, 'Quantity'] - df.loc[:, 'Quantity Tunning']

        # last_row_index = df[df['Quantity'] > 0].tail(1).index
        # order_diff = current_order_total - sel_order_size
        # df.loc[last_row_index, 'Quantity'] = df.loc[last_row_index, 'Quantity'] - order_diff

    return df


def get_suggestions_dict(options_file_in, client_lvl_cols_in):
    saved_suggestions_dict = {}

    saved_suggestions_df = level_1_a_data_acquisition.sql_retrieve_df_specified_query(options_file_in.DSN, options_file_in.sql_info['database_source'], options_file_in, query=saved_solutions_pairs_query)

    saved_suggestions_df_grouped = saved_suggestions_df[['PT_PDB_Model_Desc'] + client_lvl_cols_in].groupby(client_lvl_cols_in)
    for key, group in saved_suggestions_df_grouped:
        saved_suggestions_dict[key] = list(group['PT_PDB_Model_Desc'].values)

    return saved_suggestions_dict, saved_suggestions_df


def solution_saving(df, sel_model, client_lvl_cols_in, client_lvl_sels):
    truncate_query_part_2 = ' '.join(['and {} = \'{}\''.format(x, y) for x, y in zip(client_lvl_cols_in, client_lvl_sels) if y != '-'])

    df = client_replacement(df, client_lvl_cols_in, client_lvl_sels)  # Replaces the values of Client's Levels by the actual values selected for this solution

    level_1_e_deployment.sql_truncate(options_file.DSN, options_file, options_file.sql_info['database_source'], options_file.sql_info['optimization_solution_table'], query=truncate_query.format(sel_model) + truncate_query_part_2)

    level_1_e_deployment.sql_inject(df, options_file.DSN, options_file.sql_info['database_source'], options_file.sql_info['optimization_solution_table'], options_file,
                                    configuration_parameters + client_lvl_cols_in + ['Quantity', 'Average_Score_Euros', 'ML_VehicleData_Code'], check_date=1)

    st.write('Sugestão gravada com sucesso.')
    return


def client_replacement(df, client_lvl_cols_in, client_lvl_sels):
    # This function is needed because the configurations selected by the solver as solution have their own client's level values.
    # But these are not the values that should be saved with the solution, but instead the actual selected values by the user.

    for client_lvl, client_lvl_sel in zip(client_lvl_cols_in, client_lvl_sels):
        df.loc[:, client_lvl] = client_lvl_sel

    return df


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
        raise RerunException(RerunData(widget_state=None))

