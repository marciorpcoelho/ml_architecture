import streamlit as st
import requests
import json
import logging
import pandas as pd
import numpy as np
import cvxpy as cp
import os
import sys
import sys
import time
from datetime import date
from traceback import format_exc
from streamlit.script_runner import RerunException
from streamlit.script_request_queue import RerunData
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
sys.path.insert(1, base_path)
import modules.level_1_a_data_acquisition as level_1_a_data_acquisition
import modules.level_1_e_deployment as level_1_e_deployment
from modules.level_0_performance_report import log_record, error_upload
from modules.level_0_api_endpoint import api_endpoint_ip
import level_2_order_optimization_hyundai_options as options_file
import modules.SessionState as SessionState
from modules.level_1_b_data_processing import null_analysis
from level_2_order_optimization_hyundai_options import configuration_parameters, client_lvl_cols, client_lvl_cols_renamed, score_weights, cols_to_normalize, reverse_normalization_cols

st.set_page_config(page_title='Sugestão de Encomenda - Importador', layout="wide")

min_number_of_configuration = 10
api_backend = api_endpoint_ip + options_file.api_backend_loc

truncate_query = ''' DELETE 
FROM [BI_DTR].[dbo].[VHE_Fact_MLG_OrderOptimization_Solver_Optimization_DTR]
WHERE PT_PDB_Model_Desc = '{}'  '''

# st.markdown("<h1 style='text-align: center; color: red;'>Sugestão de Encomenda - Importador - Demo</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Sugestão de Encomenda - Importador - Demo</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Sugestão de Configurações para a encomenda mensal de viaturas Hyundai</h2>", unsafe_allow_html=True)


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

url_hyperlink = '''
    <a href= "{}" > <p style="text-align:right"> Manual de Utilizador </p></a>
'''.format(options_file.documentation_url_solver_app)
# st.markdown(url_hyperlink, unsafe_allow_html=True)

placeholder_dw_date = st.empty()
placeholder_sales_plan_date = st.empty()
placeholder_proposal_date = st.empty()

session_state = SessionState.get(first_run_flag=0, run_id=0, save_button_pressed_flag=0, order_suggestion_button_pressed_flag=0, model='', brand='', daysinstock_score_weight=score_weights['Avg_DaysInStock_Global_normalized'], sel_margin_score_weight = score_weights['TotalGrossMarginPerc_normalized'], sel_margin_ratio_score_weight = score_weights['MarginRatio_normalized'], sel_qty_sold_score_weight = score_weights['Sum_Qty_CHS_normalized'], sel_proposals_score_weight = score_weights['Proposals_VDC_normalized'], sel_oc_stock_diff_score_weight = score_weights['Stock_OC_Diff_normalized'], sel_co2_nedc_score_weight = score_weights['NEDC_normalized'])

temp_cols = ['Avg_DaysInStock_Global', 'Avg_DaysInStock_Global_normalized', '#Veículos Vendidos', 'Sum_Qty_CHS_normalized', 'Proposals_VDC', 'Proposals_VDC_normalized', 'Margin_HP', 'TotalGrossMarginPerc', 'TotalGrossMarginPerc_normalized', 'MarginRatio', 'MarginRatio_normalized', 'OC', 'Stock_VDC', 'Stock_OC_Diff', 'Stock_OC_Diff_normalized', 'NEDC', 'NEDC_normalized']
total_months_list = ['Jan', 'Fev', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def main():
    data = get_data(options_file)
    data_v2 = get_data_v2(options_file, options_file.sql_info['database_source'], options_file.sql_info['new_score_streamlit_view'], query_filter={'CommercialVersion_Flag': 1}, model_flag=1)
    sales_plan = get_data_v2(options_file, options_file.sql_info['database_source'], options_file.sql_info['sales_plan_aux'])
    co2_nedc, co2_wltp, total_sales = co2_processing(sales_plan.copy())

    proposals_last_updated_date = run_single_query(options_file.DSN, options_file.sql_info['database_source'], options_file, options_file.proposals_max_date_query).values[0][0]
    dw_last_updated_date = data_v2['Record_Date'].max()
    sales_plan_last_updated_date = sales_plan['Record_Date'].max()

    placeholder_dw_date.markdown("<p style='text-align: right;'>Última Atualização DW - {}</p>".format(dw_last_updated_date), unsafe_allow_html=True)
    placeholder_sales_plan_date.markdown("<p style='text-align: right;'>Última Atualização Plano de Vendas - {}</p>".format(sales_plan_last_updated_date), unsafe_allow_html=True)
    placeholder_proposal_date.markdown("<p style='text-align: right;'>Última Atualização Proposatas HPK - {}</p>".format(proposals_last_updated_date), unsafe_allow_html=True)

    co2_nedc_before_order = co2_nedc / total_sales
    co2_wltp_before_order = co2_wltp / total_sales
    st.write('Situação Atual de Co2 (NEDC/WLTP): {:.2f}/{:.2f} gCo2/km'.format(co2_nedc_before_order, co2_wltp_before_order))  # ToDo Make bold
    st.write('Total de Vendas (Plano de Vendas): {} viaturas'.format(int(total_sales)))  # ToDo Make bold

    data_v2 = col_normalization(data_v2.copy(), cols_to_normalize, reverse_normalization_cols)
    # data_v2['Score'] = data_v2.apply(score_calculation, args=(session_state.daysinstock_score_weight,), axis=1)

    parameters_values, parameter_restriction_vectors = [], []
    max_number_of_cars_sold_v1 = max(data['Quantity_Sold'])
    max_number_of_cars_sold_v2 = max(data_v2['Sum_Qty_CHS'])
    max_number_of_cars_sold = max(int(max_number_of_cars_sold_v1), int(max_number_of_cars_sold_v2))

    sel_brand = st.sidebar.selectbox('Marca:', ['-', 'Hyundai'], index=1, key=session_state.run_id)

    if '-' not in sel_brand:
        # st.write(list(data.loc[data['NLR_Code'] == options_file.nlr_code_desc[sel_brand], 'PT_PDB_Model_Desc'].unique()))
        # st.write(list(data_v2.loc[data_v2['NLR_Code'] == str(options_file.nlr_code_desc[sel_brand]), 'PT_PDB_Model_Desc'].unique()))
        # ToDo: Porque é que há diferenças de modelos entre as duas fontes de dados? Exemplo, data_v2 não tem H-1 6 lugares, H350 ou Ioniq

        data_models = data.loc[data['NLR_Code'] == options_file.nlr_code_desc[sel_brand], 'PT_PDB_Model_Desc'].unique()
        data_models_v2 = data_v2.loc[data_v2['NLR_Code'] == str(options_file.nlr_code_desc[sel_brand]), 'PT_PDB_Model_Desc'].unique()
        common_models = [x for x in data_models if x in data_models_v2]
        sel_model = st.sidebar.selectbox('Modelo:', ['-'] + list(common_models), index=0)
    else:
        sel_model = ''

    st.sidebar.title('Opções:')
    sel_order_size = st.sidebar.number_input('Por favor escolha o número de viaturas a encomendar:', 1, 1000, value=150)
    sel_min_number_of_configuration = st.sidebar.number_input('Por favor escolha o número mínimo de configurações (default={}):'.format(min_number_of_configuration), 1, 100, value=min_number_of_configuration)
    sel_min_sold_cars = st.sidebar.number_input('Por favor escolha um valor mínimo de viaturas vendidas por configuração (valor máximo é de {}):'.format(max_number_of_cars_sold), 1, max_number_of_cars_sold, value=5)
    st.sidebar.title('Pesos:')
    session_state.sel_daysinstock_score_weight = st.sidebar.number_input('Por favor escolha um peso para o critério de Dias em Stock: (default={:.0f}%)'.format(score_weights['Avg_DaysInStock_Global_normalized'] * 100), 1, 100, value=int(score_weights['Avg_DaysInStock_Global_normalized'] * 100), key=session_state.run_id)
    session_state.sel_margin_score_weight = st.sidebar.number_input('Por favor escolha um peso para o critério de Margem: (default={:.0f}%)'.format(score_weights['TotalGrossMarginPerc_normalized'] * 100), 1, 100, value=int(score_weights['TotalGrossMarginPerc_normalized'] * 100), key=session_state.run_id)
    session_state.sel_margin_ratio_score_weight = st.sidebar.number_input('Por favor escolha um peso para o critério de Rácio de Margem: (default={:.0f}%)'.format(score_weights['MarginRatio_normalized'] * 100), 1, 100, value=int(score_weights['MarginRatio_normalized'] * 100), key=session_state.run_id)
    session_state.sel_qty_sold_score_weight = st.sidebar.number_input('Por favor escolha um peso para o critério de Volume de Vendas: (default={:.0f}%)'.format(score_weights['Sum_Qty_CHS_normalized'] * 100), 1, 100, value=int(score_weights['Sum_Qty_CHS_normalized'] * 100), key=session_state.run_id)
    session_state.sel_proposals_score_weight = st.sidebar.number_input('Por favor escolha um peso para o critério de Propostas: (default={:.0f}%)'.format(score_weights['Proposals_VDC_normalized'] * 100), 1, 100, value=int(score_weights['Proposals_VDC_normalized'] * 100), key=session_state.run_id)
    session_state.sel_oc_stock_diff_score_weight = st.sidebar.number_input('Por favor escolha um peso para o critério de O.C. vs Stock: (default={:.0f}%)'.format(score_weights['Stock_OC_Diff_normalized'] * 100), 1, 100, value=int(score_weights['Stock_OC_Diff_normalized'] * 100), key=session_state.run_id)
    session_state.sel_co2_nedc_score_weight = st.sidebar.number_input('Por favor escolha um peso para o critério de Co2 (NEDC): (default={:.0f}%)'.format(score_weights['NEDC_normalized'] * 100), 1, 100, value=int(score_weights['NEDC_normalized'] * 100), key=session_state.run_id)

    weights_sum = session_state.sel_daysinstock_score_weight + session_state.sel_margin_score_weight + session_state.sel_margin_ratio_score_weight + session_state.sel_qty_sold_score_weight + session_state.sel_proposals_score_weight + session_state.sel_oc_stock_diff_score_weight + session_state.sel_co2_nedc_score_weight
    if weights_sum != 100:
        st.sidebar.error('Alerta: a soma dos pesos é atualmente de {}%. Por favor validar e corrigir pesos de acordo.'.format(weights_sum))

    if st.sidebar.button('Reset Scores'):
        session_state.sel_daysinstock_score_weight = score_weights['Avg_DaysInStock_Global_normalized'] * 100
        session_state.sel_margin_score_weight = score_weights['TotalGrossMarginPerc_normalized'] * 100
        session_state.sel_margin_ratio_score_weight = score_weights['MarginRatio_normalized'] * 100
        session_state.sel_qty_sold_score_weight = score_weights['Sum_Qty_CHS_normalized'] * 100
        session_state.sel_proposals_score_weight = score_weights['Proposals_VDC_normalized'] * 100
        session_state.sel_oc_stock_diff_score_weight = score_weights['Stock_OC_Diff_normalized'] * 100
        session_state.sel_co2_nedc_score_weight = score_weights['NEDC_normalized'] * 100

        session_state.run_id += 1
        raise RerunException(RerunData())

    data_v2['Score'] = data_v2.apply(score_calculation, args=(session_state.sel_daysinstock_score_weight / 100, session_state.sel_margin_score_weight / 100, session_state.sel_margin_ratio_score_weight / 100, session_state.sel_qty_sold_score_weight / 100, session_state.sel_proposals_score_weight / 100, session_state.sel_oc_stock_diff_score_weight / 100, session_state.sel_co2_nedc_score_weight / 100), axis=1)

    if sel_model != session_state.model:
        session_state.model = sel_model
        session_state.overwrite_button_pressed, session_state.save_button_pressed_flag, session_state.order_suggestion_button_pressed_flag = 0, 0, 0

    if '-' not in [sel_model] and '-' not in [sel_brand]:
        data_filtered = filter_data(data, [sel_model, sel_min_sold_cars], ['PT_PDB_Model_Desc', 'Quantity_Sold'])
        data_filtered_v2 = filter_data_v2(data_v2, [sel_model, sel_min_sold_cars], ['PT_PDB_Model_Desc', 'Sum_Qty_CHS'])

        if not data_filtered.shape[0]:
            st.write('Não foram encontrados registos para as presentes escolhas - Por favor altere o modelo/cliente/valor mínimo de viaturas por configuração.')
            return
        # st.write('Número de Configurações:', data_filtered['ML_VehicleData_Code'].nunique())

        proposals_col = 'Proposals_Count_VDC'
        stock_col = 'Stock_Count_VDC'

        session_state.order_suggestion_button_pressed_flag = 1
        # status, total_value_optimized, selection, selection_configuration_ids = solver(data_filtered, parameter_restriction_vectors, sel_order_size)
        status, total_value_optimized, selection, selection_configuration_ids = solver(data_filtered, parameter_restriction_vectors, sel_order_size)
        # status_v2, total_value_optimized_v2, selection_v2, selection_configuration_ids_v2 = solver_v2(data_filtered_v2[~data_filtered_v2['NEDC'].isnull()], parameter_restriction_vectors, sel_order_size, co2_nedc, total_sales)
        data_filtered['Quantity'] = selection
        # data_filtered_v2['Quantity'] = selection

        if status == 'optimal':
            sel_configurations = quantity_processing(data_filtered.copy(deep=True), sel_order_size, proposals_col, stock_col, sel_min_number_of_configuration)
            sel_configurations_v2 = quantity_processing_v2(data_filtered_v2.copy(deep=True), sel_order_size, sel_min_number_of_configuration)

            if sel_configurations.shape[0]:
                sel_configurations.rename(index=str, columns={'Quantity': 'Sug.Encomenda'}, inplace=True)  # ToDo: For some reason this column in particular is not changing its name by way of the renaming argument in the previous st.write. This is a temporary solution
                st.write('Sugestão Encomenda (Score Inicial):', sel_configurations[['Sug.Encomenda'] + [x for x in configuration_parameters if x not in 'PT_PDB_Model_Desc'] + ['Quantity_Sold'] + ['Average_Score_Euros']]
                         .rename(columns=options_file.column_translate_dict).reset_index(drop=True)
                         .style.format({'Score (€)': '{:.2f}', 'Sug.Encomenda': '{:.0f}', 'Propostas Entregues': '{:.0f}', 'Em Stock': '{:.0f}'})
                         )

                # st.write('Propostas Entregues para esta sugestão: ', int(sel_configurations[proposals_col].sum()))
                # st.write('Viaturas em Stock para esta sugestão: ', int(sel_configurations[stock_col].sum()))
                sel_configurations.rename(index=str, columns={'Sug.Encomenda': 'Quantity'}, inplace=True)  # ToDo: For some reason this column in particular is not changing its name by way of the renaming argument in the previous st.write. This is a temporary solution
            else:
                return

            if sel_configurations_v2.shape[0]:
                # st.write('Número de Configurações:', data_filtered_v2.shape[0])

                sel_configurations_v2.rename(index=str, columns={'Quantity': 'Sug.Encomenda', 'Sum_Qty_CHS': '#Veículos Vendidos'}, inplace=True)  # ToDo: For some reason this column in particular is not changing its name by way of the renaming argument in the previous st.write. This is a temporary solution
                st.write('Sugestão Encomenda (Novo Score):', sel_configurations_v2[['Sug.Encomenda'] + [x for x in configuration_parameters if x not in 'PT_PDB_Model_Desc'] + temp_cols + ['Score']]
                         .rename(columns=options_file.column_translate_dict).reset_index(drop=True)
                         .style.apply(highlight_cols, col_dict=options_file.col_color_dict)
                         .format(options_file.col_decimals_place_dict)
                         )
                # .style.format({'Score (€)': '{:.2f}', 'Sug.Encomenda': '{:.0f}', 'Propostas Entregues': '{:.0f}', 'Em Stock': '{:.0f}'})
                # st.write('Propostas Entregues para esta sugestão: ', int(sel_configurations_v2['Proposals_VDC'].sum()))
                # st.write('Viaturas em Stock para esta sugestão: ', int(sel_configurations_v2['Stock_VDC'].sum()))
                sel_configurations_v2.rename(index=str, columns={'Sug.Encomenda': 'Quantity'}, inplace=True)  # ToDo: For some reason this column in particular is not changing its name by way of the renaming argument in the previous st.write. This is a temporary solution

                total_sales_after_order = total_sales + sel_configurations_v2['Quantity'].sum()
                sel_configurations_v2['nedc_after_order'] = sel_configurations_v2['Quantity'] * sel_configurations_v2['NEDC']
                sel_configurations_v2['wltp_after_order'] = sel_configurations_v2['Quantity'] * sel_configurations_v2['WLTP']
                co2_nedc_after_order = co2_nedc + sel_configurations_v2['nedc_after_order'].sum()
                co2_wltp_after_order = co2_wltp + sel_configurations_v2['wltp_after_order'].sum()
                co2_nedc_per_vehicle_after_order = co2_nedc_after_order / total_sales_after_order
                co2_wltp_per_vehicle_after_order = co2_wltp_after_order / total_sales_after_order

                co2_nedc_per_vehicle_evolution = co2_nedc_per_vehicle_after_order - co2_nedc_before_order
                if co2_nedc_after_order > co2_nedc_before_order:
                    st.markdown("Situação Co2 (NEDC) após esta encomenda: {:.2f}(<span style='color:red'>+{:.2f}</span>) gCo2/km".format(co2_nedc_per_vehicle_after_order, co2_nedc_per_vehicle_evolution), unsafe_allow_html=True)
                else:
                    st.markdown("Situação Co2 (NEDC) após esta encomenda: {:.2f}(-<span style='color:green'>{:.2f}</span>) gCo2/km".format(co2_nedc_per_vehicle_after_order, co2_nedc_per_vehicle_evolution), unsafe_allow_html=True)

                co2_wltp_per_vehicle_evolution = co2_wltp_per_vehicle_after_order - co2_wltp_before_order
                if co2_wltp_after_order > co2_wltp_before_order:
                    st.markdown("Situação Co2 (WLTP) após esta encomenda: {:.2f}(<span style='color:red'>+{:.2f}</span>) gCo2/km".format(co2_wltp_per_vehicle_after_order, co2_wltp_per_vehicle_evolution), unsafe_allow_html=True)
                else:
                    st.markdown("Situação Co2 (WLTP) após esta encomenda: {:.2f}(-<span style='color:green'>{:.2f}</span>) gCo2/km".format(co2_wltp_per_vehicle_after_order, co2_wltp_per_vehicle_evolution), unsafe_allow_html=True)

                sel_configurations_v2['Configuration_Concat'] = sel_configurations_v2['PT_PDB_Model_Desc'] + ', ' + sel_configurations_v2['PT_PDB_Engine_Desc'] + ', ' + sel_configurations_v2['PT_PDB_Transmission_Type_Desc'] + ', ' + sel_configurations_v2['PT_PDB_Version_Desc'] + ', ' +  sel_configurations_v2['PT_PDB_Exterior_Color_Desc'] + ', ' + sel_configurations_v2['PT_PDB_Interior_Color_Desc']
                sel_config = st.selectbox('Configuração a explorar:', ['-'] + [x for x in sel_configurations_v2['Configuration_Concat'].unique()], index=0)
                if sel_config != '-':
                    validation_dfs = get_validation_info(sel_configurations_v2, sel_config)
                    validation_dfs_titles = ['Vendas', 'Propostas', 'Stock', 'Plano de Vendas, passo 1', 'Plano de Vendas, passo 2', 'Plano de Vendas, passo 3']

                    st.write(validation_dfs_titles[0] + ' ({}):'.format(validation_dfs[0].shape[0]), validation_dfs[0][[x for x in list(validation_dfs[0]) if x != 'Last_Modified_Date']].rename(columns=options_file.column_translate_dict))
                    left_table, right_table = st.beta_columns(2)
                    with left_table:
                        st.write(validation_dfs_titles[1] + ' ({}):'.format(validation_dfs[1].shape[0]), validation_dfs[1][[x for x in list(validation_dfs[1]) if x != 'Last_Modified_Date']].rename(columns=options_file.column_translate_dict))
                        st.write(validation_dfs_titles[3] + ':', validation_dfs[3][[x for x in list(validation_dfs[3]) if x != 'Last_Modified_Date']].rename(columns=options_file.column_translate_dict))
                        st.write(validation_dfs_titles[4] + ':', validation_dfs[4][[x for x in list(validation_dfs[4]) if x != 'Last_Modified_Date']].rename(columns=options_file.column_translate_dict))
                        st.write(validation_dfs_titles[5] + ':', validation_dfs[5][[x for x in list(validation_dfs[5]) if x != 'Last_Modified_Date']].rename(columns=options_file.column_translate_dict))
                    with right_table:
                        st.write(validation_dfs_titles[2] + ' ({}):'.format(validation_dfs[2].shape[0]), validation_dfs[2][[x for x in list(validation_dfs[2]) if x != 'Last_Modified_Date']].rename(columns=options_file.column_translate_dict))

            else:
                return

        elif status == 'infeasible':
            st.write('Não foi possível gerar uma sugestão de encomenda.')

    else:
        st.markdown("<p style='text-align: center;'>Por favor escolha uma marca e modelo para sugerir a respetiva encomenda.</p>", unsafe_allow_html=True)


def highlight_cols(s, col_dict):

    if s.name in col_dict.keys():
        return ['background-color: {}'.format(col_dict[s.name])] * len(s)
    return [''] * len(s)


@st.cache(show_spinner=False)
def get_validation_info(sel_configurations_v2, sel_config):
    validation_dfs = []

    sel_config_model = sel_configurations_v2.loc[sel_configurations_v2['Configuration_Concat'] == sel_config, 'PT_PDB_Model_Desc'].values[0]
    sel_config_engine = sel_configurations_v2.loc[sel_configurations_v2['Configuration_Concat'] == sel_config, 'PT_PDB_Engine_Desc'].values[0]
    sel_config_transmission = sel_configurations_v2.loc[sel_configurations_v2['Configuration_Concat'] == sel_config, 'PT_PDB_Transmission_Type_Desc'].values[0]
    sel_config_version = sel_configurations_v2.loc[sel_configurations_v2['Configuration_Concat'] == sel_config, 'PT_PDB_Version_Desc'].values[0]
    sel_config_ext_color = sel_configurations_v2.loc[sel_configurations_v2['Configuration_Concat'] == sel_config, 'PT_PDB_Exterior_Color_Desc'].values[0]
    sel_config_int_color = sel_configurations_v2.loc[sel_configurations_v2['Configuration_Concat'] == sel_config, 'PT_PDB_Interior_Color_Desc'].values[0]

    validation_queries = [options_file.sales_validation_query,
                          options_file.proposals_validation_query,
                          options_file.stock_validation_query,
                          options_file.sales_plan_validation_query_step_1,
                          options_file.sales_plan_validation_query_step_2,
                          options_file.sales_plan_validation_query_step_3]

    for query in validation_queries:
        validation_df = run_single_query(options_file.DSN, options_file.sql_info['database_source'], options_file, query.format(sel_config_model, sel_config_engine, sel_config_version, sel_config_transmission, sel_config_ext_color, sel_config_int_color))
        validation_dfs.append(validation_df)

    return validation_dfs


def run_single_query(dsn, database, options_file_in, query):
    query_result = level_1_a_data_acquisition.sql_retrieve_df_specified_query(dsn, database, options_file_in, query)

    return query_result


@st.cache(show_spinner=False)
def co2_processing(df):
    start_date = date(date.today().year, 1, 1)
    end_date = date.today()  # + relativedelta(months=+2)  # ToDo might be needed later on
    start_date_year = start_date.year
    # end_date_year = end_date.year
    end_date_month_number = end_date.month

    # The following condition is only for the first year of the month, where, even though January isn't complete, we default to use the sales plan for that month;
    if end_date_month_number == 1:
        df.loc[:, 'Sales_Sum'] = df.loc[(df['Sales_Plan_Year'] == start_date_year), :].loc[:, ['Jan']].sum(axis=1)  # First, filter dataframe for the current year of the sales plan, then select only the running year months;
    else:
        df.loc[:, 'Sales_Sum'] = df.loc[(df['Sales_Plan_Year'] == start_date_year), :].loc[:, total_months_list[0:end_date_month_number - 1]].sum(axis=1)  # First, filter dataframe for the current year of the sales plan, then select only the running year months;

    df.loc[:, 'Sales_Sum_Times_Co2_WLTP'] = df['Sales_Sum'] * df['WLTP_CO2']
    df.loc[:, 'Sales_Sum_Times_Co2_NEDC'] = df['Sales_Sum'] * df['NEDC_CO2']
    co2_wltp_sum = df['Sales_Sum_Times_Co2_WLTP'].sum(axis=0)
    co2_nedc_sum = df['Sales_Sum_Times_Co2_NEDC'].sum(axis=0)
    total_sales = df['Sales_Sum'].sum(axis=0)

    return co2_nedc_sum, co2_wltp_sum, total_sales


@st.cache(show_spinner=False)
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

    df, sel_vehicledata_codes = gamas_selection(df.copy(), df_pdb, current_date)
    gama_viva_mask = df['VehicleData_Code'].isin(sel_vehicledata_codes)

    df['Custo_Stock_Dist'] = df['Measure_9'] * 0.015 / 365 * df['DaysInStock_Distributor'] * (-1)
    df['Score_Euros'] = df['Fixed_Margin_II'] - df['Custo_Stock_Dist']

    df_grouped = df.groupby('ML_VehicleData_Code')
    df['Average_DaysInStock_Global'] = df_grouped['DaysInStock_Global'].transform('mean')
    df['Quantity_Sold'] = df_grouped['Chassis_Number'].transform('nunique')
    df['Average_Score_Euros'] = df_grouped['Score_Euros'].transform('mean')
    df['Gama_Viva_Flag'] = np.where(gama_viva_mask, "Sim", "Não")

    # df['NDB_Dealer_Code_alt'] = df['NDB_Dealer_Code'].str.replace(r'[A-Z]$', '')
    df = pd.merge(df, proposals_grouped, left_on=['VehicleData_Code'], right_on=['VehicleData_Code'], how='left').fillna(0)
    # df = pd.merge(df, df_proposals, left_on=['VehicleData_Code', 'NDB_Dealer_Code_alt'], right_on=['VehicleData_Code', 'NDB_Installation_Code'], how='left').fillna(0)
    df = pd.merge(df, stock_grouped, left_on=['VehicleData_Code'], right_on=['VehicleData_Code'], how='left').fillna(0)
    df = pd.merge(df, df_stock, left_on=['VehicleData_Code', 'NDB_Dealer_Code'], right_on=['VehicleData_Code', 'NDB_Dealer_Code'], how='left').fillna(0)

    df = model_lowercase(df)

    df = df[~df['PT_PDB_Model_Desc'].isin(['i20 van', 'i20 coupe', 'i30 fastback n', 'i40'])]
    df = df[df['Average_Score_Euros'] > 0]
    return df


def model_lowercase(df):

    # for model in df['PT_PDB_Model_Desc'].unique():
    #     if model == 'cr-v':
    #         df.loc[df['PT_PDB_Model_Desc'] == model, 'PT_PDB_Model_Desc'] = 'CR-V'
    #     elif model == 'hr-v':
    #         df.loc[df['PT_PDB_Model_Desc'] == model, 'PT_PDB_Model_Desc'] = 'HR-V'
    #     elif model == 'ioniq':
    #         df.loc[df['PT_PDB_Model_Desc'] == model, 'PT_PDB_Model_Desc'] = 'Ioniq'
    #     elif model[0] != 'i':
    #         df.loc[df['PT_PDB_Model_Desc'] == model, 'PT_PDB_Model_Desc'] = model.lower()
    df['PT_PDB_Model_Desc'] = df['PT_PDB_Model_Desc'].apply(lambda x: x.lower())

    return df


@st.cache(show_spinner=False, allow_output_mutation=True)
def get_data_v2(options_file_in, db, table, query_filter=None, model_flag=0):

    if query_filter is not None:
        df = level_1_a_data_acquisition.sql_retrieve_df(options_file_in.DSN, db, table, options_file_in, query_filters=query_filter)
    else:
        df = level_1_a_data_acquisition.sql_retrieve_df(options_file_in.DSN, db, table, options_file_in)

    if model_flag:
        df = model_lowercase(df)

    return df


@st.cache(show_spinner=False, allow_output_mutation=True)
def col_normalization(df, cols_to_normalize_in, cols_to_normalize_reverse_in):

    for col in cols_to_normalize_in:
        if col in cols_to_normalize_reverse_in:
            df.loc[:, col + '_normalized'] = (df[col] - df[col].max()) / (df[col].min() - df[col].max())
        else:
            df.loc[:, col + '_normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # NaN Handling
    df['NEDC_normalized'].fillna(0, inplace=True)
    df['MarginRatio_normalized'].fillna(0, inplace=True)

    return df


def score_calculation(x, sel_daysinstock_score_weight, sel_margin_score_weight, sel_margin_ratio_score_weight, sel_qty_sold_score_weight, sel_proposals_score_weight, sel_oc_stock_diff_score_weight, sel_co2_nedc_score_weight):
    # args = (sel_daysinstock_score_weight / 100, sel_margin_score_weight / 100, sel_margin_ratio_score_weight / 100, sel_qty_sold_score_weight / 100, sel_proposals_score_weight / 100, sel_oc_stock_diff_score_weight / 100, sel_co2_nedc_score_weight / 100)
    y = \
        x['Avg_DaysInStock_Global_normalized'] * sel_daysinstock_score_weight \
        + x['TotalGrossMarginPerc_normalized'] * sel_margin_score_weight \
        + x['MarginRatio_normalized'] * sel_margin_ratio_score_weight \
        + x['Sum_Qty_CHS_normalized'] * sel_qty_sold_score_weight \
        + x['Proposals_VDC_normalized'] * sel_proposals_score_weight \
        + x['Stock_OC_Diff_normalized'] * sel_oc_stock_diff_score_weight \
        + x['NEDC_normalized'] * sel_co2_nedc_score_weight

    return y


@st.cache(show_spinner=False, ttl=60*60*24*12)
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


def filter_data_v2(dataset, value_filters_list, col_filters_list):
    data_filtered = dataset.copy()

    for col_filter, filter_value in zip(col_filters_list, value_filters_list):
        if filter_value != '-' and type(filter_value) == list and len(filter_value) > 0:
            data_filtered = data_filtered.loc[data_filtered[col_filter].isin(filter_value), :]

        if filter_value != '-' and type(filter_value) != list:
            if col_filter == 'Sum_Qty_CHS':
                data_filtered = data_filtered.loc[data_filtered[col_filter].ge(filter_value), :]
            elif filter_value != '-':
                data_filtered = data_filtered.loc[data_filtered[col_filter] == filter_value, :]

    # st.write('Número de viaturas vendidas:', data_filtered['Registration_Number'].nunique())
    data_filtered.sort_values(by='Score', ascending=False, inplace=True)
    return data_filtered


# def solver(dataset, parameter_restriction_vectors, sel_order_size):
#     parameter_restriction = []
#
#     unique_ids_count = dataset['ML_VehicleData_Code'].nunique()
#     unique_ids = dataset['ML_VehicleData_Code'].unique()
#
#     scores_values = [dataset[dataset['ML_VehicleData_Code'] == x]['Average_Score_Euros'].head(1).values[0] for x in unique_ids]  # uniques() command doesn't work as intended because there are configurations (Configuration IDs) with repeated average score
#
#     selection = cp.Variable(unique_ids_count, integer=True)
#
#     order_size_restriction = cp.sum(selection) <= sel_order_size
#     total_value = selection * scores_values
#
#     problem = cp.Problem(cp.Maximize(total_value), [selection >= 0, selection <= 100,
#                                                     order_size_restriction,
#                                                     ] + parameter_restriction)
#
#     result = problem.solve(solver=cp.GLPK_MI, verbose=False, qcp=True)
#
#     return problem.status, result, selection.value, unique_ids


def solver(dataset, parameter_restriction_vectors, sel_order_size):

    data = {
        'df_solve': dataset.to_json(orient='records'),
        'parameter_restriction_vectors': parameter_restriction_vectors,
        'sel_order_size': sel_order_size
    }

    result = requests.get(api_backend, data=json.dumps(data))

    result_dict = json.loads(result.text)

    return result_dict['status'], result_dict['optimization_total_sum'], result_dict['selection'], result_dict['unique_ids']


def solver_v2(dataset, parameter_restriction_vectors, sel_order_size, co2_current_status, total_sales):

    unique_ids = dataset.index
    scores_values = [x for x in dataset['Score']]
    co2_emissions = np.array(dataset['NEDC'].values.tolist())

    selection = cp.Variable(len(unique_ids), integer=True)
    order_size_restriction = cp.sum(selection) <= sel_order_size
    total_value = selection @ scores_values

    co2_middle_calc = cp.sum(selection @ co2_emissions)

    co2_numerator = (co2_current_status + co2_middle_calc)
    co2_denominator = (total_sales + cp.sum(selection))

    co2_constraint = co2_numerator <= (104.5 * co2_denominator)  # The formula has to be built this way, otherwise, when using a simple division, makes this constraint not follow DCP rules

    problem = cp.Problem(cp.Maximize(total_value), [selection >= 0, selection <= 100,
                                                    order_size_restriction,
                                                    co2_constraint
                                                    ])

    result = problem.solve(solver=cp.GLPK_MI, verbose=False, qcp=False)

    st.write(problem.status, result, selection.value)

    # status = problem.status
    # total_value_optimized = result
    # selection = [qty for qty in selection.value]
    # selection_configuration_ids = unique_ids

    sys.exit()

    return 'optimal', 0, selection, []


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
        st.error('Aviso: Existem mais ou igual número de propostas ({}) do que viaturas a encomendar ({}). Por favor alterar algum dos parâmetros como nº mínimo de configurações a mostrar, valor mínimo de viaturas vendidas ou cliente.'.format(int(proposals_total), int(sel_order_size)))
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


def quantity_processing_v2(df, sel_order_size, sel_min_number_of_configuration):
    df = df.head(sel_min_number_of_configuration)

    total_score = df['Score'].sum()

    df.loc[:, 'Score Weight'] = df.loc[:, 'Score'] / total_score
    df.loc[:, 'Weighted Order'] = df.loc[:, 'Score Weight'] * sel_order_size
    df.loc[:, 'Quantity'] = df.loc[:, 'Weighted Order'].round()

    return df


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
        error_upload(options_file, project_identifier, format_exc(), exception_desc, error_flag=1, solution_type='OPR', sel_parameters=['sel_brand', 'sel_model', 'sel_order_size', 'sel_min_number_of_configuration', 'sel_min_sold_cars', 'sel_daysinstock_score_weight', 'sel_margin_score_weight', 'sel_margin_ratio_score_weight', 'sel_qty_sold_score_weight', 'sel_proposals_score_weight', 'sel_oc_stock_diff_score_weight', 'sel_co2_nedc_score_weight'])
        session_state.run_id += 1
        st.error('AVISO: Ocorreu um erro. Os administradores desta página foram notificados com informação do erro e este será corrigido assim que possível. Entretanto, esta aplicação será reiniciada. Obrigado pela sua compreensão.')
        time.sleep(10)
        raise RerunException(RerunData())

