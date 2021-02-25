import streamlit as st
import base64
import pandas as pd
import numpy as np
import os
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
from modules.level_1_b_data_processing import null_analysis, df_join_function
from level_2_order_optimization_hyundai_options import configuration_parameters, client_lvl_cols, client_lvl_cols_renamed, score_weights, cols_to_normalize, reverse_normalization_cols

st.set_page_config(page_title='Sugestão de Encomenda - Importador', layout="wide")

min_number_of_configuration = 10
api_backend = api_endpoint_ip + options_file.api_backend_loc

truncate_query = ''' DELETE 
FROM [BI_DTR].[dbo].[VHE_Fact_MLG_OrderOptimization_Solver_Optimization_DTR]
WHERE PT_PDB_Model_Desc = '{}'  '''

st.markdown("<h1 style='text-align: center;'>Sugestão de Encomenda - Importador</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Sugestão de Configurações para a encomenda mensal de viaturas Hyundai e Honda</h2>", unsafe_allow_html=True)

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

placeholder_dw_date = st.empty()
placeholder_sales_plan_date = st.empty()
placeholder_proposal_date = st.empty()
placeholder_margins_date = st.empty()

session_state = SessionState.get(first_run_flag=0, run_id=0, run_id_scores=0, save_button_pressed_flag=0, model='', brand='', daysinstock_score_weight=score_weights['Avg_DaysInStock_Global_normalized'], sel_margin_score_weight = score_weights['TotalGrossMarginPerc_normalized'], sel_margin_ratio_score_weight = score_weights['MarginRatio_normalized'], sel_qty_sold_score_weight = score_weights['Sum_Qty_CHS_normalized'], sel_proposals_score_weight = score_weights['Proposals_VDC_normalized'], sel_oc_stock_diff_score_weight = score_weights['Stock_OC_Diff_normalized'], sel_co2_nedc_score_weight = score_weights['NEDC_normalized'])

temp_cols = ['Avg_DaysInStock_Global', 'Avg_DaysInStock_Global_normalized', '#Veículos Vendidos', 'Sum_Qty_CHS_normalized', 'Proposals_VDC', 'Proposals_VDC_normalized', 'Margin_HP', 'TotalGrossMarginPerc', 'TotalGrossMarginPerc_normalized', 'MarginRatio', 'MarginRatio_normalized', 'OC', 'Stock_VDC', 'Stock_OC_Diff', 'Stock_OC_Diff_normalized', 'NEDC', 'NEDC_normalized']
total_months_list = ['Jan', 'Fev', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def main():
    data_v2 = get_data_v2(options_file, options_file.DSN_MLG_PRD, options_file.sql_info['database_final'], options_file.sql_info['new_score_streamlit_view'], model_flag=1)
    all_brands_sales_plan = get_data_v2(options_file, options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'], options_file.sql_info['sales_plan_aux'])
    live_ocn_df = get_data_v2(options_file, options_file.DSN_MLG_PRD, options_file.sql_info['database_final'], options_file.sql_info['current_live_ocn_table'], model_flag=1)
    end_month_index, current_year = period_calculation()

    dw_last_updated_date = data_v2['Record_Date'].max()
    placeholder_dw_date.markdown("<p style='text-align: right;'>Última Atualização DW - {}</p>".format(dw_last_updated_date), unsafe_allow_html=True)

    data_v2 = col_normalization(data_v2.copy(), cols_to_normalize, reverse_normalization_cols)
    max_number_of_cars_sold = max(data_v2['Sum_Qty_CHS'])
    sel_brand = st.sidebar.selectbox('Marca:', ['-', 'Hyundai', 'Honda'], index=0, key=session_state.run_id)

    if '-' not in sel_brand:
        co2_nedc, co2_wltp, total_sales = co2_processing(all_brands_sales_plan.loc[all_brands_sales_plan['NLR_Code'] == str(options_file.nlr_code_desc[sel_brand]), :].copy(), end_month_index, current_year)
        co2_nedc_before_order = co2_nedc / total_sales
        co2_wltp_before_order = co2_wltp / total_sales
        st.write('Situação Atual de Co2 (NEDC/WLTP): {:.2f}/{:.2f} gCo2/km'.format(co2_nedc_before_order, co2_wltp_before_order))
        if end_month_index == 1:
            sel_period_string = '{} de {}'.format('Jan', current_year)
        else:
            sel_period_string = '{} a {} de {}'.format(total_months_list[0], total_months_list[end_month_index - 1], current_year)

        st.write('Plano de Vendas Total, {}: {} viaturas'.format(sel_period_string, int(total_sales)))
        placeholder_sales_plan_single_model = st.empty()

        data_models_v2 = data_v2.loc[data_v2['NLR_Code'] == str(options_file.nlr_code_desc[sel_brand]), 'PT_PDB_Model_Desc'].unique()
        sel_model = st.sidebar.selectbox('Modelo:', ['-'] + list(sorted(data_models_v2)), index=0)

        sales_plan_last_updated_date = all_brands_sales_plan.loc[all_brands_sales_plan['NLR_Code'] == str(options_file.nlr_code_desc[sel_brand]), 'Record_Date'].max()
        proposals_last_updated_date = run_single_query(options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'], options_file, options_file.proposals_max_date_query.format(options_file.nlr_code_desc[sel_brand])).values[0][0]
        margins_last_update_date = run_single_query(options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'], options_file, options_file.margins_max_date_query.format(options_file.nlr_code_desc[sel_brand])).values[0][0]

        placeholder_sales_plan_date.markdown("<p style='text-align: right;'>Última Atualização Plano de Vendas - {}</p>".format(sales_plan_last_updated_date), unsafe_allow_html=True)
        placeholder_margins_date.markdown("<p style='text-align: right;'>Última Atualização Margens HP - {}</p>".format(margins_last_update_date), unsafe_allow_html=True)

        if sel_brand == 'Hyundai':
            placeholder_proposal_date.markdown("<p style='text-align: right;'>Última Atualização Propostas HPK - {}</p>".format(proposals_last_updated_date), unsafe_allow_html=True)
        elif sel_brand == 'Honda':
            placeholder_proposal_date.markdown("<p style='text-align: right;'>Última Atualização Propostas - {}</p>".format(proposals_last_updated_date), unsafe_allow_html=True)
        else:
            raise ValueError('Unknown Selected Brand - {}'.format(sel_brand))
    else:
        sel_model = ''

    st.sidebar.title('Opções:')
    sel_order_size = st.sidebar.number_input('Por favor escolha o número de viaturas a encomendar:', 1, 1000, value=150)
    sel_min_number_of_configuration = st.sidebar.number_input('Por favor escolha o número de configurações:', 1, 100, value=min_number_of_configuration)
    placeholder_value = st.sidebar.empty()
    sel_min_sold_cars = st.sidebar.number_input('Por favor escolha um valor mínimo de viaturas vendidas por configuração (valor máximo é de {:.0f}):'.format(max_number_of_cars_sold), 1, int(max_number_of_cars_sold), value=5)
    st.sidebar.title('Pesos:')
    session_state.sel_daysinstock_score_weight = st.sidebar.number_input('Por favor escolha um peso para o critério de Dias em Stock: (default={:.0f}%)'.format(score_weights['Avg_DaysInStock_Global_normalized'] * 100), 0, 100, value=int(score_weights['Avg_DaysInStock_Global_normalized'] * 100), key=session_state.run_id_scores)
    session_state.sel_margin_score_weight = st.sidebar.number_input('Por favor escolha um peso para o critério de Margem: (default={:.0f}%)'.format(score_weights['TotalGrossMarginPerc_normalized'] * 100), 0, 100, value=int(score_weights['TotalGrossMarginPerc_normalized'] * 100), key=session_state.run_id_scores)
    session_state.sel_margin_ratio_score_weight = st.sidebar.number_input('Por favor escolha um peso para o critério de Rácio de Margem: (default={:.0f}%)'.format(score_weights['MarginRatio_normalized'] * 100), 0, 100, value=int(score_weights['MarginRatio_normalized'] * 100), key=session_state.run_id_scores)
    session_state.sel_qty_sold_score_weight = st.sidebar.number_input('Por favor escolha um peso para o critério de Volume de Vendas: (default={:.0f}%)'.format(score_weights['Sum_Qty_CHS_normalized'] * 100), 0, 100, value=int(score_weights['Sum_Qty_CHS_normalized'] * 100), key=session_state.run_id_scores)
    session_state.sel_proposals_score_weight = st.sidebar.number_input('Por favor escolha um peso para o critério de Propostas: (default={:.0f}%)'.format(score_weights['Proposals_VDC_normalized'] * 100), 0, 100, value=int(score_weights['Proposals_VDC_normalized'] * 100), key=session_state.run_id_scores)
    session_state.sel_oc_stock_diff_score_weight = st.sidebar.number_input('Por favor escolha um peso para o critério de O.C. vs Stock: (default={:.0f}%)'.format(score_weights['Stock_OC_Diff_normalized'] * 100), 0, 100, value=int(score_weights['Stock_OC_Diff_normalized'] * 100), key=session_state.run_id_scores)
    session_state.sel_co2_nedc_score_weight = st.sidebar.number_input('Por favor escolha um peso para o critério de Co2 (NEDC): (default={:.0f}%)'.format(score_weights['NEDC_normalized'] * 100), 0, 100, value=int(score_weights['NEDC_normalized'] * 100), key=session_state.run_id_scores)

    weights_sum = session_state.sel_daysinstock_score_weight + session_state.sel_margin_score_weight + session_state.sel_margin_ratio_score_weight + session_state.sel_qty_sold_score_weight + session_state.sel_proposals_score_weight + session_state.sel_oc_stock_diff_score_weight + session_state.sel_co2_nedc_score_weight
    if weights_sum != 100:
        st.sidebar.error('Alerta: Soma dos pesos é atualmente de {}%. Por favor validar e corrigir pesos de acordo.'.format(weights_sum))

    if st.sidebar.button('Reset Pesos'):
        session_state.sel_daysinstock_score_weight = score_weights['Avg_DaysInStock_Global_normalized'] * 100
        session_state.sel_margin_score_weight = score_weights['TotalGrossMarginPerc_normalized'] * 100
        session_state.sel_margin_ratio_score_weight = score_weights['MarginRatio_normalized'] * 100
        session_state.sel_qty_sold_score_weight = score_weights['Sum_Qty_CHS_normalized'] * 100
        session_state.sel_proposals_score_weight = score_weights['Proposals_VDC_normalized'] * 100
        session_state.sel_oc_stock_diff_score_weight = score_weights['Stock_OC_Diff_normalized'] * 100
        session_state.sel_co2_nedc_score_weight = score_weights['NEDC_normalized'] * 100

        session_state.run_id_scores += 1
        raise RerunException(RerunData())

    data_v2['Score'] = data_v2.apply(score_calculation, args=(session_state.sel_daysinstock_score_weight / 100, session_state.sel_margin_score_weight / 100, session_state.sel_margin_ratio_score_weight / 100, session_state.sel_qty_sold_score_weight / 100, session_state.sel_proposals_score_weight / 100, session_state.sel_oc_stock_diff_score_weight / 100, session_state.sel_co2_nedc_score_weight / 100), axis=1)

    if sel_model != session_state.model:
        session_state.model = sel_model
        session_state.overwrite_button_pressed, session_state.save_button_pressed_flag = 0, 0

    if '-' not in [sel_model] and '-' not in [sel_brand]:
        sales_plan_sel_model_sales = run_single_query(options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'], options_file, options_file.sales_plan_current_sales_single_model.format(end_month_index - 1, sel_model)).values[0][0]
        placeholder_sales_plan_single_model.write('Plano de Vendas para {}, {}: {} viaturas'.format(sel_model, sel_period_string, int(sales_plan_sel_model_sales)))

        # data_filtered = filter_data(data, [sel_model, sel_min_sold_cars], ['PT_PDB_Model_Desc', 'Quantity_Sold'])
        data_filtered_v2 = filter_data_v2(data_v2, [sel_model, sel_min_sold_cars], ['PT_PDB_Model_Desc', 'Sum_Qty_CHS'])

        if not data_filtered_v2.shape[0]:
            st.write('Não foram encontrados registos para as presentes escolhas - Por favor altere o modelo/cliente/valor mínimo de viaturas por configuração.')
            return

        sel_configurations_v2 = quantity_processing_v2(data_filtered_v2.head(sel_min_number_of_configuration).copy(deep=True), sel_order_size)

        if sel_configurations_v2.shape[0]:
            sel_configurations_v2.rename(index=str, columns={'Quantity': 'Sug.Encomenda', 'Sum_Qty_CHS': '#Veículos Vendidos'}, inplace=True)  # ToDo: For some reason this column in particular is not changing its name by way of the renaming argument in the previous st.write. This is a temporary solution
            st.markdown("<h3 style='text-align: left;'>Sugestão de Encomenda - Score v2:</h3>", unsafe_allow_html=True)
            st.write('', sel_configurations_v2[['Sug.Encomenda'] + [x for x in configuration_parameters if x not in 'PT_PDB_Model_Desc'] + temp_cols + ['Score']]
                     .rename(columns=options_file.column_translate_dict).reset_index(drop=True)
                     .style.apply(highlight_cols, col_dict=options_file.col_color_dict)
                     .format(options_file.col_decimals_place_dict)
                     )
            sel_configurations_v2.rename(index=str, columns={'Sug.Encomenda': 'Quantity'}, inplace=True)  # ToDo: For some reason this column in particular is not changing its name by way of the renaming argument in the previous st.write. This is a temporary solution

            if sel_min_number_of_configuration > sel_configurations_v2.shape[0]:
                placeholder_value.error("Alerta: Número mínimo de configurações é superior ao número de configurações disponíveis para este modelo ({}).".format(sel_configurations_v2.shape[0]))

            total_sales_after_order = total_sales + sel_configurations_v2['Quantity'].sum()
            sel_configurations_v2['nedc_after_order'] = sel_configurations_v2['Quantity'] * sel_configurations_v2['NEDC']
            sel_configurations_v2['wltp_after_order'] = sel_configurations_v2['Quantity'] * sel_configurations_v2['WLTP']
            co2_nedc_after_order = co2_nedc + sel_configurations_v2['nedc_after_order'].sum()
            co2_wltp_after_order = co2_wltp + sel_configurations_v2['wltp_after_order'].sum()
            co2_nedc_per_vehicle_after_order = co2_nedc_after_order / total_sales_after_order
            co2_wltp_per_vehicle_after_order = co2_wltp_after_order / total_sales_after_order

            co2_nedc_per_vehicle_evolution = co2_nedc_per_vehicle_after_order - co2_nedc_before_order
            if co2_nedc_per_vehicle_evolution > 0:
                st.markdown("Situação Co2 (NEDC) após esta encomenda: {:.2f}(<span style='color:red'>+{:.2f}</span>) gCo2/km".format(co2_nedc_per_vehicle_after_order, co2_nedc_per_vehicle_evolution), unsafe_allow_html=True)
            elif co2_nedc_per_vehicle_evolution < 0:
                st.markdown("Situação Co2 (NEDC) após esta encomenda: {:.2f}(<span style='color:green'>{:.2f}</span>) gCo2/km".format(co2_nedc_per_vehicle_after_order, co2_nedc_per_vehicle_evolution), unsafe_allow_html=True)
            else:
                st.markdown("Situação Co2 (NEDC) sem alterações após esta encomenda.")

            co2_wltp_per_vehicle_evolution = co2_wltp_per_vehicle_after_order - co2_wltp_before_order
            if co2_wltp_per_vehicle_evolution > 0:
                st.markdown("Situação Co2 (WLTP) após esta encomenda: {:.2f}(<span style='color:red'>+{:.2f}</span>) gCo2/km".format(co2_wltp_per_vehicle_after_order, co2_wltp_per_vehicle_evolution), unsafe_allow_html=True)
            elif co2_wltp_per_vehicle_evolution < 0:
                st.markdown("Situação Co2 (WLTP) após esta encomenda: {:.2f}(<span style='color:green'>{:.2f}</span>) gCo2/km".format(co2_wltp_per_vehicle_after_order, co2_wltp_per_vehicle_evolution), unsafe_allow_html=True)
            else:
                st.markdown("Situação Co2 (WLTP) sem alterações após esta encomenda.")

            df_to_export = file_export_preparation(sel_configurations_v2[['Quantity', 'PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc', 'PT_PDB_Exterior_Color_Desc', 'PT_PDB_Interior_Color_Desc']].reset_index(drop=True), live_ocn_df, sel_brand)
            file_export(df_to_export.rename(columns=options_file.column_translate_dict), 'Sugestão_Encomenda_{}_{}_'.format(sel_brand, sel_model))

            sel_configurations_v2['Configuration_Concat'] = sel_configurations_v2['PT_PDB_Model_Desc'] + ', ' + sel_configurations_v2['PT_PDB_Engine_Desc'] + ', ' + sel_configurations_v2['PT_PDB_Transmission_Type_Desc'] + ', ' + sel_configurations_v2['PT_PDB_Version_Desc'] + ', ' +  sel_configurations_v2['PT_PDB_Exterior_Color_Desc'] + ', ' + sel_configurations_v2['PT_PDB_Interior_Color_Desc']
            st.markdown("<h3 style='text-align: left;'>Configuração a explorar:</h3>", unsafe_allow_html=True)
            sel_config = st.selectbox('', ['-'] + [x for x in sel_configurations_v2['Configuration_Concat'].unique()], index=0)
            if sel_config != '-':
                validation_dfs_v2 = get_validation_info(sel_configurations_v2, sel_config)
                validation_dfs_titles = ['Vendas ({}):', 'Propostas ({}, para os últimos 3 meses):', 'Stock ({}):', 'Plano de Vendas, passo 1:', 'Plano de Vendas, passo 2 - selecionando os máximos de valores de quantidade por período:',
                                         'Plano de Vendas, passo 3 - aplicando a seguinte fórmula: *Objetivo de Cobertura de Stock no mês N = 2,5 \* média das vendas de (N, N+1, N+2, N+3 e N+4)*:']

                validation_queries_display(str_title=validation_dfs_titles[0], int_count=int(validation_dfs_v2[0]['Quantity_CHS'].sum()), df_data=validation_dfs_v2[0][[x for x in list(validation_dfs_v2[0]) if x != 'Last_Modified_Date']])
                validation_queries_display(str_title=validation_dfs_titles[1], int_count=validation_dfs_v2[1].shape[0], df_data=validation_dfs_v2[1][[x for x in list(validation_dfs_v2[1]) if x != 'Last_Modified_Date']])
                validation_queries_display(str_title=validation_dfs_titles[3], df_data=validation_dfs_v2[3][[x for x in list(validation_dfs_v2[3]) if x != 'Last_Modified_Date']])
                validation_queries_display(str_title=validation_dfs_titles[4], df_data=validation_dfs_v2[4][[x for x in list(validation_dfs_v2[4]) if x != 'Last_Modified_Date']])
                validation_queries_display(str_title=validation_dfs_titles[5], df_data=validation_dfs_v2[5][[x for x in list(validation_dfs_v2[5]) if x != 'Last_Modified_Date']])
                validation_queries_display(str_title=validation_dfs_titles[2], int_count=validation_dfs_v2[2].shape[0], df_data=validation_dfs_v2[2][[x for x in list(validation_dfs_v2[2]) if x != 'Last_Modified_Date']])

        else:
            return

    else:
        st.markdown("<p style='text-align: center;'>Por favor escolha uma marca e modelo para sugerir a respetiva encomenda.</p>", unsafe_allow_html=True)


def validation_queries_display(str_title, int_count=None, df_data=pd.DataFrame()):

    st.write(str_title.format(int_count), df_data.rename(columns=options_file.column_translate_dict))

    return


def highlight_cols(s, col_dict):

    if s.name in col_dict.keys():
        return ['background-color: {}'.format(col_dict[s.name])] * len(s)
    return [''] * len(s)


@st.cache(show_spinner=False)
def get_validation_info(sel_configurations_v2, sel_config):
    validations_dfs_v2 = []

    sel_config_model = sel_configurations_v2.loc[sel_configurations_v2['Configuration_Concat'] == sel_config, 'PT_PDB_Model_Desc'].values[0]
    sel_config_engine = sel_configurations_v2.loc[sel_configurations_v2['Configuration_Concat'] == sel_config, 'PT_PDB_Engine_Desc'].values[0]
    sel_config_transmission = sel_configurations_v2.loc[sel_configurations_v2['Configuration_Concat'] == sel_config, 'PT_PDB_Transmission_Type_Desc'].values[0]
    sel_config_version = sel_configurations_v2.loc[sel_configurations_v2['Configuration_Concat'] == sel_config, 'PT_PDB_Version_Desc'].values[0]
    sel_config_ext_color = sel_configurations_v2.loc[sel_configurations_v2['Configuration_Concat'] == sel_config, 'PT_PDB_Exterior_Color_Desc'].values[0]
    sel_config_int_color = sel_configurations_v2.loc[sel_configurations_v2['Configuration_Concat'] == sel_config, 'PT_PDB_Interior_Color_Desc'].values[0]

    validation_queries_v2 = [options_file.sales_validation_query_v2,
                             options_file.proposals_validation_query_v2,
                             options_file.stock_validation_query_v2,
                             options_file.sales_plan_validation_query_step_1_v2,
                             options_file.sales_plan_validation_query_step_2_v2
                             ]
    validation_queries_dtr_v2 = [options_file.sales_plan_validation_query_step_3_v2]

    validation_tables = [options_file.sql_info['sales_validation_table'],
                         options_file.sql_info['proposals_validation_table'],
                         options_file.sql_info['stock_validation_table'],
                         options_file.sql_info['sales_plan_step_1_validation_table'],
                         options_file.sql_info['sales_plan_step_2_validation_table']
                         ]
    validation_tables_dtr = [options_file.sql_info['sales_plan_step_3_validation_table']]

    for query, validation_table in zip(validation_queries_v2, validation_tables):
        query.format(options_file.sql_info['database_final'], validation_table, sel_config_model, sel_config_engine, sel_config_version, sel_config_transmission, sel_config_ext_color, sel_config_int_color)
        validation_df_v2 = run_single_query(options_file.DSN_MLG_PRD, options_file.sql_info['database_final'], options_file, query.format(options_file.sql_info['database_final'], validation_table, sel_config_model, sel_config_engine, sel_config_version, sel_config_transmission, sel_config_ext_color, sel_config_int_color))
        validations_dfs_v2.append(validation_df_v2)

    for query, validation_table in zip(validation_queries_dtr_v2, validation_tables_dtr):
        query.format(options_file.sql_info['database_source'], validation_table, sel_config_model, sel_config_engine, sel_config_version, sel_config_transmission, sel_config_ext_color, sel_config_int_color)
        validation_df_v2 = run_single_query(options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'], options_file, query.format(options_file.sql_info['database_source'], validation_table, sel_config_model, sel_config_engine, sel_config_version, sel_config_transmission, sel_config_ext_color, sel_config_int_color))
        validations_dfs_v2.append(validation_df_v2)

    return validations_dfs_v2


def run_single_query(dsn, database, options_file_in, query):
    query_result = level_1_a_data_acquisition.sql_retrieve_df_specified_query(dsn, database, options_file_in, query)

    return query_result


def period_calculation():
    start_date = date(date.today().year, 1, 1)
    end_date = date.today()  # + relativedelta(months=+2)  # ToDo might be needed later on. Also, from dateutil.relativedelta import relativedelta
    start_date_year = start_date.year
    # end_date_year = end_date.year
    end_date_month_number = end_date.month

    # sel_months_index = total_months_list[0:end_date_month_number - 1]
    return end_date_month_number, start_date_year


@st.cache(show_spinner=False)
def co2_processing(df, end_date_month_number, current_year):
    # The following condition is only for the first year of the month, where, even though January isn't complete, we default to use the sales plan for that month;
    if end_date_month_number == 1:
        df.loc[:, 'Sales_Sum'] = df.loc[(df['Sales_Plan_Year'] == current_year), :].loc[:, ['Jan']].sum(axis=1)  # First, filter dataframe for the current year of the sales plan, then select only the running year months;
    else:
        df.loc[:, 'Sales_Sum'] = df.loc[(df['Sales_Plan_Year'] == current_year), :].loc[:, total_months_list[0:end_date_month_number - 1]].sum(axis=1)  # First, filter dataframe for the current year of the sales plan, then select only the running year months;

    df.loc[:, 'Sales_Sum_Times_Co2_WLTP'] = df['Sales_Sum'] * df['WLTP_CO2']
    df.loc[:, 'Sales_Sum_Times_Co2_NEDC'] = df['Sales_Sum'] * df['NEDC_CO2']
    co2_wltp_sum = df['Sales_Sum_Times_Co2_WLTP'].sum(axis=0)
    co2_nedc_sum = df['Sales_Sum_Times_Co2_NEDC'].sum(axis=0)
    total_sales = df['Sales_Sum'].sum(axis=0)

    return co2_nedc_sum, co2_wltp_sum, total_sales


def model_lowercase(df):

    df['PT_PDB_Model_Desc'] = df['PT_PDB_Model_Desc'].apply(lambda x: x.lower())

    return df


@st.cache(show_spinner=False, allow_output_mutation=True, ttl=60*60*24)
def get_data_v2(options_file_in, dsn, db, table, query_filter=None, model_flag=0):

    df = level_1_a_data_acquisition.sql_retrieve_df(dsn, db, table, options_file_in, query_filters=query_filter)

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


@st.cache(show_spinner=False)
def score_calculation(x, sel_daysinstock_score_weight, sel_margin_score_weight, sel_margin_ratio_score_weight, sel_qty_sold_score_weight, sel_proposals_score_weight, sel_oc_stock_diff_score_weight, sel_co2_nedc_score_weight):

    y = \
        x['Avg_DaysInStock_Global_normalized'] * sel_daysinstock_score_weight \
        + x['TotalGrossMarginPerc_normalized'] * sel_margin_score_weight \
        + x['MarginRatio_normalized'] * sel_margin_ratio_score_weight \
        + x['Sum_Qty_CHS_normalized'] * sel_qty_sold_score_weight \
        + x['Proposals_VDC_normalized'] * sel_proposals_score_weight \
        + x['Stock_OC_Diff_normalized'] * sel_oc_stock_diff_score_weight \
        + x['NEDC_normalized'] * sel_co2_nedc_score_weight

    return y


@st.cache(show_spinner=False, ttl=60*60*24*12, allow_output_mutation=True, suppress_st_warning=True)
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

    data_filtered.sort_values(by='Score', ascending=False, inplace=True)
    return data_filtered


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


def quantity_processing_v2(df, sel_order_size):
    total_score = df['Score'].sum()

    df.loc[:, 'Score Weight'] = df.loc[:, 'Score'] / total_score
    df.loc[:, 'Weighted Order'] = df.loc[:, 'Score Weight'] * sel_order_size
    df.loc[:, 'Quantity'] = df.loc[:, 'Weighted Order'].round()

    return df


def solution_saving(df, sel_model, client_lvl_cols_in, client_lvl_sels):
    truncate_query_part_2 = ' '.join(['and {} = \'{}\''.format(x, y) for x, y in zip(client_lvl_cols_in, client_lvl_sels) if y != '-'])

    df = client_replacement(df, client_lvl_cols_in, client_lvl_sels)  # Replaces the values of Client's Levels by the actual values selected for this solution

    level_1_e_deployment.sql_truncate(options_file.DSN_SRV3_PRD, options_file, options_file.sql_info['database_source'], options_file.sql_info['optimization_solution_table'], query=truncate_query.format(sel_model) + truncate_query_part_2)

    level_1_e_deployment.sql_inject(df, options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'], options_file.sql_info['optimization_solution_table'], options_file,
                                    configuration_parameters + client_lvl_cols_in + ['Quantity', 'Average_Score_Euros', 'ML_VehicleData_Code'], check_date=1)

    st.write('Sugestão gravada com sucesso.')
    return


def file_export(df, file_name):
    current_date, _ = level_1_e_deployment.time_tags(format_date='%Y%m%d')

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Gravar Sugestão</a> (carregar botão direito e Guardar Link como: {file_name + current_date}.csv)'
    st.markdown(href, unsafe_allow_html=True)


def file_export_preparation(df, ocn_df, sel_brand):

    if options_file.nlr_code_desc[sel_brand] == 702:
        df_joined = df_join_function(df,
                                     ocn_df[['Model_Code', 'OCN', 'PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc']]
                                     .set_index(['PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc']),
                                     on=['PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc'],
                                     how='left'
                                     )
    elif options_file.nlr_code_desc[sel_brand] == 706:
        df_joined = df_join_function(df,
                                     ocn_df[['Model_Code', 'PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc']]
                                     .set_index(['PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc']),
                                     on=['PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc'],
                                     how='left'
                                     )
    else:
        raise ValueError('Unknown Selected Brand - {}'.format(sel_brand))

    return df_joined


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
        raise RerunException(RerunData())

