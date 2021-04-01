import os
import sys
import shap
import time
# import logging
import pickle
import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from datetime import date
from joblib import load
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit.components.v1 as components
from traceback import format_exc
from dateutil.relativedelta import relativedelta
from streamlit.script_runner import RerunException
from streamlit.script_request_queue import RerunData
base_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', ''))
sys.path.insert(1, base_path)
import level_2_finlog_autoseguro_cost_prediction_options as options_file
from modules.level_1_a_data_acquisition import sql_retrieve_df_specified_query
from modules.level_1_b_data_processing import feat_eng, apply_ohenc
from modules.level_0_performance_report import log_record, error_upload

st.set_page_config(page_title='Previsão de Risco - Finlog Demo')

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Finlog - Demo</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Previsão de Probabilidade de Sinistro</h2>", unsafe_allow_html=True)


def main():
    enc_LL = load(base_path + options_file.enc_LL_path)
    enc_AR = load(base_path + options_file.enc_AR_path)
    enc_FI = load(base_path + options_file.enc_FI_path)
    enc_Make = load(base_path + options_file.enc_Make_path)
    enc_Fuel = load(base_path + options_file.enc_Fuel_path)
    enc_Vehicle_Tipology = load(base_path + options_file.enc_Vehicle_Tipology_path)
    # enc_Client_type = load(base_path + options_file.enc_Client_type_path)
    enc_Num_Vehicles_Total = load(base_path + options_file.enc_Num_Vehicles_Total_path)
    enc_Num_Vehicles_Finlog = load(base_path + options_file.enc_Num_Vehicles_Finlog_path)
    enc_Customer_Group = load(base_path + options_file.enc_Customer_Group_path)
    # enc_Customer_Name = load(base_path + options_file.enc_Customer_Name)

    col1, col2 = st.sidebar.beta_columns(2)

    st.sidebar.title('Parâmetros:')

    tipology_list = enc_Vehicle_Tipology.categories_[0]
    tipology_filter = st.sidebar.selectbox('Tipologia do veículo:', ['-'] + list(tipology_list), index=0)

    make_list = list(enc_Make.categories_[0])
    make_filter = st.sidebar.multiselect('Marca do veículo', make_list)
    if not len(make_filter):
        make_filter = make_list

    fuel_list = list(enc_Fuel.categories_[0])
    fuel_list = [x for x in fuel_list if str(x) != '0']
    fuel_filter = st.sidebar.multiselect('Combustível do veículo', fuel_list)
    if not len(fuel_filter):
        fuel_filter = fuel_list

    customer_group_list = enc_Customer_Group.categories_[0]
    customer_group_filter = st.sidebar.selectbox('Grupo de Empresas:', ['-'] + list(customer_group_list), index=0)

    LL_list = list(enc_LL.categories_[0])
    LL_filter = st.sidebar.multiselect('Cobertura Responsabilidade Civil (LL):', LL_list)
    if not len(LL_filter):
        LL_filter = LL_list

    AR_list = list(enc_AR.categories_[0])
    AR_filter = st.sidebar.multiselect('Cobertura Danos Próprios (AR):', AR_list)
    if not len(AR_filter):
        AR_filter = AR_list

    FI_list = list(enc_FI.categories_[0])
    FI_filter = st.sidebar.multiselect('Cobertura Quebra Isolada de Vidros (FI):', FI_list)
    if not len(FI_filter):
        FI_filter = FI_list

    contract_start_date = date.today()

    fleet_size_total_list = enc_Num_Vehicles_Total.categories_[0]
    fleet_size_total_filter = st.sidebar.selectbox('Dimensão total da frota:', ['-'] + list(fleet_size_total_list), index=0)

    fleet_size_finlog_list = enc_Num_Vehicles_Finlog.categories_[0]
    fleet_size_finlog_filter = st.sidebar.selectbox('Dimensão da frota com Auto Seguro:', ['-'] + list(fleet_size_finlog_list), index=0)

    contract_duration_limits = st.sidebar.slider('Duração do contrato', 0, 120, (0, 120), 1)
    contract_duration_min = str(contract_duration_limits[0])
    contract_duration_max = str(contract_duration_limits[1])

    km_year_limits = st.sidebar.slider('Estimativa de milhares de km por ano:', 0, 100, (0, 100), 5)
    km_year_limits_min = str(km_year_limits[0])
    km_year_limits_max = str(km_year_limits[1])

    if col1.button('Perfil 1'):
        tipology_filter = 'Ligeiros de Passageiros'
        make_filter = ['AUDI', 'CITROEN']
        fuel_filter = fuel_filter
        customer_group_filter = 'Banco BPI'
        # customer_name_filter = ['Banco BPI, S.A.', 'BANCO PORTUGUES DE INVESTIMENTO, SA']
        LL_filter = LL_filter
        AR_filter = ['Franquia 8%', 'Franquia 4%']
        FI_filter = 'Até €1.000/Ano'
        fleet_size_total_filter = '130-169'
        fleet_size_finlog_filter = '20-29'
        contract_duration_min = '48'
        contract_duration_max = '120'
        km_year_limits_min = '15'
        km_year_limits_max = '30'

        st.write('''
               Tipologia do veículo: **{}**\n
               Marca: **{}**\n
               Combustível: **{}**\n
               Grupo de Empresas: **{}**\n
               Franquia Responsabilidade Civil (LL): **{}**\n
               Franquia Danos Próprios (AR): **{}**\n
               Franquia QIV (FI): **{}**\n
               Dimensão total da frota: **{}**\n
               Dimensão da frota com Auto Seguro: **{}**\n
               Duração do contrato: **{} a {} meses**\n
               Estimativa de km (milhares) por ano: **{} a {}**\n
               **---------- ##### ----------** '''.format(tipology_filter, make_filter, fuel_filter, customer_group_filter, LL_filter, AR_filter, FI_filter, fleet_size_total_filter, fleet_size_finlog_filter, contract_duration_min, contract_duration_max, km_year_limits_min, km_year_limits_max)
                 )

    if col2.button('Perfil 2'):
        tipology_filter = 'Ligeiros de Passageiros'
        make_filter = 'RENAULT'
        fuel_filter = 'Gasóleo'
        customer_group_filter = 'Empresa'
        # customer_name_filter = 'Modelo Continente Hipermercados, S.A.'
        LL_filter = '€50.000.000'
        AR_filter = 'Franquia 2%'
        FI_filter = 'Até €1.000/Ano'
        fleet_size_total_filter = '250-449'
        fleet_size_finlog_filter = '>180'
        contract_duration_min = '2'
        contract_duration_max = '60'
        km_year_limits_min = '50'
        km_year_limits_max = '90'

        st.write('''
               Tipologia do veículo: **{}**\n
               Marca: **{}**\n
               Combustível: **{}**\n
               Grupo de Empresas: **{}**\n
               Franquia Responsabilidade Civil (LL): **{}**\n
               Franquia Danos Próprios (AR): **{}**\n
               Franquia QIV (FI): **{}**\n
               Dimensão total da frota: **{}**\n
               Dimensão da frota com Auto Seguro: **{}**\n
               Duração do contrato: **{} a {} meses**\n
               Estimativa de km (milhares) por ano: **{} a {}**\n
               **---------- ##### ----------** '''.format(tipology_filter, make_filter, fuel_filter, customer_group_filter, LL_filter, AR_filter, FI_filter, fleet_size_total_filter, fleet_size_finlog_filter, contract_duration_min, contract_duration_max, km_year_limits_min, km_year_limits_max)
                 )

    if tipology_filter == '-':
        st.text("Por favor selecione a tipologia do veículo")
    elif customer_group_filter == '-':
        st.text("Por favor selecione o grupo de empresas")
    elif fleet_size_total_filter == '-':
        st.text("Por favor selecione a dimensão total da frota")
    elif fleet_size_finlog_filter == '-':
        st.text("Por favor selecione a dimensão da frota com Auto Seguro")
    else:

        df = rows_to_predict_creation(options_file.DSN_MLG_PRD, 'BI_MLG', options_file, FI_filter, LL_filter, AR_filter, tipology_filter, make_filter, fuel_filter, customer_group_filter, fleet_size_total_filter, fleet_size_finlog_filter, contract_duration_min, contract_duration_max, km_year_limits_min, km_year_limits_max, contract_start_date)

        df = feat_eng(df)

        df = df.drop(['target_accident', 'target_cost', 'target_QIV', 'target_DP'], axis=1)

        df.Fuel = np.nan_to_num(df.Fuel)

        clf = load(base_path + '/' + options_file.MODEL_PATH)
        clf_qiv = load(base_path + '/' + options_file.MODEL_PATH_QIV)
        clf_dp = load(base_path + '/' + options_file.MODEL_PATH_DP)

        df = apply_ohenc('LL', df, enc_LL)
        df = apply_ohenc('AR', df, enc_AR)
        df = apply_ohenc('FI', df, enc_FI)
        df = apply_ohenc('Make', df, enc_Make)
        df = apply_ohenc('Fuel', df, enc_Fuel)
        df = apply_ohenc('Vehicle_Tipology', df, enc_Vehicle_Tipology)
        # df = apply_ohenc('Client_type', df, enc_Client_type)
        df = apply_ohenc('Num_Vehicles_Total', df, enc_Num_Vehicles_Total)
        df = apply_ohenc('Num_Vehicles_Finlog', df, enc_Num_Vehicles_Finlog)
        df = apply_ohenc('Customer_Group', df, enc_Customer_Group)
        # df = apply_ohenc('Customer_Name', df, enc_Customer_Name)

        df = df.astype('float32')
        prediction_proba = clf.predict_proba(df)
        prediction_proba_qiv = clf_qiv.predict_proba(df)
        prediction_proba_dp = clf_dp.predict_proba(df)

        prediction = np.mean([x[1] for x in prediction_proba])
        prediction_qiv = np.mean([x[1] for x in prediction_proba_qiv])
        prediction_dp = np.mean([x[1] for x in prediction_proba_dp])

        st.markdown("<h2 style='text-align: left;'> Sinistro: </h2>", unsafe_allow_html=True)
        st.markdown("Probabilidade de sinistro: **{:.1f}%**".format(prediction*100))

        df_test_prob_full = pd.read_csv(base_path + '/' + options_file.DATA_PROB_PATH_ALL_COST, index_col=0)

        # Casos Semelhantes
        df_test_prob = df_test_prob_full.copy()

        df_test_prob = df_test_prob[df_test_prob['pred_prob'] < 0.95]

        df_test_prob['Make_Reverse_OHE'] = enc_Make.inverse_transform(df_test_prob[[x for x in list(df_test_prob) if x.startswith('Make')]])
        # df_test_prob['Customer_Name_Reverse_OHE'] = enc_Customer_Name.inverse_transform(df_test_prob[[x for x in list(df_test_prob) if x.startswith('Customer_Name')]])
        df_test_prob['Vehicle_Tipology_Reverse_OHE'] = enc_Vehicle_Tipology.inverse_transform(df_test_prob[[x for x in list(df_test_prob) if x.startswith('Vehicle_Tipology')]])

        if type(make_filter) == str:
            make_filter = [make_filter]
        if type(fuel_filter) == str:
            fuel_filter = [fuel_filter]
        if type(LL_filter) == str:
            LL_filter = [LL_filter]
        if type(AR_filter) == str:
            AR_filter = [AR_filter]
        if type(FI_filter) == str:
            FI_filter = [FI_filter]

        df_test_prob_similar = df_test_prob[
            # (df_test_prob['Client_type'] == client_type_filter) &
            (df_test_prob['Vehicle_Tipology_Reverse_OHE'] == tipology_filter) &
            (df_test_prob['Make_Reverse_OHE'].isin(make_filter) &
            (df_test_prob['target_cost'] > 0))
        ]

        # Calculate the percentile for this simulation and present it
        num_cases_higher_prob = df_test_prob[df_test_prob['pred_prob'] > prediction].shape[0]
        num_cases_lower_prob = df_test_prob[df_test_prob['pred_prob'] < prediction].shape[0]

        case_percentile = num_cases_lower_prob/(num_cases_lower_prob + num_cases_higher_prob)
        st.write('Este perfil de cliente tem uma probabilidade de sinistro maior que **{:.1f}%** dos casos, considerando *todos os tipos de clientes e veículos*.'.format(case_percentile * 100), unsafe_allow_html=True)

        num_cases_lower_prob_similar = df_test_prob_similar[df_test_prob_similar['pred_prob'] < prediction].shape[0]
        if df_test_prob_similar.shape[0] == 0 or num_cases_lower_prob_similar == 0:

            prob_full = df_test_prob['pred_prob']

            fig, ax = plt.subplots(1, 1)
            sns.kdeplot(prob_full, shade=True, legend=False)
            ax.set(xlim=(0, 1))
            ax.axvline(prediction, 0, 1)
            ax.set_xlabel('% de Sinistro')
            ax.set_title('Dist. de prob. de sinistro:')
            st.pyplot(fig)

            st.write('Não há casos semelhantes de outros clientes para comparação.')

            shap_values_plot(clf, df.rename(columns=options_file.shap_values_column_renaming))

        else:
            num_cases_higher_prob_similar = df_test_prob_similar[df_test_prob_similar['pred_prob'] > prediction].shape[0]
            num_cases_lower_prob_similar = df_test_prob_similar[df_test_prob_similar['pred_prob'] < prediction].shape[0]

            case_percentile = num_cases_lower_prob_similar/(num_cases_lower_prob_similar + num_cases_higher_prob_similar)
            st.write('Este perfil de cliente tem uma probabilidade de sinistro maior que **{:.1f}%** dos casos, considerando apenas *clientes de perfil semelhante*.'.format(case_percentile * 100, 1), unsafe_allow_html=True)

            # Accident probability distributions
            fig, ax = plt.subplots(1, 2)
            ax[0].set_ylabel('Probabilidade')
            prob_full = df_test_prob['pred_prob']
            prob_similar = df_test_prob_similar['pred_prob']

            sns.kdeplot(prob_full, shade=True, ax=ax[0], legend=False)
            sns.kdeplot(prob_similar, shade=True, ax=ax[1], legend=False)
            ax[1].set(xlim=(0, 1))
            ax[1].set_xlabel('% de Sinistro')
            ax[1].axvline(prediction, 0, 1)
            ax[1].set_title('Dist. de prob. de sinistro \n (Mesma Tipologia, Marca e \n Tipo de Cliente)')

            ax[0].set(xlim=(0, 1))
            ax[0].set_xlabel('% de Sinistro')
            ax[0].axvline(prediction, 0, 1)
            ax[0].set_title('Dist. de prob. de sinistro \n (Sem Filtros)')

            st.pyplot(fig)
            plt.clf()

            st.markdown("<h2 style='text-align: left;'> Estudo por Cobertura: </h2>", unsafe_allow_html=True)
            cover_prob_display(prediction_qiv, prediction_dp)

            st.markdown("<h2 style='text-align: left;'> Exploração de probabilidade de sinistro: </h2>", unsafe_allow_html=True)
            shap_values_plot(clf, df.rename(columns=options_file.shap_values_column_renaming))

            df_test_prob_comparable_cases = df_test_prob[
                (df_test_prob['Vehicle_Tipology_Reverse_OHE'] == tipology_filter) &
                (df_test_prob['Make_Reverse_OHE'].isin(make_filter)) &
                (df_test_prob['target_cost'] > 0)
                ].reset_index(drop=True)

            df_test_prob_comparable_cases = df_test_prob_comparable_cases.sort_values('target_cost', ascending=True).reset_index(drop=True)
            num_rows = df_test_prob_comparable_cases.shape[0]
            cutoff_row = int(np.round(num_rows*0.9, 0))

            df_test_prob_comparable_cases = df_test_prob_comparable_cases[0:cutoff_row]

            avg_cost_accidents = np.mean(df_test_prob_comparable_cases[df_test_prob_comparable_cases['target_cost'] > 0].target_cost)

            st.markdown("<h2 style='text-align: left;'> Distribuição de custos de Sinistros: </h2>", unsafe_allow_html=True)
            st.write('O custo de um sinistro em *clientes de perfil semelhante* é em média {:.2f}€ e possui a seguinte distribuição:'.format(avg_cost_accidents), unsafe_allow_html=True)

            repair_cost_similar = df_test_prob_comparable_cases.target_cost
            fig, ax = plt.subplots(1, 1, figsize=(15, 6))
            plt.hist(repair_cost_similar, bins=25)
            plt.title('Distribuição de custos de sinistro', size=15)
            plt.xlabel('Custos por Sinistro (€)', size=15)
            plt.ylabel('Freq. Absoluta por Custo', size=15)

            colors = ['blue', 'white', 'white', 'white']
            labels = [
                      'Custos', 'Média: {:.2f}€'.format(repair_cost_similar.mean()),
                      'Mediana: {:.2f}€'.format(repair_cost_similar.median()),
                      'Mín.: {:.2f}€'.format(repair_cost_similar.min()),
                      'Máx.: {:.2f}€'.format(repair_cost_similar.max())
                      ]

            patches_list = []
            for c, l in zip(colors, labels):
                patches_list.append(mpatches.Patch(color=c, label=l))
            plt.legend(handles=patches_list, fontsize=15)

            st.pyplot(plt)
            plt.clf()

            # # get frequent repairs
            # auth_description = level_1_b_data_processing.get_auth_description(
            #     DSN = options_file.DSN_MLG,
            #     query = options_file.auth_description,
            #     segment_filter = segment_filter,
            #     make_filter = make_filter
            # )

            # generic_description_list = [
            #     'Diversos',
            #     'Rep. Sinistros',
            #     'Valor Não sujeito a IVA',
            #     'Utilitário',
            #     'Rep. Sinistros',
            #     'Franquia',
            #     'Caracterização',
            #     'Citadino',
            #     'Familiar,'
            #     'Combustível',
            #     'Derivado',
            #     'Emblemas',
            #     'Diversos (Consumíveis)',
            #     'Frente',
            #     'Decoração Viatura',
            #     'Entrega & Recolha',
            # ]

            # auth_description = auth_description[~auth_description.Description.isin(generic_description_list)].reset_index(drop = True)

            # auth_description.Custo_medio = np.round(auth_description.Custo_medio,2)

            # auth_description.rename(columns={
            #     'Description': 'Descrição',
            #     'Num_Ocorrencias': 'Número de ocorrências',
            #     'Custo_medio': 'Custo médio',
            # })

            # # Common repairs
            # st.write('As reparações mais comuns em veículos semelhantes são:')

            # st.write(auth_description.head(20))
            #st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
            print('Finished')


def cover_prob_display(prediction_qiv, prediction_dp):
    altair_plot = altair_bar_chart_plot(prediction_qiv, prediction_dp)

    col3, col4 = st.beta_columns(2)
    col3.altair_chart(altair_plot, use_container_width=True)
    qiv_over_dp_increase = 1 - prediction_qiv / prediction_dp
    dp_over_qiv_increase = 1 - prediction_dp / prediction_qiv
    if qiv_over_dp_increase > dp_over_qiv_increase:
        col4.markdown('Este perfil tem **{:.1f}%** mais probabilidade de ativação da Cobertura de Danos Próprios.'.format(qiv_over_dp_increase * 100))
    elif dp_over_qiv_increase > qiv_over_dp_increase:
        col4.markdown('Este perfil tem **{:.1f}%** mais probabilidade de ativação da Cobertura de QIV.'.format(dp_over_qiv_increase * 100))

    return


def altair_bar_chart_plot(prediction_qiv, prediction_dp):
    source_df = pd.DataFrame()
    source_df['Covers'] = ['QIV', 'Danos Próprios']
    source_df['Prob'] = [prediction_qiv*100, prediction_dp*100]
    source_df['Prob'] = source_df['Prob'].round(2)

    altair_bar_chart = alt.Chart(source_df, title='% Ativação por Cobertura').mark_bar().encode(
        x=alt.X('Covers',
                axis=alt.Axis(
                    title='Cobertura',
                    labelAngle=0,
                    )
                ),
        y=alt.Y('Prob',
                axis=alt.Axis(
                    title='% de Ativação'
                    ),
                scale=alt.Scale(
                    domain=(1, 100)
                    )
                ),
    )

    text = altair_bar_chart.mark_text(
        align='center',
        baseline='middle',
        dy=-5
    ).encode(
        text='Prob'
    )

    st.markdown("""
        <style type='text/css'>
            details {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

    return altair_bar_chart + text


@st.cache(show_spinner=False)
def load_data(data_path):

    data = pd.read_csv(data_path)
    data = data[data[['AR', 'FI', 'LL']].isnull().sum(axis=1) < 3]
    return data


def load_pickle(file_path):

    with open(base_path + '/' + file_path, 'rb') as pickled_file:
        d = pickle.load(pickled_file)

    return d


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def shap_values_plot(clf, in_df):

    try:
        df = in_df.drop(['Fuel_x0_0'], axis=0)
    except KeyError:
        df = in_df

    shap.initjs()
    explainer = shap.TreeExplainer(
        model=clf  # ,
        # data = df,
        # check_additivity=False
    )

    shap_values = explainer.shap_values(df)

    st.write('Valores a vermelho sobem a probabilidade de sinistro, enquanto valores a azul fazem o valor descer:')

    st_shap(shap.force_plot(
        base_value=explainer.expected_value[1],
        shap_values=shap_values[1][0],
        feature_names=df.columns,  # col_names,
        # matplotlib=True,
        show=False,
        figsize=(20, 3),
        link='logit')
    )

    return


def rows_to_predict_creation(dsn, db, options_file_in, FI_filter, LL_filter, AR_filter, tipology_filter, make_filter, fuel_filter, customer_group_filter, fleet_size_total_filter, fleet_size_finlog_filter, contract_duration_min, contract_duration_max, km_year_limits_min, km_year_limits_max, contract_start_date):

    sel_values_df = selected_values_df_creation(FI_filter, LL_filter, AR_filter, tipology_filter, make_filter, fuel_filter, customer_group_filter, fleet_size_total_filter, fleet_size_finlog_filter, contract_duration_min, contract_duration_max, km_year_limits_min, km_year_limits_max, contract_start_date)

    fuel_filter = query_list_string_handling(fuel_filter)
    make_filter = query_list_string_handling(make_filter)
    tipology_filter = query_list_string_handling(tipology_filter)
    # customer_name_filter = query_list_string_handling(customer_name_filter)
    customer_group_filter = query_list_string_handling(customer_group_filter)
    fleet_size_total_filter = query_list_string_handling(fleet_size_total_filter)
    fleet_size_finlog_filter = query_list_string_handling(fleet_size_finlog_filter)

    vhe_data = get_data(dsn, db, options_file_in, options_file_in.vehicle_data_query.format(fuel_filter, make_filter, tipology_filter))
    customer_data = get_data(dsn, db, options_file_in, options_file_in.customer_data_query.format(customer_group_filter, fleet_size_total_filter, fleet_size_finlog_filter, contract_duration_min, contract_duration_max, int(km_year_limits_min) * 1000 * (int(contract_duration_min) / 12)))

    # middle_df = pd.merge(sel_values_df, vhe_data, on=['Fuel', 'Make', 'Vehicle_Tipology'])
    if vhe_data.shape[0]:
        for col in options_file_in.vhe_data_col_keys:
            vhe_data[col + '_lower'] = vhe_data[col].str.lower()
            sel_values_df[col + '_lower'] = sel_values_df[col].str.lower()

        middle_df = pd.merge(sel_values_df, vhe_data, on=['Fuel_lower', 'Make_lower', 'Vehicle_Tipology_lower'], suffixes=(None, '_y'))
    elif not vhe_data.shape[0]:
        # st.write('vhe empty')
        middle_df = sel_values_df
        for col in options_file.vehicle_data_cols:
            middle_df[col] = np.nan

    if customer_data.shape[0]:
        for col in options_file_in.customer_data_col_keys:
            customer_data[col + '_lower'] = customer_data[col].str.lower()
            middle_df[col + '_lower'] = middle_df[col].str.lower()

        middle_df = pd.merge(middle_df, customer_data, on=['Customer_Group_lower', 'Num_Vehicles_Total_lower', 'Num_Vehicles_Finlog_lower'], suffixes=(None, '_y'))
    elif not customer_data.shape[0]:
        # st.write('customer empty')
        for col in options_file.customer_data_cols:
            middle_df[col] = np.nan

    cross_join_key_cols = [x for x in list(middle_df) if x.startswith('cross_join_key')]
    middle_df.drop(cross_join_key_cols, axis=1, inplace=True)

    middle_df.drop([x + '_lower' for x in options_file_in.vhe_data_col_keys + options_file_in.customer_data_col_keys if x + '_lower' in list(middle_df)], axis=1, inplace=True)
    middle_df.drop([x for x in list(middle_df) if x.endswith('_y') and x.endswith('_y') in list(middle_df)], axis=1, inplace=True)

    middle_df['contract_customer'] = np.nan
    # middle_df['Client_type'] = np.nan
    # middle_df['Customer_Name'] = np.nan
    middle_df['contract_contract'] = np.nan
    middle_df['Vehicle_No'] = np.nan
    middle_df['Accident_No'] = np.nan
    middle_df['target'] = np.nan
    middle_df['target_QIV'] = np.nan
    middle_df['target_DP'] = np.nan

    key_cols = ['contract_customer', 'contract_contract', 'Vehicle_No', 'Accident_No', 'target', 'FI', 'LL', 'AR', 'Customer_Group', 'Num_Vehicles_Total', 'Num_Vehicles_Finlog', 'Contract_km', 'contract_start_date', 'contract_end_date', 'contract_duration', 'Vehicle_Tipology', 'Make', 'Fuel']

    df_grouped = middle_df.groupby(by=key_cols, as_index=False, dropna=False).mean()

    return df_grouped


def query_list_string_handling(parameter):

    if type(parameter) == list:
        parameter = '\'' + '\', \''.join(parameter) + '\''
    else:
        parameter = '\'' + parameter + '\''

    return parameter


def selected_values_df_creation(FI_filter, LL_filter, AR_filter, tipology_filter, make_filter, fuel_filter, customer_group_filter, fleet_size_total_filter, fleet_size_finlog_filter, contract_duration_min, contract_duration_max, km_year_limits_min, km_year_limits_max, contract_start_date):
    single_value_cols = ['Vehicle_Tipology', 'Customer_Group', 'Num_Vehicles_Total', 'Num_Vehicles_Finlog', 'contract_duration', 'Contract_km', 'contract_start_date']
    df = pd.DataFrame(columns=single_value_cols)
    df['Vehicle_Tipology'] = [tipology_filter]
    df['Customer_Group'] = [customer_group_filter]
    # df['Client_type'] = client_type_filter
    df['Num_Vehicles_Total'] = fleet_size_total_filter
    df['Num_Vehicles_Finlog'] = fleet_size_finlog_filter
    df['contract_duration'] = contract_duration_min
    df['Contract_km'] = int(km_year_limits_min) * 1000 * (int(contract_duration_min) / 12)
    df['contract_start_date'] = contract_start_date.strftime("%Y-%m-%d")
    df['contract_end_date'] = contract_start_date + relativedelta(months=int(contract_duration_min))

    df['cross_join_key'] = 1

    # Multivalues cols:
    multi_value_cols = ['LL', 'AR', 'FI', 'Make', 'Fuel']
    multi_value_cols_values = [LL_filter, AR_filter, FI_filter, make_filter, fuel_filter]
    all_values_df = pd.DataFrame()
    all_values_df['cross_join_key'] = [1]

    for (sel_values, sel_value_col) in zip(multi_value_cols_values, multi_value_cols):
        if type(sel_values) == list:
            sel_val_df = pd.DataFrame(data=sel_values, columns=[sel_value_col])
        else:
            sel_val_df = pd.DataFrame(data=[sel_values], columns=[sel_value_col])

        sel_val_df['cross_join_key'] = 1
        all_values_df = pd.merge(all_values_df, sel_val_df, on='cross_join_key')

    final_df = pd.merge(all_values_df, df, on='cross_join_key')

    return final_df[[x for x in list(final_df) if x != 'cross_join_key']]


def get_data(dsn, db, options_file_in, query):

    df = sql_retrieve_df_specified_query(
        dsn=dsn,
        db=db,
        options_file=options_file_in,
        query=query
    )

    return df


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        project_identifier, exception_desc = options_file.project_id, str(sys.exc_info()[1])
        log_record('OPR Error - ' + exception_desc, project_identifier, flag=2, solution_type='OPR')
        # error_upload(options_file, project_identifier, format_exc(), exception_desc, error_flag=1, solution_type='OPR')
        st.error('AVISO: Ocorreu um erro. Os administradores desta página foram notificados com informação do erro e este será corrigido assim que possível. Entretanto, esta aplicação será reiniciada. Obrigado pela sua compreensão.')
        time.sleep(10)
        raise RerunException(RerunData())
