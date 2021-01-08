import streamlit as st
from streamlit.script_runner import RerunException
from streamlit.script_request_queue import RerunData
import os
import sys
import pyodbc
import time
import pandas as pd
from traceback import format_exc

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
sys.path.insert(1, base_path)
import level_2_pa_servicedesk_2244_options as options_file
import modules.level_1_a_data_acquisition as level_1_a_data_acquisition
import modules.level_1_e_deployment as level_1_e_deployment
from modules.level_0_performance_report import log_record, error_upload
import modules.SessionState as SessionState
from plotly import graph_objs as go

st.set_page_config(page_title='Classificação de Pedidos - Service Desk Rigor', layout="wide")
st.markdown("<h1 style='text-align: center;'>Classificação Pedidos Service Desk</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Classificação Manual de Pedidos do Service Desk</h3>", unsafe_allow_html=True)

session_state = SessionState.get(run_id=0, save_button_pressed_flag=0, overwrite_button_pressed_flag=0, update_final_table_button_pressed_flag=0, first_run=1)

truncate_query = ''' 
    DELETE 
    FROM [BI_RCG].[dbo].[{}]
    WHERE Request_Num = '{}'  
    '''

# Updates BI_SDK_Fact_Requests_Month_Detail with the new labels from BI_SDK_Fact_DW_Requests_Manual_Classification
update_query = '''
    UPDATE dbo.{}
    SET Label = Class.Label
    FROM dbo.{} AS Fact
    INNER JOIN {} AS Class ON Class.Request_Num = Fact.Request_Num
    '''

column_translate_dict = {
    'Request_Num': 'Nº Pedido',
    'Label': 'Classificação',
    'Date': 'Data'
}


# Hides the menu's hamburguer
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


def main():
    manual_classified_requests_df = get_data_non_cached(options_file, options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'], options_file.sql_info['aux_table'], columns='*')
    auto_classified_requests_df = get_data(options_file, options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'], options_file.sql_info['final_table'], columns='*', query_filters={'Classification_Flag': 1})
    current_classes = get_data(options_file, options_file.DSN_MLG_PRD, options_file.sql_info['database_final'], options_file.sql_info['keywords_table'][0], columns=['Keyword_Group'])
    auto_classified_requests_df = auto_classified_requests_df.sort_values(by='Open_Date', ascending=False)

    # sel_page = st.sidebar.radio('', ['Classificações Manuais', 'Classificações via Modelo'], index=0)

    # if sel_page == 'Classificações Manuais':
    st.sidebar.text('Histórico de classificações:')
    last_history = manual_classified_requests_df[['Request_Num', 'Label', 'Date']].rename(columns=column_translate_dict).tail(5)
    st.sidebar.table(last_history)

    sel_prediction = st.sidebar.selectbox('Escolher Classificação via modelo:', ['-'] + [x for x in auto_classified_requests_df['Label'].unique()])

    if sel_prediction != '-':
        filtered_data = filter_data(auto_classified_requests_df, [sel_prediction], ['Label'], [None])
    else:
        filtered_data = auto_classified_requests_df

    st.sidebar.text('Para atualizar a tabela final no DW:')
    if st.sidebar.button('Atualizar DataWarehouse') or session_state.update_final_table_button_pressed_flag == 1:
        session_state.update_final_table_button_pressed_flag = 1
        if not manual_classified_requests_df.shape[0]:
            st.sidebar.text('ERRO - Não existem atualmente pedidos')
            st.sidebar.text('classificados manualmente.')
            session_state.update_final_table_button_pressed_flag = 0
        else:
            update_dw(update_query.format(options_file.sql_info['initial_table_facts'], options_file.sql_info['initial_table_facts'], options_file.sql_info['aux_table']), options_file, options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'])
            session_state.update_final_table_button_pressed_flag = 0

    if manual_classified_requests_df.shape[0]:
        manual_classified_reqs = manual_classified_requests_df['Request_Num'].unique()
        filtered_data = filtered_data.loc[~filtered_data['Request_Num'].isin(manual_classified_reqs), :]

    fig = go.Figure(data=[go.Table(
        columnwidth=[120, 900, 120, 120],
        header=dict(
            values=[['Nº Pedido'], ['Descrição'], ['Data Abertura'], ['Classificação via Modelo']],
            align=['center', 'center', 'center', 'center'],
            ),
        cells=dict(
            values=[filtered_data['Request_Num'], filtered_data['Description'], filtered_data['Open_Date'], filtered_data['Label']],
            align=['center', 'right', 'center', 'center'],
            )
        )
        ])
    fig.update_layout(width=1600)
    st.write(fig)

    st.write('Nº Pedidos: {}'.format(filtered_data['Request_Num'].nunique()))

    sel_req = st.multiselect('Por favor escolha um Pedido:', filtered_data['Request_Num'].unique(), key=session_state.run_id)

    if len(sel_req) == 1:
        description = filtered_data.loc[filtered_data['Request_Num'] == sel_req[0]]['Description'].values[0]
        st.write('Descrição do pedido {}:'.format(sel_req[0]))
        st.write('"" {} ""'.format(description))  # ToDO: add markdown configuration like bold or italic
        sel_label = st.multiselect('Por favor escolha uma Categoria para o pedido {}:'.format(sel_req[0]), current_classes['Keyword_Group'].unique())

        if len(sel_label) == 1:
            previous_label = manual_classified_requests_df.loc[manual_classified_requests_df['Request_Num'] == sel_req[0], 'Label'].values

            if st.button('Gravar Classificação') or session_state.save_button_pressed_flag == 1:
                session_state.save_button_pressed_flag = 1

                if previous_label or session_state.overwrite_button_pressed_flag == 1:
                    st.write('O pedido {} já foi previamente classificado como {}'.format(sel_req[0], previous_label[0]))  # ToDO: add markdown configuration like bold or italic
                    st.write('Pretende substituir pela classe atual?')
                    if st.button('Sim'):
                        solution_saving(options_file, options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'], options_file.sql_info['aux_table'], sel_req[0], sel_label[0])
                        session_state.overwrite_button_pressed_flag, session_state.save_button_pressed_flag = 0, 0
                        session_state.run_id += 1
                        time.sleep(0.1)
                        raise RerunException(RerunData())

                else:
                    solution_saving(options_file, options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'], options_file.sql_info['aux_table'], sel_req[0], sel_label[0])
                    session_state.overwrite_button_pressed_flag, session_state.save_button_pressed_flag = 0, 0
                    session_state.run_id += 1
                    time.sleep(0.1)
                    raise RerunException(RerunData())

        elif len(sel_label) > 1:
            st.error('Por favor escolha apenas uma classe.')
    elif len(sel_req) > 1:
        st.error('Por favor escolha um pedido para classificar de cada vez.')

    # elif sel_page == 'Classificações via Modelo':
    #     model_classified_data = pd.read_csv(base_path + '/dbs/service_desk_dataset_non_classified_scored_max_prob_prepared (1).csv')
    #     model_classified_data_joined = model_classified_data.join(auto_classified_requests_df[['Request_Num', 'Description', 'Open_Date']].set_index('Request_Num'), on='Request_Num', how='left')
    #
    #     st.write('Nº Pedidos classificados via modelo: {}'.format(model_classified_data_joined['Request_Num'].nunique()))
    #
    #     st.sidebar.title('Filtros:')
    #     sel_prediction = st.sidebar.selectbox('Escolher Classificação via modelo:', ['-'] + [x for x in model_classified_data['prediction'].unique()])
    #     sel_confidence_threshold = st.sidebar.number_input('Escolha o limite mínimo de confiança:', min_value=0.0, max_value=100.0, value=0., step=1.0)
    #
    #     if sel_prediction != '-':
    #         filtered_data = filter_data(model_classified_data_joined, [sel_prediction], ['prediction'], [None])
    #     else:
    #         filtered_data = model_classified_data_joined
    #
    #     filtered_data = filter_data(filtered_data, [sel_confidence_threshold], ['Max_Prob'], ['ge'])
    #
    #     fig = go.Figure(data=[go.Table(
    #         columnwidth=[120, 600, 120, 120, 100],
    #         header=dict(
    #             values=[['Nº Pedido'], ['Descrição'], ['Data Abertura'], ['Classificação'], ['% Confiança']],
    #             align=['center', 'center', 'center', 'center', 'center'],
    #         ),
    #         cells=dict(
    #             values=[filtered_data['Request_Num'], filtered_data['Description'], filtered_data['Open_Date'], filtered_data['prediction'], filtered_data['Max_Prob'].round(2)],
    #             align=['center', 'right', 'center', 'center', 'center'],
    #         )
    #     )
    #     ])
    #     fig.update_layout(width=1500)
    #     st.write(fig)
    #
    #     if sel_prediction != '-':
    #         st.write('Nº Pedidos classificados como {}: {}'.format(sel_prediction, filtered_data['Request_Num'].nunique()))


@st.cache(show_spinner=False)
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
        elif operation_value == 'in':
            data_filtered = data_filtered.loc[data_filtered[col_filter].isin(filter_value), :]

    return data_filtered


@st.cache(show_spinner=False, ttl=60*60*24)
def get_data(options_file_in, dsn, db, view, columns, query_filters=0):
    df = level_1_a_data_acquisition.sql_retrieve_df(dsn, db, view, options_file_in, columns, query_filters)

    return df


def get_data_non_cached(options_file_in, dsn, db, view, columns, query_filters=0):
    df = level_1_a_data_acquisition.sql_retrieve_df(dsn, db, view, options_file_in, columns, query_filters)

    return df


def solution_saving(options_file_in, dsn, db, view, sel_req, sel_label):
    level_1_e_deployment.sql_truncate(dsn, options_file_in, db, view, query=truncate_query.format(view, sel_req))

    level_1_e_deployment.sql_inject_single_line(dsn, options_file_in.UID, options_file_in.PWD, db, view, [sel_req, sel_label], check_date=1)

    st.write('Classificação gravada com sucesso - {}: {}'.format(sel_req, sel_label))


def update_dw(update_query_in, options_file_in, dsn, db):

    cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'.format(dsn, options_file_in.UID, options_file_in.PWD, db), searchescape='\\')
    cursor = cnxn.cursor()

    cursor.execute(update_query_in)

    cnxn.commit()
    cursor.close()
    cnxn.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        project_identifier, exception_desc = options_file.project_id, str(sys.exc_info()[1])
        log_record('OPR Error - ' + exception_desc, project_identifier, flag=2, solution_type='OPR')
        # error_upload(options_file, project_identifier, format_exc(), exception_desc, error_flag=1, solution_type='OPR')
        session_state.run_id += 1
        st.error('AVISO: Ocorreu um erro. Os administradores desta página foram notificados com informação do erro e este será corrigido assim que possível. Entretanto, esta aplicação será reiniciada. Obrigado pela sua compreensão.')
        time.sleep(10)
        raise RerunException(RerunData())

