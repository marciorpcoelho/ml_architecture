import streamlit as st
from streamlit.script_runner import RerunException
from streamlit.script_request_queue import RerunData
import os
import sys
import pyodbc
import time
from traceback import format_exc

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
sys.path.insert(1, base_path)
import level_2_pa_servicedesk_2244_options as options_file
import modules.level_1_a_data_acquisition as level_1_a_data_acquisition
import modules.level_1_e_deployment as level_1_e_deployment
from modules.level_0_performance_report import log_record, error_upload
import modules.SessionState as SessionState
from plotly import graph_objs as go

st.beta_set_page_config(page_title='Classificação de Pedidos - Service Desk Rigor')

session_state = SessionState.get(run_id=0, save_button_pressed_flag=0, overwrite_button_pressed_flag=0, update_final_table_button_pressed_flag=0, first_run=1)

truncate_query = ''' 
    DELETE 
    FROM [BI_RCG].[dbo].[BI_SDK_Fact_DW_Requests_Manual_Classification]
    WHERE Request_Num = '{}'  
    '''

# Updates BI_SDK_Fact_Requests_Month_Detail with the new labels from BI_SDK_Fact_DW_Requests_Manual_Classification
update_query = '''
    UPDATE dbo.BI_SDK_Fact_Requests_Month_Detail
    SET Label = Class.Label
    FROM dbo.BI_SDK_Fact_Requests_Month_Detail AS Fact
    INNER JOIN BI_SDK_Fact_DW_Requests_Manual_Classification AS Class ON Class.Request_Num = Fact.Request_Num
    '''

column_translate_dict = {
    'Request_Num': 'Nº Pedido',
    'Label': 'Classificação',
    'Date': 'Data'
}

"""
# Classificação Pedidos Service Desk
Classificação Manual de Pedidos do Service Desk
"""

# Hides the menu's hamburguer
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


def main():
    manual_classified_requests_df = get_data_non_cached(options_file, options_file.DSN, options_file.sql_info['database_source'], options_file.sql_info['aux_table'], columns='*')
    auto_classified_requests_df = get_data(options_file, options_file.DSN, options_file.sql_info['database_source'], options_file.sql_info['final_table'], columns='*', query_filters={'Label': 'Não Definido'})
    current_classes = get_data(options_file, options_file.DSN_MLG, options_file.sql_info['database_final'], options_file.sql_info['keywords_table'][0], columns=['Keyword_Group'])

    auto_classified_requests_df = auto_classified_requests_df.sort_values(by='Open_Date', ascending=False)

    st.sidebar.text('Histórico de classificação:')
    last_history = manual_classified_requests_df[['Request_Num', 'Label', 'Date']].rename(columns=column_translate_dict).tail(5)
    st.sidebar.table(last_history)

    st.sidebar.text('Para atualizar a tabela final no DW:')
    if st.sidebar.button('Atualizar DataWarehouse') or session_state.update_final_table_button_pressed_flag == 1:
        session_state.update_final_table_button_pressed_flag = 1
        if not manual_classified_requests_df.shape[0]:
            st.sidebar.text('ERRO - Não existem atualmente pedidos')
            st.sidebar.text('classificados manualmente.')
            session_state.update_final_table_button_pressed_flag = 0
        else:
            update_dw(update_query, options_file, options_file.DSN, options_file.sql_info['database_source'])
            session_state.update_final_table_button_pressed_flag = 0

    if manual_classified_requests_df.shape[0]:
        manual_classified_reqs = manual_classified_requests_df['Request_Num'].unique()
        auto_classified_requests_df = auto_classified_requests_df.loc[~auto_classified_requests_df['Request_Num'].isin(manual_classified_reqs), :]

    fig = go.Figure(data=[go.Table(
        columnwidth=[120, 600, 120],
        header=dict(
            values=[['Nº Pedido'], ['Descrição'], ['Data Abertura']],
            align=['center', 'center', 'center'],
            ),
        cells=dict(
            values=[auto_classified_requests_df['Request_Num'], auto_classified_requests_df['Description'], auto_classified_requests_df['Open_Date']],
            align=['center', 'right', 'center'],
            )
        )
        ])
    fig.update_layout(width=800)
    st.write(fig)

    st.write('Nº Pedidos com classe Não Definido: {}'.format(auto_classified_requests_df['Request_Num'].nunique()))

    sel_req = st.multiselect('Por favor escolha um Pedido:', auto_classified_requests_df['Request_Num'].unique(), key=session_state.run_id)

    if len(sel_req) == 1:
        description = auto_classified_requests_df.loc[auto_classified_requests_df['Request_Num'] == sel_req[0]]['Description'].values[0]
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
                        solution_saving(options_file, options_file.DSN, options_file.sql_info['database_source'], options_file.sql_info['aux_table'], sel_req[0], sel_label[0])
                        session_state.overwrite_button_pressed_flag, session_state.save_button_pressed_flag = 0, 0
                        session_state.run_id += 1
                        time.sleep(0.1)
                        raise RerunException(RerunData())

                else:
                    solution_saving(options_file, options_file.DSN, options_file.sql_info['database_source'], options_file.sql_info['aux_table'], sel_req[0], sel_label[0])
                    session_state.overwrite_button_pressed_flag, session_state.save_button_pressed_flag = 0, 0
                    session_state.run_id += 1
                    time.sleep(0.1)
                    raise RerunException(RerunData())

        elif len(sel_label) > 1:
            st.error('Por favor escolha apenas uma classe.')
    elif len(sel_req) > 1:
        st.error('Por favor escolha um pedido para classificar de cada vez.')


@st.cache(show_spinner=False, ttl=60*60*24)
def get_data(options_file_in, dsn, db, view, columns, query_filters=0):
    df = level_1_a_data_acquisition.sql_retrieve_df(dsn, db, view, options_file_in, columns, query_filters)

    return df


def get_data_non_cached(options_file_in, dsn, db, view, columns, query_filters=0):
    df = level_1_a_data_acquisition.sql_retrieve_df(dsn, db, view, options_file_in, columns, query_filters)

    return df


def solution_saving(options_file_in, dsn, db, view, sel_req, sel_label):
    level_1_e_deployment.sql_truncate(dsn, options_file_in, db, view, query=truncate_query.format(sel_req))

    level_1_e_deployment.sql_inject_single_line(dsn, options_file_in.UID, options_file_in.PWD, db, view, [sel_req, sel_label], check_date=1)

    st.write('Sugestão gravada com sucesso - {}: {}'.format(sel_req, sel_label))


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
        error_upload(options_file, project_identifier, format_exc(), exception_desc, error_flag=1, solution_type='OPR')
        session_state.run_id += 1
        st.error('AVISO: Ocorreu um erro. Os administradores desta página foram notificados com informação do erro e este será corrigido assim que possível. Entretanto, esta aplicação será reiniciada. Obrigado pela sua compreensão.')
        time.sleep(10)
        raise RerunException(RerunData())

