import streamlit as st
import pandas as pd
import os
import sys
import time
from traceback import format_exc
from streamlit.script_runner import RerunException
from streamlit.script_request_queue import RerunData
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
sys.path.insert(1, base_path)
import level_2_order_optimization_hyundai_options as options_file
import modules.level_1_a_data_acquisition as level_1_a_data_acquisition
import modules.SessionState as SessionState
import modules.level_1_e_deployment as level_1_e_deployment
from modules.level_0_performance_report import log_record, error_upload
from plotly import graph_objs as go

st.beta_set_page_config(page_title='Correspondência de Gamas - Importador')

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

session_state = SessionState.get(run_id=0, sel_brand='-', sel_model='-', draws_gama_morta=pd.DataFrame(), high_confidence_matches=pd.DataFrame(), not_high_confidence_matches=pd.DataFrame(), sel_df='-', df_sim=pd.DataFrame(), sel_table='-', validate_button_pressed=0, unmatched_data_filtered=pd.DataFrame())

"""
# Correspondência Gamas - Importador
Correspondência entre Gamas Mortas e Vivas
"""

url_hyperlink = '''
    <a href= "{}" > <p style="text-align:right"> Manual de Utilizador </p></a>
'''.format(options_file.documentation_url_gamas_match_app)
st.markdown(url_hyperlink, unsafe_allow_html=True)


def main():
    data = get_data(options_file)
    unmatched_data = get_data_non_cached(options_file, 0)
    matched_data = get_data_non_cached(options_file, 1)

    sel_goal = st.sidebar.radio('Modo de utilização:', ['Gamas por Corresponder', 'Gamas Correspondidas'], index=0)

    if sel_goal == 'Gamas por Corresponder':
        session_state.validate_button_pressed = 0

        if not unmatched_data.shape[0]:
            st.write('Não existem gamas sem correspondência.')
            return

        else:
            sel_brand = st.sidebar.selectbox('Marca:', ['-'] + list(unmatched_data['PT_PDB_Franchise_Desc'].unique()), index=0)

            if sel_brand != '-':
                unmatched_data = unmatched_data.loc[unmatched_data['PT_PDB_Franchise_Desc'] == sel_brand.upper(), :]
                # print('2 - ', unmatched_data.shape)

                unique_models = [x for x in list(unmatched_data['PT_PDB_Model_Desc'].unique()) if x not in ['H-1', 'H-1 3 lugares', 'H-1 6 lugares', 'H350', 'i20 Coupe', 'i20 VAN']]

                sel_model = st.sidebar.selectbox('Modelo', ['-'] + unique_models, index=0)
                # sel_confidence_threshold = st.sidebar.slider('Grau de Semelhança', min_value=0.0, max_value=1.0, value=0.8, step=0.01)

                if sel_model != '-':
                    if sel_brand != session_state.sel_brand or sel_model != session_state.sel_model:
                        # session_state.sel_brand = sel_brand
                        # session_state.sel_model = sel_model

                        session_state.unmatched_data_filtered = filter_data(unmatched_data, [sel_model], ['PT_PDB_Model_Desc'])
                        matched_data_filtered = filter_data(data, [sel_model, sel_brand], ['PT_PDB_Model_Desc', 'PT_PDB_Franchise_Desc'])

                        st.write('Existem as seguintes gamas por corresponder para a marca {} e modelo {}:'.format(sel_brand, sel_model))

                        display_gamas_mortas = session_state.unmatched_data_filtered.loc[session_state.unmatched_data_filtered['PT_PDB_Commercial_Version_Flag'] == -1, :]['PT_PDB_Commercial_Version_Desc_Old'].rename('Gama').reset_index(drop=True)
                        if display_gamas_mortas.shape[0]:
                            st.write('Gamas Mortas:')
                            st.table(display_gamas_mortas)

                        display_gamas_vivas = session_state.unmatched_data_filtered.loc[session_state.unmatched_data_filtered['PT_PDB_Commercial_Version_Flag'] == 1, :]['PT_PDB_Commercial_Version_Desc_Old'].rename('Gama').reset_index(drop=True)
                        if display_gamas_vivas.shape[0]:
                            st.write('Gamas Vivas:')
                            st.table(display_gamas_vivas)

                        # session_state.gama_viva_per_model = matched_data_filtered['PT_PDB_Commercial_Version_Desc_New'].unique()
                        session_state.gama_viva_per_model = list(matched_data_filtered.loc[matched_data_filtered['PT_PDB_Commercial_Version_Flag'] == 1, 'PT_PDB_Commercial_Version_Desc_Old'].unique()) + list(session_state.unmatched_data_filtered.loc[session_state.unmatched_data_filtered['PT_PDB_Commercial_Version_Flag'] == 1, 'PT_PDB_Commercial_Version_Desc_Old'].unique())
                        # session_state.gama_morta_per_model = matched_data_filtered['PT_PDB_Commercial_Version_Desc_Old'].unique()
                        session_state.gama_morta_per_model = list(matched_data_filtered.loc[matched_data_filtered['PT_PDB_Commercial_Version_Flag'] == -1, 'PT_PDB_Commercial_Version_Desc_Old'].unique()) + list(session_state.unmatched_data_filtered.loc[session_state.unmatched_data_filtered['PT_PDB_Commercial_Version_Flag'] == -1, 'PT_PDB_Commercial_Version_Desc_Old'].unique())

                        unmatched_gamas = list(session_state.unmatched_data_filtered['PT_PDB_Commercial_Version_Desc_Old'].unique())
                        sel_gama = st.selectbox('Por favor escolha uma Gama:', ['-'] + list(session_state.unmatched_data_filtered['PT_PDB_Commercial_Version_Desc_Old'].unique()), index=0, key=session_state.run_id)

                        if sel_gama != '-':
                            sel_gama_flag = session_state.unmatched_data_filtered.loc[session_state.unmatched_data_filtered['PT_PDB_Commercial_Version_Desc_Old'] == sel_gama, 'PT_PDB_Commercial_Version_Flag'].values[0]

                            session_state.df_sim = calculate_cosine_similarity(sel_gama, sel_gama_flag, session_state.gama_viva_per_model, session_state.gama_morta_per_model, unmatched_gamas)

                            # print(matched_data_filtered[matched_data_filtered['PT_PDB_Commercial_Version_Flag'] == 1])
                            # print(session_state.unmatched_data_filtered[session_state.unmatched_data_filtered['PT_PDB_Commercial_Version_Flag'] == -1])

                            if session_state.df_sim.shape[0]:
                                suggestions = session_state.df_sim[['PT_PDB_Commercial_Version_Desc_Old', 'similarity_cosine']].sort_values(by=['similarity_cosine'], ascending=False).head(5).reset_index()
                            else:
                                suggestions = pd.DataFrame()

                            if sel_gama_flag == -1:
                                if session_state.df_sim.shape[0]:
                                    st.table(suggestions[['PT_PDB_Commercial_Version_Desc_Old', 'similarity_cosine']].rename(index=str, columns={'PT_PDB_Commercial_Version_Desc_Old': 'Gama Viva', 'similarity_cosine': 'Grau de Semelhança'}))
                                else:
                                    st.write('Sem Sugestões de Correspondência.')
                                sel_gama_match = st.selectbox('Por favor escolha a correspondente Gama:', ['-', 's/ correspondência'] + list([x for x in session_state.gama_viva_per_model if x not in [' ', '']]), index=0, key=session_state.run_id)

                            elif sel_gama_flag == 1:
                                if session_state.df_sim.shape[0]:
                                    st.table(suggestions[['PT_PDB_Commercial_Version_Desc_Old', 'similarity_cosine']].rename(index=str, columns={'PT_PDB_Commercial_Version_Desc_Old': 'Gama Morta', 'similarity_cosine': 'Grau de Semelhança'}))
                                else:
                                    st.write('Sem Sugestões de Correspondência.')
                                sel_gama_match = st.selectbox('Por favor escolha a correspondente Gama:', ['-', 's/ correspondência'] + list([x for x in session_state.gama_morta_per_model if x not in [' ', '']]), index=0, key=session_state.run_id)

                            if sel_gama_match != '-':
                                if st.button('Validar') or session_state.validate_button_pressed == 1:
                                    session_state.validate_button_pressed = 1

                                    if sel_gama_flag == -1:
                                        if sel_gama_match == 's/ correspondência':
                                            st.write('A gama morta: \n{} não possui correspondência.'.format(sel_gama))
                                        else:
                                            st.write('A gama morta: \n{} corresponde à gama viva \n{}'.format(sel_gama, sel_gama_match))
                                    else:
                                        if sel_gama_match == 's/ correspondência':
                                            st.write('A gama viva: \n{} não possui correspondência.'.format(sel_gama))
                                        else:
                                            st.write('A gama viva: \n{} corresponde à gama morta \n{}'.format(sel_gama, sel_gama_match))

                                    save_function(sel_gama, sel_gama_match, sel_brand, sel_model)

                                    session_state.validate_button_pressed = 0
                                    session_state.run_id += 1
                                    time.sleep(0.1)
                                    raise RerunException(RerunData())

    elif sel_goal == 'Gamas Correspondidas':
        session_state.validate_button_pressed = 0
        sel_brand = st.sidebar.selectbox('Marca:', ['-'] + list(matched_data['PT_PDB_Franchise_Desc'].unique()), index=0)

        if sel_brand != '-':
            matched_data = matched_data.loc[matched_data['PT_PDB_Franchise_Desc'] == sel_brand.upper(), :]
            print('2 - ', matched_data.shape)

            unique_models = [x for x in list(matched_data['PT_PDB_Model_Desc'].unique()) if x not in ['H-1', 'H-1 3 lugares', 'H-1 6 lugares', 'H350', 'i20 Coupe', 'i20 VAN']]

            sel_model = st.sidebar.selectbox('Modelo', ['-'] + unique_models, index=0)
            # sel_confidence_threshold = st.sidebar.slider('Grau de Semelhança', min_value=0.0, max_value=1.0, value=0.8, step=0.01)

            if sel_model != '-':
                if sel_brand != session_state.sel_brand or sel_model != session_state.sel_model:
                    # session_state.sel_brand = sel_brand
                    # session_state.sel_model = sel_model

                    matched_data_filtered = filter_data(data, [sel_model, sel_brand], ['PT_PDB_Model_Desc', 'PT_PDB_Franchise_Desc'])
                    session_state.gama_viva_per_model = matched_data_filtered['PT_PDB_Commercial_Version_Desc_New'].unique()

                    st.write('Existem as seguintes gamas correspondidas para a marca {} e modelo {}:'.format(sel_brand, sel_model))

                    row_even_color = 'lightgrey'
                    row_odd_color = 'white'

                    if matched_data_filtered.shape[0]:
                        matched_data_temp = matched_data_filtered.loc[~matched_data_filtered['PT_PDB_Commercial_Version_Desc_Old'].isin(matched_data_filtered['PT_PDB_Commercial_Version_Desc_New'].unique()), :]
                        st.subheader('Correspondências:')
                        fig = go.Figure(data=[go.Table(
                            columnwidth=[500, 500],
                            header=dict(
                                values=[['Gama Morta'], ['Gama Viva']],
                                align=['center', 'center'],
                            ),
                            cells=dict(
                                values=[matched_data_temp['PT_PDB_Commercial_Version_Desc_Old'], matched_data_temp['PT_PDB_Commercial_Version_Desc_New']],
                                align=['center', 'center'],
                                fill_color=[[row_odd_color, row_even_color] * matched_data_temp.shape[0]],
                            )
                        )
                        ])

                        fig.update_layout(width=1100)
                        st.write(fig)

                        sel_gama = st.selectbox('Por favor escolha uma Gama Morta:', ['-'] + list(matched_data_temp['PT_PDB_Commercial_Version_Desc_Old'].unique()), index=0, key=session_state.run_id)
                        if sel_gama != '-':
                            matched_sel_gama = matched_data_filtered.loc[matched_data_filtered['PT_PDB_Commercial_Version_Desc_Old'] == sel_gama, :]['PT_PDB_Commercial_Version_Desc_New'].values[0]

                            if len(matched_sel_gama) > 1:
                                st.write('A Gama Viva correspondente é: {}. Se desejar alterar, escolha entre as seguintes:'.format(matched_sel_gama))
                            else:
                                st.write('Para escolher uma nova correspondência, escolha entre as seguintes:')

                            session_state.df_sim = calculate_cosine_similarity(sel_gama, -1, session_state.gama_viva_per_model, [])

                            suggestions = session_state.df_sim[['PT_PDB_Commercial_Version_Desc_Old', 'similarity_cosine']].sort_values(by=['similarity_cosine'], ascending=False).head(5).reset_index()
                            st.write('Sugestões:')
                            st.table(suggestions[['PT_PDB_Commercial_Version_Desc_Old', 'similarity_cosine']].rename(index=str, columns={'PT_PDB_Commercial_Version_Desc_Old': 'Gama Morta', 'similarity_cosine': 'Grau de Semelhança'}))

                            sel_gama_match = st.selectbox('Por favor escolha a correspondente Gama:', ['-', 's/ correspondência'] + list([x for x in session_state.gama_viva_per_model if x not in [' ', '']]), index=0, key=session_state.run_id)

                            if sel_gama_match != '-':
                                if st.button('Validar') or session_state.validate_button_pressed == 1:
                                    session_state.validate_button_pressed = 1

                                    st.write('A gama morta: \n{} corresponde à gama morta \n{}'.format(sel_gama, sel_gama_match))
                                    save_function(sel_gama, sel_gama_match, sel_brand, sel_model)

                                    session_state.validate_button_pressed = 0
                                    session_state.run_id += 1
                                    time.sleep(0.1)
                                    raise RerunException(RerunData())


def save_function(gama_morta, gama_viva, sel_brand, sel_model):
    if gama_viva == 's/ correspondência':
        gama_viva_sql = 'NULL'
    else:
        gama_viva_sql = '\'' + gama_viva.replace('\'', '\'\'') + '\''

    query = '''
    UPDATE [BI_DTR].[dbo].[VHE_MapDMS_Vehicle_Commercial_Versions_DTR]
    SET PT_PDB_Commercial_Version_Desc_New = {}, Classification_Flag = 1
    WHERE PT_PDB_Commercial_Version_Desc_Old = '{}'
    and PT_PDB_Franchise_Desc = '{}'
    and PT_PDB_Model_Desc = '{}' '''.format(gama_viva_sql, gama_morta.replace('\'', '\'\''), sel_brand, sel_model)

    level_1_e_deployment.sql_query(query, options_file.DSN, options_file.sql_info['database_source'], options_file.sql_info['commercial_version_matching'], options_file)
    return


def calculate_cosine_similarity(gama, gama_flag, gama_viva, gama_morta, unmatched_gamas=None):
    # start = time.time()
    print('gama', gama)
    print('gama_flag', gama_flag)
    # print('df \n', df)
    print('unmatched_gamas', unmatched_gamas)

    if unmatched_gamas is None:
        unmatched_gamas = []

    if gama_flag == -1:  # Gama Morta
        unique_designacao_comercial_morta_original = [gama]
        unique_designacao_comercial_viva_original = [x for x in gama_viva if x not in ['', ' ']]
        print('unique_designacao_comercial_morta_original', unique_designacao_comercial_morta_original)
        print('unique_designacao_comercial_viva_original', unique_designacao_comercial_viva_original)

        df_end = pd.DataFrame()
        for designacao_comercial_viva_original in unique_designacao_comercial_viva_original:

            df_middle = pd.DataFrame()
            df_middle['PT_PDB_Commercial_Version_Desc_New'] = unique_designacao_comercial_morta_original
            df_middle['PT_PDB_Commercial_Version_Desc_Old'] = designacao_comercial_viva_original

            for key, row in df_middle.iterrows():
                vec_designacao_comercial_viva = CountVectorizer().fit_transform([row['PT_PDB_Commercial_Version_Desc_Old']] + [row['PT_PDB_Commercial_Version_Desc_New']]).toarray()
                row['similarity_cosine'] = cosine_sim_vectors(vec_designacao_comercial_viva[0], vec_designacao_comercial_viva[1])
                df_end = df_end.append(row)

        df_end.reset_index(inplace=True)

    elif gama_flag == 1:  # Gama Viva  # WORKS
        unique_designacao_comercial_morta_original = [x for x in gama_morta if x not in ['', ' '] + unmatched_gamas]
        unique_designacao_comercial_viva_original = [gama]

        df_end = pd.DataFrame()
        for designacao_comercial_morta_original in unique_designacao_comercial_morta_original:

            df_middle = pd.DataFrame()
            df_middle['PT_PDB_Commercial_Version_Desc_New'] = unique_designacao_comercial_viva_original
            df_middle['PT_PDB_Commercial_Version_Desc_Old'] = designacao_comercial_morta_original

            for key, row in df_middle.iterrows():
                vec_designacao_comercial_viva = CountVectorizer().fit_transform([row['PT_PDB_Commercial_Version_Desc_Old']] + [row['PT_PDB_Commercial_Version_Desc_New']]).toarray()
                row['similarity_cosine'] = cosine_sim_vectors(vec_designacao_comercial_viva[0], vec_designacao_comercial_viva[1])
                df_end = df_end.append(row)

        df_end.reset_index(inplace=True)

    # print('cosine sim elapsed time: {}'.format(time.time() - start))
    return df_end


def cosine_sim_vectors(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)

    return cosine_similarity(vec1, vec2)[0][0]


def filter_data(dataset, value_filters_list, col_filters_list):
    data_filtered = dataset.copy()

    for col_filter, filter_value in zip(col_filters_list, value_filters_list):
        if filter_value != '-':
            data_filtered = data_filtered.loc[data_filtered[col_filter] == filter_value, :]

    return data_filtered


@st.cache(show_spinner=False, ttl=60*60*12)
def get_data(options_file_in):
    # gamas_match = pd.read_excel(input_file)
    gamas = level_1_a_data_acquisition.sql_retrieve_df(options_file_in.DSN, options_file_in.sql_info['database_source'], options_file_in.sql_info['commercial_version_matching'], options_file_in)

    # print('1 - ', gamas.shape)
    gamas['PT_PDB_Model_Desc'] = gamas['PT_PDB_Model_Desc'].str.lower()
    gamas.dropna(subset=['PT_PDB_Commercial_Version_Desc_Old'], inplace=True)
    gamas.drop_duplicates(subset=['PT_PDB_Model_Desc', 'PT_PDB_Commercial_Version_Desc_Old', 'PT_PDB_Commercial_Version_Desc_New'], inplace=True)  # This removes duplicates matching rows, even the ones without corresponding Gama Viva. There is however a case where the same Gama Morta has two matches: null and a corresponding Gama Viva - 1.4 TGDi DCT Style MY19'5 + TA for model i30 SW
    gamas['PT_PDB_Commercial_Version_Desc_New'].fillna(' ', inplace=True)

    return gamas


def get_data_non_cached(options_file_in, classification_flag):
    gamas = level_1_a_data_acquisition.sql_retrieve_df(options_file_in.DSN, options_file_in.sql_info['database_source'], options_file_in.sql_info['commercial_version_matching'], options_file_in, query_filters={'Classification_Flag': classification_flag})

    # print('3 - ', gamas.shape)
    gamas['PT_PDB_Model_Desc'] = gamas['PT_PDB_Model_Desc'].str.lower()
    gamas.dropna(subset=['PT_PDB_Commercial_Version_Desc_Old'], inplace=True)
    gamas.drop_duplicates(subset=['PT_PDB_Model_Desc', 'PT_PDB_Commercial_Version_Desc_Old', 'PT_PDB_Commercial_Version_Desc_New'], inplace=True)  # This removes duplicates matching rows, even the ones without corresponding Gama Viva. There is however a case where the same Gama Morta has two matches: null and a corresponding Gama Viva - 1.4 TGDi DCT Style MY19'5 + TA for model i30 SW
    gamas['PT_PDB_Commercial_Version_Desc_New'].fillna(' ', inplace=True)

    return gamas


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
