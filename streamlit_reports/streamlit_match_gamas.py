import streamlit as st
import pandas as pd
import os
import sys
import time
from traceback import format_exc
from streamlit.ScriptRunner import RerunException
from streamlit.ScriptRequestQueue import RerunData
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
sys.path.insert(1, base_path)
import level_2_order_optimization_hyundai_options as options_file
import modules.SessionState as SessionState
import modules.level_1_e_deployment as level_1_e_deployment
from modules.level_0_performance_report import log_record, error_upload
from plotly import graph_objs as go

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# url_hyperlink = '''
#     <a href= "{}" > <p style="text-align:right"> Documentação </p></a>
# '''.format(options_file.url_doc)
# st.markdown(url_hyperlink, unsafe_allow_html=True)

session_state = SessionState.get(run_id=0, sel_brand='-', sel_model='-', draws_gama_morta=pd.DataFrame(), high_confidence_matches=pd.DataFrame(), not_high_confidence_matches=pd.DataFrame(), sel_df='-', df_sim=pd.DataFrame(), sel_table='-', validate_button_pressed=0)

"""
# Correspondência Gamas - Importador
Correspondência entre Gamas Mortas e Vivas
"""


def main():
    sel_brand = st.sidebar.selectbox('Marca:', ['-', 'Hyundai', 'Honda'], index=0)

    if sel_brand != '-':
        data = get_data(options_file.gamas_match_temp_file.format(sel_brand.lower()))
        unique_models = list(data['Modelo'].unique())

        sel_model = st.sidebar.selectbox('Modelo', ['-'] + unique_models, index=0)
        sel_confidence_threshold = st.sidebar.slider('Grau de Semelhança', min_value=0.0, max_value=1.0, value=0.8, step=0.01)

        if sel_model != '-':
            if sel_brand != session_state.sel_brand or sel_model != session_state.sel_model:
                session_state.sel_brand = sel_brand
                session_state.sel_model = sel_model

                data_filtered = filter_data(data, [sel_model], ['Modelo'])
                session_state.gama_viva_per_model = data_filtered['Gama Viva'].unique()

                session_state.df_sim = calculate_cosine_similarity(data_filtered)
                idx_max_similarity = session_state.df_sim.groupby(['Gama Morta'])['similarity_cosine'].transform(max) == session_state.df_sim['similarity_cosine']
                df_max_similarity = session_state.df_sim[idx_max_similarity]

                session_state.draws = df_max_similarity.groupby(['Gama Morta']).filter(lambda x: len(x) > 1).sort_values(by='similarity_cosine', ascending=False)
                draws_gama_morta = session_state.draws['Gama Morta'].unique()
                session_state.high_confidence_matches = df_max_similarity.loc[(df_max_similarity['similarity_cosine'] >= sel_confidence_threshold - 0.001) & (~df_max_similarity['Gama Morta'].isin(draws_gama_morta)), ['Gama Morta', 'Gama Viva', 'similarity_cosine']].sort_values(by='similarity_cosine', ascending=False)
                session_state.not_high_confidence_matches = df_max_similarity.loc[(df_max_similarity['similarity_cosine'] < sel_confidence_threshold - 0.001) & (~df_max_similarity['Gama Morta'].isin(draws_gama_morta)), ['Gama Morta', 'Gama Viva', 'similarity_cosine']].sort_values(by='similarity_cosine', ascending=False)

            row_even_color = 'lightgrey'
            row_odd_color = 'white'

            if session_state.high_confidence_matches.shape[0]:
                st.subheader('Alta Semelhança:')
                fig = go.Figure(data=[go.Table(
                    columnwidth=[500, 500, 100],
                    header=dict(
                        values=[['Gama Morta'], ['Gama Viva'], ['Semelhança']],
                        align=['center', 'center', 'center'],
                    ),
                    cells=dict(
                        values=[session_state.high_confidence_matches['Gama Morta'], session_state.high_confidence_matches['Gama Viva'], session_state.high_confidence_matches['similarity_cosine'].round(2)],
                        align=['center', 'center', 'center'],
                        fill_color=[[row_odd_color, row_even_color] * session_state.high_confidence_matches.shape[0]],
                    )
                )
                ])
                st.button('Validar Alta Semelhança')

                fig.update_layout(width=1100)
                st.write(fig)

            if session_state.draws_gama_morta.shape[0]:
                st.subheader('Empates:')
                fig = go.Figure(data=[go.Table(
                    columnwidth=[500, 500, 100],
                    header=dict(
                        values=[['Gama Morta'], ['Gama Viva'], ['Semelhança']],
                        align=['center', 'center', 'center'],
                    ),
                    cells=dict(
                        values=[session_state.draws['Gama Morta'], session_state.draws['Gama Viva'], session_state.draws['similarity_cosine'].round(2)],
                        align=['center', 'center', 'center'],
                        fill_color=[[row_odd_color, row_even_color] * session_state.draws.shape[0]],
                    )
                )
                ])

                fig.update_layout(width=1100)
                st.write(fig)

            if session_state.not_high_confidence_matches.shape[0]:
                st.subheader('Baixa Semelhança:')
                fig = go.Figure(data=[go.Table(
                    columnwidth=[500, 500, 100],
                    header=dict(
                        values=[['Gama Morta'], ['Gama Viva'], ['Semelhança']],
                        align=['center', 'center', 'center'],
                    ),
                    cells=dict(
                        values=[session_state.not_high_confidence_matches['Gama Morta'], session_state.not_high_confidence_matches['Gama Viva'], session_state.not_high_confidence_matches['similarity_cosine'].round(2)],
                        align=['center', 'center', 'center'],
                        fill_color=[[row_odd_color, row_even_color] * session_state.not_high_confidence_matches.shape[0]],
                    )
                )
                ])
                st.button('Validar Baixa Semelhança')

                fig.update_layout(width=1100)
                st.write(fig)

            sel_table = st.sidebar.selectbox('Tabela a editar:', ['-'] + ['Alta Semelhança', 'Empates', 'Baixa Semelhança'], index=0)

            if sel_table != '-':
                if sel_table == 'Alta Semelhança':
                    sel_df = session_state.high_confidence_matches
                elif sel_table == 'Baixa Semelhança':
                    sel_df = session_state.not_high_confidence_matches
                elif sel_table == 'Empates':
                    sel_df = session_state.draws

                session_state.sel_table = sel_table
                session_state.sel_df = sel_df

                sel_gama_morta = st.selectbox('Por favor escolha uma Gama Morta:', ['-'] + list(session_state.sel_df['Gama Morta'].unique()), index=0, key=session_state.run_id)
                if sel_gama_morta != '-':
                    # st.table('Sugestões:', df_sim.loc[df_sim['Gama Morta'] == sel_gama_morta, ['Gama Viva', 'similarity_cosine']].sort_values(by=['similarity_cosine'], ascending=False).head(5))
                    suggestions = session_state.df_sim.loc[session_state.df_sim['Gama Morta'] == sel_gama_morta, :].sort_values(by=['similarity_cosine'], ascending=False).head(5).reset_index()
                    st.table(suggestions[['Gama Viva', 'similarity_cosine']])

                    sel_gama_viva = st.selectbox('Por favor escolha a correspondente Gama Viva', ['-', 's/ correspondência'] + list([x for x in session_state.gama_viva_per_model if x != ' ']), index=0, key=session_state.run_id)

                    if sel_gama_viva != '-':
                        if st.button('Validar') or session_state.validate_button_pressed == 1:
                            session_state.validate_button_pressed = 1

                            st.write('A gama morta: \n{} corresponde à Gama Viva \n{}'.format(sel_gama_morta, sel_gama_viva))
                            save_function(sel_gama_morta, sel_gama_viva, sel_brand, sel_model)

                            session_state.validate_button_pressed = 0
                            session_state.run_id += 1
                            time.sleep(0.1)
                            raise RerunException(RerunData(widget_state=None))


def save_function(gama_morta, gama_viva, sel_brand, sel_model):

    df_solution = solution_dataframe_creation(gama_morta, gama_viva, sel_brand, sel_model)

    level_1_e_deployment.sql_inject(df_solution, options_file.DSN, options_file.sql_info['database_source'], options_file.sql_info['commercial_version_matching'], options_file, list(df_solution))

    st.write('Correspondência gravada com sucesso.')
    return


def solution_dataframe_creation(gama_morta, gama_viva, sel_brand, sel_model):
    df_solution = pd.DataFrame()
    current_year_month = level_1_e_deployment.time_tags(format_date="%Y%m")[0]
    if gama_viva == 's/ correspondência':
        gama_viva = None

    df_solution['Client_Id'] = [7]
    df_solution['PT_PDB_Franchise_Desc'] = sel_brand
    df_solution['PT_PDB_Model_Desc'] = sel_model
    df_solution['PT_PDB_Commercial_Version_Desc_Old'] = gama_morta
    df_solution['PT_PDB_Commercial_Version_Desc_New'] = gama_viva
    df_solution['Classification_Flag'] = 1
    df_solution['Initial_Period'] = current_year_month  # Convert to int?
    df_solution['Final_Period'] = 99912  # Place Holder

    st.write(df_solution)

    return df_solution


def calculate_cosine_similarity(df):
    unique_designacao_comercial_morta_original = list(df['Gama Morta'].unique())
    unique_designacao_comercial_viva_original = list(df['Gama Viva'].unique())

    try:
        unique_designacao_comercial_viva_original.remove(' ')
    except ValueError:
        pass

    df_end = pd.DataFrame()
    for designacao_comercial_morta_original in unique_designacao_comercial_morta_original:

        df_middle = pd.DataFrame()
        df_middle['Gama Viva'] = unique_designacao_comercial_viva_original
        df_middle['Gama Morta'] = designacao_comercial_morta_original
        correct_gama_viva = df.loc[df['Gama Morta'] == designacao_comercial_morta_original, 'Gama Viva'].values
        df_middle['correct_gama_viva'] = [correct_gama_viva] * len(unique_designacao_comercial_viva_original)

        for key, row in df_middle.iterrows():
            vec_designacao_comercial_viva = CountVectorizer().fit_transform([row['Gama Morta']] + [row['Gama Viva']]).toarray()
            row['similarity_cosine'] = cosine_sim_vectors(vec_designacao_comercial_viva[0], vec_designacao_comercial_viva[1])
            df_end = df_end.append(row)

    df_end.reset_index(inplace=True)
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


@st.cache
def get_data(input_file):
    gamas_match = pd.read_excel(input_file)
    gamas_match.dropna(subset=['Gama Morta'], inplace=True)
    gamas_match.drop_duplicates(subset=['Modelo', 'Gama Morta', 'Gama Viva'], inplace=True)  # This removes duplicates matching rows, even the ones without corresponding Gama Viva. There is however a case where the same Gama Morta has two matches: null and a corresponding Gama Viva - 1.4 TGDi DCT Style MY19'5 + TA for model i30 SW
    gamas_match['Gama Viva'].fillna(' ', inplace=True)

    return gamas_match


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
