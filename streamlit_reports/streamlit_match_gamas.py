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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
sys.path.insert(1, base_path)
import level_2_order_optimization_hyundai_options as options_file
import modules.SessionState as SessionState
from plotly import graph_objs as go
import dash_table


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

session_state = SessionState.get(run_id=0)

"""
# Correspondência Gamas - Importador
Correspondência entre Gamas Mortas e Vivas
"""


def main():
    sel_brand = st.sidebar.selectbox('Marca:', ['-', 'Hyundai', 'Honda'], index=0, key=session_state.run_id)

    if sel_brand != '-':
        session_state.sel_brand = sel_brand
        data = get_data(options_file.gamas_match_temp_file.format(sel_brand.lower()))
        unique_models = list(data['Modelo'].unique())

        sel_model = st.sidebar.selectbox('Modelo', ['-'] + unique_models, index=0, key=session_state.run_id)
        sel_confidence_threshold = st.sidebar.slider('Grau de Semelhança', min_value=0.0, max_value=1.0, value=0.8, step=0.01)

        if sel_model != '-' or session_state.sel_brand != sel_brand:
            session_state.sel_model = sel_model

            data_filtered = filter_data(data, [sel_model], ['Modelo'])
            gama_viva_per_model = data_filtered['Gama Viva'].unique()

            df_sim = calculate_cosine_similarity(data_filtered)
            idx_max_similarity = df_sim.groupby(['Gama Morta'])['similarity_cosine'].transform(max) == df_sim['similarity_cosine']
            df_max_similarity = df_sim[idx_max_similarity]

            draws = df_max_similarity.groupby(['Gama Morta']).filter(lambda x: len(x) > 1).sort_values(by='similarity_cosine', ascending=False)
            draws_gama_morta = draws['Gama Morta'].unique()
            high_confidence_matches = df_max_similarity.loc[(df_max_similarity['similarity_cosine'] >= sel_confidence_threshold - 0.001) & (~df_max_similarity['Gama Morta'].isin(draws_gama_morta)), ['Gama Morta', 'Gama Viva', 'similarity_cosine']].sort_values(by='similarity_cosine', ascending=False)
            not_high_confidence_matches = df_max_similarity.loc[(df_max_similarity['similarity_cosine'] < sel_confidence_threshold - 0.001) & (~df_max_similarity['Gama Morta'].isin(draws_gama_morta)), ['Gama Morta', 'Gama Viva', 'similarity_cosine']].sort_values(by='similarity_cosine', ascending=False)

            rowEvenColor = 'lightgrey'
            rowOddColor = 'white'
            st.subheader('Alta Semelhança:')
            fig = go.Figure(data=[go.Table(
                columnwidth=[500, 500, 100],
                header=dict(
                    values=[['Gama Morta'], ['Gama Viva'], ['Semelhança']],
                    align=['center', 'center', 'center'],
                ),
                cells=dict(
                    values=[high_confidence_matches['Gama Morta'], high_confidence_matches['Gama Viva'], high_confidence_matches['similarity_cosine'].round(2)],
                    align=['center', 'center', 'center'],
                    fill_color=[[rowOddColor, rowEvenColor] * high_confidence_matches.shape[0]],
                )
            )
            ])
            st.button('Validar Alta Semelhança')

            fig.update_layout(width=1100)
            st.write(fig)

            st.subheader('Empates:')
            fig = go.Figure(data=[go.Table(
                columnwidth=[500, 500, 100],
                header=dict(
                    values=[['Gama Morta'], ['Gama Viva'], ['Semelhança']],
                    align=['center', 'center', 'center'],
                ),
                cells=dict(
                    values=[draws['Gama Morta'], draws['Gama Viva'], draws['similarity_cosine'].round(2)],
                    align=['center', 'center', 'center'],
                    fill_color=[[rowOddColor, rowEvenColor] * draws.shape[0]],
                )
            )
            ])

            fig.update_layout(width=1100)
            st.write(fig)

            st.subheader('Baixa Semelhança:')
            fig = go.Figure(data=[go.Table(
                columnwidth=[500, 500, 100],
                header=dict(
                    values=[['Gama Morta'], ['Gama Viva'], ['Semelhança']],
                    align=['center', 'center', 'center'],
                ),
                cells=dict(
                    values=[not_high_confidence_matches['Gama Morta'], not_high_confidence_matches['Gama Viva'], not_high_confidence_matches['similarity_cosine'].round(2)],
                    align=['center', 'center', 'center'],
                    fill_color=[[rowOddColor, rowEvenColor] * not_high_confidence_matches.shape[0]],
                )
            )
            ])
            st.button('Validar Baixa Semelhança')

            fig.update_layout(width=1100)
            st.write(fig)

            sel_table = st.sidebar.selectbox('Tabela a editar:', ['-'] + ['Alta Semelhança', 'Empates', 'Baixa Semelhança'], index=0, key=session_state.run_id)

            if sel_table != '-' or session_state.sel_brand != sel_brand or session_state.sel_model != sel_model:
                if sel_table == 'Alta Semelhança':
                    sel_df = high_confidence_matches
                elif sel_table == 'Baixa Semelhança':
                    sel_df = not_high_confidence_matches
                elif sel_table == 'Empates':
                    sel_df = draws

                sel_gama_morta = st.selectbox('Por favor escolha uma Gama Morta:', ['-'] + list(sel_df['Gama Morta'].unique()), index=0, key=session_state.run_id)
                if sel_gama_morta != '-':
                    # st.table('Sugestões:', df_sim.loc[df_sim['Gama Morta'] == sel_gama_morta, ['Gama Viva', 'similarity_cosine']].sort_values(by=['similarity_cosine'], ascending=False).head(5))
                    suggestions = df_sim.loc[df_sim['Gama Morta'] == sel_gama_morta, :].sort_values(by=['similarity_cosine'], ascending=False).head(5).reset_index()
                    st.table(suggestions[['Gama Viva', 'similarity_cosine']])

                    sel_gama_viva = st.selectbox('Por favor escolha a correspondente Gama Viva', ['-'] + list(gama_viva_per_model), index=0, key=session_state.run_id)

                    if sel_gama_viva != '-':
                        st.write('A gama morta: \n{} corresponde à Gama Viva \n{}'.format(sel_gama_morta, sel_gama_viva))


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
        print(exception_desc, '\n', format_exc())
