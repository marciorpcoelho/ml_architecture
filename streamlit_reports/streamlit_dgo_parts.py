import pandas as pd
import time
import streamlit as st
import os
import sys
import itertools
from collections import Counter
from plotly import graph_objs as go
from traceback import format_exc
from streamlit.script_runner import RerunException
from streamlit.script_request_queue import RerunData

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
sys.path.insert(1, base_path)

import level_2_pa_part_reference_options as options_file
import modules.level_1_a_data_acquisition as level_1_a_data_acquisition
import modules.level_1_e_deployment as level_1_e_deployment
from modules.level_0_performance_report import log_record, error_upload
import modules.SessionState as SessionState

st.beta_set_page_config(page_title='Classificação de Peças - DGO')

'''
## Aplicação de Apoio à Classificação de Famílias de Peças - DGO
'''

url_hyperlink = '''
    <a href= "{}" > <p style="text-align:right"> Manual de Utilizador </p></a>
'''.format(options_file.documentation_url_app)
st.markdown(url_hyperlink, unsafe_allow_html=True)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


session_state = SessionState.get(sel_family_desc='-', run_id=0, sel_model_class='-', data_filtered_sel=pd.DataFrame(), data_filtered_sim=pd.DataFrame(), sel_text='', sel_text_option='', data_text_filtered_sel=pd.DataFrame(), data_text_filtered_sim=pd.DataFrame(), data=pd.DataFrame(columns=['Part_Ref', 'Part_Description', 'Part_Cost', 'Part_PVP', 'Product_Group_DW_desc', 'Classification_desc', 'Classification_Prob']))


def main():
    df_product_group = get_data_product_group_sql(options_file.others_families_dict, options_file)
    cm_family_lvl_1 = get_data_sql(options_file, options_file.sql_info['database_final'], options_file.sql_info['matrix_lvl_1'])
    cm_family_lvl_2 = get_data_sql(options_file, options_file.sql_info['database_final'], options_file.sql_info['matrix_lvl_2'])
    cm_family_dict_lvl_1 = cm_replacements(cm_family_lvl_1)
    cm_family_dict_lvl_2 = cm_replacements(cm_family_lvl_2)

    family_dict_sorted = family_dict_sorting(cm_family_dict_lvl_1, cm_family_dict_lvl_2)

    sel_page = st.sidebar.radio('Tarefa:', ['Análise de Classificações', 'Correções às Famílias Atuais'], index=0)

    if sel_page == 'Correções às Famílias Atuais':
        data_original = get_dataset_sql(options_file.others_families_dict, options_file, options_file.classified_app_query)
        data = product_group_description(data_original, df_product_group)

        lower_performance_families = family_dict_sorted.keys()
        lower_performance_families_values = [x[0] for x in family_dict_sorted.values()]

        df_current_cm = pd.DataFrame()
        df_current_cm['Product_Group_DW'] = lower_performance_families
        df_current_cm['Percentage_Predicted'] = lower_performance_families_values

        df_current_cm = df_current_cm.merge(df_product_group[['Product_Group_Code', 'PT_Product_Group_Desc']], left_on='Product_Group_DW', right_on='Product_Group_Code', how='left').drop('Product_Group_Code', axis=1).rename(columns={'PT_Product_Group_Desc': 'Product_Group_DW_desc'})
        df_current_cm.loc[df_current_cm['Product_Group_DW_desc'].isnull(), 'Product_Group_DW_desc'] = df_current_cm.loc[df_current_cm['Product_Group_DW_desc'].isnull(), 'Product_Group_DW']

        st.sidebar.table(df_current_cm[['Product_Group_DW_desc', 'Percentage_Predicted']].rename(columns=options_file.column_translate_dict).head(15))
        sel_family_desc = st.sidebar.selectbox('Por favor escolha a família de peças para alterar:', ['-'] + [x for x in df_current_cm['Product_Group_DW_desc'].unique()])  # ToDo: Maybe give more options?

        if sel_family_desc != '-':
            if session_state.sel_family_desc != sel_family_desc:
                session_state.sel_family_desc = sel_family_desc
                session_state.run_id += 1

            if sel_family_desc not in options_file.others_families_dict.values():
                sel_family_code = df_product_group.loc[df_product_group['PT_Product_Group_Desc'] == sel_family_desc, 'Product_Group_Code'].values[0]
            else:
                sel_family_code = sel_family_desc

            sim_family_code = family_dict_sorted[sel_family_code][1][0]

            if sim_family_code not in options_file.others_families_dict.values():
                sim_family_desc = df_product_group.loc[df_product_group['Product_Group_Code'] == sim_family_code, 'PT_Product_Group_Desc'].values[0]
            else:
                sim_family_desc = sim_family_code

            st.write('A Família escolhida - {} - foi confundida com a família {}, em cerca de {:.2f}% das suas classificações.'.format(sel_family_desc, sim_family_desc, family_dict_sorted[sel_family_code][2][0] * 100))

            session_state.data_filtered_sel = filter_data(data, [sel_family_desc], ['Product_Group_DW_desc'], [None])
            session_state.data_filtered_sim = filter_data(data, [sim_family_desc], ['Product_Group_DW_desc'], [None])

            min_cost, max_cost, min_pvp, max_pvp = cost_and_pvp_limits()

            df_common_keywords = common_keywords_calculation(sel_family_desc)
            st.write(df_common_keywords.head(50))

            sel_text = st.text_input('Pesquisar pela(s) palavra(s):', '', key=session_state.run_id)
            sel_text_option = st.radio('Escolha a forma de pesquisa:', ('contains', 'starts'), format_func=radio_button_options, key=session_state.run_id)

            sel_costs = st.sidebar.slider('Por favor escolha os valores limite de custo:', min_cost, max_cost, (min_cost, max_cost), 10.0)
            sel_pvps = st.sidebar.slider('Por favor escolha os valores limite de venda:', min_pvp, max_pvp, (min_pvp, max_pvp), 10.0)
            # sel_original_class = st.selectbox('Por favor escolha a família original a filtrar:', ['-'] + filtered_original_classes, index=0)

            if sel_text != '':
                # if sel_costs[1] != max_cost or sel_costs[0] != min_cost or sel_pvps[1] != max_pvp or sel_pvps[0] != min_pvp:
                session_state.data_filtered_sim = filter_data(session_state.data_filtered_sim, [sel_costs[1], sel_costs[0], sel_pvps[1], sel_pvps[0]], ['Part_Cost', 'Part_Cost', 'Part_PVP', 'Part_PVP'], ['le', 'ge', 'le', 'ge'])
                session_state.data_filtered_sel = filter_data(session_state.data_filtered_sel, [sel_costs[1], sel_costs[0], sel_pvps[1], sel_pvps[0]], ['Part_Cost', 'Part_Cost', 'Part_PVP', 'Part_PVP'], ['le', 'ge', 'le', 'ge'])

                description_filtering(sel_text_option, sel_text)

                if session_state.data_text_filtered_sel.shape[0]:
                    fig = go.Figure(data=[go.Table(
                        columnwidth=[],
                        header=dict(
                            values=[options_file.column_translate_dict['Part_Ref'], options_file.column_translate_dict['Part_Description'], options_file.column_translate_dict['Part_Cost'], options_file.column_translate_dict['Part_PVP'], options_file.column_translate_dict['Product_Group_DW_desc'], options_file.column_translate_dict['Classification_desc'], options_file.column_translate_dict['Classification_Prob']],
                            align=['center', 'center', 'center', 'center'],
                        ),
                        cells=dict(
                            values=[session_state.data_text_filtered_sel['Part_Ref'], session_state.data_text_filtered_sel['Part_Description'], session_state.data_text_filtered_sel['Part_Cost'].round(2), session_state.data_text_filtered_sel['Part_PVP'].round(2), session_state.data_text_filtered_sel['Product_Group_DW_desc'], session_state.data_text_filtered_sel['Classification_desc'], session_state.data_text_filtered_sel['Classification_Prob'].round(2)],
                            align=['center', 'left', 'center', 'center'],
                            ),
                        )]
                    )
                    fig.update_layout(width=1500, height=500, title='Família Escolhida: {} - Nº de Peças encontradas: {}'.format(sel_family_desc, session_state.data_text_filtered_sel.shape[0]))
                    st.write(fig)

                    sel_family_sel_overwrite = st.selectbox('Por favor escolha a família para as peças selecionadas: ', ['-'] + sorted([x for x in df_product_group['PT_Product_Group_Desc'].unique()]), key=session_state.run_id+1)
                    if st.button('Validar alteração', key=0):
                        if sel_family_sel_overwrite == '-':
                            st.error('Por favor selecione uma família de peças.')
                        else:
                            update_family(session_state.data_text_filtered_sel, sel_family_sel_overwrite, df_product_group)
                            save_classification_rule(df_product_group, session_state.sel_text, sel_text_option, sel_family_sel_overwrite, sel_costs[1], max_cost, sel_costs[0], min_cost, sel_pvps[1], max_pvp, sel_pvps[0], min_pvp)
                else:
                    st.write(options_file.warning_message_app_dict[sel_text_option].format(sel_family_desc, session_state.sel_text))

                if session_state.data_text_filtered_sim.shape[0]:
                    fig = go.Figure(data=[go.Table(
                        columnwidth=[],
                        header=dict(
                            values=[options_file.column_translate_dict['Part_Ref'], options_file.column_translate_dict['Part_Description'], options_file.column_translate_dict['Part_Cost'], options_file.column_translate_dict['Part_PVP'], options_file.column_translate_dict['Product_Group_DW_desc'], options_file.column_translate_dict['Classification_desc'], options_file.column_translate_dict['Classification_Prob']],
                            align=['center', 'center', 'center', 'center', 'center', 'center'],
                        ),
                        cells=dict(
                            values=[session_state.data_text_filtered_sim['Part_Ref'], session_state.data_text_filtered_sim['Part_Description'], session_state.data_text_filtered_sim['Part_Cost'].round(2), session_state.data_text_filtered_sim['Part_PVP'].round(2), session_state.data_text_filtered_sim['Product_Group_DW_desc'], session_state.data_text_filtered_sim['Classification_desc'], session_state.data_text_filtered_sim['Classification_Prob'].round(2)],
                            align=['center', 'left', 'center', 'center', 'center', 'center'],
                            ),
                        )]
                    )
                    fig.update_layout(width=1500, height=500, title='Família Semelhante: {} - Nº de Peças encontradas: {}'.format(sim_family_desc, session_state.data_text_filtered_sim.shape[0]))
                    st.write(fig)

                    sel_family_sim_overwrite = st.selectbox('Por favor escolha a família para as peças selecionadas: ', ['-'] + sorted([x for x in df_product_group['PT_Product_Group_Desc'].unique()]), key=session_state.run_id)
                    if st.button('Validar alteração', key=1):
                        if sel_family_sim_overwrite == '-':
                            st.error('Por favor selecione uma família de peças.')
                        else:
                            update_family(session_state.data_text_filtered_sim, sel_family_sim_overwrite, df_product_group)
                            save_classification_rule(df_product_group, session_state.sel_text, sel_text_option, sel_family_sim_overwrite, sel_costs[1], max_cost, sel_costs[0], min_cost, sel_pvps[1], max_pvp, sel_pvps[0], min_pvp)
                else:
                    st.write(options_file.warning_message_app_dict[sel_text_option].format(sim_family_desc, session_state.sel_text))

        else:
            st.write('Por favor escolha uma família de peças.')

    elif sel_page == 'Análise de Classificações':
        data_original = get_dataset_sql(options_file.others_families_dict, options_file, options_file.non_classified_app_query)
        data = product_group_description(data_original, df_product_group)

        sample_data = data.head(50)
        fig = go.Figure(data=[go.Table(
            columnwidth=[],
            header=dict(
                values=[options_file.column_translate_dict['Part_Ref'], options_file.column_translate_dict['Part_Description'], options_file.column_translate_dict['Part_Cost'], options_file.column_translate_dict['Part_PVP'], options_file.column_translate_dict['Product_Group_DW_desc'], options_file.column_translate_dict['Classification_desc'], options_file.column_translate_dict['Classification_Prob']],
                align=['center', 'center', 'center', 'center'],
            ),
            cells=dict(
                # values=[session_state.data['Part_Ref'].head(50), session_state.data['Part_Description'].head(50), session_state.data['Part_Cost'].round(2).head(50), session_state.data['Part_PVP'].round(2).head(50), session_state.data['Product_Group_DW_desc'].head(50), session_state.data['Classification_desc'].head(50), session_state.data['Classification_Prob'].round(2).head(50)],
                values=[sample_data['Part_Ref'], sample_data['Part_Description'], sample_data['Part_Cost'].round(2), sample_data['Part_PVP'].round(2), sample_data['Product_Group_DW_desc'], sample_data['Classification_desc'], sample_data['Classification_Prob'].round(2)],
                align=['center', 'left', 'center', 'center'],
            ),
        )]
        )
        fig.update_layout(width=1500, height=500, title='Amostra de classificações:')
        st.write(fig)

        # st.dataframe(data[['Part_Ref', 'Part_Description', 'Part_Cost', 'Part_PVP', 'Product_Group_DW_desc', 'Classification_desc', 'Classification_Prob']].head(50))

        sel_part_ref = st.multiselect('Por favor escolha a referência a alterar:', [x for x in data['Part_Ref'].unique()])

        # sel_text = st.text_input('Pesquisar pela(s) palavra(s):', '')
        # sel_text_option = st.radio('Escolha a forma de pesquisa:', ('contains', 'starts'), format_func=radio_button_options)

        # data_cost_min, data_cost_max = session_state.data['Part_Cost'].min().item(), session_state.data['Part_Cost'].max().item()
        # data_pvp_min, data_pvp_max = session_state.data['Part_PVP'].min().item(), session_state.data['Part_PVP'].max().item()
        #
        # sel_costs = st.sidebar.slider('Por favor escolha os valores limite de custo:', data_cost_min, data_cost_max, (data_cost_min, data_cost_max), 10.0)
        # sel_pvps = st.sidebar.slider('Por favor escolha os valores limite de venda:', data_pvp_min, data_pvp_max, (data_pvp_min, data_pvp_max), 10.0)

        # if sel_text != '':
        # session_state.data = filter_data(data, [sel_costs[1], sel_costs[0], sel_pvps[1], sel_pvps[0]], ['Part_Cost', 'Part_Cost', 'Part_PVP', 'Part_PVP'], ['le', 'ge', 'le', 'ge'])

        # if sel_text_option == 'starts' and sel_text_option != session_state.sel_text_option or sel_text_option == 'starts' and sel_text != session_state.sel_text:
        #     session_state.sel_text = sel_text
        #     session_state.sel_text_option = sel_text_option
        #     data = data.loc[data['Part_Description'].str.startswith(sel_text), :]
        #
        # elif sel_text_option == 'contains' and sel_text_option != session_state.sel_text_option or sel_text_option == 'contains' and sel_text != session_state.sel_text:
        #     sel_text_regex = sel_text_regex_conversion(sel_text)
        #     session_state.sel_text = sel_text
        #     session_state.sel_text_option = sel_text_option
        #     data = data.loc[data['Part_Description'].str.contains(sel_text_regex, case=False, regex=True), :]

        # if sel_part_ref != '-':
        sel_family_overwrite = st.selectbox('Por favor escolha a família para as peças selecionadas: ', ['-'] + sorted([x for x in df_product_group['PT_Product_Group_Desc'].unique()]), key=1)
        if st.button('Alterar', key=0):
            if sel_family_overwrite == '-':
                st.error('Por favor escolha uma família de peças.')
            else:
                update_family(data[data['Part_Ref'].isin(sel_part_ref)], sel_family_overwrite, df_product_group)
                # save_classification_rule(df_product_group, session_state.sel_text, sel_text_option, sel_family_overwrite, sel_costs[1], data_cost_max, sel_costs[0], data_cost_min, sel_pvps[1], data_pvp_max, sel_pvps[0], data_pvp_min)


def update_family(df, new_family_classification, df_product_group):
    new_family_classification_code = family_code_convertion(new_family_classification, df_product_group)

    sel_refs = [part_ref for part_ref in df['Part_Ref']]
    sel_refs_query = '\'' + "', '".join(sel_refs) + '\''
    # sel_product_group_dw = [product_group_dw for product_group_dw in df['Product_Group_DW']]

    # case_query = ' '.join(["WHEN Part_Ref = \'{}\' THEN \'{}\'".format(part_ref, product_group_dw) for part_ref, product_group_dw in sel_refs])

    query = options_file.update_product_group_dw_app_query.format(options_file.sql_info['parts_classification_table'], new_family_classification_code, sel_refs_query)
    # st.write(query)
    level_1_e_deployment.sql_query(query, options_file.DSN_MLG, options_file.sql_info['database_final'], options_file.sql_info['parts_classification_table'], options_file)

    return


def family_code_convertion(new_family_classification, df_product_group):
    if new_family_classification not in options_file.others_families_dict.values() and new_family_classification != '-':
        new_family_classification_code = df_product_group.loc[df_product_group['PT_Product_Group_Desc'] == new_family_classification, 'Product_Group_Code'].values[0]
    else:
        new_family_classification_code = new_family_classification

    return new_family_classification_code


def save_classification_rule(df_product_group, text, text_option, sel_family_sel_overwrite, sel_cost_max, max_cost, sel_cost_min, min_cost, sel_pvp_max, max_pvp, sel_pvp_min, min_pvp):
    family_code = family_code_convertion(sel_family_sel_overwrite, df_product_group)
    time_tag, _ = level_1_e_deployment.time_tags(format_date="%Y%m%d")

    st.write(text, text_option, family_code, sel_cost_max, max_cost, sel_cost_min, min_cost, sel_pvp_max, max_pvp, sel_pvp_min, min_pvp, time_tag)

    df_rules = pd.DataFrame()
    df_rules['Matching_Rule'] = [text_option]
    df_rules['Word'] = text
    df_rules['Product_Group_DW'] = family_code
    df_rules['Sel_Max_Cost'] = sel_cost_max
    df_rules['Max_Cost'] = max_cost
    df_rules['Sel_Min_Cost'] = sel_cost_min
    df_rules['Min_Cost'] = min_cost
    df_rules['Sel_Max_PVP'] = sel_pvp_max
    df_rules['Max_PVP'] = max_pvp
    df_rules['Sel_Min_PVP'] = sel_pvp_min
    df_rules['Min_PVP'] = min_pvp
    df_rules['Date'] = time_tag

    level_1_e_deployment.sql_inject(df_rules, options_file.DSN_MLG, options_file.sql_info['database_final'], options_file.sql_info['parts_classification_rules'], options_file, columns=list(df_rules))
    return


def sel_text_regex_conversion(sel_text):
    sel_text_list = sel_text.split()  # Splits selected works by space (default split value)
    all_sel_text_list_combinations = list(itertools.permutations(sel_text_list, len(sel_text_list)))  # Creates all combinations with the selected words, taking order into account
    sel_text_regex = '|'.join(['.*'.join(x) for x in all_sel_text_list_combinations])  # merges different combinations with .* and then merges them with | between them

    return sel_text_regex


def description_filtering(sel_text_option, sel_text):
    if sel_text_option == 'starts' and sel_text_option != session_state.sel_text_option or sel_text_option == 'starts' and sel_text != session_state.sel_text:
        session_state.sel_text = sel_text
        session_state.sel_text_option = sel_text_option
        session_state.data_text_filtered_sel = session_state.data_filtered_sel.loc[session_state.data_filtered_sel['Part_Description'].str.startswith(sel_text), :]
        session_state.data_text_filtered_sim = session_state.data_filtered_sim.loc[session_state.data_filtered_sim['Part_Description'].str.startswith(sel_text), :]

    elif sel_text_option == 'contains' and sel_text_option != session_state.sel_text_option or sel_text_option == 'contains' and sel_text != session_state.sel_text:
        sel_text_regex = sel_text_regex_conversion(sel_text)
        session_state.sel_text = sel_text
        session_state.sel_text_option = sel_text_option
        session_state.data_text_filtered_sel = session_state.data_filtered_sel.loc[session_state.data_filtered_sel['Part_Description'].str.contains(sel_text_regex, case=False, regex=True), :]
        session_state.data_text_filtered_sim = session_state.data_filtered_sim.loc[session_state.data_filtered_sim['Part_Description'].str.contains(sel_text_regex, case=False, regex=True), :]


@st.cache(show_spinner=False)
def cost_and_pvp_limits():
    data_filtered_sel_cost_min, data_filtered_sel_cost_max = session_state.data_filtered_sel['Part_Cost'].min(), session_state.data_filtered_sel['Part_Cost'].quantile(0.95)
    data_filtered_sel_pvp_min, data_filtered_sel_pvp_max = session_state.data_filtered_sel['Part_PVP'].min(), session_state.data_filtered_sel['Part_PVP'].quantile(0.95)

    data_filtered_sim_cost_min, data_filtered_sim_cost_max = session_state.data_filtered_sim['Part_Cost'].min(), session_state.data_filtered_sim['Part_Cost'].quantile(0.95)
    data_filtered_sim_pvp_min, data_filtered_sim_pvp_max = session_state.data_filtered_sim['Part_PVP'].min(), session_state.data_filtered_sim['Part_PVP'].quantile(0.95)

    min_cost = min(data_filtered_sel_cost_min, data_filtered_sim_cost_min).item()  # .item() converts the numpy data type (np.float64) to a python native type (float)
    max_cost = max(data_filtered_sel_cost_max, data_filtered_sim_cost_max).item()
    min_pvp = min(data_filtered_sel_pvp_min, data_filtered_sim_pvp_min).item()
    max_pvp = max(data_filtered_sel_pvp_max, data_filtered_sim_pvp_max).item()

    return min_cost, max_cost, min_pvp, max_pvp


@st.cache(show_spinner=False)
def common_keywords_calculation(sel_family):
    text_sel = " ".join(description.strip() for description in session_state.data_filtered_sel['Part_Description'])
    text_sim = " ".join(description.strip() for description in session_state.data_filtered_sim['Part_Description'])

    document_1_words = [word for word in text_sel.split() if word not in options_file.stop_words_list]
    document_1_words_counter = Counter(document_1_words)
    document_2_words = [word for word in text_sim.split() if word not in options_file.stop_words_list]
    document_2_words_counter = Counter(document_2_words)

    common = set(document_1_words).intersection(set(document_2_words))

    document_1_words_counter_common = {key: document_1_words_counter[key] for key in common}
    document_2_words_counter_common = {key: document_2_words_counter[key] for key in common}
    common_dict = {key: document_1_words_counter_common[key] + document_2_words_counter_common[key] for key in common if len(key) > 1}
    common_dict_sorted = {k: v for k, v in sorted(common_dict.items(), key=lambda item: item[1], reverse=True)}

    df_common_keywords = pd.DataFrame()
    df_common_keywords['Palavras em Comum'] = [x for x in common_dict_sorted.keys()]
    df_common_keywords['#Palavras em Comum'] = [x for x in common_dict_sorted.values()]

    return df_common_keywords


@st.cache(show_spinner=False)
def get_dataset_sql(others_dict, options_file_in, query):
    df = level_1_a_data_acquisition.sql_retrieve_df_specified_query(options_file_in.DSN_MLG, options_file_in.sql_info['database_final'], options_file_in, query)
    df['Part_Description'] = df['Part_Description'].astype('str')

    # for key in others_dict.keys():
    #     df.loc[df['Product_Group_DW'] == str(key), 'Product_Group_DW'] = others_dict[key]
    #     df.loc[df['Classification'] == str(key), 'Classification'] = others_dict[key]

    # df = df.replace({'Product_Group_DW': others_dict})
    # df = df.replace({'Classification': others_dict})

    df['Product_Group_DW'] = df['Product_Group_DW'].map(others_dict).fillna(df['Product_Group_DW'])
    df['Classification'] = df['Classification'].map(others_dict).fillna(df['Classification'])

    return df


@st.cache(show_spinner=False)
def get_data_product_group_sql(others_dict, options_file_in):
    df = level_1_a_data_acquisition.sql_retrieve_df_specified_query(options_file_in.DSN, options_file_in.sql_info['database_BI_AFR'], options_file_in, options_file_in.product_group_app_query)
    df['Product_Group_Code'] = df['Product_Group_Code'].astype('str')

    df = df[df['Product_Group_Code'] != '77']
    df.loc[df['Product_Group_Code'] == '75', 'Product_Group_Code'] = '75/77'
    df.loc[df['PT_Product_Group_Desc'] == 'Lazer', 'PT_Product_Group_Desc'] = 'Lazer/Marroquinaria'

    for key in others_dict.keys():
        df.loc[df['Product_Group_Code'] == str(key), 'PT_Product_Group_Desc'] = others_dict[key]

    # df['PT_Product_Group_Desc'] = df['PT_Product_Group_Desc'].map(others_dict).fillna(df['PT_Product_Group_Desc'])

    return df


def get_data_sql(options_file_in, db, view):
    df = level_1_a_data_acquisition.sql_retrieve_df(options_file_in.DSN_MLG, db, view, options_file_in)

    df = df.set_index('Actual')

    return df


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

    return data_filtered


@st.cache(show_spinner=False)
def cm_replacements(df):
    family_names = [x for x in list(df) if not x.startswith('Totals')]
    df['Family_%'] = [df.iloc[value, value] / df.iloc[value, df.shape[1]-1] for value in range(df.shape[0])]

    df_copy = df.copy(deep=True)
    family_confusion_dict = {}

    for value in range(df.shape[0]):
        sel_family = df.index[value]
        sel_family_performance = df.iloc[value, df.shape[1]-1]

        if sel_family_performance < 0.8:
            df_copy.iloc[value, value] = -1
            # st.write('1 - ', sel_family)

            # top_conf_family = df_copy[family_names].idxmax(axis='columns').values[value]
            # st.write('2 - ', top_conf_family)

            top_conf_family_test = df_copy[family_names].apply(lambda s: s.abs().nlargest(2).index.tolist(), axis=1).iloc[value]
            # st.write('2a - ', top_conf_family_test)

            # top_conf_family_performance = df_copy[family_names].max(axis='columns').values[value] / df_copy['Totals'].values[value]
            # st.write('3 - ', top_conf_family_performance)

            top_conf_family_performance_test = [df_copy.loc[df_copy.index[value], x] / df_copy['Totals'].values[value] for x in top_conf_family_test]
            # st.write('3a - ', top_conf_family_performance_test)

            # st.write(sel_family, sel_family_performance)
            # st.write(df_copy['Totals'].values[value])

            # family_confusion_dict[sel_family] = [sel_family_performance, top_conf_family, top_conf_family_performance]
            family_confusion_dict[sel_family] = [sel_family_performance, top_conf_family_test, top_conf_family_performance_test]

    return family_confusion_dict


@st.cache(show_spinner=False)
def product_group_description(df, df_product_group):

    df = df.merge(df_product_group[['Product_Group_Code', 'PT_Product_Group_Desc']], left_on='Product_Group_DW', right_on='Product_Group_Code', how='left').drop('Product_Group_Code', axis=1).rename(columns={'PT_Product_Group_Desc': 'Product_Group_DW_desc'})
    df = df.merge(df_product_group[['Product_Group_Code', 'PT_Product_Group_Desc']], left_on='Classification', right_on='Product_Group_Code', how='left').drop('Product_Group_Code', axis=1).rename(columns={'PT_Product_Group_Desc': 'Classification_desc'})
    df.loc[df['Product_Group_DW_desc'].isnull(), 'Product_Group_DW_desc'] = df.loc[df['Product_Group_DW_desc'].isnull(), 'Product_Group_DW']
    df.loc[df['Classification_desc'].isnull(), 'Classification_desc'] = df.loc[df['Classification_desc'].isnull(), 'Classification']

    return df


@st.cache(show_spinner=False)
def family_dict_sorting(family_dict_lvl_1, family_dict_lvl_2):

    family_dict = {**family_dict_lvl_1, **family_dict_lvl_2}

    family_dict_sorted = {k: v for k, v in sorted(family_dict.items(), key=lambda item: item[1])}

    return family_dict_sorted


def radio_button_options(option):

    radio_options_dict = {
        'contains': 'Descrição contém palavra escolhida',
        'starts': 'Descrição começa com a palavra escolhida'
    }

    return radio_options_dict[option]


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
