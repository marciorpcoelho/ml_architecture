import pandas as pd
import time
import streamlit as st
import os
import sys
import base64
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


session_state = SessionState.get(sel_all_refs_flag=True, sel_family_desc='-', run_id=0, sel_model_class='-', data_filtered_sel=pd.DataFrame(), data_filtered_sim=pd.DataFrame(), sel_text='', sel_text_option='', data_text_filtered_sel=pd.DataFrame(), data_text_filtered_sim=pd.DataFrame(), data=pd.DataFrame(columns=['Part_Ref', 'Part_Description', 'Part_Cost', 'Part_PVP', 'Product_Group_DW_desc', 'Classification_desc', 'Classification_Prob']))


def main():
    df_product_group = get_data_product_group_sql(options_file.others_families_dict, options_file)
    cm_family_lvl_1 = get_data_sql(options_file, options_file.sql_info['database_final'], options_file.sql_info['matrix_lvl_1'])
    cm_family_lvl_2 = get_data_sql(options_file, options_file.sql_info['database_final'], options_file.sql_info['matrix_lvl_2'])
    cm_family_dict_lvl_1 = cm_replacements(cm_family_lvl_1)
    cm_family_dict_lvl_2 = cm_replacements(cm_family_lvl_2)

    family_dict_sorted = family_dict_sorting(cm_family_dict_lvl_1, cm_family_dict_lvl_2)

    sel_page = st.sidebar.radio('Tarefa:', ['Análise de Classificações', 'Correções às Famílias Atuais', 'Exportação de Classificações'], index=1)

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

                    sel_family_sel_overwrite = st.selectbox('Por favor escolha a família para as peças selecionadas: ', ['-'] + [x for x in df_product_group['PT_Product_Group_Desc'].unique()], key=session_state.run_id+1, format_func=lambda x: df_product_group.loc[df_product_group['PT_Product_Group_Desc'] == x, 'Product_Group_Merge'].values[0] if x != '-' else '-')
                    if st.button('Validar alteração', key=0):
                        if sel_family_sel_overwrite == '-':
                            st.error('Por favor selecione uma família de peças.')
                        else:
                            update_family(session_state.data_text_filtered_sel.copy(), sel_family_sel_overwrite, df_product_group)
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

                    sel_family_sim_overwrite = st.selectbox('Por favor escolha a família para as peças selecionadas: ', ['-'] + [x for x in df_product_group['PT_Product_Group_Desc'].unique()], key=session_state.run_id, format_func=lambda x: df_product_group.loc[df_product_group['PT_Product_Group_Desc'] == x, 'Product_Group_Merge'].values[0] if x != '-' else '-')
                    if st.button('Validar alteração', key=1):
                        if sel_family_sim_overwrite == '-':
                            st.error('Por favor selecione uma família de peças.')
                        else:
                            update_family(session_state.data_text_filtered_sim.copy(), sel_family_sim_overwrite, df_product_group)
                            save_classification_rule(df_product_group, session_state.sel_text, sel_text_option, sel_family_sim_overwrite, sel_costs[1], max_cost, sel_costs[0], min_cost, sel_pvps[1], max_pvp, sel_pvps[0], min_pvp)
                else:
                    st.write(options_file.warning_message_app_dict[sel_text_option].format(sim_family_desc, session_state.sel_text))

        else:
            st.write('Por favor escolha uma família de peças.')

    elif sel_page == 'Análise de Classificações':
        data_original = get_dataset_sql(options_file.others_families_dict, options_file, options_file.non_classified_app_query)
        data = product_group_description(data_original, df_product_group)

        sel_text = st.text_input('Pesquisar pela(s) palavra(s):', '', key=session_state.run_id)
        sel_text_option = st.radio('Escolha a forma de pesquisa:', ('contains', 'starts'), format_func=radio_button_options)
        sel_all_refs_flag = st.sidebar.checkbox('Selecionar todas as referências.', value=True)

        if sel_text != '':
            # if sel_text_option == 'starts' and sel_text_option != session_state.sel_text_option or sel_text_option == 'starts' and sel_text != session_state.sel_text or sel_text_option == 'starts' and sel_all_refs_flag != session_state.sel_all_refs_flag:
            if sel_text_option == 'starts':
                data_df = data.loc[data['Part_Description'].str.startswith(sel_text), :]
            # elif sel_text_option == 'contains' and sel_text_option != session_state.sel_text_option or sel_text_option == 'contains' and sel_text != session_state.sel_text or sel_text_option == 'contains' and sel_all_refs_flag != session_state.sel_all_refs_flag:
            elif sel_text_option == 'contains':
                sel_text_regex = sel_text_regex_conversion(sel_text)

                data_df = data.loc[data['Part_Description'].str.contains(sel_text_regex, case=False, regex=True), :]
            table_title = 'Peças que {} - {}'.format(radio_button_options('table_' + sel_text_option), sel_text)
        else:
            data_df = data.head(50)
            table_title = 'Amostra de classificações'

        table_title += ' ({} referências):'.format(data_df.shape[0])
        if data_df.shape[0] > 0:
            fig = go.Figure(data=[go.Table(
                columnwidth=[],
                header=dict(
                    values=[options_file.column_translate_dict['Part_Ref'], options_file.column_translate_dict['Part_Description'], options_file.column_translate_dict['Part_Cost'], options_file.column_translate_dict['Part_PVP'], options_file.column_translate_dict['Product_Group_DW_desc'], options_file.column_translate_dict['Classification_desc'], options_file.column_translate_dict['Classification_Prob']],
                    align=['center', 'center', 'center', 'center'],
                ),
                cells=dict(
                    # values=[session_state.data['Part_Ref'].head(50), session_state.data['Part_Description'].head(50), session_state.data['Part_Cost'].round(2).head(50), session_state.data['Part_PVP'].round(2).head(50), session_state.data['Product_Group_DW_desc'].head(50), session_state.data['Classification_desc'].head(50), session_state.data['Classification_Prob'].round(2).head(50)],
                    values=[data_df['Part_Ref'], data_df['Part_Description'], data_df['Part_Cost'].round(2), data_df['Part_PVP'].round(2), data_df['Product_Group_DW_desc'], data_df['Classification_desc'], data_df['Classification_Prob'].round(2)],
                    align=['center', 'left', 'center', 'center'],
                ),
            )]
            )
            fig.update_layout(width=1500, height=500, title=table_title)
            st.write(fig)
        else:
            st.error('Não existem peças nas condições referidas. Por favor altere o(s) valor(es) do(s) filtro(s).')

        sel_part_ref = ''
        if not sel_all_refs_flag:
            sel_part_ref = st.multiselect('Por favor escolha a(s) referência(s) a alterar:', [x for x in data_df['Part_Ref'].unique()], key=session_state.run_id)
            data_df = filter_data(data_df, [sel_part_ref], ['Part_Ref'], ['in'])

        # data_cost_min, data_cost_max = session_state.data['Part_Cost'].min().item(), session_state.data['Part_Cost'].max().item()
        # data_pvp_min, data_pvp_max = session_state.data['Part_PVP'].min().item(), session_state.data['Part_PVP'].max().item()
        #
        # sel_costs = st.sidebar.slider('Por favor escolha os valores limite de custo:', data_cost_min, data_cost_max, (data_cost_min, data_cost_max), 10.0)
        # sel_pvps = st.sidebar.slider('Por favor escolha os valores limite de venda:', data_pvp_min, data_pvp_max, (data_pvp_min, data_pvp_max), 10.0)

        # if sel_text != '':
        # session_state.data = filter_data(data, [sel_costs[1], sel_costs[0], sel_pvps[1], sel_pvps[0]], ['Part_Cost', 'Part_Cost', 'Part_PVP', 'Part_PVP'], ['le', 'ge', 'le', 'ge'])

        # if sel_part_ref != '-':
        sel_family_overwrite = st.selectbox('Por favor escolha a família para as peças selecionadas: ', ['-'] + [x for x in df_product_group['PT_Product_Group_Desc'].unique()], key=session_state.run_id, format_func=lambda x: df_product_group.loc[df_product_group['PT_Product_Group_Desc'] == x, 'Product_Group_Merge'].values[0] if x != '-' else '-')
        if st.button('Alterar'):
            if not data_df.shape[0]:
                st.error('Por favor escolha as referências a alterar.')
                return

            if sel_family_overwrite != '-':
                update_family(data_df.copy(), sel_family_overwrite, df_product_group)
                session_state.run_id += 1
                time.sleep(0.1)
                raise RerunException(RerunData())
            elif sel_family_overwrite == '-':
                st.error('Por favor escolha uma família de peças.')

    elif sel_page == 'Exportação de Classificações':
        st.write('Nesta página é possível exportar uma família completa de peças, de acordo com a classificação mais recente do modelo de machine learning.')
        data = get_dataset_sql(options_file.others_families_dict, options_file, options_file.classified_app_query)
        current_date, _ = level_1_e_deployment.time_tags(format_date='%Y%m%d')

        available_families = df_product_group['Product_Group_Code'].unique()

        sel_family_desc = st.selectbox('Por favor escolha a família para as peças selecionadas: ', ['-'] + [x for x in available_families], key=session_state.run_id + 1, format_func=lambda x: df_product_group.loc[df_product_group['Product_Group_Code'] == x, 'Product_Group_Merge'].values[0] if x != '-' else '-')
        if sel_family_desc != '-' and int(sel_family_desc) in options_file.others_families_dict.keys():
            sel_family_desc = options_file.others_families_dict[int(sel_family_desc)]

        if sel_family_desc != '-':
            data_filter = filter_data(data, [sel_family_desc], ['Classification'], [None])
            if data_filter.shape[0]:
                st.write('Classificações ({} referências):'.format(data_filter.shape[0]), data_filter[['Part_Ref', 'Part_Description', 'Part_Cost', 'Part_PVP', 'Product_Group_DW', 'Classification', 'Classification_Prob']].rename(columns=options_file.column_translate_dict).head(50))
                file_export(data_filter[['Part_Ref', 'Part_Description', 'Part_Cost', 'Part_PVP', 'Product_Group_DW', 'Classification', 'Classification_Prob']].rename(columns=options_file.column_translate_dict), 'Classificações_família_{}_{}'.format(sel_family_desc, current_date))
            else:
                st.error('Não existem atualmente peças classificadas para a família escolhida.')


def file_export(df, file_name):

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Gravar Classificações</a> (carregar botão direito e Guardar Link como: {file_name}.csv)'
    st.markdown(href, unsafe_allow_html=True)


def update_family(df, new_family_classification, df_product_group):
    new_family_classification_code = family_code_convertion(new_family_classification, df_product_group)

    df['New_Product_Group_DW'] = new_family_classification_code
    df.rename(columns={'Product_Group_DW': 'Old_Product_Group_DW'}, inplace=True)
    level_1_e_deployment.sql_inject(df, options_file.DSN_MLG, options_file.sql_info['database_final'], options_file.sql_info['parts_classification_refs'], options_file, ['Part_Ref', 'Part_Description', 'Part_Cost', 'Part_PVP', 'Client_ID', 'Old_Product_Group_DW', 'New_Product_Group_DW', 'Classification', 'Classification_Prob'], check_date=1)

    st.write('Famílias das referências selecionadas alteradas com sucesso.')

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

    # st.write(text, text_option, family_code, sel_cost_max, max_cost, sel_cost_min, min_cost, sel_pvp_max, max_pvp, sel_pvp_min, min_pvp, time_tag)

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
    df.drop('Date', axis=1, inplace=True)

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

    df['Product_Group_Merge'] = df['PT_Product_Group_Level_1_Desc'] + ', ' + df['PT_Product_Group_Level_2_Desc'] + ', ' + df['PT_Product_Group_Desc']
    df.sort_values(by='Product_Group_Merge', inplace=True)
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
        elif operation_value == 'in':
            data_filtered = data_filtered.loc[data_filtered[col_filter].isin(filter_value), :]

    return data_filtered


@st.cache(show_spinner=False)
def cm_replacements(df):
    df.drop('Date', axis=1, inplace=True)
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
        'contains': 'Descrição contém palavra(s) escolhida(s)',
        'starts': 'Descrição começa pela palavra escolhida',
        'table_contains': 'contêm a(s) palavra(s)',
        'table_starts': 'começam pela palavra',
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
