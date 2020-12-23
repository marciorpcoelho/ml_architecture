import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from traceback import format_exc
import level_2_pa_servicedesk_2244_options as options_file
from modules.level_1_a_data_acquisition import read_csv, sql_retrieve_df, sql_mapping_retrieval
from modules.level_1_b_data_processing import stop_words_removal, summary_description_null_checkup, top_words_processing, text_preprocess, literal_removal, string_to_list, df_join_function, null_handling, lowercase_column_conversion, null_analysis, remove_rows, value_replacement, value_substitution, duplicate_removal, language_detection, string_replacer, similar_words_handling
from modules.level_1_c_data_modelling import new_request_type
from modules.level_1_e_deployment import save_csv, sql_inject
from modules.level_0_performance_report import error_upload, log_record, project_dict, performance_info, performance_info_append
from sklearn.decomposition import PCA
my_dpi = 96
pd.set_option('display.expand_frame_repr', False)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S @ %d/%m/%y', filename=options_file.log_files['full_log'], filemode='a')
logging.Logger('errors')
logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))

similar_process_flag = 1
sel_request_num = 'RE-125884'


def main():
    log_record('Projeto: {}'.format(project_dict[options_file.project_id]), options_file.project_id)
    input_file_facts, input_file_durations, input_file_clients, input_file_pbi_categories, input_file_manual_classification, input_keywords_df = 'dbs/db_facts_initial.csv', 'dbs/db_facts_duration.csv', 'dbs/db_clients_initial.csv', 'dbs/db_pbi_categories_initial.csv', 'dbs/db_manual_classification.csv', 'dbs/db_keywords_df.csv'
    query_filters = [{'Cost_Centre': '6825', 'Record_Type': ['1', '2']}, {'Cost_Centre': '6825'}]

    df_facts, df_facts_duration, df_clients, df_pbi_categories, df_manual_classifications, keywords_df, keyword_dict, ranking_dict = data_acquisition([input_file_facts, input_file_durations, input_file_clients, input_file_pbi_categories, input_file_manual_classification, input_keywords_df], query_filters, local=0)
    df, df_top_words = data_processing(df_facts, df_facts_duration, df_clients, df_pbi_categories, keywords_df)
    df = data_modelling(df, df_top_words, df_manual_classifications, keyword_dict, ranking_dict)

    deployment(df)
    performance_info(options_file.project_id, options_file, model_choice_message='N/A')

    log_record('Conclusão com sucesso - Projeto: {}'.format(project_dict[options_file.project_id]), options_file.project_id)


def data_acquisition(input_files, query_filters, local=0):
    performance_info_append(time.time(), 'Section_A_Start')
    df_facts, df_facts_duration, df_clients, df_pbi_categories, df_manual_classifications, keywords_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    log_record('Início Secção A...', options_file.project_id)

    if local:
        df_facts = read_csv(input_files[0], index_col=0, parse_dates=options_file.date_columns, infer_datetime_format=True)
        df_facts_duration = read_csv(input_files[1], index_col=0)
        df_clients = read_csv(input_files[2], index_col=0)
        df_pbi_categories = read_csv(input_files[3], index_col=0)
        df_manual_classifications = read_csv(input_files[4], index_col=0)
        keywords_df = read_csv(input_files[5], index_col=0)
    elif not local:
        df_facts = sql_retrieve_df(options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'],  options_file.sql_info['initial_table_facts'], options_file,  options_file.sql_fact_columns, query_filters=query_filters[0], parse_dates=options_file.date_columns)
        df_facts_duration = sql_retrieve_df(options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'], options_file.sql_info['initial_table_facts_durations'], options_file, options_file.sql_facts_durations_columns, query_filters=query_filters[1])
        df_clients = sql_retrieve_df(options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'], options_file.sql_info['initial_table_clients'], options_file, options_file.sql_dim_contacts_columns)
        df_pbi_categories = sql_retrieve_df(options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'], options_file.sql_info['initial_table_pbi_categories'], options_file, options_file.sql_pbi_categories_columns, query_filters=query_filters[1])
        df_manual_classifications = sql_retrieve_df(options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'], options_file.sql_info['aux_table'], options_file)
        keywords_df = sql_retrieve_df(options_file.DSN_MLG_PRD, options_file.sql_info['database_final'], options_file.sql_info['keywords_table_str'], options_file, columns=['Keywords_PT', 'Keywords_ES']).dropna()

        save_csv([df_facts, df_facts_duration, df_clients, df_pbi_categories, df_manual_classifications, keywords_df], ['dbs/db_facts_initial', 'dbs/db_facts_duration', 'dbs/db_clients_initial', 'dbs/db_pbi_categories_initial', 'dbs/db_manual_classification', 'dbs/db_keywords_df'])

    keyword_dict, ranking_dict = sql_mapping_retrieval(options_file.DSN_MLG_PRD, options_file.sql_info['database_final'], options_file.sql_info['keywords_table'], 'Keyword_Group', options_file, multiple_columns=1)
    keyword_dict = keyword_dict[0]

    log_record('Fim Secção A...', options_file.project_id)
    performance_info_append(time.time(), 'Section_A_End')

    return df_facts, df_facts_duration, df_clients, df_pbi_categories, df_manual_classifications, keywords_df, keyword_dict, ranking_dict


def data_processing(df_facts, df_facts_duration, df_clients, df_pbi_categories, keywords_df):
    performance_info_append(time.time(), 'Section_B_Start')
    log_record('Início Secção B...', options_file.project_id)

    dict_strings_to_replace = {('Description', 'filesibmcognoscbindatacqertmodelsfdfdeeacebedeabeeabbedrtm'): 'files ibm cognos', ('Description', 'cognosapv'): 'cognos apv', ('Description', 'caetanoautopt'): 'caetano auto pt',
                               ('Description', 'autolinecognos'): 'autoline cognos', ('Description', 'realnao'): 'real nao', ('Description', 'booksytner'): 'book sytner'}  # ('Description', 'http://'): 'http://www.', ('Summary', 'http://'): 'http://www.'

    # Remove PBI's categories requests
    log_record('Contagem inicial de pedidos: {}'.format(df_facts['Request_Num'].nunique()), options_file.project_id)
    pbi_categories = remove_rows(df_pbi_categories.copy(), [df_pbi_categories[~df_pbi_categories['Category_Name'].str.contains('Power BI')].index], options_file.project_id)['Category_Id'].values  # Selects the Category ID's which belong to PBI
    log_record('Contagem de pedidos PBI: {}'.format(df_facts[df_facts['Category_Id'].isin(pbi_categories)]['Request_Num'].nunique()), options_file.project_id)
    df_facts = remove_rows(df_facts, [df_facts.loc[df_facts['Category_Id'].isin(pbi_categories)].index], options_file.project_id)  # Removes the rows which belong to PBI;
    log_record('Após o filtro de pedidos PBI, a nova contagem é de: {}'.format(df_facts['Request_Num'].nunique()), options_file.project_id)

    # Lowercase convertion of Summary and Description
    df_facts = lowercase_column_conversion(df_facts, columns=['Summary', 'Description'])

    # Addition of Client/Assignee Information and imputation of some missing values
    df_facts = df_join_function(df_facts, df_facts_duration.set_index('Request_Num'), on='Request_Num')
    df_facts = df_join_function(df_facts, df_clients.set_index('Contact_Id'), on='Contact_Customer_Id')
    df_facts = value_replacement(df_facts, options_file.assignee_id_replacements)
    df_facts = df_join_function(df_facts, df_clients.set_index('Contact_Id'), on='Contact_Assignee_Id', lsuffix='_Customer', rsuffix='_Assignee')
    df_facts = value_replacement(df_facts, options_file.sla_resolution_hours_replacements)

    # Collection of all Client/Assignee possible names
    unique_clients_names_decoded = string_to_list(df_facts, ['Name_Customer'])
    unique_clients_login_decoded = string_to_list(df_facts, ['Login_Name_Customer'])
    unique_assignee_names_decoded = string_to_list(df_facts, ['Name_Assignee'])
    unique_assignee_login_decoded = string_to_list(df_facts, ['Login_Name_Assignee'])

    # Imputation of missing values for Name_Assignee Column
    df_facts = null_handling(df_facts, {'Name_Assignee': 'Fechados pelo Cliente'})

    # Replaces resolve date by close date when the first is null and second exists
    df_facts = value_substitution(df_facts, non_null_column='Close_Date', null_column='Resolve_Date')

    # df_facts = df_facts.groupby('Request_Num').apply(close_and_resolve_date_replacements)  # Currently doing nothing, hence why it's commented

    # Removes duplicate request numbers
    df_facts = duplicate_removal(df_facts, ['Request_Num'])

    # Removes new lines, tabs, etc;
    df_facts = literal_removal(df_facts, 'Description')

    # Replaces string errors, specified in the provided dictionary
    df_facts = string_replacer(df_facts, dict_strings_to_replace)

    df_facts = value_replacement(df_facts, {'Description': options_file.regex_dict['url']})
    df_facts = value_replacement(df_facts, {'Summary': options_file.regex_dict['url']})
    df_facts = value_substitution(df_facts, non_null_column='Summary', null_column='Description')  # Replaces description by summary when the first is null and second exists

    df_facts = language_detection(df_facts, 'Description', 'Language')
    df_facts = string_replacer(df_facts, {('Language', 'ca'): 'es', ('Category_Id', 'pcat:'): ''})

    df_facts = summary_description_null_checkup(df_facts)  # Cleans requests which have the Summary and Description null

    stop_words_list = options_file.words_to_remove_from_description + unique_clients_names_decoded + unique_clients_login_decoded + unique_assignee_names_decoded + unique_assignee_login_decoded
    df_facts['Description'] = df_facts['Description'].apply(stop_words_removal, args=(stop_words_list,))

    if similar_process_flag:
        df_facts = similar_words_handling(df_facts, keywords_df, options_file.testing_dict)

    df_facts = text_preprocess(df_facts, unique_clients_names_decoded + unique_clients_login_decoded + unique_assignee_names_decoded + unique_assignee_login_decoded, options_file)

    df_facts = value_replacement(df_facts, options_file.language_replacements)

    # Checkpoint B.1 - Key Words data frame creation

    df_facts, df_top_words = top_words_processing(df_facts, description_col='StemmedDescription')

    log_record('Após o processamento a contagem de pedidos é de: {}'.format(df_facts['Request_Num'].nunique()), options_file.project_id)
    log_record('Fim Secção B.', options_file.project_id)
    performance_info_append(time.time(), 'Section_B_End')

    return df_facts, df_top_words


def data_modelling(df, df_top_words, df_manual_classification, keyword_dict, ranking_dict):
    performance_info_append(time.time(), 'Section_C_Start')
    log_record('Início Secção C...', options_file.project_id)

    df = new_request_type(df, df_top_words, df_manual_classification, keyword_dict, ranking_dict, options_file)

    log_record('Fim Secção C.', options_file.project_id)
    performance_info_append(time.time(), 'Section_C_End')

    return df


def deployment(df):
    performance_info_append(time.time(), 'Section_E_Start')
    log_record('Início Secção E...', options_file.project_id)
    df = df.astype(object).where(pd.notnull(df), None)

    sql_inject(df, options_file.DSN_SRV3_PRD, options_file.sql_info['database_source'], options_file.sql_info['final_table'], options_file, ['Request_Num', 'StemmedDescription', 'Description', 'Language', 'Open_Date', 'Label'], truncate=1)

    log_record('Fim Secção E.', options_file.project_id)
    performance_info_append(time.time(), 'Section_E_End')

    return


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        project_identifier, exception_desc = options_file.project_id, str(sys.exc_info()[1])
        log_record(exception_desc, project_identifier, flag=2)
        error_upload(options_file, project_identifier, format_exc(), exception_desc, error_flag=1)
        log_record('Falhou - Projeto: ' + str(project_dict[project_identifier]) + '.', project_identifier)

