import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from traceback import format_exc
import level_2_pa_servicedesk_2244_options as options_file
from modules.level_1_a_data_acquisition import read_csv, sql_retrieve_df, sql_mapping_retrieval
from modules.level_1_b_data_processing import summary_description_null_checkup, top_words_processing, threshold_grouping, value_count_histogram, date_cols, ohe, data_type_conversion, min_max_scaling, min_max_scaling_reverse, constant_columns_removal, remove_columns, object_column_removal, text_preprocess, literal_removal, string_to_list, df_join_function, null_handling, lowercase_column_conversion, null_analysis, remove_rows, value_replacement, value_substitution, duplicate_removal, language_detection, string_replacer, close_and_resolve_date_replacements
from modules.level_1_c_data_modelling import clustering_training, new_request_type
from modules.level_1_d_model_evaluation import cluster_metrics_plots, radial_chart_preprocess, make_spider
from modules.level_1_e_deployment import save_csv, sql_inject
from modules.level_0_performance_report import error_upload, log_record, project_dict, performance_info, performance_info_append
from wordcloud import WordCloud
from sklearn.decomposition import PCA
my_dpi = 96
pd.set_option('display.expand_frame_repr', False)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S @ %d/%m/%y', filename=options_file.log_files['full_log'], filemode='a')
logging.Logger('errors')
logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))

### Options
max_number_of_clusters = 10
clustering = 0


def main():
    log_record('Projeto: {}'.format(project_dict[options_file.project_id]), options_file.project_id)
    input_file_facts, input_file_durations, input_file_clients, input_file_pbi_categories, input_file_manual_classification = 'dbs/db_facts_initial.csv', 'dbs/db_facts_duration.csv', 'dbs/db_clients_initial.csv', 'dbs/db_pbi_categories_initial.csv', 'dbs/db_manual_classification.csv'
    query_filters = [{'Cost_Centre': '6825', 'Record_Type': ['1', '2']}, {'Cost_Centre': '6825'}]

    df_facts, df_facts_duration, df_clients, df_pbi_categories, df_manual_classifications = data_acquisition([input_file_facts, input_file_durations, input_file_clients, input_file_pbi_categories], query_filters, local=0)
    df, df_top_words = data_processing(df_facts, df_facts_duration, df_clients, df_pbi_categories)

    # df, df_top_words = pd.read_csv('output/df_cleaned.csv', index_col=0), pd.read_csv('output/df_top_words.csv', index_col=0)
    df = data_modelling(df, df_top_words, df_manual_classifications)

    if clustering:
        df_clustered, df_cluster_centers = data_modelling_cluster(df, max_number_of_clusters)
        # df_clustered = pd.read_csv('output/df_clustered.csv', index_col=0)
        # df_cluster_centers = pd.read_csv('output/df_cluster_centers.csv', index_col=0)
        df_cluster_centers_scaled = pd.read_csv('output/df_cluster_centers_scaled.csv', index_col=0)
        model_evaluation(df_clustered, df_cluster_centers, df_cluster_centers_scaled)

    # pca_analysis()
    deployment(df)
    performance_info(options_file.project_id, options_file, model_choice_message='N/A', unit_count=df.shape[0])

    log_record('Conclusão com sucesso - Projeto: {}'.format(project_dict[options_file.project_id]), options_file.project_id)


def data_modelling(df, df_top_words, df_manual_classification):
    performance_info_append(time.time(), 'Section_C_Start')
    log_record('Início Secção C...', options_file.project_id)

    df = new_request_type(df, df_top_words, df_manual_classification, options_file)

    log_record('Fim Secção C.', options_file.project_id)
    performance_info_append(time.time(), 'Section_C_End')

    return df


def pca_analysis():
    df_cleaned = pd.read_csv('output/df_clustered.csv', index_col=0)
    df_top_words = pd.read_csv('output/df_top_words_clustered.csv', index_col=0)

    g = df_cleaned.columns.to_series().groupby(df_cleaned.dtypes).groups
    dtype_dict = {k.name: v for k, v in g.items()}

    non_object_columns = list(dtype_dict['int64'].values) + list(dtype_dict['float64'].values)

    df_non_object_columns = df_cleaned[non_object_columns]

    df_non_object_columns_cleaned = df_non_object_columns.dropna(axis=0)

    pca = PCA(n_components=10)
    pca.fit(df_non_object_columns_cleaned)
    x = pca.transform(df_non_object_columns_cleaned)
    eigenvalues = pca.explained_variance_ratio_

    eigenvalues_sum = [eigenvalues[0:i+1].sum() for i in range(len(eigenvalues))]
    # plt.plot(range(1, len(eigenvalues)+1), eigenvalues_sum)
    plt.scatter(x[:, 1], x[:, 2], c=df_non_object_columns_cleaned['labels'])
    # plt.xlabel('Nº of Principal Components')
    # plt.ylabel('Variance Covered')
    # plt.title('PCA Analysis')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x[:, 0], x[:, 1], x[:, 3], c=df_non_object_columns_cleaned['labels'])
    plt.show()


def data_acquisition(input_files, query_filters, local=0):
    performance_info_append(time.time(), 'Section_A_Start')
    log_record('Início Secção A...', options_file.project_id)

    if local:
        df_facts = read_csv(input_files[0], index_col=0, parse_dates=options_file.date_columns, infer_datetime_format=True)
        df_facts_duration = read_csv(input_files[1], index_col=0)
        df_clients = read_csv(input_files[2], index_col=0)
        df_pbi_categories = read_csv(input_files[3], index_col=0)
        df_manual_classifications = read_csv(input_files[4], index_col=0)
    elif not local:
        df_facts = sql_retrieve_df(options_file.DSN, options_file.sql_info['database_source'],  options_file.sql_info['initial_table_facts'], options_file,  options_file.sql_fact_columns, query_filters=query_filters[0], parse_dates=options_file.date_columns)
        df_facts_duration = sql_retrieve_df(options_file.DSN, options_file.sql_info['database_source'], options_file.sql_info['initial_table_facts_durations'], options_file, options_file.sql_facts_durations_columns, query_filters=query_filters[1])
        df_clients = sql_retrieve_df(options_file.DSN, options_file.sql_info['database_source'], options_file.sql_info['initial_table_clients'], options_file, options_file.sql_dim_contacts_columns)
        df_pbi_categories = sql_retrieve_df(options_file.DSN, options_file.sql_info['database_source'], options_file.sql_info['initial_table_pbi_categories'], options_file, options_file.sql_pbi_categories_columns, query_filters=query_filters[1])
        df_manual_classifications = sql_retrieve_df(options_file.DSN, options_file.sql_info['database_source'], options_file.sql_info['aux_table'], options_file)

        save_csv([df_facts, df_facts_duration, df_clients, df_pbi_categories], ['dbs/db_facts_initial', 'dbs/db_facts_duration', 'dbs/db_clients_initial', 'dbs/db_pbi_categories_initial'])

    log_record('Fim Secção A...', options_file.project_id)
    performance_info_append(time.time(), 'Section_A_End')

    return df_facts, df_facts_duration, df_clients, df_pbi_categories, df_manual_classifications


def data_processing(df_facts, df_facts_duration, df_clients, df_pbi_categories):
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

    df_facts = duplicate_removal(df_facts, ['Request_Num'])

    df_facts = literal_removal(df_facts, 'Description')
    df_facts = string_replacer(df_facts, dict_strings_to_replace)

    df_facts = value_replacement(df_facts, {'Description': options_file.regex_dict['url']})
    df_facts = value_replacement(df_facts, {'Summary': options_file.regex_dict['url']})
    df_facts = value_substitution(df_facts, non_null_column='Summary', null_column='Description')  # Replaces description by summary when the first is null and second exists

    df_facts = language_detection(df_facts, 'Description', 'Language')
    df_facts = string_replacer(df_facts, {('Language', 'ca'): 'es', ('Category_Id', 'pcat:'): ''})

    df_facts = summary_description_null_checkup(df_facts)  # Cleans requests which have the Summary and Description null

    df_facts = text_preprocess(df_facts, unique_clients_names_decoded + unique_clients_login_decoded + unique_assignee_names_decoded + unique_assignee_login_decoded, options_file)

    df_facts = value_replacement(df_facts, options_file.language_replacements)

    # df_facts.to_csv('output/df_facts.csv')
    save_csv([df_facts], ['output/df_facts'])

    # Checkpoint B.1 - Key Words data frame creation
    # df_cleaned = clustering_preprocessing(df_facts)

    df_facts, df_top_words = top_words_processing(df_facts)

    log_record('Após o processamento a contagem de pedidos é de: {}'.format(df_facts['Request_Num'].nunique()), options_file.project_id)
    log_record('Fim Secção B.', options_file.project_id)
    performance_info_append(time.time(), 'Section_B_End')

    return df_facts, df_top_words


# def clustering_preprocessing(df_facts):
#     _, top_words_frequency = word_frequency(df_facts, threshold=30)
#     df_top_words, df_cleaned = words_dataframe_creation(df_facts, top_words_frequency)
#
#     try:
#         df_top_words.drop(['\''], axis=1, inplace=True)  # ToDo: will need to deal with this before it reaches this section of the code
#     except KeyError:
#         pass
#
#     df_cleaned = df_join_function(df_cleaned, df_top_words, on='Request_Num')
#
#     _, object_columns, non_object_columns = object_column_removal(df_cleaned)
#
#     # Checkpoint B.2 - Category Column treatment
#     datetime_columns_to_create = {'close_': 'Close_Date', 'open_': 'Open_Date', 'resolve_': 'Resolve_Date', 'assignee_': 'Assignee_Date'}
#     df_cleaned = date_cols(df_cleaned, datetime_columns_to_create)
#
#     df_cleaned = df_cleaned.groupby('Login_Name_Customer').apply(threshold_grouping, column='Login_Name_Customer', value='Outros', threshold=50)
#     df_cleaned = df_cleaned.groupby('Login_Name_Assignee').apply(threshold_grouping, column='Login_Name_Assignee', value='Outros', threshold=50)
#     df_cleaned = df_cleaned.groupby('Location_Name_Customer').apply(threshold_grouping, column='Location_Name_Customer', value='Outros', threshold=50)
#     df_cleaned = df_cleaned.groupby('Site_Name_Customer').apply(threshold_grouping, column='Site_Name_Customer', value='Outros', threshold=50)
#     df_cleaned = df_cleaned.groupby('Company_Group_Name_Customer').apply(threshold_grouping, column='Company_Group_Name_Customer', value='Outros', threshold=50)
#
#     # df_cleaned = remove_columns(df_cleaned, ['Location_Id_Customer', 'Contact_Customer_Id', 'Category_Id', 'Close_Date', 'Open_Date', 'Resolve_Date', 'Assignee_Date', 'Request_Num', 'Request_Id', 'Status_Id', 'Site_Name', 'Comments', 'Description', 'Summary',
#     #                                          'SLA_Resolution_Flag', 'Location_Id_Assignee', 'Company_Group_Customer', 'Contact_Type_Customer', 'Site_Id_Customer', 'Name_Customer', 'Comments_Customer', 'Name_Assignee', 'Comments_Assignee', 'Location_Name_Assignee', 'Company_Group_Name_Assignee',
#     #                                          'Contact_Assignee_Id', 'Site_Id_Assignee', 'Contact_Type_Assignee', 'Company_Group_Assignee', 'Site_Name_Assignee', 'StemmedDescription', 'SLA_StdTime_Resolution_Hours', 'SLA_StdTime_Assignee_Hours'])  # First Attempt
#
#     # df_cleaned = remove_columns(df_cleaned, ['Language', 'Location_Id_Customer', 'Contact_Customer_Id', 'Category_Id', 'Close_Date', 'Open_Date', 'Resolve_Date', 'Assignee_Date', 'Request_Num', 'Request_Id', 'Status_Id', 'Site_Name', 'Comments', 'Description', 'Summary',
#     #                                          'SLA_Resolution_Flag', 'Location_Id_Assignee', 'Company_Group_Customer', 'Contact_Type_Customer', 'Site_Id_Customer', 'Name_Customer', 'Comments_Customer', 'Name_Assignee', 'Comments_Assignee', 'Location_Name_Assignee', 'Company_Group_Name_Assignee',
#     #                                          'Contact_Assignee_Id', 'Site_Id_Assignee', 'Contact_Type_Assignee', 'Company_Group_Assignee', 'Site_Name_Assignee', 'StemmedDescription', 'SLA_StdTime_Resolution_Hours', 'SLA_StdTime_Assignee_Hours'])  # Second Attempt
#
#     df_cleaned = remove_columns(df_cleaned, ['Site_Name_Customer', 'Location_Name_Customer', 'Language', 'Location_Id_Customer', 'Contact_Customer_Id', 'Category_Id', 'Close_Date', 'Open_Date', 'Resolve_Date', 'Assignee_Date', 'Request_Num', 'Request_Id', 'Status_Id', 'Site_Name', 'Comments', 'Description', 'Summary',
#                                              'SLA_Resolution_Flag', 'Location_Id_Assignee', 'Company_Group_Customer', 'Contact_Type_Customer', 'Site_Id_Customer', 'Name_Customer', 'Comments_Customer', 'Name_Assignee', 'Comments_Assignee', 'Location_Name_Assignee', 'Company_Group_Name_Assignee',
#                                              'Contact_Assignee_Id', 'Site_Id_Assignee', 'Contact_Type_Assignee', 'Company_Group_Assignee', 'Site_Name_Assignee', 'StemmedDescription', 'SLA_StdTime_Resolution_Hours', 'SLA_StdTime_Assignee_Hours'])  # Third Attempt
#
#     df_cleaned = constant_columns_removal(df_cleaned)
#
#     non_keyword_columns = ['Contact_Assignee_Id', 'Login_Name_Assignee', 'Login_Name_Customer', 'SLA_Id', 'SLA_Resolution_Flag', 'WeekDay_Id', 'Request_Type', 'Priority_Id',
#                            'Contact_Assignee_Id', 'Site_Name_Customer', 'WaitingTime_Resolution_Minutes', 'SLA_Resolution_Minutes', 'WaitingTime_Assignee_Minutes', 'SLA_Assignee_Minutes', 'Location_Id_Customer', 'Site_Id_Customer',
#                            'Contact_Type_Customer', 'Company_Group_Customer', 'Location_Id_Assignee', 'Site_Id_Assignee', 'Contact_Type_Assignee', 'Company_Group_Assignee']
#
#     # for column in non_keyword_columns:
#     #     try:
#     #         print(column, df_cleaned[column].nunique())
#     #     except KeyError:
#     #         print('Column not found: {}'.format(column))
#
#     df_cleaned.dropna(axis=0, inplace=True)
#
#     # df_cleaned = ohe(df_cleaned, ['Language', 'WeekDay_Id', 'Login_Name_Customer', 'Login_Name_Assignee', 'Location_Name_Customer', 'Site_Name_Customer', 'Company_Group_Name_Customer'])  # First Attempt, with which the clusters were separated by language (PT/ES) and Location and Company (both CA/Ibericar)
#     # df_cleaned = ohe(df_cleaned, ['WeekDay_Id', 'Login_Name_Customer', 'Login_Name_Assignee', 'Location_Name_Customer', 'Site_Name_Customer', 'Company_Group_Name_Customer'])  # Second Attempt, removing Language;
#     df_cleaned = ohe(df_cleaned, ['WeekDay_Id', 'Login_Name_Customer', 'Login_Name_Assignee', 'Company_Group_Name_Customer'])  # Third Attempt, removing Location and Site Name;
#
#     df_cleaned['Resolution_Duration'] = df_cleaned['SLA_Resolution_Minutes'] - df_cleaned['WaitingTime_Resolution_Minutes']
#     df_cleaned['Assignee_Duration'] = df_cleaned['SLA_Assignee_Minutes'] - df_cleaned['WaitingTime_Assignee_Minutes']
#
#     data_type_conversion(df_cleaned, 'int64')
#
#     print(df_cleaned.head(10))


def data_modelling_cluster(df, max_number_of_clusters):
    # df = pd.read_csv('output/df_facts.csv', index_col=0)
    #
    # _, top_words_frequency = word_frequency(df, threshold=30)
    # df_top_words, df_cleaned = words_dataframe_creation(df, top_words_frequency)
    #
    # # df_top_words.to_csv('output/service_desk_df_cleaned_top_words.csv')
    # # df_top_words = pd.read_csv('output/service_desk_df_cleaned_top_words.csv', index_col=0)
    # df_top_words.drop(['\''], axis=1, inplace=True)
    #
    # df_cleaned = df_cleaned.join(df_top_words, on='Request_Num')

    # g = df.columns.to_series().groupby(df.dtypes).groups
    # dtype_dict = {k.name: v for k, v in g.items()}
    #
    # non_object_columns = list(dtype_dict['int64'].values) + list(dtype_dict['float64'].values)
    #
    # df_inter = df[non_object_columns]
    #
    # df_non_object_columns = df_inter.dropna(axis=0)

    df_scaled, scaler = min_max_scaling(df)

    models, scores, score_names = clustering_training(df_scaled, max_number_of_clusters)
    cluster_metrics_plots(len(models), scores, score_names)

    # Choosing model with 3 clusters:
    # df_top_words['labels'] = models[1].labels_
    df['labels'] = models[1].labels_
    df_cluster_center = radial_chart_preprocess(df, models[1])
    df_cluster_center.to_csv('output/df_cluster_centers_scaled.csv')

    array_cluster_center_converted = min_max_scaling_reverse(df_cluster_center, scaler)

    df.to_csv('output/df_clustered.csv')
    pd.DataFrame(array_cluster_center_converted, columns=list(df)[:-1]).to_csv('output/df_cluster_centers.csv')

    # df_cleaned['labels'] = models[1].labels_
    # df_cleaned.to_csv('output/df_clustered.csv')

    # print(df_top_words.head())
    # print(df_cleaned.head())

    # df_top_words = pd.read_csv('output/df_top_words_clustered.csv', index_col=0)
    # print(df_top_words.head())
    # print(df_top_words['labels'].value_counts())

    # cluster_word_cloud(df_top_words)

    return df, df_cluster_center


def model_evaluation(df, df_cluster_center, df_cluster_center_scaled):
    # print(df.head())
    # print(df_cluster_center_scaled.head())
    # print(df_cluster_center.head())

    # non_keyword_columns = ['Contact_Customer_Id', 'SLA_Id', 'SLA_Resolution_Flag', 'WeekDay_Id', 'Request_Type', 'Category_Id', 'Priority_Id',
    #  'Contact_Assignee_Id', 'WaitingTime_Resolution_Minutes', 'SLA_Resolution_Minutes', 'WaitingTime_Assignee_Minutes', 'SLA_Assignee_Minutes', 'Location_Id_Customer', 'Site_Id_Customer',
    #  'Contact_Type_Customer', 'Company_Group_Customer', 'Location_Id_Assignee', 'Site_Id_Assignee', 'Contact_Type_Assignee', 'Company_Group_Assignee']
    print(list(df))

    # non_keyword_columns = ['Contact_Customer_Id', 'SLA_Id', 'SLA_Resolution_Flag', 'WeekDay_Id', 'Request_Type', 'Category_Id', 'Priority_Id',
    #  'Contact_Assignee_Id', 'WaitingTime_Resolution_Minutes', 'SLA_Resolution_Minutes', 'WaitingTime_Assignee_Minutes', 'SLA_Assignee_Minutes', 'Location_Id_Customer', 'Site_Id_Customer',
    #  'Contact_Type_Customer', 'Company_Group_Customer', 'Location_Id_Assignee', 'Site_Id_Assignee', 'Contact_Type_Assignee', 'Company_Group_Assignee']

    non_keyword_columns_v1 = ['Priority_Id', 'SLA_Id', 'Request_Type', 'SLA_Resolution_Minutes', 'SLA_Assignee_Minutes', 'WaitingTime_Assignee_Minutes', 'WaitingTime_Resolution_Minutes', 'close_day', 'close_month', 'close_year', 'open_day', 'open_month', 'open_year', 'resolve_day', 'resolve_month', 'resolve_year', 'assignee_day', 'assignee_month', 'assignee_year',
     'Language_pt', 'Language_es', 'Language_da', 'Language_fr', 'Language_en', 'Language_ro', 'Language_it', 'Language_no', 'Language_lt', 'Language_sq', 'Language_et', 'Language_so', 'Language_id', 'WeekDay_Id_4.0', 'WeekDay_Id_5.0', 'WeekDay_Id_6.0', 'WeekDay_Id_2.0', 'WeekDay_Id_3.0', 'WeekDay_Id_1.0', 'WeekDay_Id_7.0', 'Login_Name_Customer_scga6301', 'Login_Name_Customer_jfer', 'Login_Name_Customer_losbc', 'Login_Name_Customer_Outros', 'Login_Name_Customer_scnp',
     'Login_Name_Customer_rfoveb', 'Login_Name_Customer_fgro', 'Login_Name_Customer_jafa', 'Login_Name_Customer_jrga', 'Login_Name_Customer_paur', 'Login_Name_Customer_jsor', 'Login_Name_Customer_ibra', 'Login_Name_Customer_jcoe', 'Login_Name_Customer_rfssc', 'Login_Name_Customer_ajsco', 'Login_Name_Customer_lmrm', 'Login_Name_Customer_erui', 'Login_Name_Customer_tpfs', 'Login_Name_Customer_jagme', 'Login_Name_Customer_ccmmp', 'Login_Name_Customer_lflbf', 'Login_Name_Customer_hjma',
     'Login_Name_Customer_apcp', 'Login_Name_Customer_acmc', 'Login_Name_Customer_mlpglc', 'Login_Name_Customer_mjsdms', 'Login_Name_Customer_fdapc', 'Login_Name_Customer_srja', 'Login_Name_Customer_maal', 'Login_Name_Assignee_mimp', 'Login_Name_Assignee_fplo', 'Login_Name_Assignee_algs', 'Login_Name_Assignee_aasf', 'Login_Name_Assignee_safs', 'Login_Name_Assignee_Outros', 'Login_Name_Assignee_pjaca', 'Login_Name_Assignee_fpfb', 'Login_Name_Assignee_accma', 'Login_Name_Assignee_tmsm',
     'Location_Name_Customer_PORTIANGA-COMERCIO INTERNAC PARTI', 'Location_Name_Customer_IBERICAR HOLDING ANDALUCIA, SL', 'Location_Name_Customer_Outros', 'Location_Name_Customer_CAETANO-AUTO', 'Location_Name_Customer_CAETANO RETAIL SGPS', 'Location_Name_Customer_IBERICAR REICOMSA', 'Location_Name_Customer_IBERICAR CENTRO Y CATALUNA', 'Location_Name_Customer_CAETANO BAVIERA', 'Location_Name_Customer_IBERICAR GALICIA', 'Location_Name_Customer_HYUNDAI PORTUGAL, SA.',
     'Location_Name_Customer_RIGOR', 'Location_Name_Customer_CAETANO DRIVE, SPORT E URBAN', 'Location_Name_Customer_TOYOTA CAETANO', 'Location_Name_Customer_CAETANOBUS', 'Site_Name_Customer_PORTIANGA-COMERCIO INTERNAC PARTI', 'Site_Name_Customer_IBERICAR HOLDING ANDALUCIA, SL', 'Site_Name_Customer_Outros', 'Site_Name_Customer_CAETANO-AUTO', 'Site_Name_Customer_CAETANO RETAIL SGPS', 'Site_Name_Customer_IBERICAR REICOMSA', 'Site_Name_Customer_IBERICAR CENTRO Y CATALUNA ',
     'Site_Name_Customer_CAETANO BAVIERA', 'Site_Name_Customer_IBERICAR GALICIA', 'Site_Name_Customer_HYUNDAI PORTUGAL, SA.', 'Site_Name_Customer_RIGOR', 'Site_Name_Customer_CAETANO DRIVE, SPORT E URBAN', 'Site_Name_Customer_TOYOTA CAETANO', 'Site_Name_Customer_CAETANOBUS', 'Company_Group_Name_Customer_Grupo SC', 'Company_Group_Name_Customer_Grupo Ibericar', 'Company_Group_Name_Customer_Grupo Caetano-Auto', 'Company_Group_Name_Customer_Grupo Caetano Retail Portugal',
     'Company_Group_Name_Customer_Outros', 'Company_Group_Name_Customer_Grupo Baviera', 'Company_Group_Name_Customer_Grupo Rigor', 'Company_Group_Name_Customer_Grupo Caetanobus', 'Resolution_Duration', 'Assignee_Duration']

    non_keyword_columns = [x for x in list(df) if x in non_keyword_columns_v1]
    # legends, patches_first_half, patches_second_half = [], [], []
    # length = int(len(non_keyword_columns) / 2)
    # for i in range(0, length):
    #     # legends.append('{} - {}'.format(list(string.ascii_uppercase)[i], non_keyword_columns[i]))
    #     patches_first_half.append(mpatches.Patch(color='red', label='{} - {}'.format(list(string.ascii_uppercase)[i], non_keyword_columns[i])))
    # for i in range(length, len(non_keyword_columns)):
    #     patches_second_half.append(mpatches.Patch(color='red', label='{} - {}'.format(list(string.ascii_uppercase)[i], non_keyword_columns[i])))

    # plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)

    my_palette = plt.cm.get_cmap("Set2", len(df_cluster_center.index))

    for row in range(df_cluster_center_scaled.shape[0]):
        make_spider(df_cluster_center_scaled[non_keyword_columns], row=row, title='Group {} - {} Requests'.format(df_cluster_center_scaled.index[row], df[df['labels'] == df_cluster_center_scaled.index[row]].shape[0]), color=my_palette(row))

    # plt.gca().add_artist(plt.legend(handles=patches_second_half, bbox_to_anchor=(3.5, 1), loc='best'))
    # plt.legend(handles=patches_first_half, bbox_to_anchor=(2, 1), loc='best')
    plt.show()


def cluster_word_cloud(df):
    occurrences = [dict() for _ in range(4)]

    for i in range(4):
        df_label = df[df['labels'] == i]
        df_label = constant_columns_removal(df_label, options_file.project_id, value=0)

        list_words = list(df_label)[:-1]
        # print(i, '\n', df_label.describe().T)
        for word in list_words:
            occurrences[i][word] = df_label.loc[:, word].sum(axis=0)

    def random_color_func(word=None, font_size=None, position=None,
                          orientation=None, font_path=None, random_state=None):
        h = int(360.0 * tone / 255.0)
        s = int(100.0 * 255.0 / 255.0)
        l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
        return "hsl({}, {}%, {}%)".format(h, s, l)

    def make_wordcloud(listing, increment):
        ax1 = fig.add_subplot(4, 2, increment)
        words = dict()
        trunc_occurrences = listing
        for s in trunc_occurrences:
            words[s[0]] = s[1]
        # ________________________________________________________
        wordcloud = WordCloud(width=1000, height=400, background_color='lightgrey',
                              max_words=1628, relative_scaling=1,
                              color_func=random_color_func,
                              normalize_plurals=False)
        wordcloud.generate_from_frequencies(words)
        ax1.imshow(wordcloud, interpolation="bilinear")
        ax1.axis('off')
        plt.title('cluster nº{}'.format(increment - 1))

    fig = plt.figure(1, figsize=(14, 14))
    color = [0, 160, 130, 95, 280, 40, 330, 110, 25]
    for i in range(4):
        list_cluster_occurrences = occurrences[i]

        tone = color[i]
        listing = []
        for key, value in list_cluster_occurrences.items():
            listing.append([key, value])
        listing.sort(key=lambda x: x[1], reverse=True)
        make_wordcloud(listing, i+1)

    plt.show()


# def clustering_application(df, n_clusters_max):
#     silhouette_scores, calinski_scores, inertia_scores, centroids = [], [], [], []
#     print('Calculating plots with different number of clusters...')
#     models = [KMeans(n_clusters=n, init='k-means++', max_iter=10, n_init=100, n_jobs=-1).fit(df) for n in range(2, n_clusters_max)]
#
#     for m in models:
#         print('Evaluating {} clusters...'.format(len(np.unique(list(m.labels_)))))
#         s = silhouette_score(df, m.labels_, random_state=42)
#         ch = calinski_harabaz_score(df, m.labels_)
#         centroids.append(m.cluster_centers_)
#         inertia_scores.append(m.inertia_)
#         silhouette_scores.append(s)
#         calinski_scores.append(ch)
#
#     dist = [np.min(cdist(df, c, 'euclidean'), axis=1) for c in centroids]
#     totss = sum(pdist(df) ** 2) / df.shape[0]
#     totwithinss = [sum(d ** 2) for d in dist]
#     between_clusters = (totss - totwithinss) / totss * 100
#     scores = [silhouette_scores, calinski_scores, inertia_scores, between_clusters]
#     score_names = ['Silhouette Scores', 'Calinski Scores', 'Inertia Scores', 'Elbow Method']
#
#     return models, scores, score_names


# def cluster_metrics_plots(number_of_models_trained, scores, score_names):
#
#     fig, ax = plt.subplots(2, 2, figsize=(18, 12))
#     plt.setp(ax, xticks=range(0, number_of_models_trained), xticklabels=range(2, number_of_models_trained + 2))
#
#     ax[0, 0].plot(scores[0])
#     ax[0, 0].set_title(score_names[0])
#     ax[0, 0].grid()
#
#     ax[1, 0].plot(scores[1])
#     ax[1, 0].set_title(score_names[1])
#     ax[1, 0].grid()
#
#     ax[0, 1].plot(scores[2])
#     ax[0, 1].set_title(score_names[2])
#     ax[0, 1].grid()
#
#     ax[1, 1].plot(scores[3])
#     ax[1, 1].set_title(score_names[3])
#     ax[1, 1].grid()
#
#     plt.show()


def word_histogram(listing):

    listing = sorted(listing, key=lambda x: x[1], reverse=True)
    number_of_words = 125
    plt.rc('font', weight='normal')
    fig, ax = plt.subplots(figsize=(7, 25))
    y_axis = [i[1] for i in listing[:number_of_words]]
    x_axis = [k for k, i in enumerate(listing[:number_of_words])]
    x_label = [i[0] for i in listing[:number_of_words]]
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=7)
    plt.yticks(x_axis, x_label)
    plt.xlabel("#Tickets", fontsize=18, labelpad=10)
    ax.barh(x_axis, y_axis, align='center')
    ax = plt.gca()
    ax.invert_yaxis()
    plt.grid()
    plt.title("Word's Frequency", bbox={'facecolor': 'k', 'pad': 5}, color='w', fontsize=25)
    plt.show()


def deployment(df):
    performance_info_append(time.time(), 'Section_E_Start')
    log_record('Início Secção E...', options_file.project_id)
    df = df.astype(object).where(pd.notnull(df), None)

    sql_inject(df, options_file.DSN, options_file.sql_info['database_source'], options_file.sql_info['final_table'], options_file, ['Request_Num', 'StemmedDescription', 'Description', 'Language', 'Label'], truncate=1)

    # sql_join(df, options_file.DSN, options_file.sql_info['database_source'], options_file.sql_info['initial_table_facts'], options_file)

    log_record('Fim Secção E.', options_file.project_id)
    performance_info_append(time.time(), 'Section_E_End')

    return


def cdf(listing, name):
    listing = sorted(listing, key=lambda x: x[1], reverse=True)

    ser = pd.Series(np.sort([i[1] for i in listing]))
    cum_dist = np.linspace(0., 1., len(ser))
    ser_cdf = pd.Series(cum_dist, index=ser.sort_values())
    ser_cdf.plot()
    plt.xlabel(name)
    plt.ylabel('CDF')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        project_identifier, exception_desc = options_file.project_id, str(sys.exc_info()[1])
        log_record(exception_desc, project_identifier, flag=2)
        error_upload(options_file, project_identifier, format_exc(), exception_desc, error_flag=1)
        log_record('Falhou - Projeto: ' + str(project_dict[project_identifier]) + '.', project_identifier)

