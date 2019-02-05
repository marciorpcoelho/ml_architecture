import sys
import pandas as pd
import logging
import nltk
import string
import unidecode
from nltk.stem.snowball import SnowballStemmer
import level_2_pa_servicedesk_2244_options as options_file
from level_1_a_data_acquisition import read_csv, sql_retrieve_df
from level_1_b_data_processing import lowercase_column_convertion, null_analysis, zero_analysis, inf_analysis, remove_rows, value_replacement, date_replacement, duplicate_removal, language_detection
from level_1_e_deployment import save_csv, sql_inject
from level_0_performance_report import error_upload, log_record, project_dict
pd.set_option('display.expand_frame_repr', False)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S @ %d/%m/%y', filename=options_file.log_files['full_log'], filemode='a')
logging.Logger('errors')
logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))


def main():
    log_record('Project: PA @ Service Desk', options_file.project_id)
    input_file_facts, input_file_durations, input_file_clients, pbi_categories = 'dbs/db_facts_initial.csv', 'dbs/db_facts_duration.csv', 'dbs/db_clients_initial.csv', 'dbs/db_pbi_categories_initial.csv'
    query_filters = [{'Cost_Centre': '6825', 'Record_Type': ['1', '2']}, {'Cost_Centre': '6825'}]

    df_facts, df_facts_duration, df_clients, df_pbi_categories = data_acquisition([input_file_facts, input_file_durations, input_file_clients, pbi_categories], query_filters, local=0)
    df = data_processing(df_facts, df_facts_duration, df_clients, df_pbi_categories)
    deployment(df)


def data_acquisition(input_files, query_filters, local=0):
    log_record('Started Step A...', options_file.project_id)

    if local:
        df_facts = read_csv(input_files[0], index_col=0, parse_dates=options_file.date_columns, infer_datetime_format=True)
        df_facts_duration = read_csv(input_files[1], index_col=0)
        df_clients = read_csv(input_files[2], index_col=0)
        df_pbi_categories = read_csv(input_files[3], index_col=0)
    elif not local:
        df_facts = sql_retrieve_df(options_file.DSN, options_file.sql_info['database_source'],  options_file.sql_info['initial_table_facts'], options_file,  options_file.sql_fact_columns, query_filters=query_filters[0], parse_dates=options_file.date_columns)
        df_facts_duration = sql_retrieve_df(options_file.DSN, options_file.sql_info['database_source'], options_file.sql_info['initial_table_facts_durations'], options_file, options_file.sql_facts_durations, query_filters=query_filters[1])
        df_clients = sql_retrieve_df(options_file.DSN, options_file.sql_info['database_source'], options_file.sql_info['initial_table_clients'], options_file)
        df_pbi_categories = sql_retrieve_df(options_file.DSN, options_file.sql_info['database_source'], options_file.sql_info['initial_table_pbi_categories'], options_file, options_file.sql_pbi_categories_columns, query_filters=query_filters[1])

        save_csv([df_facts, df_facts_duration, df_clients, df_pbi_categories], ['dbs/db_facts_initial', 'dbs/db_facts_duration', 'dbs/db_clients_initial', 'dbs/db_pbi_categories_initial'])

    log_record('Finished Step A...', options_file.project_id)
    return df_facts, df_facts_duration, df_clients, df_pbi_categories


def data_processing(df_facts, df_facts_duration, df_clients, df_pbi_categories):
    log_record('Started Step B...', options_file.project_id)

    print('Total Initial Requests:', df_facts['Request_Num'].nunique())
    pbi_categories = remove_rows(df_pbi_categories, [df_pbi_categories[~df_pbi_categories['Category_Name'].str.contains('Power BI')].index])['Category_Id'].values  # Selects the Category ID's which belong to PBI
    print('The number of PBI requests are:', df_facts[df_facts['Category_Id'].isin(pbi_categories)]['Request_Num'].nunique())
    df_facts = remove_rows(df_facts, [df_facts.loc[df_facts['Category_Id'].isin(pbi_categories)].index])  # Removes the rows which belong to PBI;
    # print('After PBI Filtering, the number of requests for this year is:', df_facts[df_facts['Open_Date'] > '2019-01-01']['Request_Num'].nunique())
    print('After PBI Filtering, the number of requests is:', df_facts['Request_Num'].nunique())
    df_facts = lowercase_column_convertion(df_facts, columns=['Summary', 'Description'])
    # df_facts = remove_rows(df_facts, [df_facts[df_facts.Description.isnull() | df_facts.Summary.isnull()].index])  # Removes rows without summary or description (there are no requests with Description without Summary)
    print('Total Requests after Treatment:', df_facts['Request_Num'].nunique())

    df_facts = df_facts.join(df_facts_duration.set_index('Request_Num'), on='Request_Num')
    df_facts = df_facts.join(df_clients.set_index('Contact_Id'), on='Contact_Customer_Id')
    df_facts = value_replacement(df_facts, ['Request_Num', 'Request_Num', 'Request_Num', 'Request_Num', 'Request_Num', 'Request_Num', 'Request_Num', 'Request_Num'], ['Contact_Assignee_Id', 'Contact_Assignee_Id', 'Contact_Assignee_Id', 'Contact_Assignee_Id', 'Contact_Assignee_Id', 'Contact_Assignee_Id', 'Contact_Assignee_Id', 'Contact_Assignee_Id'], ['RE-107512', 'RE-114012', 'RE-175076', 'RE-191719', 'RE-74793', 'RE-80676', 'RE-84389', 'RE-157518'], [-107178583, -107178583, 1746469363, 129950480, 1912342313, 1912342313, 1912342313, -172602144])
    df_facts = df_facts.join(df_clients.set_index('Contact_Id'), on='Contact_Assignee_Id', lsuffix='_Customer', rsuffix='_Assignee')

    df_facts = value_replacement(df_facts, options_file.cols_with_characteristic, options_file.cols_to_replace, options_file.category_id, options_file.values_to_replace_by)  #
    # df_facts = value_replacement(df_facts, ['Request_Num', 'Request_Num', 'Request_Num'], ['Contact_Assignee_Id', 'Contact_Assignee_Id', 'Contact_Assignee_Id'], ['RE-74793', 'RE-84389', 'RE-80676', 'RE-191719', 'RE-74793', 'RE-80676', 'RE-84389'], ['Felisbela Lopes', 'Felisbela Lopes', 'Susana Silveira', 'Paulo Alves', 'António Fonseca', 'António Fonseca', 'António Fonseca'])
    df_facts.loc[df_facts['Name_Assignee'].isnull(), 'Name_Assignee'] = 'Fechados pelo Cliente'
    df_facts = date_replacement(df_facts)

    # df_facts = df_facts.groupby('Request_Num').apply(close_and_resolve_date_replacements)  # Currently doing nothing, hence why it's commented

    # print('Open Requests:', df_facts.loc[df_facts['Resolve_Date'].isnull()]['Request_Num'].nunique())
    # print('Open Requests:', df_facts.loc[df_facts['Resolve_Date'].isnull()]['Request_Num'].unique())

    df_facts = duplicate_removal(df_facts, ['Request_Num'])

    url_pattern = r'http://(.*)'
    df_facts.loc[~df_facts['Description'].isnull(), 'Description'] = df_facts[~df_facts['Description'].isnull()]['Description'].map(lambda s: s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\\u', ' '))
    df_facts.loc[~df_facts['Description'].isnull(), 'Description'] = df_facts[~df_facts['Description'].isnull()]['Description'].str.replace(url_pattern, ' ')  # Finds and replaces the pattern defined by pattern

    # df_facts[~df_facts['Description'].isnull()]['Description'] = df_facts[~df_facts['Description'].isnull()]['Description'].map(lambda s: s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\\u', ' '))
    # df_facts[~df_facts['Description'].isnull()]['Description'] = df_facts[~df_facts['Description'].isnull()]['Description'].str.replace(url_pattern, ' ')  # Finds and replaces the pattern defined by pattern

    # df_facts['Description'] = df_facts['Description'].map(lambda s: s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\\u', ' '))
    # df_facts['Description'] = df_facts['Description'].str.replace(url_pattern, ' ')  # Finds and replaces the pattern defined by pattern
    # print('After', df_facts.shape)

    print('Total Requests:', df_facts['Request_Num'].nunique(), ', Row Count:', df_facts.shape[0])
    stemmer_pt = SnowballStemmer('porter')
    # stemmer_es = SnowballStemmer('spanish')

    df_facts = language_detection(df_facts, 'Description', 'Language')
    df_facts['StemmedDescription'] = str()
    df_facts = value_replacement(df_facts, ['Language'], ['Language'], ['ca'], ['es'])

    # Punctuation Removal
    for key, row in df_facts.iterrows():
        description, stemmed_word = row['Description'], []
        digit_remover = str.maketrans('', '', string.digits)
        punctuation_remover = str.maketrans('', '', string.punctuation)
        try:
            tokenized = nltk.tokenize.word_tokenize(description)
            for word in tokenized:
                word = word.translate(digit_remover).translate(punctuation_remover)
                word = unidecode.unidecode(word)
                if ',' in word:
                    print(word)
                if word in ['\'\'', '``', '“', '”', '']:
                    continue
                else:
                    stemmed_word.append(stemmer_pt.stem(word))
                # else:
                #     if language == 'pt':
                #         stemmed_word.append(stemmer_pt.stem(word))
                #     elif language == 'es':
                #         stemmed_word.append(stemmer_pt.stem(word))
        except TypeError:
            pass
        df_facts.at[key, 'StemmedDescription'] = ' '.join([x for x in stemmed_word if x not in options_file.words_to_remove_from_description])

    df_facts.loc[df_facts['Name_Assignee'].isnull(), 'Name_Assignee'] = 'Fechados pelo Cliente'
    df_facts.loc[(df_facts['Contact_Customer_Id'] == 1316563093) | (df_facts['Contact_Customer_Id'] == -650110013) | (df_facts['Contact_Customer_Id'] == 1191100018) | (df_facts['Contact_Customer_Id'] == -849867232) |
                 (df_facts['Contact_Customer_Id'] == 80794334) | (df_facts['Contact_Customer_Id'] == -1511754133) | (df_facts['Contact_Customer_Id'] == 1566878955) | (df_facts['Contact_Customer_Id'] == -250410311) |
                 (df_facts['Contact_Customer_Id'] == 1959237887), 'Language'] = 'es'  # Javier Soria, 'Juan Fernandez', 'Juan Gomez', 'Juan Sanchez', 'Cesar Malvido', 'Eduardo Ruiz', 'Ignacio Bravo', 'Marc Illa', 'Toni Silva'
    print('Total Requests:', df_facts['Request_Num'].nunique(), ', Row Count:', df_facts.shape[0])
    df_facts.to_csv('output/df_facts.csv')

    log_record('Finished Step B.', options_file.project_id)
    return df_facts


def close_and_resolve_date_replacements(x):

    if len(x) > 1 and len(x) > sum(x['Assignee_Date'].isnull()) >= 1:
        x.dropna(subset=['Assignee_Date'], axis=0, inplace=True)

    if len(x) > 1 and len(x) > sum(x['Close_Date'].isnull()) >= 1:
        x.dropna(subset=['Close_Date'], axis=0, inplace=True)

    if len(x) > 1 and len(x) > sum(x['Resolve_Date'].isnull()) >= 1:
        x.dropna(subset=['Resolve_Date'], axis=0, inplace=True)

    return x


def deployment(df):
    log_record('Started Step E...', options_file.project_id)
    null_analysis(df)
    df = df.astype(object).where(pd.notnull(df), None)

    sql_inject(df, options_file.DSN_MLG, options_file.sql_info['database_final'], options_file.sql_info['final_table'], options_file, list(df), truncate=1)

    log_record('Finished Step E.', options_file.project_id)
    return


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        project_identifier = 2244
        log_record(exception.args[0], project_identifier, flag=2)
        error_upload(options_file, project_identifier, options_file.log_files['full_log'], error_flag=1)
        log_record('Failed - Project: ' + str(project_dict[project_identifier]) + '.', project_identifier)

