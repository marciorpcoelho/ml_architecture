import pandas as pd
import numpy as np
import sys
import time
import pyodbc
import level_1_b_data_processing as level_b
import level_0_performance_report
from level_1_e_deployment import sql_get_last_vehicle_count, sql_inject_single_line


def read_csv(*args, **kwargs):

    df = pd.read_csv(*args, **kwargs)

    return df


def vehicle_count_checkup(df, options_file, sql_check=0):
    current_vehicle_count = df['Nº Stock'].nunique()
    last_vehicle_count = sql_get_last_vehicle_count(options_file.DSN_MLG, options_file, options_file.sql_info['database'], options_file.sql_info['vhe_number_history'])

    if not sql_check:
        if current_vehicle_count < 80:
            raise ValueError('Apenas ' + str(current_vehicle_count) + ' veículos foram encontrados. Por favor verificar na base de dados.')
    elif sql_check:
        if current_vehicle_count < last_vehicle_count:
            raise ValueError('Atual contagem de veículos ({}) é inferior à ultima contagem ({}). Por favor verificar na base de dados.'.format(current_vehicle_count, last_vehicle_count))
        elif current_vehicle_count == last_vehicle_count:
            level_0_performance_report.log_record('Atual contagem de veículos ({}) sem incrementos desde a última vez que o modelo foi treinado ({}). Por favor confirmar se o comportamento é o esperado.'.format(current_vehicle_count, last_vehicle_count), options_file.project_id, flag=1)
        else:
            time_tag_date = time.strftime("%Y-%m-%d")
            values = [str(current_vehicle_count), time_tag_date]
            sql_inject_single_line(options_file.DSN_MLG, options_file.UID, options_file.PWD, options_file.sql_info['database'], options_file.sql_info['vhe_number_history'], values)
            level_0_performance_report.log_record('Updating vehicle count: {}.'.format(current_vehicle_count), options_file.project_id, flag=0)
    return


# This function works really well for one single function on scheduler (provided i had sys.stdout.flush() to the end of each file). But if more than one functions are running at the same time (different threads) the stdout
# saved is all mixed and saved on the file of the last function; - trying now with logging module
def log_files(project_name, output_dir='logs/'):
    sys.stdout = open(output_dir + project_name + '.txt', 'a')
    sys.stderr = open(output_dir + project_name + '.txt', 'a')


def sql_retrieve_df(dsn, db, view, options_file, columns='*', query_filters=0, column_renaming=0, **kwargs):
    start = time.time()
    query, query_filters_string_list = None, []

    if columns != '*':
        columns = str(columns)[1:-1].replace('\'', '')

    try:
        cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'.format(dsn, options_file.UID, options_file.PWD, db), searchescape='\\')

        if type(query_filters) == int:
            query = 'SELECT ' + columns + ' FROM ' + view + ' WITH (NOLOCK)'
        elif type(query_filters) == dict:
            for key in query_filters:
                if type(query_filters[key]) == list:
                    testing_string = '\'%s\'' % "\', \'".join(query_filters[key])
                    query_filters_string_list.append(key + ' in (' + testing_string + ')')
                else:
                    query_filters_string_list.append(key + ' = \'%s\'' % str(query_filters[key]))
            query = 'SELECT ' + columns + ' FROM ' + view + ' WITH (NOLOCK) WHERE ' + ' and '.join(query_filters_string_list)
        df = pd.read_sql(query, cnxn, **kwargs)
        if column_renaming:
            level_b.column_rename(df, list(options_file.sql_to_code_renaming.keys()), list(options_file.sql_to_code_renaming.values()))

        cnxn.close()

        print('Elapsed time: {:.2f} seconds.'.format(time.time() - start))
        return df

    except (pyodbc.ProgrammingError, pyodbc.OperationalError):
        return  # ToDo need to figure a better way of handling these errors


# When you have a full query to use and no need to prepare anything. Will merge with sql_retrieve_df in the future;
def sql_retrieve_df_specified_query(dsn, db, options_file, query):
    start = time.time()

    try:
        cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'.format(dsn, options_file.UID, options_file.PWD, db), searchescape='\\')

        df = pd.read_sql(query, cnxn)
        cnxn.close()

        print('Elapsed time: {:.2f} seconds.'.format(time.time() - start))
        return df

    except (pyodbc.ProgrammingError, pyodbc.OperationalError):
        return


def sql_mapping_retrieval(dsn, db, mapping_tables, mapped_column_name, options_file, multiple_columns=0):
    dictionary_list = []
    dictionary_ranking = {}

    for mapping_table in mapping_tables:
        parameter_dict = {}
        df = sql_retrieve_df(dsn, db, mapping_table, options_file)
        if not multiple_columns:
            for key in df[mapped_column_name].unique():
                parameter_dict[key] = list(df[df[mapped_column_name] == key]['Original_Value'].values)
        if multiple_columns:
            for key in df[mapped_column_name].unique():
                listing = df[df[mapped_column_name] == key][[x for x in list(df)[:-1] if x not in mapped_column_name]].values  # Added [:-1] to consider the cases where the last column is the priority rank
                parameter_dict[key] = np.unique([item for sublist in listing for item in sublist if item is not None])
                ranking = int(df[df[mapped_column_name] == key][list(df)[-1]].unique()[0])
                dictionary_ranking[key] = ranking
        dictionary_list.append(parameter_dict)

    return dictionary_list, dictionary_ranking


def dw_data_retrieval(pse_code, current_date, options_info, update):

    print('PSE_Code = {}'.format(pse_code))

    sales_info = ['dbs/df_sales', options_info.sales_query]
    purchases_info = ['dbs/df_purchases', options_info.purchases_query]
    stock_info = ['dbs/df_stock', options_info.stock_query]
    reg_info = ['dbs/df_reg', options_info.reg_query]
    reg_al_info = ['dbs/df_reg_al_client', options_info.reg_autoline_clients]

    dfs = []

    for dimension in [sales_info, purchases_info, stock_info, reg_info, reg_al_info]:

        if update:
            file_name = dimension[0] + '_' + str(pse_code) + '_' + str(current_date)
        else:
            file_name = dimension[0] + '_' + str(pse_code) + '_02_08_19'  # Last time I ran this script and saved these files

        try:
            df = read_csv(file_name + '.csv', index_col=0)
            print('{} file found.'.format(file_name))
        except FileNotFoundError:
            print('{} file not found. Retrieving data from SQL...'.format(file_name))
            df = sql_retrieve_df_specified_query(options_info.DSN_PRD, options_info.sql_info['database'], options_info, dimension[1])
            df.to_csv(file_name + '.csv')

        dfs.append(df)

    df_sales = dfs[0]
    df_purchases = dfs[1]
    df_stock = dfs[2]
    df_reg = dfs[3]
    df_reg_al_clients = dfs[4]

    df_purchases['Movement_Date'] = pd.to_datetime(df_purchases['Movement_Date'], format='%Y%m%d')
    df_purchases['WIP_Date_Created'] = pd.to_datetime(df_purchases['WIP_Date_Created'], format='%Y%m%d')

    df_sales['SLR_Document_Date'] = pd.to_datetime(df_sales['SLR_Document_Date'], format='%Y%m%d')
    df_sales['WIP_Date_Created'] = pd.to_datetime(df_sales['WIP_Date_Created'], format='%Y%m%d')
    df_sales['Movement_Date'] = pd.to_datetime(df_sales['Movement_Date'], format='%Y%m%d')

    df_stock['Record_Date'] = pd.to_datetime(df_stock['Record_Date'], format='%Y%m%d')
    df_stock.rename(index=str, columns={'Quantity': 'Stock_Qty'}, inplace=True)

    return df_sales, df_purchases, df_stock, df_reg, df_reg_al_clients


def autoline_data_retrieval(pse_code):

    try:
        df_al = read_csv('dbs/auto_line_part_ref_history_' + str(pse_code) + '_20190731.csv', usecols=['Data Mov', 'Refª da peça', 'Descrição', 'Unit', 'Nº de factura', 'WIP nº', 'Sugestão nº  (Enc)', 'Conta', 'Nº auditoria stock', 'Preço de custo', 'P. V. P'])
        df_al.rename(index=str, columns={'Data Mov': 'Movement_Date', 'Refª da peça': 'Part_Ref', 'Descrição': 'Part_Desc', 'Nº de factura': 'SLR_Document_Number', 'WIP nº': 'WIP_Number', 'Sugestão nº  (Enc)': 'Encomenda', 'Conta': 'SLR_Document_Account', 'Nº auditoria stock': 'Audit_Number'}, inplace=True)
        df_al['Movement_Date'] = pd.to_datetime(df_al['Movement_Date'], format='%d/%m/%Y')
        df_al.sort_values(by='Movement_Date', inplace=True)
        print('dbs/auto_line_part_ref_history_' + str(pse_code) + '_20190731 found.')

        return df_al
    except FileNotFoundError:
        raise FileNotFoundError('AutoLine file for PSE_Code={} was not found!'.format(pse_code))


