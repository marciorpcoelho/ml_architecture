import pandas as pd
import numpy as np
import sys
import os
import time
import pyodbc
import modules.level_1_b_data_processing as level_1_b_data_processing
# from modules.level_1_b_data_processing import df_join_function, value_substitution, column_rename
import modules.level_0_performance_report as level_0_performance_report
# from modules.level_0_performance_report import log_record
import modules.level_1_e_deployment as level_1_e_deployment
# from modules.level_1_e_deployment import sql_get_last_vehicle_count, sql_inject_single_line

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')) + '\\'


def read_csv(*args, **kwargs):

    df = pd.read_csv(*args, **kwargs)

    return df


def vehicle_count_checkup(df, options_file, sql_check=0):
    current_vehicle_count = df['Nº Stock'].nunique()
    last_vehicle_count = level_1_e_deployment.sql_get_last_vehicle_count(options_file.DSN_MLG, options_file, options_file.sql_info['database'], options_file.sql_info['vhe_number_history'])

    if not sql_check:
        if current_vehicle_count < 100:
            raise ValueError('Apenas ' + str(current_vehicle_count) + ' veículos foram encontrados. Por favor verificar na base de dados.')
    elif sql_check:
        if current_vehicle_count < last_vehicle_count:
            raise ValueError('Atual contagem de veículos ({}) é inferior à ultima contagem ({}). Por favor verificar na base de dados.'.format(current_vehicle_count, last_vehicle_count))
        elif current_vehicle_count == last_vehicle_count:
            level_0_performance_report.log_record('Atual contagem de veículos ({}) sem incrementos desde a última vez que o modelo foi treinado ({}). Por favor confirmar se o comportamento é o esperado.'.format(current_vehicle_count, last_vehicle_count), options_file.project_id, flag=1)
        else:
            time_tag_date = time.strftime("%Y-%m-%d")
            values = [str(current_vehicle_count), time_tag_date]
            level_1_e_deployment.sql_inject_single_line(options_file.DSN_MLG, options_file.UID, options_file.PWD, options_file.sql_info['database'], options_file.sql_info['vhe_number_history'], values)
            level_0_performance_report.log_record('A atualizar contagem de viaturas: {}.'.format(current_vehicle_count), options_file.project_id, flag=0)
    return


def missing_customer_info_treatment(df_sales):

    df_vehicles_wo_clients = pd.read_excel(base_path + 'dbs/viaturas_sem_cliente_final rb.xlsx', usecols=['Chassis_Number', 'Registration_Number', 'conc / nº cliente navision'], dtype={'conc / nº cliente navision': str}).dropna()
    df_vehicles_wo_clients.rename(index=str, columns={'conc / nº cliente navision': 'SLR_Account_CHS_Key'}, inplace=True)
    df_vehicles_wo_clients['SLR_Account_CHS_Key'] = '702_' + df_vehicles_wo_clients['SLR_Account_CHS_Key']
    df_sales = level_1_b_data_processing.df_join_function(df_sales, df_vehicles_wo_clients[['Chassis_Number', 'SLR_Account_CHS_Key']].set_index('Chassis_Number'), on='Chassis_Number', rsuffix='_new', how='left')
    df_sales = level_1_b_data_processing.value_substitution(df_sales, non_null_column='SLR_Account_CHS_Key_new', null_column='SLR_Account_CHS_Key')  # Replaces description by summary when the first is null and second exists
    df_sales.drop(['SLR_Account_CHS_Key_new'], axis=1, inplace=True)

    return df_sales


# This function works really well for one single function on scheduler (provided i had sys.stdout.flush() to the end of each file). But if more than one functions are running at the same time (different threads) the stdout
# saved is all mixed and saved on the file of the last function; - trying now with logging module
def log_files(project_name, output_dir=base_path + 'logs/'):
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
            level_1_b_data_processing.column_rename(df, list(options_file.sql_to_code_renaming.keys()), list(options_file.sql_to_code_renaming.values()))

        cnxn.close()

        print('Elapsed time: {:.2f} seconds.'.format(time.time() - start))
        return df

    except (pyodbc.ProgrammingError, pyodbc.OperationalError) as error:
        level_0_performance_report.log_record('Erro ao obter os dados do DW - {}'.format(error), options_file.project_id, flag=2)
        return


# When you have a full query to use and no need to prepare anything. Will merge with sql_retrieve_df in the future;
def sql_retrieve_df_specified_query(dsn, db, options_file, query):
    start = time.time()

    try:
        cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'.format(dsn, options_file.UID, options_file.PWD, db), searchescape='\\')

        df = pd.read_sql(query, cnxn)
        cnxn.close()

        print('Elapsed time: {:.2f} seconds.'.format(time.time() - start))
        return df

    except (pyodbc.ProgrammingError, pyodbc.OperationalError) as error:
        level_0_performance_report.log_record('Erro ao obter os dados do DW - {}'.format(error), options_file.project_id, flag=2)
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

    print('Retrieving data for PSE_Code = {}'.format(pse_code))

    sales_info = [base_path + 'dbs/df_sales', options_info.sales_query]
    purchases_info = [base_path + 'dbs/df_purchases', options_info.purchases_query]
    stock_info = [base_path + 'dbs/df_stock', options_info.stock_query]
    reg_info = [base_path + 'dbs/df_reg', options_info.reg_query]
    reg_al_info = [base_path + 'dbs/df_reg_al_client', options_info.reg_autoline_clients]
    product_group_dw = [base_path + 'dbs/df_product_group_dw', options_info.dim_product_group_dw]

    dfs = []

    for dimension in [sales_info, purchases_info, stock_info, reg_info, reg_al_info, product_group_dw]:

        if update:
            file_name = dimension[0] + '_' + str(pse_code) + '_' + str(current_date)
        else:
            file_name = dimension[0] + '_' + str(pse_code) + '_20191031'  # Last time I ran this script and saved these files

        try:
            df = read_csv(file_name + '.csv', index_col=0)
            # print('{} file found.'.format(file_name))
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
    df_product_group_dw = dfs[5]

    df_purchases['Movement_Date'] = pd.to_datetime(df_purchases['Movement_Date'], format='%Y%m%d')
    df_purchases['WIP_Date_Created'] = pd.to_datetime(df_purchases['WIP_Date_Created'], format='%Y%m%d')

    df_sales['SLR_Document_Date'] = pd.to_datetime(df_sales['SLR_Document_Date'], format='%Y%m%d')
    df_sales['WIP_Date_Created'] = pd.to_datetime(df_sales['WIP_Date_Created'], format='%Y%m%d')
    df_sales['Movement_Date'] = pd.to_datetime(df_sales['Movement_Date'], format='%Y%m%d')

    df_stock['Record_Date'] = pd.to_datetime(df_stock['Record_Date'], format='%Y%m%d')
    df_stock.rename(index=str, columns={'Quantity': 'Stock_Qty'}, inplace=True)

    return df_sales, df_purchases, df_stock, df_reg, df_reg_al_clients, df_product_group_dw


def autoline_data_retrieval(pse_code, current_date):
    start = time.time()

    try:
        df_al = read_csv(base_path + 'dbs/auto_line_part_ref_history_{}_{}.csv'.format(pse_code, current_date), usecols=['Data Mov', 'Refª da peça', 'Descrição', 'Unit', 'Nº de factura', 'WIP nº', 'Sugestão nº  (Enc)', 'Conta', 'Nº auditoria stock', 'Preço de custo', 'P. V. P', 'GPr'])
    except FileNotFoundError:
        try:
            df_1 = pd.read_excel(base_path + 'dbs/auto_line_fb1_{}_{}.xlsx'.format(pse_code, current_date))
            df_2 = pd.read_excel(base_path + 'dbs/auto_line_fb2_{}_{}.xlsx'.format(pse_code, current_date))
            df_al = pd.concat([df_1, df_2])

            df_al.to_csv(base_path + 'dbs/auto_line_part_ref_history_{}_{}.csv'.format(pse_code, current_date))
            print('dbs/auto_line_part_ref_history_{}_{} created and saved.'.format(pse_code, current_date))

        except FileNotFoundError:
            print('AutoLine file for PSE_Code={} and date={} was not found!'.format(pse_code, current_date))

    df_al.rename(index=str, columns={'Data Mov': 'Movement_Date', 'Refª da peça': 'Part_Ref', 'Descrição': 'Part_Desc', 'Nº de factura': 'SLR_Document_Number', 'WIP nº': 'WIP_Number', 'Sugestão nº  (Enc)': 'Encomenda', 'Conta': 'SLR_Document_Account', 'Nº auditoria stock': 'Audit_Number', 'GPr': 'TA'}, inplace=True)
    df_al['Movement_Date'] = pd.to_datetime(df_al['Movement_Date'], format='%d/%m/%Y')
    df_al.sort_values(by='Movement_Date', inplace=True)

    print('Elapsed Time: {:.2f} seconds.'.format(time.time() - start))

    return df_al

