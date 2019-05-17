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


def sql_mapping_retrieval(dsn, db, mapping_tables, mapped_column_name, options_file, multiple_columns=0):
    dictionary_list = []

    for mapping_table in mapping_tables:
        parameter_dict = {}
        df = sql_retrieve_df(dsn, db, mapping_table, options_file)
        if not multiple_columns:
            for key in df[mapped_column_name].unique():
                parameter_dict[key] = list(df[df[mapped_column_name] == key]['Original_Value'].values)
        if multiple_columns:
            for key in df[mapped_column_name].unique():
                listing = df[df[mapped_column_name] == key][[x for x in list(df)[:-1] if x not in mapped_column_name]].values  # Changed to consider the cases where the last column is the priority rank
                parameter_dict[key] = np.unique([item for sublist in listing for item in sublist if item is not None])
        dictionary_list.append(parameter_dict)

    return dictionary_list
