import pandas as pd
import sys
import time
import pyodbc
import level_1_b_data_processing as level_b


def read_csv(*args, **kwargs):

    df = pd.read_csv(*args, **kwargs)

    return df


def vehicle_count_checkup(df):
    vehicle_count = df['Nº Stock'].nunique()
    if vehicle_count < 5000:
        raise ValueError('Apenas ' + str(vehicle_count) + ' foram encontrados. Por favor verificar os dados na base de dados.')


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
        cnxn = pyodbc.connect('DSN=' + dsn + ';UID=' + options_file.UID + ';PWD=' + options_file.PWD + ';DATABASE=' + db)

        if type(query_filters) == int:
            query = 'SELECT ' + columns + ' FROM ' + view
        elif type(query_filters) == dict:
            for key in query_filters:
                if type(query_filters[key]) == list:
                    testing_string = '\'%s\'' % "\', \'".join(query_filters[key])
                    query_filters_string_list.append(key + ' in (' + testing_string + ')')
                else:
                    query_filters_string_list.append(key + ' = \'%s\'' % str(query_filters[key]))
            query = 'SELECT ' + columns + ' FROM ' + view + ' WHERE ' + ' and '.join(query_filters_string_list)
        df = pd.read_sql(query, cnxn, **kwargs)
        if column_renaming:
            level_b.column_rename(df, list(options_file.sql_to_code_renaming.keys()), list(options_file.sql_to_code_renaming.values()))

        cnxn.close()

        print('Elapsed time: %.2f' % (time.time() - start), 'seconds.')
        return df

    except (pyodbc.ProgrammingError, pyodbc.OperationalError):
        return  # ToDo need to figure a better way of handling these errors
