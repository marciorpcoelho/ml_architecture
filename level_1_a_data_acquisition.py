import pandas as pd
import sys
import time
import pyodbc
import logging
from level_2_optionals_baviera_performance_report_info import log_record
from level_2_optionals_baviera_options import DSN, UID, PWD, sql_to_code_renaming, sql_info
from level_1_b_data_processing import column_rename


def read_csv(column_renaming=0, *args, **kwargs):

    df = pd.read_csv(*args, **kwargs)

    if column_renaming:
        column_rename(df, list(sql_to_code_renaming.keys()), list(sql_to_code_renaming.values()))

    return df


# This function works really well for one single function on scheduler (provided i had sys.stdout.flush() to the end of each file). But if more than one functions are running at the same time (different threads) the stdout
# saved is all mixed and saved on the file of the last function; - trying now with logging module
def log_files(project_name, output_dir='logs/'):
    sys.stdout = open(output_dir + project_name + '.txt', 'a')
    sys.stderr = open(output_dir + project_name + '.txt', 'a')


def sql_retrieve_df(database, view, columns='*'):
    start = time.time()
    # logging.info('Retrieving data from SQL Server, DB ' + database + ' and view ' + view + '...')
    log_record('Retrieving data from SQL Server, DB ' + database + ' and view ' + view + '...', sql_info['database'], sql_info['log_record'])

    if columns != '*':
        columns = str(columns)[1:-1].replace('\'', '')

    cnxn = pyodbc.connect('DSN=' + DSN + ';UID=' + UID + ';PWD=' + PWD + ';DATABASE=' + database)

    query = 'SELECT ' + columns + ' FROM ' + view

    df = pd.read_sql(query, cnxn)

    print('Elapsed time: %.2f' % (time.time() - start), 'seconds.')
    return df
