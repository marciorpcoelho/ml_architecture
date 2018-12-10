import os
import time
import pyodbc
import pandas as pd
from datetime import datetime
from level_2_optionals_baviera_options import update_frequency_days, DSN, UID, PWD


def save_csv(dfs, names):
    # Checks for file existence and deletes it if exists, then saves it

    for i, df in enumerate(dfs):
        name = names[i] + '.csv'
        if os.path.isfile(name):
            os.remove(name)
        df.to_csv(name)


def sql_inject(df, database, view, columns, time_to_last_update=update_frequency_days, truncate=0, check_date=0):

    start = time.time()
    columns_string = str()
    values_string = 'values ('

    if truncate:
        sql_truncate(database, view)

    cnxn = pyodbc.connect('DSN=' + DSN + ';UID=' + UID + ';PWD=' + PWD + ';DATABASE=' + database)
    cursor = cnxn.cursor()

    if check_date:
        columns += ['Date']

    for item in columns:
        columns_string += '[' + item + '], '
        values_string += '?, '

    columns_string = columns_string[:-2] + '' + columns_string[-1:]
    values_string = values_string[:-2] + ')' + values_string[-1:]

    try:
        if check_date:
            time_result = sql_date_comparison(df, database, view, 'Date', time_to_last_update)
            if time_result:
                print('Uploading to SQL Server to DB ' + database + ' and view ' + view + '...')
                for index, row in df.iterrows():
                    print('inject_placeholder')
                    # cursor.execute("INSERT INTO " + view + "(" + columns_string + ') ' + values_string, [row[value] for value in columns])
                upload = 1
            elif not time_result:
                print('Newer data already exists.')
                upload = 0
        if not check_date:
            print('Uploading to SQL Server to DB ' + database + ' and view ' + view + '...')
            for index, row in df.iterrows():
                print('inject_placeholder')
                # cursor.execute("INSERT INTO " + view + "(" + columns_string + ') ' + values_string, [row[value] for value in columns])
            upload = 1

        print('Elapsed time: %.2f' % (time.time() - start), 'seconds.')
    except pyodbc.ProgrammingError:
        upload = 0
        save_csv([df], ['output/' + view + '_backup'])

    cnxn.commit()
    cursor.close()
    cnxn.close()

    return upload


def sql_truncate(database, view):
    print('Truncating view ' + view + ' from DB ' + database)
    cnxn = pyodbc.connect('DSN=' + DSN + ';UID=' + UID + ';PWD=' + PWD + ';DATABASE=' + database)
    query = "TRUNCATE TABLE " + view
    cursor = cnxn.cursor()
    cursor.execute(query)

    cnxn.commit()
    cursor.close()
    cnxn.close()


def sql_date_comparison(df, database, view, date_column, time_to_last_update):
    time_tag = time.strftime("%d/%m/%y")
    current_date = datetime.strptime(time_tag, '%d/%m/%y')

    df['Date'] = [time_tag] * df.shape[0]
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    last_date = sql_date_checkup(database, view, date_column)

    if (current_date - last_date).days >= time_to_last_update:
        return 1
    else:
        return 0


def sql_date_checkup(database, view, date_column):

    cnxn = pyodbc.connect('DSN=' + DSN + ';UID=' + UID + ';PWD=' + PWD + ';DATABASE=' + database)
    cursor = cnxn.cursor()

    cursor.execute('SELECT MAX(' + '[' + date_column + ']' + ') FROM ' + database + '.dbo.' + view)

    result = cursor.fetchone()

    try:
        result_date = datetime.strptime(result[0], '%Y-%m-%d')
    except TypeError:
        result_date = datetime.strptime('1960-01-01', '%Y-%m-%d')  # Just in case the database is empty

    return result_date


def sql_last_update_date(database, view):

    cnxn = pyodbc.connect('DSN=' + DSN + ';UID=' + UID + ';PWD=' + PWD + ';DATABASE=' + database)
    cursor = cnxn.cursor()

    cursor.execute('SELECT OBJECT_NAME(OBJECT_ID) AS TableName, last_user_update FROM sys.dm_db_index_usage_stats WHERE OBJECT_ID=OBJECT_ID(\'' + view + '\')')

    result = cursor.fetchone()

    return result[1]


def sql_age_comparison(database, view, update_frequency):
    time_tag = time.strftime("%d/%m/%y")
    current_date = datetime.strptime(time_tag, '%d/%m/%y')
    last_date = sql_date_checkup(database, view, 'Date')

    if (current_date - last_date).days >= update_frequency:
        return 1
    else:
        return 0


def sql_retrieve_df(database, view, columns):
    start = time.time()
    print('Retrieving data from SQL Server, DB ' + database + ' and view ' + view + '...')

    sel_cols = str(columns)[1:-1].replace('\'', '')

    cnxn = pyodbc.connect('DSN=' + DSN + ';UID=' + UID + ';PWD=' + PWD + ';DATABASE=' + database)

    query = 'SELECT ' + sel_cols + ' FROM ' + view

    df = pd.read_sql(query, cnxn)

    print('Elapsed time: %.2f' % (time.time() - start), 'seconds.')
    return df
