import os
import time
import pyodbc


def save_csv(dfs, names):
    # Checks for file existence and deletes it if exists, then saves it

    for i, df in enumerate(dfs):
        name = names[i] + '.csv'
        if os.path.isfile(name):
            os.remove(name)
        df.to_csv(name)


def sql_inject(df, database, view, columns):
    start = time.time()
    columns_string = str()
    values_string = 'values ('

    print('Uploading to SQL Server to DB ' + database + ' and view ' + view + '...')

    cnxn = pyodbc.connect('DSN=MLG;UID=interplataformas;PWD=inf2008;DATABASE=' + database)
    cursor = cnxn.cursor()

    # for item in list(columns.values()):
    #     columns_string += '[' + item + '], '
    #     values_string += '?, '

    for item in columns:
        columns_string += '[' + item + '], '
        values_string += '?, '

    columns_string = columns_string[:-2] + '' + columns_string[-1:]
    values_string = values_string[:-2] + ')' + values_string[-1:]

    # for index, row in df.iterrows():
    #     cursor.execute("INSERT INTO " + view + "(" + columns_string + ') ' + values_string, [row[value] for value in list(columns.values())])

    for index, row in df.iterrows():
        cursor.execute("INSERT INTO " + view + "(" + columns_string + ') ' + values_string, [row[value] for value in columns])

    cnxn.commit()
    cursor.close()
    cnxn.close()

    print('Elapsed time: %.2f' % (time.time() - start), 'seconds...')


def sql_truncate(database, view):
    print('Truncating view ' + view + ' from DB ' + database)
    cnxn = pyodbc.connect('DSN=MLG;UID=interplataformas;PWD=inf2008;DATABASE=' + database)
    query = "TRUNCATE TABLE " + view
    cursor = cnxn.cursor()
    cursor.execute(query)

    cnxn.commit()
    cursor.close()
    cnxn.close()
