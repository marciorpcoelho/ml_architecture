import os
import re
import time
import pyodbc
import logging
import pandas as pd
from datetime import datetime

import modules.level_0_performance_report as level_0_performance_report

if 'nt' in os.name:
    OS_PLATFORM = 'WINDOWS'
    # base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')) + '\\'
elif 'posix' in os.name:
    OS_PLATFORM = 'LINUX'
    # base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))


def odbc_connection_creation(dsn, uid, pwd, db):
    # Creates an ODBC connection to the specified SQL Server
    odbc_cnxn = None

    if OS_PLATFORM == 'WINDOWS':
        odbc_cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'.format(dsn, uid, pwd, db), searchescape='\\')
    elif OS_PLATFORM == 'LINUX':
        odbc_cnxn = pyodbc.connect('Driver=ODBC Driver 17 for SQL Server;Server=tcp:' + str(dsn) + ';UID=' + str(uid) + ';PWD=' + str(pwd) + ';DATABASE=' + str(db), searchescape='\\')

    return odbc_cnxn


def save_csv(dfs, names, **kwargs):
    # Checks for file existence and deletes it if exists, then saves it

    for df, name in zip(dfs, names):
        name += '.csv'
        if os.path.isfile(name):
            os.remove(name)
        df.to_csv(name, **kwargs)

    return


def time_tags(format_date="%Y-%m-%d", format_time="%H:%M:%S"):

    time_tag_date = time.strftime(format_date)
    time_tag_hour = time.strftime(format_time)

    return time_tag_date, time_tag_hour


def sql_inject_single_line(dsn, uid, pwd, database, view, values, check_date=0):

    if check_date:
        values.append(time_tags()[0])
        values_string = '\'%s\'' % '\', \''.join(values)
    else:
        values_string = '\'%s\'' % '\', \''.join(values)

    try:
        cnxn = odbc_connection_creation(dsn, uid, pwd, database)
        cursor = cnxn.cursor()

        cursor.execute('INSERT INTO [{}].dbo.[{}] VALUES ({})'.format(database, view, values_string))

        cnxn.commit()
        cursor.close()
        cnxn.close()
    except (pyodbc.ProgrammingError, pyodbc.OperationalError):
        logging.warning('Unable to access SQL Server.')
        return


def apostrophe_escape(line):

    return line.replace('\'', '"')


def sql_delete(dsn, database, view, options_file, query_filters):
    level_0_performance_report.log_record('A apagar registos da tabela {} na BD {}...'.format(view, database), options_file.project_id)
    query, query_filters_string_list = None, []

    cnxn = odbc_connection_creation(dsn, options_file.UID, options_file.PWD, database)

    if not query_filters:
        query = 'DELETE FROM ' + view
    elif type(query_filters) == dict:
        for key in query_filters:
            if type(query_filters[key]) == list:
                testing_string = '\'%s\'' % "\', \'".join([str(x) for x in query_filters[key]])
                query_filters_string_list.append(key + ' in (' + testing_string + ')')
            else:
                query_filters_string_list.append(key + ' = \'%s\'' % str(query_filters[key]))
        query = 'DELETE FROM ' + view + ' WHERE ' + ' and '.join(query_filters_string_list)

    cursor = cnxn.cursor()
    affected_rows_count = cursor.execute(query).rowcount

    level_0_performance_report.log_record('{} registo(s) apagado(s) da tabela {} na BD {}.'.format(affected_rows_count, view, database), options_file.project_id)

    cnxn.commit()
    cursor.close()
    cnxn.close()

    return


def sql_inject(df, dsn, database, view, options_file, columns, truncate=0, check_date=0):  # v1
    time_to_last_update = options_file.update_frequency_days

    start = time.time()

    if truncate:
        sql_truncate(dsn, options_file, database, view)

    cnxn = odbc_connection_creation(dsn, options_file.UID, options_file.PWD, database)
    cursor = cnxn.cursor()

    if check_date:
        columns += ['Date']

    columns_string, values_string = sql_string_preparation_v1(columns)

    try:
        if check_date:
            df.loc[:, 'Date'] = time_tags()[0]
            time_result = sql_date_comparison(dsn, options_file, database, view, 'Date', time_to_last_update)

            if time_result:
                level_0_performance_report.log_record('A fazer upload para SQL, Database {} e view {}...'.format(database, view), options_file.project_id)

                for index, row in df.iterrows():
                    # continue
                    cursor.execute("INSERT INTO " + view + "(" + columns_string + ') ' + values_string, [row[value] for value in columns])

                level_0_performance_report.log_record('Inseridas {} linhas na DB {} e na tabela {}.'.format(df.shape[0], database, view), options_file.project_id)
            elif not time_result:
                level_0_performance_report.log_record('Já existem dados mais recentes.', options_file.project_id)
        if not check_date:
            level_0_performance_report.log_record('A fazer upload para SQL, Database {} e view {}...'.format(database, view), options_file.project_id)
            for index, row in df.iterrows():
                # continue
                cursor.execute("INSERT INTO " + view + "(" + columns_string + ') ' + values_string, [row[value] for value in columns])
            level_0_performance_report.log_record('Inseridas {} linhas na DB {} e na tabela {}.'.format(df.shape[0], database, view), options_file.project_id)

        print('Duração: {:.2f} segundos.'.format(time.time() - start))
    except (pyodbc.ProgrammingError, pyodbc.DataError) as error:
        save_csv([df], [base_path + '/output/' + view + '_backup'])
        level_0_performance_report.log_record('Erro ao fazer upload - {} - A gravar localmente...'.format(error), options_file.project_id, flag=1)

    cnxn.commit()
    cursor.close()
    cnxn.close()

    return


def sql_inject_v2(df, dsn, database, view, options_file, columns, truncate=0, check_date=0):  # v2
    time_to_last_update = options_file.update_frequency_days

    query_convert_datetimes = ''
    start = time.time()

    if truncate:
        sql_truncate(dsn, options_file, database, view)

    cnxn = odbc_connection_creation(dsn, options_file.UID, options_file.PWD, database)
    cursor = cnxn.cursor()

    if check_date:
        columns += ['Date']

    columns_string = sql_string_preparation_v2(columns)

    insert_ = '''
    INSERT INTO {}
    ({})
    VALUES'''.format(view, columns_string)

    try:
        if check_date:
            df['Date'] = time_tags()[0]
            time_result = sql_date_comparison(dsn, options_file, database, view, 'Date', time_to_last_update)
            if time_result:
                values_string = [str(tuple(x)) for x in df.values]
                level_0_performance_report.log_record('A fazer upload para SQL, Database {} e view {}...'.format(database, view), options_file.project_id)

                for batch in chunker(values_string, 1000):
                    rows = ','.join(batch)
                    rows = re.sub(level_0_performance_report.regex_dict['null_replacement'], 'NULL', rows)
                    rows = re.sub(level_0_performance_report.regex_dict['timestamp_removal'], '', rows)
                    insert_rows = insert_ + rows
                    cursor.execute(insert_rows)

            elif not time_result:
                level_0_performance_report.log_record('Já existem dados mais recentes.', options_file.project_id)
        if not check_date:
            values_string = [str(tuple(x)) for x in df.values]
            level_0_performance_report.log_record('A fazer upload para SQL, Database {} e view {}...'.format(database, view), options_file.project_id)

            for batch in chunker(values_string, 1000):
                rows = ','.join(batch)
                rows = re.sub(level_0_performance_report.regex_dict['null_replacement'], 'NULL', rows)
                rows = re.sub(level_0_performance_report.regex_dict['timestamp_removal'], '', rows)
                insert_rows = insert_ + rows + query_convert_datetimes
                cursor.execute(insert_rows)

        print('Duração: {:.2f} segundos.'.format(time.time() - start))
    except (pyodbc.ProgrammingError, pyodbc.DataError) as error:
        save_csv([df], [base_path + '/output/' + view + '_backup'])
        level_0_performance_report.log_record('Erro ao fazer upload - {} - A gravar localmente...'.format(error), options_file.project_id, flag=1)

    cnxn.commit()
    cursor.close()
    cnxn.close()

    return


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def sql_join(df, dsn, database, view, options_file):
    start = time.time()
    level_0_performance_report.log_record('Joining to SQL Server to DB {} and view {}...'.format(database, view), options_file.project_id)

    cnxn = odbc_connection_creation(dsn, options_file.UID, options_file.PWD, database)
    cursor = cnxn.cursor()

    query = '''update a
        set Label=b.Label,StemmedDescription=b.StemmedDescription,Language=b.Language
        from [scsqlsrv3\prd].BI_RCG.dbo.BI_SDK_Fact_Requests_Month_Detail  as a
        inner join [scrcgaisqld1\dev01].BI_MLG.dbo.SDK_Fact_BI_PA_ServiceDesk as b on a.request_num=b.request_num '''.replace('\'', '\'\'')
    cursor.execute(query)

    print('Duração: {:.2f} segundos.'.format(time.time() - start))

    cnxn.commit()
    cursor.close()
    cnxn.close()

    return


def sql_query(query, dsn, database, options_file):
    start = time.time()

    cnxn = odbc_connection_creation(dsn, options_file.UID, options_file.PWD, database)
    cursor = cnxn.cursor()
    cursor.execute(query)

    print('Duração: {:.2f} segundos.'.format(time.time() - start))

    cnxn.commit()
    cursor.close()
    cnxn.close()

    return


def sql_string_preparation_v2(values_list):
    columns_string = '[%s]' % "], [".join(values_list)

    return columns_string


def sql_string_preparation_v1(values_list):
    columns_string = '[%s]' % "], [".join(values_list)

    values_string = ['?'] * len(values_list)
    values_string = 'values (%s)' % ', '.join(values_string)

    return columns_string, values_string


def sql_truncate(dsn, options_file, database, view, query=None):
    level_0_performance_report.log_record('A truncar view {} da DB {}.'.format(view, database), options_file.project_id)

    cnxn = odbc_connection_creation(dsn, options_file.UID, options_file.PWD, database)

    if query is None:
        query = "TRUNCATE TABLE " + view

    cursor = cnxn.cursor()
    cursor.execute(query)

    cnxn.commit()
    cursor.close()
    cnxn.close()


def sql_date_comparison(dsn, options_file, database, view, date_column, time_to_last_update):
    time_tag_date, _ = time_tags()
    current_date = datetime.strptime(time_tag_date, '%Y-%m-%d')

    last_date = sql_date_checkup(dsn, options_file, database, view, date_column)

    if (current_date - last_date).days >= time_to_last_update:
        return 1
    else:
        return 0


def sql_date_checkup(dsn, options_file, database, view, date_column):

    try:
        cnxn = odbc_connection_creation(dsn, options_file.UID, options_file.PWD, database)
        cursor = cnxn.cursor()

        cursor.execute('SELECT MAX(' + '[' + date_column + ']' + ') FROM ' + database + '.dbo.' + view + 'WITH (NOLOCK)')

        result = cursor.fetchone()
        result_date = datetime.strptime(result[0], '%Y-%m-%d')

        cursor.close()
        cnxn.close()
    except (pyodbc.ProgrammingError, pyodbc.OperationalError, TypeError):
        result_date = datetime.strptime('1960-01-01', '%Y-%m-%d')  # Just in case the database is empty

    return result_date


def sql_second_highest_date_checkup(dsn, options_file, database, view, date_column='Date'):
    cnxn = odbc_connection_creation(dsn, options_file.UID, options_file.PWD, database)

    query = 'with second_date as (SELECT MAX([' + str(date_column) + ']) as max_date ' \
            'FROM [' + str(database) + '].[dbo].[' + str(view) + '] ' \
            'WHERE [' + str(date_column) + '] < CONVERT(date, GETDATE()) and Project_Id = \'' + str(options_file.project_id) + '\') ' \
            'SELECT Error_log.* ' \
            'FROM [' + str(database) + '].[dbo].[' + str(view) + '] as Error_log ' \
            'cross join second_date ' \
            'WHERE second_date.max_date = Error_log.[' + str(date_column) + '] and [Dataset] = \'Test\' and Project_Id = \'' + str(options_file.project_id) + '\''

    df = pd.read_sql(query, cnxn, index_col='Algorithms')

    cnxn.close()
    return df


# Uploads parameter's mappings to SQL
def sql_mapping_upload(dsn, options_file, dictionaries):
    parameters_name = ['Rims_Size', 'Sales_Place', 'Sales_Place_v2', 'Model', 'Version', 'Interior_Type', 'Color_Ext', 'Color_Int', 'Motor_Desc']
    cnxn = odbc_connection_creation(dsn, options_file.UID, options_file.PWD, 'BI_MLG')
    cursor = cnxn.cursor()

    for (parameter, dictionary) in zip(parameters_name, dictionaries):
        df_map = pd.DataFrame(columns=['Original_Value', 'Mapped_Value'])
        view = 'VHE_MapBI_' + str(parameter)

        all_values, all_keys = [], []

        all_values, all_keys = key_and_value_generator(dictionary, all_values, all_keys)  # Will use this method, as the time gains are marginal (if any) when compared to an item comprehension approach and it is more readable;

        all_values = [item for sublist in all_values for item in sublist]
        all_keys = [item for sublist in all_keys for item in sublist]

        df_map['Original_Value'] = all_values
        df_map['Mapped_Value'] = all_keys

        columns_string = sql_string_preparation_v2(list(df_map))
        values_string = [str(tuple(x)) for x in df_map.values]

        sql_truncate(dsn, options_file, 'BI_MLG', view)
        print('A fazer upload para SQL, Database ' + 'BI_MLG' + ' e view ' + view + '...')

        insert_ = '''INSERT INTO {}
        ({}) VALUES '''.format(view, columns_string)

        for batch in chunker(values_string, 1000):
            rows = ','.join(batch)
            insert_rows = insert_ + str(rows)
            cursor.execute(insert_rows)

    cnxn.commit()
    cursor.close()
    cnxn.close()


def key_and_value_generator(dictionary, all_values, all_keys):

    for key in dictionary.keys():
        values = dictionary[key]
        all_values.append(values)
        all_keys.append([key] * len(values))

    return all_values, all_keys


def sql_get_last_project_unit_count(dsn, options_file, database, view, date_column='Date'):
    cnxn = odbc_connection_creation(dsn, options_file.UID, options_file.PWD, database)
    crsr = cnxn.cursor()

    query = 'SELECT TOP (1) * ' \
            'FROM [' + str(database) + '].[dbo].[' + str(view) + '] WITH (NOLOCK) ' \
            'WHERE Project_Id = ' + str(options_file.project_id) + \
            ' ORDER BY [' + str(date_column) + '] DESC'

    crsr.execute(query)
    result = crsr.fetchone()[0]

    cnxn.close()
    return result
