import os
import time
import pyodbc
import logging
import pandas as pd
from datetime import datetime
import level_0_performance_report


def save_csv(dfs, names):
    # Checks for file existence and deletes it if exists, then saves it

    # for i, df in enumerate(dfs):
    #     name = names[i] + '.csv'
    #     if os.path.isfile(name):
    #         os.remove(name)
    #     df.to_csv(name)

    for df, name in zip(dfs, names):
        name += '.csv'
        if os.path.isfile(name):
            os.remove(name)
        df.to_csv(name)

    return


# def sql_log_inject(line, project_id, flag, performance_info_dict):
#
#     try:
#         cnxn = pyodbc.connect('DSN=' + performance_info_dict['DSN'] + ';UID=' + performance_info_dict['UID'] + ';PWD=' + performance_info_dict['PWD'] + ';DATABASE=' + performance_info_dict['DB'])
#         cursor = cnxn.cursor()
#         time_tag_date = time.strftime("%Y-%m-%d")
#         time_tag_hour = time.strftime("%H:%M:%S")
#
#         line = apostrophe_escape(line)
#         cursor.execute('INSERT INTO [' + str(performance_info_dict['DB']) + '].dbo.[' + str(performance_info_dict['log_view']) + '] VALUES (\'' + str(line) + '\', ' + str(flag) + ', \'' + str(time_tag_hour) + '\', \'' + str(time_tag_date) + '\', ' + str(project_id) + ')')
#
#         cnxn.commit()
#         cursor.close()
#         cnxn.close()
#     except (pyodbc.ProgrammingError, pyodbc.OperationalError):
#         logging.warning('Unable to access SQL Server.')
#         return


def log_inject(line, project_id, flag, performance_info_dict):

    time_tag_date = time.strftime("%Y-%m-%d")
    time_tag_hour = time.strftime("%H:%M:%S")
    line = apostrophe_escape(line)

    values = [str(line), str(flag), time_tag_hour, time_tag_date, str(project_id)]

    sql_inject_single_line(performance_info_dict['DSN'], performance_info_dict['UID'], performance_info_dict['PWD'], performance_info_dict['DB'], performance_info_dict['log_view'], values)


def sql_inject_single_line(dsn, uid, pwd, database, view, values):
    values_string = '\'%s\'' % '\', \''.join(values)

    try:
        cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'.format(dsn, uid, pwd, database), searchescape='\\')
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


def sql_inject(df, dsn, database, view, options_file, columns, truncate=0, check_date=0):
    time_to_last_update = options_file.update_frequency_days

    start = time.time()

    if truncate:
        sql_truncate(dsn, options_file, database, view)

    cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'.format(dsn, options_file.UID, options_file.PWD, database), searchescape='\\')
    cursor = cnxn.cursor()

    if check_date:
        columns += ['Date']

    columns_string, values_string = sql_string_preparation(columns)

    try:
        if check_date:
            time_result = sql_date_comparison(df, dsn, options_file, database, view, 'Date', time_to_last_update)
            if time_result:
                # level_0_performance_report.log_record('Uploading to SQL Server to DB ' + database + ' and view ' + view + '...', options_file.project_id)
                level_0_performance_report.log_record('Uploading to SQL Server to DB {} and view {}...'.format(database, view), options_file.project_id)

                for index, row in df.iterrows():
                    # continue
                    cursor.execute("INSERT INTO " + view + "(" + columns_string + ') ' + values_string, [row[value] for value in columns])
            elif not time_result:
                level_0_performance_report.log_record('Newer data already exists.', options_file.project_id)
        if not check_date:
            # level_0_performance_report.log_record('Uploading to SQL Server to DB ' + database + ' and view ' + view + '...', options_file.project_id)
            level_0_performance_report.log_record('Uploading to SQL Server to DB {} and view {}...'.format(database, view), options_file.project_id)
            for index, row in df.iterrows():
                # continue
                cursor.execute("INSERT INTO " + view + "(" + columns_string + ') ' + values_string, [row[value] for value in columns])

        # print('Elapsed time: %.2f' % (time.time() - start), 'seconds.')
        print('Elapsed time: {:.2f} seconds.'.format(time.time() - start))
    except pyodbc.ProgrammingError:
        save_csv([df], ['output/' + view + '_backup'])
        level_0_performance_report.log_record('Error in uploading to database. Saving locally...', options_file.project_id, flag=1)

    cnxn.commit()
    cursor.close()
    cnxn.close()

    return


def sql_string_preparation(values_list):
    columns_string = '[%s]' % "], [".join(values_list)

    values_string = ['?'] * len(values_list)
    values_string = 'values (%s)' % ', '.join(values_string)

    return columns_string, values_string


def sql_truncate(dsn, options_file, database, view):
    level_0_performance_report.log_record('Truncating view {} from DB {}.'.format(view, database), options_file.project_id)
    cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'.format(dsn, options_file.UID, options_file.PWD, database), searchescape='\\')
    query = "TRUNCATE TABLE " + view
    cursor = cnxn.cursor()
    cursor.execute(query)

    cnxn.commit()
    cursor.close()
    cnxn.close()


def sql_date_comparison(df, dsn, options_file, database, view, date_column, time_to_last_update):
    time_tag = time.strftime("%d/%m/%y")
    current_date = datetime.strptime(time_tag, '%d/%m/%y')

    df['Date'] = [time_tag] * df.shape[0]
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    last_date = sql_date_checkup(dsn, options_file, database, view, date_column)

    if (current_date - last_date).days >= time_to_last_update:
        return 1
    else:
        return 0


def sql_date_checkup(dsn, options_file, database, view, date_column):

    try:
        cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'.format(dsn, options_file.UID, options_file.PWD, database), searchescape='\\')
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
    cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'.format(dsn, options_file.UID, options_file.PWD, database), searchescape='\\')

    query = 'with second_date as (SELECT MAX([' + str(date_column) + ']) as max_date ' \
            'FROM [' + str(database) + '].[dbo].[' + str(view) + '] ' \
            'WHERE [' + str(date_column) + '] < CONVERT(date, GETDATE()) and Project_Id = \'' + str(options_file.project_id) + '\') ' \
            'SELECT Error_log.* ' \
            'FROM [' + str(database) + '].[dbo].[' + str(view) + '] as Error_log ' \
            'cross join second_date ' \
            'WHERE second_date.max_date = Error_log.[' + str(date_column) + '] and [Dataset] = \'Test\''

    df = pd.read_sql(query, cnxn, index_col='Algorithms')

    cnxn.close()
    return df


def sql_age_comparison(dsn, options_file, database, view, update_frequency):
    time_tag = time.strftime("%d/%m/%y")
    current_date = datetime.strptime(time_tag, '%d/%m/%y')
    last_date = sql_date_checkup(dsn, options_file, database, view, 'Date')

    if (current_date - last_date).days >= update_frequency:
        return 1
    else:
        return 0


# Uploads parameter's mappings to SQL
def sql_mapping_upload(dsn, options_file, dictionaries):
    # dictionaries = [options_file.jantes_dict, options_file.sales_place_dict, options_file.sales_place_dict_v2, options_file.model_dict, options_file.versao_dict, options_file.tipo_int_dict, options_file.color_ext_dict, options_file.color_int_dict, options_file.motor_dict_v2]
    parameters_name = ['Rims_Size', 'Sales_Place', 'Sales_Place_v2', 'Model', 'Version', 'Interior_Type', 'Color_Ext', 'Color_Int', 'Motor_Desc']
    cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE=BI_MLG'.format(dsn, options_file.UID, options_file.PWD), searchescape='\\')
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

        columns_string, values_string = sql_string_preparation(list(df_map))

        sql_truncate(dsn, options_file, 'BI_MLG', view)
        print('Uploading to SQL Server to DB ' + 'BI_MLG' + ' and view ' + view + '...')
        for index, row in df_map.iterrows():
            cursor.execute("INSERT INTO " + view + "(" + columns_string + ') ' + values_string, [row[value] for value in list(df_map)])

    cnxn.commit()
    cursor.close()
    cnxn.close()


def key_and_value_generator(dictionary, all_values, all_keys):

    for key in dictionary.keys():
        values = dictionary[key]
        all_values.append(values)
        all_keys.append([key] * len(values))

    return all_values, all_keys


def sql_get_last_vehicle_count(dsn, options_file, database, view, date_column='Date'):
    cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE=BI_MLG'.format(dsn, options_file.UID, options_file.PWD, database), searchescape='\\')
    crsr = cnxn.cursor()

    query = 'SELECT TOP (1) *' \
            'FROM [' + str(database) + '].[dbo].[' + str(view) + '] ' \
            'WITH (NOLOCK) ORDER BY [' + str(date_column) + '] DESC'

    crsr.execute(query)
    result = crsr.fetchone()[0]

    cnxn.close()
    return result
