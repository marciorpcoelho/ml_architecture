import pandas as pd
import re
import logging
# from level_2_optionals_baviera_options import sql_info, column_performance_sql_renaming, regex_dict
import level_2_optionals_baviera_options
# from level_1_e_deployment import sql_inject, sql_log_inject
import level_1_e_deployment
import level_1_b_data_processing
pd.set_option('display.expand_frame_repr', False)

times_global = []
names_global = []
warnings_global = []


def performance_info_append(timings, name):

    times_global.append(timings)
    names_global.append(name)


def performance_warnings_append(warning):

    warnings_global.append(warning)


def performance_info(vehicle_count, running_times_upload_flag):

    df_performance, df_warnings = pd.DataFrame(), pd.DataFrame()
    if not len(warnings_global):
        df_warnings['Warnings'] = [0]
        df_warnings['Warning_Flag'] = [0]
    else:
        df_warnings['Warnings'] = warnings_global
        df_warnings['Warning_Flag'] = [1] * len(warnings_global)

    for step in names_global:
        timings = times_global[names_global.index(step)]
        if type(timings) == list:
            df_performance[step] = timings
        else:
            df_performance[step] = [timings] * vehicle_count

    df_performance = level_1_b_data_processing.column_rename(df_performance, list(level_2_optionals_baviera_options.column_performance_sql_renaming.keys()), list(level_2_optionals_baviera_options.column_performance_sql_renaming.values()))

    if running_times_upload_flag:
        level_1_e_deployment.sql_inject(df_performance, level_2_optionals_baviera_options.sql_info['database'], level_2_optionals_baviera_options.sql_info['performance_running_time'], list(df_performance), time_to_last_update=0, check_date=1)
    level_1_e_deployment.sql_inject(df_warnings, level_2_optionals_baviera_options.sql_info['database'], level_2_optionals_baviera_options.sql_info['warning_log'], list(df_warnings), time_to_last_update=0, check_date=1)


def error_upload(log_file, error_flag=0):
    df_error = pd.DataFrame(columns={'Error_Full', 'Error_Only', 'File_Loc', 'Line', 'Error_Flag'})

    if error_flag:
        error_full, error_only = parse_line(log_file)
        rx = re.compile(level_2_optionals_baviera_options.regex_dict['between_quotes'])
        error_files = rx.findall(error_full[0])

        rx = re.compile(level_2_optionals_baviera_options.regex_dict['lines_number'])
        error_line_number = rx.findall(error_full[0])

        error_line_number = [x.replace(',', '').replace(' ', '') for x in error_line_number]
        error_full_series = [error_full[0]] * len(error_files)
        error_only_series = [error_only[0]] * len(error_files)

        df_error['Error_Full'] = error_full_series
        df_error['Error_Only'] = error_only_series
        df_error['File_Loc'] = error_files
        df_error['Line'] = error_line_number
        df_error['Error_Flag'] = error_flag
    elif not error_flag:
        df_error.loc[0, :] = ['', '', '', '', 0]

    level_1_e_deployment.sql_inject(df_error, level_2_optionals_baviera_options.sql_info['database'], level_2_optionals_baviera_options.sql_info['error_log'], list(df_error), time_to_last_update=0, check_date=1)


def parse_line(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        rx = re.compile(level_2_optionals_baviera_options.regex_dict['error_only'])
        error_only = rx.findall(content)

        rx = re.compile(level_2_optionals_baviera_options.regex_dict['error_full'])
        error_full = rx.findall(content.replace('\n', ' '))

        return error_full, error_only


def log_record(message, database, view, flag=0):
    # flag code: message: 0, warning: 1, error: 2

    if flag == 0:
        logging.info(message)
    elif flag == 1:
        logging.warning(message)
    elif flag == 2:
        logging.exception('#')

    level_1_e_deployment.sql_log_inject(message, flag, database, view)
