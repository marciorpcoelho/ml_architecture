import pandas as pd
import re
import sys
from level_2_optionals_baviera_options import sql_info, column_performance_sql_renaming, regex_dict
from level_1_e_deployment import sql_inject
# from level_1_b_data_processing import column_rename
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


def performance_info(vehicle_count):

    df_performance, df_warnings = pd.DataFrame(), pd.DataFrame()
    df_warnings['Warnings'] = warnings_global

    for step in names_global:
        timings = times_global[names_global.index(step)]
        if type(timings) == list:
            df_performance[step] = timings
        else:
            df_performance[step] = [timings] * vehicle_count

    df_performance = level_1_b_data_processing.column_rename(df_performance, list(column_performance_sql_renaming.keys()), list(column_performance_sql_renaming.values()))

    sql_inject(df_performance, sql_info['database'], sql_info['performance_running_time'], list(df_performance), time_to_last_update=0, check_date=1)
    sql_inject(df_warnings, sql_info['database'], sql_info['warning_log'], list(df_warnings), time_to_last_update=0, check_date=1)


# def error_upload(log_file, error=0):
#     df_error = pd.DataFrame(columns={'Error_Full', 'Error_Only', 'Warnings', 'File_Loc', 'Line', 'Error_Flag', 'Warning_Flag'})
#
#     if error:
#         error_full, error_only = parse_line(log_file)
#         rx = re.compile(regex_dict['between_quotes'])
#         error_files = rx.findall(error_full[0])
#         rx = re.compile(regex_dict['lines_number'])
#         error_line_number = rx.findall(error_full[0])
#         error_line_number = [x.replace(',', '').replace(' ', '') for x in error_line_number]
#         error_full_series = [error_full[0]] * len(error_files)
#         error_only_series = [error_only[0]] * len(error_files)
#     elif not error:
#         error_full_series, error_only_series, error_files, error_line_number = '', '', '', ''
#
#     warnings, warning_flag = warning_checkup(log_file)
#
#     df_error['Error_Full'] = error_full_series
#     df_error['Error_Only'] = error_only_series
#     df_error['File_Loc'] = error_files
#     df_error['Line'] = error_line_number
#     df_error['Error_Flag'] = [error * len(error_files)]
#     df_error['Warnings'] = [warnings * len(error_files)]
#     df_error['Warning_Flag'] = warning_flag
#
#     sql_inject(df_error, sql_info['database'], sql_info['error_log'], list(df_error), time_to_last_update=0, check_date=1)


def error_upload(log_file, error_flag=0):
    df_error = pd.DataFrame(columns={'Error_Full', 'Error_Only', 'Warnings', 'File_Loc', 'Line', 'Error_Flag', 'Warning_Flag'})

    if error_flag:
        error_full, error_only = parse_line(log_file)
        rx = re.compile(regex_dict['between_quotes'])
        error_files = rx.findall(error_full[0])

        rx = re.compile(regex_dict['lines_number'])
        error_line_number = rx.findall(error_full[0])

        error_line_number = [x.replace(',', '').replace(' ', '') for x in error_line_number]
        error_full_series = [error_full[0]] * len(error_files)
        error_only_series = [error_only[0]] * len(error_files)
    elif not error_flag:
        error_full_series, error_only_series, error_files, error_line_number = None, None, None, None

    df_error['Error_Full'] = error_full_series
    df_error['Error_Only'] = error_only_series
    df_error['File_Loc'] = error_files
    df_error['Line'] = error_line_number
    df_error['Error_Flag'] = error_flag

    sql_inject(df_error, sql_info['database'], sql_info['error_log'], list(df_error), time_to_last_update=0, check_date=1)


def parse_line(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        rx = re.compile(regex_dict['error_only'])
        error_only = rx.findall(content)

        rx = re.compile(regex_dict['error_full'])
        error_full = rx.findall(content.replace('\n', ' '))

        return error_full, error_only
