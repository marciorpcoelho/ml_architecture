import pandas as pd
import time
import level_2_optionals_baviera_options
from level_1_e_deployment import sql_inject
# from level_1_b_data_processing import column_rename
import level_1_b_data_processing
pd.set_option('display.expand_frame_repr', False)

times_global = []
names_global = []


def performance_info_append(time, name):

    times_global.append(time)
    names_global.append(name)


def performance_info(vehicle_count):

    time_tag = time.strftime("%d/%m/%y")
    df_performance = pd.DataFrame()

    for step in names_global:
        timings = times_global[names_global.index(step)]
        if type(timings) == list:
            df_performance[step] = timings
        else:
            df_performance[step] = [timings] * vehicle_count

    df_performance['Date'] = [time_tag] * vehicle_count
    df_performance['Date'] = pd.to_datetime(df_performance['Date'])
    # df_performance.to_csv('output/' + 'performance_v2.csv')

    df_performance = level_1_b_data_processing.column_rename(df_performance, list(level_2_optionals_baviera_options.column_performance_sql_renaming.keys()), list(level_2_optionals_baviera_options.column_performance_sql_renaming.values()))
    df_performance.to_csv('output/' + 'performance_v2_16_11.csv')

    sql_inject(df_performance, 'BI_MLG', 'LOG_Performance_Running_Time', list(df_performance))
    # ToDo: need to add a check in the date field


