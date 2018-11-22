import pandas as pd
from level_2_optionals_baviera_options import sql_info, column_performance_sql_renaming
from level_1_e_deployment import sql_inject
import level_1_b_data_processing
pd.set_option('display.expand_frame_repr', False)

times_global = []
names_global = []


def performance_info_append(timings, name):

    times_global.append(timings)
    names_global.append(name)


def performance_info(vehicle_count):

    df_performance = pd.DataFrame()

    for step in names_global:
        timings = times_global[names_global.index(step)]
        if type(timings) == list:
            df_performance[step] = timings
        else:
            df_performance[step] = [timings] * vehicle_count

    df_performance = level_1_b_data_processing.column_rename(df_performance, list(column_performance_sql_renaming.keys()), list(column_performance_sql_renaming.values()))

    sql_inject(df_performance, sql_info['database'], sql_info['performance_running_time'], list(df_performance), time_to_last_update=0, check_date=1)







