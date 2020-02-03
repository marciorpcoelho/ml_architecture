import pandas as pd
import logging
import os
import time
import pyodbc
from multiprocessing import cpu_count
from py_dotenv import read_dotenv
pd.set_option('display.expand_frame_repr', False)
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')) + '\\'
dotenv_path = base_path + 'info.env'
read_dotenv(dotenv_path)

times_global = []
names_global = []
warnings_global = []
pool_workers_count = cpu_count()

# Universal Information
EMAIL = os.getenv('EMAIL')
EMAIL_PASS = os.getenv('EMAIL_PASS')
performance_id = 0000

performance_sql_info = {'DSN': os.getenv('DSN_MLG'),
                        'UID': os.getenv('UID'),
                        'PWD': os.getenv('PWD'),
                        'DB': 'BI_MLG',
                        'log_view': 'LOG_Information',
                        'error_log': 'LOG_Performance_Errors',
                        'warning_log': 'LOG_Performance_Warnings',
                        'model_choices': 'LOG_Performance_Model_Choices',
                        'mail_users': 'LOG_MailUsers',
                        'performance_running_time': 'LOG_Performance_Running_Time',
                        'performance_algorithm_results': 'LOG_Performance_Algorithms_Results',
                        'sp_send_dbmail': 'usp_LOG_Mail_Execution',
                        'sp_send_dbmail_input_parameters_name': ['mail_subject', 'mail_body_part1', 'mail_body_part2', 'project_id'],
                        }

performance_sql_queries = {
    'mail_users_query': 'SELECT * FROM {}.dbo.{}'.format(performance_sql_info['DB'], performance_sql_info['mail_users'])
}

regex_dict = {
    'error_full': r'((?:\#[^\#\r\n]*){1})$',  # Catches the error message from the eof up to the unique value #
    'error_only': r'[\n](.*){1}$',
    'between_quotes': r'\s{1}\"(.*?.py|.*?.pyx)\"',
    'lines_number': r'\s[0-9]{1,}\,',
    'null_replacement': r'\'NULL\'',
    'timestamp_removal': r'Timestamp'
}

project_dict = {2244: 'PA@Service Desk',
                2162: 'Otimização Encomenda Baviera Viaturas',
                2259: 'Otimização Encomenda Baviera APV',
                2406: 'Otimização Encomenda Hyundai/Honda',
                0000: 'Performance Analysis'
                }

project_sql_dict = {2244: 'Project_SD',
                    2162: 'Project_VHE_BMW',
                    2259: 'Project_VHE_APV',
                    2406: 'Project_VHE_DTR',
                    }

project_pbi_performance_link = 'https://bit.ly/2SJIYJy'


dict_models_name_conversion = {
    'dt': ['Decision Tree'],
    'rf': ['Random Forest'],
    'lr': ['Logistic Regression'],
    'knn': ['KNN'],
    'svm': ['SVM'],
    'ab': ['Adaboost'],
    'xgb': ['XGBoost'],
    'lgb': ['LightGBM'],
    'gc': ['Gradient'],
    'bayes': ['Bayesian'],
    'ann': ['ANN'],
    'voting': ['Voting'],
    'lreg': ['Linear Regression'],
    'lasso_cv': ['LassoCV'],
    'ridge': ['Ridge'],
    'll_cv': ['LassoLarsCV'],
    'elastic_cv': ['ElasticNetCV'],
    'svr': ['Support Vector Regression'],
}


def performance_info_append(timings, name):

    times_global.append(timings)
    names_global.append(name)


def performance_warnings_append(warning):

    warnings_global.append(warning)


def performance_info(project_id, options_file, model_choice_message, unit_count):

    df_performance, df_warnings = pd.DataFrame(), pd.DataFrame()
    current_date = time.strftime("%Y-%m-%d")

    if not len(warnings_global):
        warning_flag = 0
        df_warnings['Warnings'] = ['0']
    else:
        warning_flag = 1
        df_warnings['Warnings'] = warnings_global

    df_warnings['Warning_Flag'] = warning_flag
    df_warnings['Project_Id'] = project_id
    df_warnings['Date'] = current_date

    if project_id == 2162:
        for (step, timings) in zip(names_global, times_global):
            if type(timings) == list:
                df_performance[step] = timings
            else:
                df_performance[step] = [timings] * unit_count

    else:
        for (step, timings) in zip(names_global, times_global):
            df_performance[step] = [timings]

    df_performance['Date'] = current_date
    df_performance['Project_Id'] = project_id
    performance_report_sql_inject(df_performance, performance_sql_info['DSN'], performance_sql_info['DB'], performance_sql_info['performance_running_time'], options_file, list(df_performance))
    performance_report_sql_inject(df_warnings, performance_sql_info['DSN'], performance_sql_info['DB'], performance_sql_info['warning_log'], options_file, list(df_warnings))

    email_notification(project_id, warning_flag=warning_flag, warning_desc=warnings_global, error_desc='', model_choice_message=model_choice_message)


def email_notification(project_id, warning_flag, warning_desc, error_desc, error_flag=0, model_choice_message=0):
    run_conclusion, warning_conclusion, conclusion_message = None, None, None

    mail_subject = '#PRJ-{}: {} - Relatório'.format(project_id, project_dict[project_id])
    link = project_pbi_performance_link

    if error_flag:
        run_conclusion = 'não terminou devido ao erro: {}.'.format(error_desc)
        conclusion_message = ''
    elif not error_flag:
        run_conclusion = 'terminou com sucesso.'
        conclusion_message = '\r\nA sua conclusão foi: {}'.format(model_choice_message)

    if warning_flag:
        warning_conclusion = '\n Foram encontrados os seguintes alertas: \r\n - {}'.format('\r\n - '.join(x for x in warning_desc))
    elif not warning_flag:
        warning_conclusion = 'Não foram encontrados quaisquer alertas.\n'

    mail_body_part1 = '''Bom dia'''
    mail_body_part2 = '''\n \nO projeto {} {} \n{} \n{}
                      \n \nPara mais informações, por favor consulta o seguinte relatório: {} 
                      \nCumprimentos, \nDatabase Mail, v2.0'''.format(project_dict[project_id], run_conclusion, warning_conclusion, conclusion_message, link)

    try:
        sp_query = sp_query_creation(performance_sql_info['sp_send_dbmail'], performance_sql_info['sp_send_dbmail_input_parameters_name'], [mail_subject, mail_body_part1, mail_body_part2, project_id])
        generic_query_execution(sp_query)
    except (pyodbc.ProgrammingError, pyodbc.OperationalError) as error:
        log_record('Erro ao executar SP {} - {}'.format(performance_sql_info['sp_send_dbmail'], error), project_id, flag=2)


def generic_query_execution(query):

    cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'.format(performance_sql_info['DSN'], performance_sql_info['UID'], performance_sql_info['PWD'], performance_sql_info['DB']), searchescape='\\')
    cursor = cnxn.cursor()

    cursor.execute(query)

    cnxn.commit()
    cursor.close()
    cnxn.close()

    return


def sp_query_creation(sp_name, sp_input_parameters_name_list, sp_output_parameters_value_list):

    query_exec = 'EXEC {} '.format(sp_name)
    query_parameters = ', '.join(['@{} = \'{}\''.format(x, str(y).replace('\'', '"')) for x, y in zip(sp_input_parameters_name_list, sp_output_parameters_value_list)])
    sp_query = query_exec + query_parameters

    return sp_query


def error_upload(options_file, project_id, error_full, error_only, error_flag=0):
    df_error = pd.DataFrame(columns={'Error_Full', 'Error_Only', 'Error_Flag', 'Project_Id', 'Date'})
    # error_only = None
    current_date = time.strftime("%Y-%m-%d")

    if error_flag:
        df_error['Error_Full'] = [error_full]
        df_error['Error_Only'] = [error_only]
        df_error['Error_Flag'] = error_flag
        df_error['Project_Id'] = project_id
        df_error['Date'] = current_date

        if not len(warnings_global):
            warning_flag = 0
            warning_desc = 0
        else:
            warning_flag = 1
            warning_desc = warnings_global

        email_notification(project_id, warning_flag=warning_flag, warning_desc=warning_desc, error_desc=error_only, error_flag=1, model_choice_message=0)
    elif not error_flag:
        df_error.loc[0, ['Error_Full', 'Error_Only', 'Error_Flag', 'Project_Id', 'Date']] = [None, None, 0, project_id, current_date]

    performance_report_sql_inject(df_error, performance_sql_info['DSN'], performance_sql_info['DB'], performance_sql_info['error_log'], options_file, list(df_error))

    if error_flag:
        return error_flag, error_only[0]
    else:
        return error_flag, 0


def performance_report_sql_inject(df, dsn, database, view, options_file, columns):
    start = time.time()

    cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'.format(dsn, options_file.UID, options_file.PWD, database), searchescape='\\')
    cursor = cnxn.cursor()

    columns_string = '[%s]' % "], [".join(columns)
    values_string = ['?'] * len(columns)
    values_string = 'values (%s)' % ', '.join(values_string)

    try:
        log_record('A fazer upload para SQL, Database {} e view {}...'.format(database, view), options_file.project_id)

        for index, row in df.iterrows():
            cursor.execute("INSERT INTO " + view + "(" + columns_string + ') ' + values_string, [row[value] for value in columns])

        print('Duração: {:.2f} segundos.'.format(time.time() - start))
    except (pyodbc.ProgrammingError, pyodbc.DataError) as error:
        df.to_csv(base_path + 'output/{}_backup.csv'.format(view))
        log_record('Erro ao fazer upload - {} - A gravar localmente...'.format(error), options_file.project_id, flag=1)

    cnxn.commit()
    cursor.close()
    cnxn.close()

    return


def log_record(message, project_id, flag=0):
    # Flag Code: message: 0, warning: 1, error: 2

    if flag == 0:
        logging.info(message)
    elif flag == 1:
        logging.warning(message)
        performance_warnings_append(message)
    elif flag == 2:
        logging.exception('#')

    performance_report_sql_inject_single_line(message, flag, performance_sql_info, project_id)


def performance_report_sql_inject_single_line(line, flag, performance_sql_info_in, project_id):

    time_tag_date = time.strftime("%Y-%m-%d")
    time_tag_hour = time.strftime("%H:%M:%S")
    line = line.replace('\'', '"')

    values = [str(line), str(flag), time_tag_hour, time_tag_date, str(project_id)]

    values_string = '\'%s\'' % '\', \''.join(values)

    try:
        cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'.format(performance_sql_info_in['DSN'], performance_sql_info_in['UID'], performance_sql_info_in['PWD'], performance_sql_info_in['DB']), searchescape='\\')
        cursor = cnxn.cursor()

        cursor.execute('INSERT INTO [{}].dbo.[{}] VALUES ({})'.format(performance_sql_info_in['DB'], performance_sql_info_in['log_view'], values_string))

        cnxn.commit()
        cursor.close()
        cnxn.close()
    except (pyodbc.ProgrammingError, pyodbc.OperationalError):
        logging.warning('Unable to access SQL Server.')
        return
