import pandas as pd
import re
import smtplib
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
                    }

project_pbi_performance_link = {2244: 'https://bit.ly/2X8twFU',
                                2162: 'https://bit.ly/2U1dznN',
                                }


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


def performance_info(project_id, options_file, model_choice_message, unit_count, running_times_upload_flag):

    df_performance, df_warnings = pd.DataFrame(), pd.DataFrame()
    if not len(warnings_global):
        df_warnings['Warnings'] = [0]
        df_warnings['Warning_Flag'] = [0]
        df_warnings['Project_Id'] = [project_id]
        warning_flag = 0
    else:
        df_warnings['Warnings'] = warnings_global
        df_warnings['Warning_Flag'] = [1] * len(warnings_global)
        df_warnings['Project_Id'] = [project_id] * len(warnings_global)
        warning_flag = 1

    if project_id == 2162:
        for (step, timings) in zip(names_global, times_global):
            if type(timings) == list:
                df_performance[step] = timings
            else:
                df_performance[step] = [timings] * unit_count

        df_performance['Project_Id'] = project_id

        if running_times_upload_flag:
            performance_report_sql_inject(df_performance, performance_sql_info['DSN'], performance_sql_info['DB'], performance_sql_info['performance_running_time'], options_file, list(df_performance))

    if project_id == 2244:
        for (step, timings) in zip(names_global, times_global):
            df_performance[step] = [timings]

        df_performance['Project_Id'] = project_id
        performance_report_sql_inject(df_performance, performance_sql_info['DSN'], performance_sql_info['DB'], performance_sql_info['performance_running_time'], options_file, list(df_performance))

    performance_report_sql_inject(df_warnings, performance_sql_info['DSN'], performance_sql_info['DB'], performance_sql_info['warning_log'], options_file, list(df_warnings))

    error_flag, error_only = error_upload(options_file, project_id, options_file.log_files['full_log'])
    email_notification(project_id, warning_flag=warning_flag, warning_desc=warnings_global, error_desc=error_only, error_flag=error_flag, model_choice_message=model_choice_message)


def email_notification(project_id, warning_flag, warning_desc, error_desc, error_flag=0, model_choice_message=0):
    run_conclusion, warning_conclusion, conclusion_message = None, None, None
    fromaddr = EMAIL
    df_mail_users = performance_report_sql_retrieve_df(performance_sql_queries['mail_users_query'], performance_sql_info['DSN'], performance_sql_info['UID'], performance_sql_info['PWD'], performance_sql_info['DB'], performance_id)
    users = df_mail_users['UserName'].unique()
    toaddrs = df_mail_users['UserEmail'].unique()
    flags_to_send = df_mail_users[project_sql_dict[project_id]].values

    mail_subject = str(project_dict[project_id]) + ' - Relatório'
    link = project_pbi_performance_link[project_id]

    if error_flag:
        run_conclusion = 'não terminou devido ao erro: {}.'.format(error_desc)
        conclusion_message = ''
    elif not error_flag:
        run_conclusion = 'terminou com sucesso.'
        conclusion_message = '\r\nA sua conclusão foi: ' + str(model_choice_message)

    if warning_flag:
        warning_conclusion = '\n Foram encontrados os seguintes alertas: \r\n - {}'.format('\r\n - '.join(x for x in warning_desc))
    elif not warning_flag:
        warning_conclusion = 'Não foram encontrados quaisquer alertas.\n'

    for (flag, user, toaddr) in zip(flags_to_send, users, toaddrs):
        if flag:
            mail_body = 'Bom dia {}, ' \
                         '\n \nO projeto {} {} \n{} \n{}' \
                         '\n \nPara mais informações, por favor consulta o seguinte relatório: {} \n' \
                         '\nCumprimentos, \nRelatório Automático, v1.4' \
                         .format(user, project_dict[project_id], run_conclusion, warning_conclusion, conclusion_message, link)
            message = 'Subject: {}\n\n{}'.format(mail_subject, mail_body).encode('latin-1')

            try:
                server = smtplib.SMTP('smtp.gmail.com')
                server.ehlo()
                server.starttls()
                server.login(EMAIL, EMAIL_PASS)
                server.sendmail(fromaddr, toaddr, message)
                server.quit()
            except TimeoutError:
                return


def performance_report_sql_retrieve_df(query, dsn, uid, pwd, db, project_id, **kwargs):

    cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'.format(dsn, uid, pwd, db), searchescape='\\')

    try:
        df = pd.read_sql(query, cnxn, **kwargs)

        return df
    except (pyodbc.ProgrammingError, pyodbc.OperationalError) as error:
        log_record('Erro ao obter os dados do DW - {}'.format(error), project_id, flag=1)

    cnxn.close()


def error_upload(options_file, project_id, log_file, error_flag=0):
    df_error = pd.DataFrame(columns={'Error_Full', 'Error_Only', 'File_Loc', 'Error_Flag', 'Project_Id'})
    error_only = None

    if error_flag:
        error_full, error_only = parse_line(log_file)
        rx = re.compile(regex_dict['between_quotes'])
        error_files = rx.findall(error_full[0])

        error_full_series = [error_full[0]] * len(error_files)
        error_only_series = [error_only[0]] * len(error_files)
        project_id_series = [project_id] * len(error_files)

        df_error['Error_Full'] = error_full_series
        df_error['Error_Only'] = error_only_series
        df_error['File_Loc'] = error_files
        df_error['Error_Flag'] = error_flag
        df_error['Project_Id'] = project_id_series

        if not len(warnings_global):
            warning_flag = 0
            warning_desc = 0
        else:
            warning_flag = 1
            warning_desc = warnings_global

        email_notification(project_id, warning_flag=warning_flag, warning_desc=warning_desc, error_desc=error_only, error_flag=1, model_choice_message=0)
    elif not error_flag:
        df_error.loc[0, ['Error_Full', 'Error_Only', 'File_Loc', 'Error_Flag', 'Project_Id']] = [None, None, None, 0, project_id]

    performance_report_sql_inject(df_error, performance_sql_info['DSN'], performance_sql_info['DB'], performance_sql_info['error_log'], options_file, list(df_error))

    if error_flag:
        return error_flag, error_only[0]
    else:
        return error_flag, 0


def performance_report_sql_inject(df, dsn, database, view, options_file, columns):
    start = time.time()

    cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'.format(dsn, options_file.UID, options_file.PWD, database), searchescape='\\')
    cursor = cnxn.cursor()

    # columns += ['Date']
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


def parse_line(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        rx = re.compile(regex_dict['error_only'])
        error_only = rx.findall(content)

        rx = re.compile(regex_dict['error_full'])
        error_full = rx.findall(content.replace('\n', ' '))

        return error_full, error_only


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
