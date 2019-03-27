import pandas as pd
import re
import smtplib
import logging
import os
import level_1_e_deployment
import level_1_b_data_processing
import level_1_a_data_acquisition
from multiprocessing import cpu_count
from py_dotenv import read_dotenv
pd.set_option('display.expand_frame_repr', False)
dotenv_path = 'info.env'
read_dotenv(dotenv_path)

times_global = []
names_global = []
warnings_global = []
pool_workers_count = cpu_count()

# Universal Information
EMAIL = os.getenv('EMAIL')
EMAIL_PASS = os.getenv('EMAIL_PASS')

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

regex_dict = {
    'error_full': r'((?:\#[^\#\r\n]*){1})$',  # Catches the error message from the eof up to the unique value #
    'error_only': r'[\n](.*){1}$',
    'between_quotes': r'\s{1}\"(.*?.py|.*?.pyx)\"',
    'lines_number': r'\s[0-9]{1,}\,',
}

project_dict = {2244: 'PA@Service Desk',
                2162: 'Otimização Encomenda Baviera (BMW)',
                }

project_sql_dict = {2244: 'Project_SD',
                    2162: 'Project_VHE_BMW',
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
        for step in names_global:
            timings = times_global[names_global.index(step)]
            if type(timings) == list:
                df_performance[step] = timings
            else:
                df_performance[step] = [timings] * unit_count

        df_performance = level_1_b_data_processing.column_rename(df_performance, list(options_file.column_performance_sql_renaming.keys()), list(options_file.column_performance_sql_renaming.values()))

        if running_times_upload_flag:
            level_1_e_deployment.sql_inject(df_performance, performance_sql_info['DSN'], performance_sql_info['DB'], performance_sql_info['performance_running_time'], options_file, list(df_performance), check_date=1)
        # level_1_e_deployment.sql_inject(df_warnings, performance_sql_info['DSN'], performance_sql_info['DB'], performance_sql_info['warning_log'], options_file, list(df_warnings), check_date=1)

    level_1_e_deployment.sql_inject(df_warnings, performance_sql_info['DSN'], performance_sql_info['DB'], performance_sql_info['warning_log'], options_file, list(df_warnings), check_date=1)

    # if project_id == 2244:
    #     level_1_e_deployment.sql_inject(df_warnings, performance_sql_info['DSN'], performance_sql_info['DB'], performance_sql_info['warning_log'], options_file, list(df_warnings), check_date=1)

    error_flag, error_only = error_upload(options_file, project_id, options_file.log_files['full_log'])
    email_notification(options_file, project_id, warning_flag=warning_flag, warning_desc=warnings_global, error_desc=error_only, error_flag=error_flag, model_choice_message=model_choice_message)


def email_notification(options_file, project_id, warning_flag, warning_desc, error_desc, error_flag=0, model_choice_message=0):
    run_conclusion, warning_conclusion, conclusion_message = None, None, None
    fromaddr = 'mrpc@gruposalvadorcaetano.pt'
    df_mail_users = level_1_a_data_acquisition.sql_retrieve_df(performance_sql_info['DSN'], performance_sql_info['DB'], performance_sql_info['mail_users'], options_file)
    users = df_mail_users['UserName'].unique()
    toaddrs = df_mail_users['UserEmail'].unique()
    flags_to_send = df_mail_users[project_sql_dict[project_id]].values

    mail_subject = str(project_dict[project_id]) + ' - Relatório'
    link = 'https://bit.ly/2U1dznN'

    if error_flag:
        run_conclusion = 'não terminou devido ao erro: {}.'.format(error_desc)
        conclusion_message = ''
    elif not error_flag:
        run_conclusion = 'terminou com sucesso.'
        conclusion_message = 'A sua conclusão foi: ' + str(model_choice_message)

    if warning_flag:
        warning_conclusion = 'Foram encontrados os seguintes alertas: {}'.format([x for x in warning_desc])
    elif not warning_flag:
        warning_conclusion = 'Não foram encontrados quaisquer alertas.'

    for (flag, user, toaddr) in zip(flags_to_send, users, toaddrs):
        if flag:
            mail_body = 'Bom dia {}, ' \
                         '\nO projeto {} {} \n{} \n{}' \
                         '\nPara mais informações, por favor consulta o seguinte relatório: {} \n' \
                         '\nCumprimentos, \nRelatório Automático, v1.3.1' \
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


def error_upload(options_file, project_id, log_file, error_flag=0):
    df_error = pd.DataFrame(columns={'Error_Full', 'Error_Only', 'File_Loc', 'Line', 'Error_Flag', 'Project_Id'})
    error_only = None

    if error_flag:
        error_full, error_only = parse_line(log_file)
        rx = re.compile(regex_dict['between_quotes'])
        error_files = rx.findall(error_full[0])

        rx = re.compile(regex_dict['lines_number'])
        error_line_number = rx.findall(error_full[0])

        error_line_number = [x.replace(',', '').replace(' ', '') for x in error_line_number]
        error_full_series = [error_full[0]] * len(error_files)
        error_only_series = [error_only[0]] * len(error_files)
        project_id_series = [project_id] * len(error_files)

        df_error['Error_Full'] = error_full_series
        df_error['Error_Only'] = error_only_series
        df_error['File_Loc'] = error_files
        df_error['Line'] = error_line_number
        df_error['Error_Flag'] = error_flag
        df_error['Project_Id'] = project_id_series

        if not len(warnings_global):
            warning_flag = 0
            warning_desc = 0
        else:
            warning_flag = 1
            warning_desc = warnings_global

        email_notification(options_file, project_id, warning_flag=warning_flag, warning_desc=warning_desc, error_desc=error_only, error_flag=1, model_choice_message=0)
    elif not error_flag:
        df_error.loc[0, ['Error_Full', 'Error_Only', 'File_Loc', 'Line', 'Error_Flag', 'Project_Id']] = [None, None, None, None, 0, project_id]

    level_1_e_deployment.sql_inject(df_error, performance_sql_info['DSN'], performance_sql_info['DB'], performance_sql_info['error_log'], options_file, list(df_error), check_date=1)

    if error_flag:
        return error_flag, error_only[0]
    else:
        return error_flag, 0


def parse_line(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        rx = re.compile(regex_dict['error_only'])
        error_only = rx.findall(content)

        rx = re.compile(regex_dict['error_full'])
        error_full = rx.findall(content.replace('\n', ' '))

        return error_full, error_only


def log_record(message, project_id, flag=0):
    # flag code: message: 0, warning: 1, error: 2

    if flag == 0:
        logging.info(message)
    elif flag == 1:
        logging.warning(message)
        performance_warnings_append(message)
    elif flag == 2:
        logging.exception('#')

    level_1_e_deployment.log_inject(message, project_id, flag, performance_sql_info)
