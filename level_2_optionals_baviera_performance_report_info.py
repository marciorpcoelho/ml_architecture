import pandas as pd
import re
import smtplib
import logging
import level_2_optionals_baviera_options
import level_1_e_deployment
import level_1_b_data_processing
import level_1_a_data_acquisition
pd.set_option('display.expand_frame_repr', False)

times_global = []
names_global = []
warnings_global = []


def performance_info_append(timings, name):

    times_global.append(timings)
    names_global.append(name)


def performance_warnings_append(warning):

    warnings_global.append(warning)


def performance_info(model_choice_message, vehicle_count, running_times_upload_flag):

    df_performance, df_warnings = pd.DataFrame(), pd.DataFrame()
    if not len(warnings_global):
        df_warnings['Warnings'] = [0]
        df_warnings['Warning_Flag'] = [0]
        warning_flag = 0
    else:
        df_warnings['Warnings'] = warnings_global
        df_warnings['Warning_Flag'] = [1] * len(warnings_global)
        warning_flag = 1

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

    error_flag, error_only = error_upload(level_2_optionals_baviera_options.log_files['full_log'])
    email_notification(warning_flag=warning_flag, warning_desc=warnings_global, error_desc=error_only, error_flag=error_flag, model_choice_message=model_choice_message)


def email_notification(warning_flag, warning_desc, error_desc, error_flag=0, model_choice_message=0):
    run_conclusion, warning_conclusion, conclusion_message = None, None, None
    fromaddr = 'mrpc@gruposalvadorcaetano.pt'
    df_mail_users = level_1_a_data_acquisition.sql_retrieve_df(level_2_optionals_baviera_options.sql_info['database'], level_2_optionals_baviera_options.sql_info['mail_users'])
    # users = df_mail_users['UserName'].unique()
    users = ['Marcio']
    # toaddrs = df_mail_users['UserEmail'].unique()
    toaddrs = ['marcio.coelho@rigorcg.pt']

    mail_subject = 'Otimização Encomenda - Relatório'

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

    for (user, toaddr) in zip(users, toaddrs):
        mail_body = 'Bom dia ' + str(user) + \
                    ', \nO projeto Otimização Encomenda (BMW) ' + str(run_conclusion) + \
                    ' \n' + str(warning_conclusion) + \
                    ' \n' + str(conclusion_message) + \
                    '\n Para mais informações, por favor consulta o seguinte relatório: ' + \
                    str('https://app.powerbi.com/groups/3d13efce-f4f6-4bb1-bf1f-f8a1076f1c0b/reports/a5dcce26-9b8d-4d83-9781-092685ca4385?ctid=cc1c517a-b933-41da-8549-2d5c307156fb') + \
                    ' \n\n Cumprimentos, \n Relatório Automático Otimização Encomenda (BMW), v1.2'
        message = 'Subject: {}\n\n{}'.format(mail_subject, mail_body).encode('latin-1')

        try:
            server = smtplib.SMTP('smtp-mail.outlook.com')
            server.ehlo()
            server.starttls()
            server.login(level_2_optionals_baviera_options.EMAIL, level_2_optionals_baviera_options.EMAIL_PASS)
            server.sendmail(fromaddr, toaddr, message)
            server.quit()
        except TimeoutError:
            return


def error_upload(log_file, error_flag=0):
    df_error = pd.DataFrame(columns={'Error_Full', 'Error_Only', 'File_Loc', 'Line', 'Error_Flag'})
    error_only = None

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

        email_notification(warning_flag=0, warning_desc=0, error_desc=error_only, error_flag=1, model_choice_message=0)
    elif not error_flag:
        df_error.loc[0, :] = ['', '', '', '', 0]

    level_1_e_deployment.sql_inject(df_error, level_2_optionals_baviera_options.sql_info['database'], level_2_optionals_baviera_options.sql_info['error_log'], list(df_error), time_to_last_update=0, check_date=1)

    if error_flag:
        return error_flag, error_only[0]
    else:
        return error_flag, 0


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
