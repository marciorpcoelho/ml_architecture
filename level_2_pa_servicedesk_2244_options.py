import os
from py_dotenv import read_dotenv
from multiprocessing import cpu_count
dotenv_path = 'info.env'
read_dotenv(dotenv_path)

DSN = os.getenv('DSN_SD')
DSN_MLG = os.getenv('DSN_MLG')
UID = os.getenv('UID')
PWD = os.getenv('PWD')
pool_workers_count = cpu_count()

# Options:
update_frequency_days = 0
max_number_of_clusters = 11


sql_info = {
    'database_source': 'BI_RCG',
    'database_final': 'BI_MLG',
    'log_record': 'LOG_PA@ServiceDesk',
    'initial_table_facts': 'BI_SDK_Fact_Requests_Month_Detail',
    'initial_table_facts_durations': 'BI_SDK_Fact_Requests',
    'initial_table_clients': 'BI_SDK_Dim_Contacts',
    'initial_table_pbi_categories': 'BI_SDK_Dim_Requests_Categories',
    'final_table': 'SDK_Fact_BI_PA_ServiceDesk',
}

log_files = {
    'full_log': 'logs/ia_servicedesk_2244.txt',
}

date_columns = ['Open_Date', 'Assignee_Date', 'Close_Date', 'Resolve_Date']

project_id = 2244

sql_fact_columns = {
    'WeekDay_Id',
    'Request_Id',
    'Request_Num',
    'Summary',
    'Description',
    'Open_Date',
    'Assignee_Date',
    'Close_Date',
    'Resolve_Date',
    'Contact_Customer_Id',
    'Category_Id',
    'SLA_Id',
    'SLA_StdTime_Assignee_Hours',
    'SLA_StdTime_Resolution_Hours',
    'Request_Type',
    'Priority_Id',
    'SLA_Assignee_Flag',
    'SLA_Resolution_Flag',
    'Request_Age_Code',
    'Cost_Centre',
    'Record_Type',
}

sql_pbi_categories_columns = {
    'Category_Id',
    'Category_Name',
}

sql_facts_durations_columns = {
    'Request_Num',
    'Status_Id',
    'Contact_Assignee_Id',
    'SLA_Assignee_Minutes',
    'SLA_Resolution_Minutes',
    'WaitingTime_Resolution_Minutes',
    'WaitingTime_Assignee_Minutes',
}

sql_dim_contacts_columns = {
        'Contact_Id',
        'Name',
        'Login_Name',
        'Contact_Type',
        'Location_Id',
        'Location_Name',
        'Comments',
        'Site_Id',
        'Site_Name',
        'Company_Group',
        'Company_Group_Name',
}

sla_resolution_hours_replacements = {
    'Category_Id': ['pcat:1246020', 'pcat:1246021', 'pcat:1246019', 'pcat:1246354', 'pcat:1246353'],
    'SLA_StdTime_Resolution_Hours': [24.0, 16.0, 8.0, 8.0, 16.0]
}

assignee_id_replacements = {
    'Request_Num': ['RE-107512', 'RE-114012', 'RE-175076', 'RE-191719', 'RE-74793', 'RE-80676', 'RE-84389', 'RE-157518'],
    'Contact_Assignee_Id': [-107178583, -107178583, 1746469363, 129950480, 1912342313, 1912342313, 1912342313, -172602144]
}

language_replacements = {
    'Contact_Customer_Id': [1316563093, -650110013, 1191100018, -849867232, 80794334, -1511754133, 1566878955, -250410311, 1959237887],
    # Javier Soria, 'Juan Fernandez', 'Juan Gomez', 'Juan Sanchez', 'Cesar Malvido', 'Eduardo Ruiz', 'Ignacio Bravo', 'Marc Illa', 'Toni Silva'
    'Language': ['es'] * 9
}


regex_dict = {
    # 'url_1': r'http://(.*)',
    # 'url_2': r'http://(.*)php',
    # 'url_3': r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',
    'url': r'(http|ftp|https)://([\w_-]+(?:(?:[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',
}


words_to_remove_from_description = ['felisbel', 'felisbela', 'susana', 'paul', 'manuel', 'fernando', 'rui', 'brito', 'lui', 'maia', 'lc', 'joao', 'miguel', 'marlene', 'maria', 'boa',
                                    'tard', 'jlopez', 'buen', 'de', 'a', 'o', 'que', 'e', 'do', 'da', 'os', 'ao', 'dos', 'um', 'se', 'das', 'uma', 'obrig', 'a', 'p.f',
                                    'na', 'as', 'por', 'dia', 'é', 'agradeco', 'cumprimento', 'para', 'el', 'la', 'y', 'lo', 'un', 'una', 'k', 'v', 'gracia', 'me', 'obrigado', 'cp',
                                    'estar', 'entr', 'le', 'seja', 'foi', 'hay', 'es', 'ma', 'saludo', 'del', 'al', 'tengo', 'tenho', 'ja', 'cmpt', 'n', 's', 'r', 'c', 'obg', 'favor',
                                    'ou', 'poi', 'agradecia', 'est', 'obrigada', 'bueno', 'bom', 'con', 'ter', 'bon', 'boa', 'esta', 'pelo', 'tenemos', 'como', 'sao', 'fazer', 'ver', 'estamo',
                                    'azevedo', 'oliveira', '-', '\'', 'tpsilva', 'mb', '12ts23', '12ts26', '10ua27', 'joliveira', 'salvador', 'piedra', 'diego', 'rodriguez', 'estebaranz']

column_performance_sql_renaming = {
    'start_section_a': 'Section_A_Start',
    'start_section_b': 'Section_B_Start',
    'start_section_c': 'Section_C_Start',
    'start_section_e': 'Section_E_Start',
    'end_section_a': 'Section_A_End',
    'end_section_b': 'Section_B_End',
    'end_section_c': 'Section_C_End',
    'end_section_e': 'Section_E_End',
}
