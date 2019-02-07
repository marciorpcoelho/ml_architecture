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

sql_facts_durations = {
    'Request_Num',
    'Status_Id',
    'Contact_Assignee_Id',
    'SLA_Assignee_Minutes',
    'SLA_Resolution_Minutes',
    'WaitingTime_Resolution_Minutes',
    'WaitingTime_Assignee_Minutes',
}

cols_with_characteristic = ['Category_Id', 'Category_Id', 'Category_Id', 'Category_Id', 'Category_Id']
cols_to_replace = ['SLA_StdTime_Resolution_Hours', 'SLA_StdTime_Resolution_Hours', 'SLA_StdTime_Resolution_Hours', 'SLA_StdTime_Resolution_Hours', 'SLA_StdTime_Resolution_Hours']
category_id = ['pcat:1246020', 'pcat:1246021', 'pcat:1246019', 'pcat:1246354', 'pcat:1246353']
values_to_replace_by = [24.0, 16.0, 8.0, 8.0, 16.0]

regex_dict = {
    'url': r'http(.*)',
}

words_to_remove_from_description = ['felisbel', 'felisbela', 'susana', 'paul', 'manuel', 'fernando', 'rui', 'brito', 'lui', 'maia', 'lc', 'joao', 'miguel', 'marlene', 'maria'
                                    'boa', 'tard', 'jlopez', 'buen', 'de', 'a', 'o', 'que', 'no', 'e', 'em', 'do', 'com', 'da', 'os',
                                    'ao', 'dos', 'um', 'se', 'das', 'uma', 'obrig', 'a', 'p.f', 'na', 'as', 'por', 'dia', 'é', 'agradeco', 'cumprimento', 'para',
                                    'en', 'el', 'la', 'y', 'lo', 'un', 'una', 'k', 'v', 'gracia', 'me', 'obrigado', 'cp', 'estar', 'entr', 'le', 'seja', 'foi', 'dr',
                                    'hay', 'es', 'ma', 'saludo', 'del', 'al', 'tengo', 'tenho', 'ja', 'cmpt', 'n', 's', 'r', 'c', 'obg', 'favor', 'ou', 'poi', 'agradecia',
                                    'est', 'obrigada', 'bueno', 'bom', 'con', 'ter', 'bon', 'boa', 'esta', 'pelo', 'tenemos', 'como', 'sao', 'fazer', 'ver', 'estamo']
