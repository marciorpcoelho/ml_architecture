import os
from py_dotenv import read_dotenv
from multiprocessing import cpu_count
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
dotenv_path = '/info.env'
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
read_dotenv(base_path + dotenv_path)

if 'nt' in os.name:
    DSN_SRV3_PRD = os.getenv('DSN_SRV3_Prd')
    DSN_MLG_PRD = os.getenv('DSN_MLG_Prd')
elif 'posix' in os.name:
    DSN_SRV3_PRD = os.getenv('DSN_SRV3_Prd_Linux')
    DSN_MLG_PRD = os.getenv('DSN_MLG_Prd_Linux')
UID = os.getenv('UID')
PWD = os.getenv('PWD')
pool_workers_count = cpu_count()

# Options:
update_frequency_days = 0
max_number_of_clusters = 11


sql_info = {
    'database_source': 'BI_RCG',
    'database_final': 'BI_MLG',
    'initial_table_facts': 'BI_SDK_Fact_Requests_Month_Detail',
    'initial_table_facts_durations': 'BI_SDK_Fact_Requests',
    'initial_table_clients': 'BI_SDK_Dim_Contacts',
    'initial_table_pbi_categories': 'BI_SDK_Dim_Requests_Categories',
    # 'final_table': 'SDK_Fact_BI_PA_ServiceDesk',
    'final_table': 'BI_SDK_Fact_DW_Requests_Classification',
    'aux_table': 'BI_SDK_Fact_DW_Requests_Manual_Classification',
    'keywords_table': ['SDK_Setup_Keywords'],  # This is a mapping table, uses the sql_mapping_retrieval function, and therefore should be a list, not a string;
    'keywords_table_str': 'SDK_Setup_Keywords'
}

log_files = {
    'full_log': 'logs/ia_servicedesk_2244.txt',
}

date_columns = ['Open_Date', 'Creation_Date', 'Assignee_Date_Orig', 'Assignee_Date', 'Close_Date', 'Resolve_Date', 'Resolve_Date_Orig']

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

model_training_fact_cols = {
    'Request_Num',
    'Summary',
    'Description',
    'Status_Id',
    'Creation_Date',
    'Creation_TimeSpent',
    'Open_Date', 'Assignee_Date_Orig',
    'Assignee_Date', 'Resolve_Date_Orig',
    'Resolve_Date', 'Close_Date', 'TimeSpent_Minutes',
    'Contact_Assignee_Id', 'Application_Id', 'Contact_Customer_Id', 'Category_Id', 'SLA_Id',
    'SLA_StdTime_Assignee_Hours', 'SLA_StdTime_Resolution_Hours', 'Request_Type_Orig', 'Request_Type',
    'Priority_Id',
    'SLA_Violation_Orig', 'SLA_Assignee_Flag', 'SLA_Resolution_Flag', 'SLA_Assignee_Minutes',
    'SLA_Resolution_Minutes', 'SLA_Close_Minutes', 'WaitingTime_Assignee_Minutes', 'WaitingTime_Assignee_Minutes_Supplier',
    'WaitingTime_Assignee_Minutes_Internal', 'WaitingTime_Assignee_Minutes_Customer', 'WaitingTime_Resolution_Minutes',
    'WaitingTime_Resolution_Minutes_Supplier', 'WaitingTime_Resolution_Minutes_Internal', 'WaitingTime_Resolution_Minutes_Customer',
    'SLA_Assignee_Minutes_Above', 'SLA_Resolution_Minutes_Above', 'SLA_Violation', 'SLA_Assignee_Violation', 'SLA_Resolution_Violation',
    'Next_1Day_Minutes', 'Next_2Days_Minutes', 'Next_3Days_Minutes', 'NumReq_ReOpen', 'NumReq_ReOpen_Nr',
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


testing_dict = {'aceder': ['acedem', 'acceder', 'acede'],
                'acesso': ['aceso', 'accesso', 'acessos'],
                'actualizar': ['actualiza', 'atualizar', 'atualiza'],
                'agendamento': ['agendamentos'],
                'balanco': ['balancço', 'balance', 'balanço'],
                'cognos': ['kognos', 'cogno', 'cognos=', 'cognos7', 'cognos-', 'cognus'],
                'colaborador': ['colabolador', 'colaborados', 'colaboradora'],
                'copia': ['copias', 'copiar', 'cópia'],
                'cubo': ['cbo', '-cubo', 'cudo', 'cuno', 'cubos'],
                'curso': ['curso-'],
                'desbalanceamento': ['desbalanceamentos', 'balancemento'],
                'eficiencia': ['eficiençia', 'eficiência'],
                'financeira': ['financira', 'financeiras', 'financeiro'],
                'marcado': ['marcada', 'marcados'],
                'nao': ["'nao", 'dao', 'não'],
                'oficina': ['oficinal', 'oficinas'],
                'orcamento': ['orçamento'],
                'parametrizacao': ['parametrizaçao', 'parametriza', 'parametrizaão', 'parametrizado', 'parametrizadas', 'parametrizada', 'parametrização'],
                'parametrizacoes': ['parametrizes', 'parametriza-los', 'parametrizações'],
                'pasta': ['pastas'],
                'pecas': ['peças'],
                'portal': ['porta'],
                'produtivo': ['protutivo', 'produtivos'],
                'recurso': ['resurso', 'recursos'],
                'relatorio': ['relatorios', 'relatório'],
                'stock': ['stock…', 'stocks'],
                'utilizador': ['utilizador…', 'utlizador', 'utilizadora', 'ultilizador', 'utilizados', 'utilizado'],
                'vdvs': ['vd/vs'],
                'venda': ['vendo', 'venta', 'vends', 'vende', 'vendas'],
                'vendas': ['vendos', 'vendas-', 'vends', 'ventas', 'venda'],
                'versao': ['verso', 'versão'],
                'viatura': ['viaturas']
                }

gridsearch_parameters = {
    'lr': [LogisticRegression, [{'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'newton-cg', 'liblinear'], 'max_iter': [2000], 'multi_class': ['ovr', 'multinomial']}]],
    'lgb': [lgb.LGBMClassifier, [{'num_leaves': [15, 31, 50, 100], 'n_estimators': [50, 100, 200], 'max_depth': ['50', '100'], 'objective': ['multiclass']}]],

}