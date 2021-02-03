import os
from py_dotenv import read_dotenv
from multiprocessing import cpu_count
dotenv_path = '/info.env'
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
read_dotenv(base_path + dotenv_path)

if 'nt' in os.name:
    DSN_SRV3_PRD = os.getenv('DSN_SRV3_Prd')
    DSN_MLG_PRD = os.getenv('DSN_MLG_Prd')
    DSN_MLG_DEV = os.getenv('DSN_MLG_Dev')
elif 'posix' in os.name:
    DSN_SRV3_PRD = os.getenv('DSN_SRV3_Prd_Linux')
    DSN_MLG_PRD = os.getenv('DSN_MLG_Prd_Linux')
UID = os.getenv('UID')
PWD = os.getenv('PWD')


project_id = 2844
update_frequency_days = 0

sql_info = {
    'database_source': 'BI_MLG',
    'source_table': 'VHE_Fact_PA_PSE_Info'
}

temp_file_loc = 'dbs/cr_vhe_pse_visits.csv'
temp_file_grouped_loc = 'dbs/cr_vhe_pse_visits_grouped.csv'

