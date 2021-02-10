import os
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from py_dotenv import read_dotenv
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree, linear_model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LassoCV, Ridge, LassoLarsCV, ElasticNetCV
from sklearn.svm import SVR, SVC

dotenv_path = '/info.env'
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
read_dotenv(base_path + dotenv_path)

if 'nt' in os.name:
    DSN_SRV3_PRD = os.getenv('DSN_SRV3_Prd')
    DSN_MLG_PRD = os.getenv('DSN_MLG_Prd')
    DSN_MLG_DEV = os.getenv('DSN_MLG_Dev')
elif 'posix' in os.name:
    DSN = os.getenv('DSN_Prd_Linux')
    DSN_MLG = os.getenv('DSN_MLG_Linux')
    DSN_SRV3_PRD = os.getenv('DSN_SRV3_Prd_Linux')
    DSN_MLG_PRD = os.getenv('DSN_MLG_Prd_Linux')
UID = os.getenv('UID')
PWD = os.getenv('PWD')

sql_info = {
    'database_mlg': 'BI_MLG',
}

log_files = {
    'full_log': 'logs/finlog_cost_prediction_2527.txt'
}

project_id = 2527
update_frequency_days = 0

DATA_PATH = '../dbs/dataset_train_20200817_v6.csv'  # File created with code in get_train_dataset.py
DATA_PROB_PATH = '../dbs/df_train_test_prob_for_plotting_20200817.csv'  # File created with jupyter notebook Finlog_20200810_candidate_V4

# Encoders created with jupyter notebook Finlog_20200810_candidate_V4.ipynb
MODEL_PATH = '../models/model.joblib'

enc_LL_path = '../models/enc_LL.joblib'
enc_AR_path = '../models/enc_AR.joblib'
enc_PI_path = '../models/enc_PI.joblib'
enc_LA_path = '../models/enc_LA.joblib'
enc_FI_path = '../models/enc_FI.joblib'
enc_Make_path = '../models/enc_Make.joblib'
enc_Fuel_path = '../models/enc_Fuel.joblib'
enc_Vehicle_Segment_path = '../models/enc_Vehicle_Segment.joblib'
enc_Vehicle_Tipology_path = '../models/enc_Vehicle_Tipology.joblib'
enc_Client_type_path = '../models/enc_Client_type.joblib'
enc_Num_Vehicles_Total_path = '../models/enc_Num_Vehicles_Total.joblib'
enc_Num_Vehicles_Finlog_path = '../models/enc_Num_Vehicles_Finlog.joblib'
enc_Customer_Group_path = '../models/enc_Customer_Group.joblib'
Customer_Group_dict_path = '../models/customer_group_dict'


apply_query_v2 = '''
    WITH filters AS (
        SELECT DISTINCT
        --NULL AS Vehicle_No,
        '{}' AS FI,
        '{}' AS LL,
        '{}' AS AR,
        '{}' AS Vehicle_Tipology,
        '{}' AS Make,
        '{}' AS Fuel,
        --
        '{}' AS Client_type,
        '{}' AS Num_Vehicles_Total,
        '{}' AS Num_Vehicles_Finlog,
        '{}' AS contract_duration,
        '{}' AS Contract_km,
        '{}' AS contract_start_date
    FROM [BI_MLG].[dbo].[VHE_Finlog_Vehicle_Data]), 
    cte AS (
        SELECT DISTINCT
        NULL AS contract_customer,
        NULL AS Customer_Name,
        NULL AS contract_contract,
        NULL AS Vehicle_No,
        NULL AS Accident_No,
        NULL AS target,
        filters.FI,
        filters.LL,
        filters.AR,
        filters.Client_type,
        filters.Num_Vehicles_Total,
        filters.Num_Vehicles_Finlog,
        customer_data.Mean_repair_value_FULL,
        customer_data.Sum_repair_value_FULL,
        customer_data.Sum_contrat_km_FULL,
        customer_data.Num_Accidents_FULL,
        customer_data.Mean_contract_duration_FULL,
        customer_data.Mean_monthly_repair_cost_FULL,
        customer_data.Mean_repair_value_5YEAR,
        customer_data.Sum_repair_value_5YEAR,
        customer_data.Sum_contrat_km_5YEAR,
        customer_data.Num_Accidents_5YEAR,
        customer_data.Mean_contract_duration_5YEAR,
        customer_data.Mean_monthly_repair_cost_5YEAR,
        customer_data.Mean_repair_value_1YEAR,
        customer_data.Sum_repair_value_1YEAR,
        customer_data.Sum_contrat_km_1YEAR,
        customer_data.Num_Accidents_1YEAR,
        customer_data.Mean_contract_duration_1YEAR,
        customer_data.Mean_monthly_repair_cost_1YEAR,
        --
        filters.Contract_km * 1000 AS Contract_km,
        filters.contract_start_date,
        --contract_end_date,
        CAST(DATEADD(month, CAST(filters.contract_duration AS int), filters.contract_start_date) AS DATE)  AS contract_end_date,
        filters.contract_duration,
        filters.Vehicle_Tipology,
        filters.Make,
        filters.Fuel,
        vehicle_data.Weight_Empty AS Weight_Empty,
        vehicle_data.Insurable_Value AS Insurable_Value,
        vehicle_data.Engine_CC AS Engine_CC,
        vehicle_data.Power_kW AS Power_kW,
        vehicle_data.Max_speed AS Max_speed,
        vehicle_data.Max_Add_Load AS Max_Add_Load
    FROM filters
    LEFT JOIN [BI_MLG].[dbo].[VHE_Finlog_Vehicle_Data] vehicle_data ON
    vehicle_data.Fuel = filters.Fuel AND
    vehicle_data.Make = filters.Make AND
    vehicle_data.Vehicle_Tipology = filters.Vehicle_Tipology
    LEFT JOIN [BI_MLG].[dbo].[VHE_Finlog_Customer_Data] customer_data ON
    customer_data.Client_type = filters.Client_type AND
    customer_data.Num_Vehicles_Total = filters.Num_Vehicles_Total AND
    customer_data.Num_Vehicles_Finlog = filters.Num_Vehicles_Finlog
    )
    SELECT
        contract_customer,
        Customer_Name,
        contract_contract,
        Vehicle_No,
        Accident_No,
        target,
        FI,
        LL,
        AR,
        Client_type,
        Num_Vehicles_Total,
        Num_Vehicles_Finlog,
        --
        AVG(Mean_repair_value_FULL) AS Mean_repair_value_FULL,
        AVG(Sum_repair_value_FULL) AS Sum_repair_value_FULL,
        AVG(Sum_contrat_km_FULL) AS Sum_contrat_km_FULL,
        AVG(Num_Accidents_FULL) AS Num_Accidents_FULL,
        AVG(Mean_contract_duration_FULL) AS Mean_contract_duration_FULL,
        AVG(Mean_monthly_repair_cost_FULL) AS Mean_monthly_repair_cost_FULL,
        AVG(Mean_repair_value_5YEAR) AS Mean_repair_value_5YEAR,
        AVG(Sum_repair_value_5YEAR) AS Sum_repair_value_5YEAR,
        AVG(Sum_contrat_km_5YEAR) AS Sum_contrat_km_5YEAR,
        AVG(Num_Accidents_5YEAR) AS Num_Accidents_5YEAR,
        AVG(Mean_contract_duration_5YEAR) AS Mean_contract_duration_5YEAR,
        AVG(Mean_monthly_repair_cost_5YEAR) AS Mean_monthly_repair_cost_5YEAR,
        AVG(Mean_repair_value_1YEAR) AS Mean_repair_value_1YEAR,
        AVG(Sum_repair_value_1YEAR) AS Sum_repair_value_1YEAR,
        AVG(Sum_contrat_km_1YEAR) AS Sum_contrat_km_1YEAR,
        AVG(Num_Accidents_1YEAR) AS Num_Accidents_1YEAR,
        AVG(Mean_contract_duration_1YEAR) AS Mean_contract_duration_1YEAR,
        AVG(Mean_monthly_repair_cost_1YEAR) AS Mean_monthly_repair_cost_1YEAR,
        --
        Contract_km,
        contract_start_date,
        contract_end_date,
        contract_duration,
        Vehicle_Tipology,
        Make,
        Fuel,
        --
        AVG(Weight_Empty) AS Weight_Empty,
        AVG(Insurable_Value) AS Insurable_Value,
        AVG(Engine_CC) AS Engine_CC,
        AVG(Power_kW) AS Power_kW,
        AVG(Max_speed) AS Max_speed,
        AVG(Max_Add_Load) AS Max_Add_Load
    FROM cte
    GROUP BY
        contract_customer,
        Customer_Name,
        contract_contract,
        Vehicle_No,
        Accident_No,
        target,
        FI,
        LL,
        AR,
        Client_type,
        Num_Vehicles_Total,
        Num_Vehicles_Finlog,
        Contract_km,
        contract_start_date,
        contract_end_date,
        contract_duration,
        Vehicle_Tipology,
        Make,
        Fuel
'''


apply_query = '''
    WITH filters AS (
        SELECT DISTINCT
        --NULL AS Vehicle_No,
        '${FI_filter}' AS FI,
        '${LA_filter}' AS LA,
        '${LL_filter}' AS LL,
        '${PI_filter}' AS PI,
        '${AR_filter}' AS AR,
        '${segment_filter}' AS Vehicle_Segment,
        '${make_filter}' AS Make,
        '${fuel_filter}' AS Fuel,
        --
        '${client_type_filter}' AS Client_type,
        '${fleet_size_total_filter}' AS Num_Vehicles_Total,
        '${fleet_size_finlog_filter}' AS Num_Vehicles_Finlog,
        '${contract_duration}' AS contract_duration,
        '${km_year}' AS Contract_km,
        '${contract_start_date}' AS contract_start_date
        FROM [BI_MLG].[dbo].[VHE_Finlog_Vehicle_Data]), 
        cte AS (
        SELECT DISTINCT
        NULL AS contract_customer,
        NULL AS Customer_Name,
        NULL AS contract_contract,
        NULL AS Vehicle_No,
        NULL AS Accident_No,
        NULL AS target,
        filters.FI,
        filters.LA,
        filters.LL,
        filters.PI,
        filters.AR,
        filters.Client_type,
        filters.Num_Vehicles_Total,
        filters.Num_Vehicles_Finlog,
        customer_data.Mean_repair_value_FULL,
        customer_data.Sum_repair_value_FULL,
        customer_data.Sum_contrat_km_FULL,
        customer_data.Num_Accidents_FULL,
        customer_data.Mean_contract_duration_FULL,
        customer_data.Mean_monthly_repair_cost_FULL,
        customer_data.Mean_repair_value_5YEAR,
        customer_data.Sum_repair_value_5YEAR,
        customer_data.Sum_contrat_km_5YEAR,
        customer_data.Num_Accidents_5YEAR,
        customer_data.Mean_contract_duration_5YEAR,
        customer_data.Mean_monthly_repair_cost_5YEAR,
        customer_data.Mean_repair_value_1YEAR,
        customer_data.Sum_repair_value_1YEAR,
        customer_data.Sum_contrat_km_1YEAR,
        customer_data.Num_Accidents_1YEAR,
        customer_data.Mean_contract_duration_1YEAR,
        customer_data.Mean_monthly_repair_cost_1YEAR,
        --
        filters.Contract_km * 1000 AS Contract_km,
        filters.contract_start_date,
        --contract_end_date,
        CAST(DATEADD(month, CAST(filters.contract_duration AS int), filters.contract_start_date) AS DATE)  AS contract_end_date,
        filters.contract_duration,
        filters.Vehicle_Segment,
        filters.Make,
        filters.Fuel,
        vehicle_data.Weight_Empty AS Weight_Empty,
        vehicle_data.Insurable_Value AS Insurable_Value,
        vehicle_data.Engine_CC AS Engine_CC,
        vehicle_data.Power_kW AS Power_kW,
        vehicle_data.Max_speed AS Max_speed,
        vehicle_data.Max_Add_Load AS Max_Add_Load
    FROM filters
    LEFT JOIN [BI_MLG].[dbo].[VHE_Finlog_Vehicle_Data] vehicle_data ON
        vehicle_data.Fuel = filters.Fuel AND
        vehicle_data.Make = filters.Make AND
        vehicle_data.Vehicle_Segment = filters.Vehicle_Segment
    
    LEFT JOIN [BI_MLG].[dbo].[VHE_Finlog_Customer_Data] customer_data ON
        customer_data.Client_type = filters.Client_type AND
        customer_data.Num_Vehicles_Total = filters.Num_Vehicles_Total AND
        customer_data.Num_Vehicles_Finlog = filters.Num_Vehicles_Finlog)
    SELECT
        contract_customer,
        Customer_Name,
        contract_contract,
        Vehicle_No,
        Accident_No,
        target,
        FI,
        LA,
        LL,
        PI,
        AR,
        Client_type,
        Num_Vehicles_Total,
        Num_Vehicles_Finlog,
        --
        AVG(Mean_repair_value_FULL) AS Mean_repair_value_FULL,
        AVG(Sum_repair_value_FULL) AS Sum_repair_value_FULL,
        AVG(Sum_contrat_km_FULL) AS Sum_contrat_km_FULL,
        AVG(Num_Accidents_FULL) AS Num_Accidents_FULL,
        AVG(Mean_contract_duration_FULL) AS Mean_contract_duration_FULL,
        AVG(Mean_monthly_repair_cost_FULL) AS Mean_monthly_repair_cost_FULL,
        AVG(Mean_repair_value_5YEAR) AS Mean_repair_value_5YEAR,
        AVG(Sum_repair_value_5YEAR) AS Sum_repair_value_5YEAR,
        AVG(Sum_contrat_km_5YEAR) AS Sum_contrat_km_5YEAR,
        AVG(Num_Accidents_5YEAR) AS Num_Accidents_5YEAR,
        AVG(Mean_contract_duration_5YEAR) AS Mean_contract_duration_5YEAR,
        AVG(Mean_monthly_repair_cost_5YEAR) AS Mean_monthly_repair_cost_5YEAR,
        AVG(Mean_repair_value_1YEAR) AS Mean_repair_value_1YEAR,
        AVG(Sum_repair_value_1YEAR) AS Sum_repair_value_1YEAR,
        AVG(Sum_contrat_km_1YEAR) AS Sum_contrat_km_1YEAR,
        AVG(Num_Accidents_1YEAR) AS Num_Accidents_1YEAR,
        AVG(Mean_contract_duration_1YEAR) AS Mean_contract_duration_1YEAR,
        AVG(Mean_monthly_repair_cost_1YEAR) AS Mean_monthly_repair_cost_1YEAR,
        --
        Contract_km,
        contract_start_date,
        contract_end_date,
        contract_duration,
        Vehicle_Segment,
        Make,
        Fuel,
        --
        AVG(Weight_Empty) AS Weight_Empty,
        AVG(Insurable_Value) AS Insurable_Value,
        AVG(Engine_CC) AS Engine_CC,
        AVG(Power_kW) AS Power_kW,
        AVG(Max_speed) AS Max_speed,
        AVG(Max_Add_Load) AS Max_Add_Load
    FROM cte
    GROUP BY
        contract_customer,
        Customer_Name,
        contract_contract,
        Vehicle_No,
        Accident_No,
        target,
        FI,
        LA,
        LL,
        PI,
        AR,
        Client_type,
        Num_Vehicles_Total,
        Num_Vehicles_Finlog,
        Contract_km,
        contract_start_date,
        contract_end_date,
        contract_duration,
        Vehicle_Segment,
        Make,
        Fuel
'''

auth_description = '''
    WITH cte AS (
    SELECT
        vehicle.Make,
        vehicle.Vehicle_Segment,
        auth.*
    FROM [BI_MLG].[dbo].[VHE_Authorization_Line] auth
    LEFT JOIN [BI_MLG].[dbo].[VHE_Finlog_Vehicle_Data] vehicle ON
    auth.Vehicle = vehicle.Vehicle_No
    WHERE
    auth.[Invoice Type] IN ('SINISTRO', 'VSSINISTRO', 'VS ROUBO') --AND
    --vehicle.Vehicle_Segment = '${segment_filter}' AND
    --vehicle.Make = '${make_filter}'
    )
    SELECT
        cte.Description,
        COUNT(1) AS Num_Ocorrencias,
        AVG(Amount) AS Custo_medio
    FROM cte
    GROUP BY Description
    ORDER BY COUNT(1) DESC;
'''


get_train_dataset_query = '''
    SELECT
        customer.contract_customer,
        customer.Customer_Name,
        customer.contract_contract,
        customer.Vehicle_No,
        customer.Accident_No,
        customer.target,
        customer.Customer_Group,
        vehicle.FI,
        --vehicle.LA,
        vehicle.LL,
        --vehicle.PI,
        vehicle.AR,
        --customer_accident_history.Vehicle_Segment,
        customer.Client_type,
        --
        customer.Num_Vehicles_Total,
        customer.Num_Vehicles_Finlog,
        --
        customer.Mean_repair_value_FULL,
        customer.Sum_repair_value_FULL,
        customer.Sum_contrat_km_FULL,
        customer.Num_Accidents_FULL,
        customer.Mean_contract_duration_FULL,
        customer.Mean_monthly_repair_cost_FULL,
        --
        customer.Mean_repair_value_5YEAR,
        customer.Sum_repair_value_5YEAR,
        customer.Sum_contrat_km_5YEAR,
        customer.Num_Accidents_5YEAR,
        customer.Mean_contract_duration_5YEAR,
        customer.Mean_monthly_repair_cost_5YEAR,
        --
        customer.Mean_repair_value_1YEAR,
        customer.Sum_repair_value_1YEAR,
        customer.Sum_contrat_km_1YEAR,
        customer.Num_Accidents_1YEAR,
        customer.Mean_contract_duration_1YEAR,
        customer.Mean_monthly_repair_cost_1YEAR,
        --
        customer.Contract_km,
        customer.contract_start_date,
        customer.contract_end_date,
        customer.contract_duration,
        --vehicle.Vehicle_Segment,
        vehicle.Vehicle_Tipology,
        vehicle.Make,
        vehicle.Fuel,
        vehicle.Weight_Empty,
        vehicle.Insurable_Value,
        vehicle.Engine_CC,
        vehicle.Power_kW,
        vehicle.Max_speed,
        vehicle.Max_Add_Load
    FROM
        [BI_MLG].[dbo].[VHE_Finlog_Customer_Data] customer
    LEFT JOIN
        [BI_MLG].[dbo].[VHE_Finlog_Vehicle_Data] vehicle
    ON
        vehicle.Vehicle_No = customer.Vehicle_No
    WHERE
        (vehicle.FI IS NOT NULL OR
        vehicle.LL IS NOT NULL OR
        vehicle.AR IS NOT NULL) AND
        LEFT(vehicle.AR_code, 2) = 'AS' AND
        customer.contract_start_date > '1900-01-01 00:00:00.000' AND
        --customer.contract_end_date < '2020-07-10 00:00:00.000' AND
        DATALENGTH(vehicle.[Vehicle_No]) > 0 AND
        DATALENGTH(vehicle.Make) > 0 AND
        DATALENGTH(customer.Contract_km) > 0 AND
        LEFT(vehicle.LL, 3) != 'OLD' AND
        LEFT(vehicle.AR, 3) != 'OLD' AND
        LEFT(vehicle.FI, 3) != 'OLD';
'''

customer_data_query = '''
    SELECT DISTINCT
        customer_data.Mean_repair_value_FULL,
        customer_data.Sum_repair_value_FULL,
        customer_data.Sum_contrat_km_FULL,
        customer_data.Num_Accidents_FULL,
        customer_data.Mean_contract_duration_FULL,
        customer_data.Mean_monthly_repair_cost_FULL,
        customer_data.Mean_repair_value_5YEAR,
        customer_data.Sum_repair_value_5YEAR,
        customer_data.Sum_contrat_km_5YEAR,
        customer_data.Num_Accidents_5YEAR,
        customer_data.Mean_contract_duration_5YEAR,
        customer_data.Mean_monthly_repair_cost_5YEAR,
        customer_data.Mean_repair_value_1YEAR,
        customer_data.Sum_repair_value_1YEAR,
        customer_data.Sum_contrat_km_1YEAR,
        customer_data.Num_Accidents_1YEAR,
        customer_data.Mean_contract_duration_1YEAR,
        customer_data.Mean_monthly_repair_cost_1YEAR,
        customer_data.Client_type,
        customer_data.Num_Vehicles_Total,
        customer_data.Num_Vehicles_Finlog,
        1 as cross_join_key
    FROM [BI_MLG].[dbo].[VHE_Finlog_Customer_Data] as customer_data
    WHERE 1=1
    and customer_data.Client_type in ({})
    and customer_data.Num_Vehicles_Total in ({})
    and customer_data.Num_Vehicles_Finlog in ({})
'''

customer_data_cols = [
        'Mean_repair_value_FULL',
        'Sum_repair_value_FULL',
        'Sum_contrat_km_FULL',
        'Num_Accidents_FULL',
        'Mean_contract_duration_FULL',
        'Mean_monthly_repair_cost_FULL',
        'Mean_repair_value_5YEAR',
        'Sum_repair_value_5YEAR',
        'Sum_contrat_km_5YEAR',
        'Num_Accidents_5YEAR',
        'Mean_contract_duration_5YEAR',
        'Mean_monthly_repair_cost_5YEAR',
        'Mean_repair_value_1YEAR',
        'Sum_repair_value_1YEAR',
        'Sum_contrat_km_1YEAR',
        'Num_Accidents_1YEAR',
        'Mean_contract_duration_1YEAR',
        'Mean_monthly_repair_cost_1YEAR']

vehicle_data_query = '''
    SELECT DISTINCT
        VHE.Weight_Empty AS Weight_Empty,
        VHE.Insurable_Value AS Insurable_Value,
        VHE.Engine_CC AS Engine_CC,
        VHE.Power_kW AS Power_kW,
        VHE.Max_speed AS Max_speed,
        VHE.Max_Add_Load AS Max_Add_Load,
        VHE.Fuel,
        VHE.Make,
        VHE.Vehicle_Tipology,
        1 as cross_join_key
    FROM [BI_MLG].[dbo].[VHE_Finlog_Vehicle_Data] as VHE
    WHERE 1=1
    and VHE.Fuel in ({})
    and VHE.Make in ({})
    and VHE.Vehicle_Tipology in ({})
'''

vehicle_data_cols = ['Weight_Empty', 'Insurable_Value', 'Engine_CC', 'Power_kW', 'Max_speed', 'Max_Add_Load']

vhe_data_col_keys = ['Fuel', 'Make', 'Vehicle_Tipology']
customer_data_col_keys = ['Client_type', 'Num_Vehicles_Total', 'Num_Vehicles_Finlog']


