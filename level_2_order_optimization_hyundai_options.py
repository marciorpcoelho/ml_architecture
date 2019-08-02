import os
from py_dotenv import read_dotenv
dotenv_path = 'info.env'
read_dotenv(dotenv_path)

DSN_PRD = os.getenv('DSN_Prd')
DSN_MLG = os.getenv('DSN_MLG')
UID = os.getenv('UID')
PWD = os.getenv('PWD')

project_id = 2406
update_frequency_days = 0

sql_info = {
    'database_source': 'BI_DTR',
    'database_final': 'BI_MLG',
    'product_db': 'VHE_Dim_VehicleData_DTR',
    'sales': 'VHE_Fact_BI_Sales_DTR',
    'stock': 'VHE_Fact_BI_Stock_DTR',
    'final_table': 'VHE_Fact_BI_Sales_DTR_Temp',
}

sales_query_filtered = '''
        SELECT *
        FROM [BI_DTR].dbo.[VHE_Fact_BI_Sales_DTR] WITH (NOLOCK)
        where Registration_Flag = '1' and Chassis_Flag = '1' and VehicleData_Code <> '1'
            and Sales_Type_Code_DMS in ('RAC', 'STOCK', 'UVENDA', 'VENDA') and Sales_Type_Dealer_Code <> 'Demo'
        UNION ALL
        SELECT *
        FROM [BI_DW_History].dbo.[VHE_Fact_BI_Sales_DTR] WITH (NOLOCK) 
        where Registration_Flag = '1' and Chassis_Flag = '1' and VehicleData_Code <> '1'
            and Sales_Type_Code_DMS in ('RAC', 'STOCK', 'UVENDA', 'VENDA') and Sales_Type_Dealer_Code <> 'Demo' '''


sales_query = '''
        SELECT *
        FROM [BI_DTR].dbo.[VHE_Fact_BI_Sales_DTR] WITH (NOLOCK) 
        UNION ALL 
        SELECT *
        FROM [BI_DW_History].dbo.[VHE_Fact_BI_Sales_DTR] WITH (NOLOCK)'''

stock_query = '''
        select *
        from VHE_Fact_BI_Stock_DTR WITH (NOLOCK)'''

product_db = '''
        select *
        from VHE_Dim_VehicleData_DTR WITH (NOLOCK)'''


# Motorização
motor_translation = {
    '1.0': ['1.0 lpgi'],
    '1.0i': ['1.0 t-gdi', '1.0i', '1.0l'],
    '1.1d': ['1.1 crdi'],
    '1.2i': ['1.2i'],
    '1.3i': ['1.3l'],
    '1.4d': ['1.4 crdi'],
    '1.4i': ['1.4 t-gdi'],
    '1.5i': ['1.5l'],
    # '1.6': [],
    '1.6d': ['1.6l', '1.6 crdi'],
    '1.6i': ['1.6 t-gdi'],
    '1.7d': ['1.7 crdi'],
    '2.0d': ['2.0 crdi'],
    '2.0i': ['2.0L', '2.0 t-gdi'],
    '2.2d': ['2.2 crdi'],
    '2.5d': ['2.5 crdi'],
    'eletrico': ['motor elétrico']
}