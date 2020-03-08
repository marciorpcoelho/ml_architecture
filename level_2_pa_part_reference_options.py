import os
from py_dotenv import read_dotenv
from multiprocessing import cpu_count
dotenv_path = '/info.env'
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
read_dotenv(base_path + dotenv_path)

if 'nt' in os.name:
    DSN = os.getenv('DSN_Prd')
    DSN_MLG = os.getenv('DSN_MLG')
elif 'posix' in os.name:
    DSN = os.getenv('DSN_Prd_Linux')
    DSN_MLG = os.getenv('DSN_MLG_Linux')
UID = os.getenv('UID')
PWD = os.getenv('PWD')


sql_info = {
    'database_afr': 'BI_RCG',
    'database_crp': 'BI_CRP',
    'database_ibe': 'BI_IBE',
    'database_ca': 'BI_CA',
    'database_gsc': 'BI_GSC',
    'database_final': 'BI_MLG',
    }


afr_current_stock_query = '''SELECT DISTINCT Part_Ref, Part_Desc, Product_Group_DW, Client_Id
FROM BI_AFR.dbo.PSE_Fact_BI_Parts_Stock_Month
WHERE Part_Ref <> '' and Part_Desc is not Null and Part_Desc <> '' and Stock_Month = '202002'
GROUP BY Part_Ref, Part_Desc, Product_Group_DW, Client_Id'''

crp_current_stock_query = '''SELECT DISTINCT Part_Ref, Part_Desc, Product_Group_DW, Client_Id
FROM BI_CRP.dbo.PSE_Fact_BI_Parts_Stock_Month WITH (NOLOCK)
WHERE Part_Ref <> '' and Part_Desc is not Null and Part_Desc <> '' and Stock_Month = '202002'
GROUP BY Part_Ref, Part_Desc, Product_Group_DW, Client_Id'''

ibe_current_stock_query = '''SELECT DISTINCT Part_Ref, Part_Desc, Product_Group_DW, Client_Id
FROM BI_IBE.dbo.PSE_Fact_BI_Parts_Stock_Month WITH (NOLOCK)
WHERE Part_Ref <> '' and Part_Desc is not Null and Part_Desc <> '' and Stock_Month = '202002'
GROUP BY Part_Ref, Part_Desc, Product_Group_DW, Client_Id'''

ca_current_stock_query = '''SELECT DISTINCT Part_Ref, Part_Desc, Product_Group_DW, Client_Id
FROM BI_CA.dbo.PSE_Fact_BI_Parts_Stock_Month WITH (NOLOCK)
WHERE Part_Ref <> '' and Part_Desc is not Null and Part_Desc <> '' and Stock_Month = '202002'
GROUP BY Part_Ref, Part_Desc, Product_Group_DW, Client_Id'''

dim_product_group_query = '''SELECT [Product_Group_Code]
      ,[Product_Group_Level_1_Code]
      ,[Product_Group_Level_2_Code]
      ,[PT_Product_Group_Level_1_Desc]
      ,[PT_Product_Group_Level_2_Desc]
      ,[PT_Product_Group_Desc]
      ,[ES_Product_Group_Level_1_Desc]
      ,[ES_Product_Group_Level_2_Desc]
      ,[ES_Product_Group_Desc]
      ,[EN_Product_Group_Level_1_Desc]
      ,[EN_Product_Group_Level_2_Desc]
      ,[EN_Product_Group_Desc]
      ,[FR_Product_Group_Level_1_Desc]
      ,[FR_Product_Group_Level_2_Desc]
      ,[FR_Product_Group_Desc]
      ,[Product_Group_Display_Order]
  FROM [BI_AFR].[dbo].[PSE_Dim_Product_Groups_GSC]'''

dim_clients_query = '''SELECT Client_Id, Client_Name, Description, Description_Mobile, BI_Database
FROM [BI_GSC].[dbo].[GBL_Setup_Clients]'''






