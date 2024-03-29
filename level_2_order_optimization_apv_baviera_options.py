import os
from py_dotenv import read_dotenv
dotenv_path = '/info.env'
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
read_dotenv(base_path + dotenv_path)

update_frequency_days = 0
api_backend_loc = 'optimizations/apv_parts_baviera/'

if 'nt' in os.name:
    DSN_SRV3_PRD = os.getenv('DSN_SRV3_Prd')
    DSN_MLG_PRD = os.getenv('DSN_MLG_Prd')
    DSN_MLG_DEV = os.getenv('DSN_MLG_Dev')
elif 'posix' in os.name:
    DSN_SRV3_PRD = os.getenv('DSN_SRV3_Prd_Linux')
    DSN_MLG_PRD = os.getenv('DSN_MLG_Prd_Linux')
UID = os.getenv('UID')
PWD = os.getenv('PWD')

sql_info = {
    'database': 'BI_CRP',
    'database_histy': 'BI_DW_History',
    'database_final': 'BI_MLG',
    'dms_database': 'DMS_MLG_01',
    'sales_table': '',
    'purchases_table': '',
    'stock_table': '',
    'final_table': 'PSE_Fact_PA_OrderOptimization',
    'optimization_solution_table': 'PSE_Fact_PA_OrderOptimization_Solver_Optimization',
    'ta_table': 'PSE_Fact_PA_OrderOptimization_TA_Groups',
    'goals_table': 'PSE_Fact_PA_OrderOptimization_Goals',
}

sql_info_stock_dbs = {
    'AFR_SPG_01': 'DMS_AFR_01',
    'AFR_SPG_03': 'DMS_AFR_03',
    'AFR_SPG_05': 'DMS_AFR_05',
    'AFR_SPG_06': 'DMS_AFR_06',
    'AFR_SPG_07': 'DMS_AFR_07',
    'AFR_SPG_08': 'DMS_AFR_08',
    'CRP_SPG_04': 'DMS_CRP_04',
    'IBE_SPG_02': 'DMS_IBE_02',
    'CA_SPG_01': 'DMS_CA_01',
    'CRP_ATL_01': 'DMS_CRP_01',
    'IBE_ATL_01': 'DMS_IBE_01',
}

sql_info_stock_tables = {
    'AFR_SPG_01': ['SM_Parts_Stock', 'SM_Parts_Stock_Warehouses'],
    'AFR_SPG_03': ['SM_Parts_Stock', 'SM_Parts_Stock_Warehouses'],
    'AFR_SPG_05': ['SM_Parts_Stock', 'SM_Parts_Stock_Warehouses'],
    'AFR_SPG_06': ['SM_Parts_Stock', 'SM_Parts_Stock_Warehouses'],
    'AFR_SPG_07': ['SM_Parts_Stock', 'SM_Parts_Stock_Warehouses'],
    'AFR_SPG_08': ['SM_Parts_Stock', 'SM_Parts_Stock_Warehouses'],
    'CRP_SPG_04': ['SM_Parts_Stock', 'SM_Parts_Stock_Warehouses'],
    'CA_SPG_01': ['SM_Parts_Stock', 'SM_Parts_Stock_Warehouses'],
    'IBE_SPG_02': ['SM_Parts_Stock', 'SM_Parts_Stock_Warehouses'],
    'CRP_ATL_01': ['SM_Parts_Stock'],
    'IBE_ATL_01': ['SM_Parts_Stock'],
}

project_id = 2259
urgent_purchases_flags = [4, 5]
# min_date = '20180131'  # This is a placeholder for the minimum possible date - It already searches for the last processed date.
min_date = '20201001'  # This refers to the date where this data change DMS Source
documentation_url_solver_apv = 'https://gruposalvadorcaetano.sharepoint.com/:b:/s/rigor/6825_DGAA/ERAmjgJYWztNiMhPUZQHAVIBWZFd_xqdQDzKSbWB7VtTkQ?e=xXzBwu'

log_files = {
    'full_log': 'logs/apv_baviera_2259.txt'
}

configuration_parameters_full = ['Motor_Desc', 'Alarm', 'AC_Auto', 'Open_Roof', 'Auto_Trans', 'Colour_Ext', 'Colour_Int', 'LED_Lights', 'Xenon_Lights', 'Rims_Size', 'Model_Code', 'Navigation', 'Park_Front_Sens', 'Roof_Bars', 'Interior_Type', 'Version']
extra_parameters = ['Average_Score_Euros', 'Number_Cars_Sold', 'Average_Score_Euros_Local', 'Number_Cars_Sold_Local', 'Sales_Place']

bmw_ta_mapping = {
    'BMW_Bonus_Group_1': ['1', '2'],  # Peças + Óleos
    'BMW_Bonus_Group_2': ['-'],  # Vendas de 4º Nível - Can be ignored
    'BMW_Bonus_Group_3': ['Chemical'],  # Químicos
    'BMW_Bonus_Group_4': ['3', '5', '7'],  # Acessórios + Jantes + Lifestyle
    'BMW_Bonus_Group_5': ['8'],  # Pneus
    'Outros': ['4', '6', '9'],
    'NO_SAME_BRAND_TA': ['NO_SAME_BRAND_TA'],
    'NO_TA': ['NO_TA'],
}

mini_ta_mapping = {
    'MINI_Bonus_Group_1': ['1', '2'],  # Peças + Óleos
    'MINI_Bonus_Group_2': ['-'],  # MINI Regeneration - No idea on this one...
    'MINI_Bonus_Group_3': ['3', '5', '7'],  # Acessórios + Jantes + Lifestyle
    'MINI_Bonus_Group_4': ['8'],  # Pneus
    'Outros': ['4', '6', '9'],
    'NO_SAME_BRAND_TA': ['NO_SAME_BRAND_TA'],
    'NO_TA': ['NO_TA'],
}

# The cost goals for group 1 and 2, as well as the sale goals for groups 3, 4 and 5 are taken from the provided files.
# The remaining sale/cost goals are fluctuations of 5% from the previous values
# There currently are no goals for MINI groups
# group_goals = {
#     'dtss_goal': 15,  # Weekdays only!
#     'BMW_Bonus_Group_1': [252051, 264654],  # Cost, Sales
#     'BMW_Bonus_Group_2': [56055, 58858],  # Cost, Sales
#     'BMW_Bonus_Group_3': [5075, 5329],  # Cost, Sales
#     'BMW_Bonus_Group_4': [9539, 10016],  # Cost, Sales
#     'BMW_Bonus_Group_5': [8675, 9109],  # Cost, Sales
#     'MINI_Bonus_Group_1': [0, 0],
#     'MINI_Bonus_Group_2': [0, 0],
#     'MINI_Bonus_Group_3': [0, 0],
#     'MINI_Bonus_Group_4': [0, 0],
# }

group_goals = {
    'dtss_goal': 15,  # Weekdays only!
    'number_of_unique_parts': 50,
    'number_of_total_parts': 50,
    # 'BMW_Bonus_Group_1': [640345 - 399127, 253279],  # Cost, Sales (Values are for a single month, using the goal for the 3-month period minus the already sold in the first two months
    # 'BMW_Bonus_Group_2': [155713 - 93374, 3655],  # Cost, Sales
    # 'BMW_Bonus_Group_3': [12893 - 10808, 2189],  # Cost, Sales
    # 'BMW_Bonus_Group_4': [25445 - 16152, 9758],  # Cost, Sales
    # 'BMW_Bonus_Group_5': [23143 - 14146, 9447],  # Cost, Sales
    # 'BMW_Bonus_Group_1': [229979 * 1.05],  # Purchase - September Goal
    # 'BMW_Bonus_Group_3': [4630 * 1.05],  # Purchase
    # 'BMW_Bonus_Group_4': [9139 * 1.05],  # Sales
    # 'BMW_Bonus_Group_5': [8132 * 1.05],  # Sales
    # 'BMW_Bonus_Group_1': [223713 - 208239],  # Purchase - September Goal
    # 'BMW_Bonus_Group_3': [4504 - 2226],  # Purchase
    # 'BMW_Bonus_Group_4': [8890 - 8250],  # Sales
    # 'BMW_Bonus_Group_5': [8085 - 3320],  # Sales
    # 'BMW_Bonus_Group_1_limit': [223713 * 1.05 - 208239],  # Purchase - September Goal
    # 'BMW_Bonus_Group_3_limit': [4504 * 1.05 - 2226],  # Purchase
    # 'BMW_Bonus_Group_4_limit': [8890 * 1.05 - 8250],  # Sales
    # 'BMW_Bonus_Group_5_limit': [8085 * 1.05 - 3320],  # Sales
    # 'MINI_Bonus_Group_1': [0],
    # 'MINI_Bonus_Group_3': [0],
    # 'MINI_Bonus_Group_4': [0],
    # 'MINI_Bonus_Group_1_limit': [0],
    # 'MINI_Bonus_Group_3_limit': [0],
    # 'MINI_Bonus_Group_4_limit': [0],
}

group_goals_type = {
    'BMW_Bonus_Group_1': 'Cost',  # Purchases
    'BMW_Bonus_Group_3': 'Cost',  # Purchases
    'BMW_Bonus_Group_4': 'PVP',  # Sales
    'BMW_Bonus_Group_5': 'PVP',  # Sales
    'MINI_Bonus_Group_1': 'PVP',  # Sales
    'MINI_Bonus_Group_3': 'PVP',  # Sales
    'MINI_Bonus_Group_4': 'PVP',  # Sales
}

goal_types = ['Cost', 'PVP']

part_groups_desc_mapping = {
    'BMW_Bonus_Group_1': 'BMW - Peças + Óleos',
    'BMW_Bonus_Group_2': 'BMW - Vendas Balcão',
    'BMW_Bonus_Group_3': 'BMW - Químicos',
    'BMW_Bonus_Group_4': 'BMW - Acessórios + Jantes + Lifestyle',
    'BMW_Bonus_Group_5': 'BMW - Pneus',
    'MINI_Bonus_Group_1': 'MINI - Peças + Óleos',
    'MINI_Bonus_Group_2': 'MINI Regeneration',   # MINI Regeneration - No idea on this one...
    'MINI_Bonus_Group_3': 'MINI - Acessórios + Jantes + Lifestyle',  # Acessórios + Jantes + Lifestyle
    'MINI_Bonus_Group_4': 'MINI - Pneus',   # Pneus
}

pse_code_desc_mapping = {
    '15': 'V.N. Gaia',
    '44': 'P. Nações',
    '38': 'S.M. Feira',
    '47': 'Cascais',
    '53': 'M.P. Azevedo',
    '67': 'Maia',
    '45': 'B. Roma',
    '48': 'Faro',
    '49': 'Portimão',
    '39': 'Aveiro',
    # '0Q': 'Motorrad - Entrecampos',
    # '0J': 'Setúbal',
    # '0M': 'Coimbra',
    # '0O': 'Viseu',
    # '0P': 'Viseu - Colisão',
}

pse_code_groups = [['15', '47', '38', '39', '45'], ['44', '53', '48', '49', '67']]


# 'BMW - Peças + Óleos'
# part_groups_desc = ['BMW - Pneus', 'BMW - Acessórios + Jantes + Lifestyle', 'BMW - Químicos', 'MINI - Pneus', 'MINI - Acessórios + Jantes + Lifestyle', 'MINI Regeneration', 'MINI - Peças + Óleos']
part_groups_desc = ['BMW - Peças + Óleos', 'BMW - Pneus', 'BMW - Acessórios + Jantes + Lifestyle', 'BMW - Químicos', 'MINI - Pneus', 'MINI - Acessórios + Jantes + Lifestyle', 'MINI - Peças + Óleos']

sales_query = '''
    Select Sales.SLR_Document_Date,  
        Sales.Movement_Date,  
        Sales.SLR_Document, 
        LTRIM(RTRIM(Sales.Part_Ref)) as Part_Ref,  
        Sales.Part_Desc,
        Sales.Product_Group, 
        Sales.Client_Id, 
        Sales.NLR_Code, 
        Sales.PSE_Code, 
        Sales.WIP_Number,  
        Sales.WIP_Date_Created,  
        SUM(Quantity*AdvPay_ValueIncluded) as Qty_Sold,  
        SUM(PVP_1 * Quantity * AdvPay_ValueIncluded) as PVP, 
        SUM((Posting_Sell_Value* AdvPay_ValueIncluded) + Menu_Difference) as Sale_Value, 
        SUM(Cost_Value*AdvPay_ValueIncluded) as Cost_Sale, 
        SUM(Posting_Discount_Value* AdvPay_ValueIncluded) as Discount_Value, 
        SUM(((Posting_Sell_Value*AdvPay_ValueIncluded) + Menu_Difference) - (Posting_Discount_Value*AdvPay_ValueIncluded) - (Cost_Value*AdvPay_ValueIncluded)) as Gross_Margin  
    FROM [BI_CRP].[dbo].[PSE_Sales] AS Sales WITH (NOLOCK)  
    WHERE Client_Id = 3  and Line_Type = 'P' AND NLR_Code = '701' and PSE_Code in ({}) --and LEFT(Part_Ref, 2) in ('MN', 'BM')  
    GROUP BY Sales.SLR_Document_Date,  
        Sales.Movement_Date,  
        Sales.SLR_Document, 
        Sales.Part_Ref,  
        Sales.Part_Desc, 
        Sales.Product_Group, 
        Sales.Client_Id, 
        Sales.NLR_Code, 
        Sales.PSE_Code, 
        Sales.WIP_Number, 
        Sales.WIP_Date_Created   
    union all  
    Select Sales.SLR_Document_Date,  
        Sales.Movement_Date,  
        Sales.SLR_Document, 
        LTRIM(RTRIM(Sales.Part_Ref)) as Part_Ref,  
        Sales.Part_Desc,  
        Sales.Product_Group, 
        Sales.Client_Id, 
        Sales.NLR_Code, 
        Sales.PSE_Code, 
        Sales.WIP_Number, 
        Sales.WIP_Date_Created,  
        SUM(Quantity*AdvPay_ValueIncluded) as Qty_Sold,  
        SUM(PVP_1*Quantity* AdvPay_ValueIncluded) as PVP, 
        SUM((Posting_Sell_Value* AdvPay_ValueIncluded) + Menu_Difference) as Sale_Value, 
        SUM(Cost_Value*AdvPay_ValueIncluded) as Cost_Sale, 
        SUM(Posting_Discount_Value* AdvPay_ValueIncluded) as Discount_Value, 
        SUM(((Posting_Sell_Value*AdvPay_ValueIncluded) + Menu_Difference) - (Posting_Discount_Value*AdvPay_ValueIncluded) - (Cost_Value*AdvPay_ValueIncluded)) as Gross_Margin  
    FROM [BI_DW_History].[dbo].[PSE_Sales] AS Sales WITH (NOLOCK)  
    WHERE Client_Id = 3  and Line_Type='P' AND NLR_Code = '701' and PSE_Code in ({}) --and LEFT(Part_Ref, 2) in ('MN', 'BM') 
    GROUP BY Sales.SLR_Document_Date,  
        Sales.Movement_Date,  
        Sales.SLR_Document, 
        Sales.Part_Ref,  
        Sales.Part_Desc, 
        Sales.Product_Group, 
        Sales.Client_Id, 
        Sales.NLR_Code, 
        Sales.PSE_Code, 
        Sales.WIP_Number,  
        Sales.WIP_Date_Created '''

purchases_query = '''
    SELECT Movement_Date,  
        PLR_Document,  
        Quantity,  
        Cost_Value, 
        LTRIM(RTRIM(Part_Ref)) as Part_Ref, 
        Part_Desc, 
        Product_Group_DW,  
        Order_Type_DW,  
        WIP_Number, 
        WIP_Date_Created  
    FROM [BI_CRP].[dbo].[PSE_parts_purchase] WITH (NOLOCK)  
    WHERE NLR_Code = '701'  AND Parts_Included=1 and PSE_Code in ({}) --and LEFT(Part_Ref, 2) in ('MN', 'BM')  
    union all  
    SELECT Movement_Date,  
        PLR_Document, 
        Quantity,  
        Cost_Value, 
        LTRIM(RTRIM(Part_Ref)) as Part_Ref, 
        Part_Desc, 
        Product_Group_DW,  
        Order_Type_DW,  
        WIP_Number, 
        WIP_Date_Created  
    FROM [BI_DW_History].[dbo].[PSE_parts_purchase] WITH (NOLOCK)  
    WHERE NLR_Code = '701' AND Parts_Included=1 and PSE_Code in ({}) --and LEFT(Part_Ref, 2) in ('MN', 'BM') '''

reg_query = '''
    SELECT Movement_Date,  
        Cost_Value,  
        SLR_Document, 
        Record_Date,  
        WIP_Date_Created,  
        Quantity, 
        LTRIM(RTRIM(Part_Ref)) as Part_Ref,  
        Part_Desc  
    FROM [BI_CRP].[dbo].[PSE_Parts_Adjustments] WITH (NOLOCK)  
    WHERE NLR_Code = '701' and PSE_Code in ({}) --and LEFT(Part_Ref, 2) in ('MN', 'BM')'''

reg_autoline_clients = '''
    SELECT *  
    FROM [BI_CRP].dbo.[PSE_Mapping_Adjustments_SLR_Accounts] WITH (NOLOCK)  
    WHERE nlr_code = '701' '''

df_solve_query = '''
    SELECT * 
    FROM [BI_MLG].dbo.[{}] '''.format(sql_info['final_table'])

dim_product_group_dw = '''
    SELECT Product_Group_Code,
        Product_Group_Level_1_Code,
        Product_Group_Level_2_Code,
        PT_Product_Group_Level_1_Desc,
        PT_Product_Group_Level_2_Desc,
        PT_Product_Group_Desc
    FROM [BI_CRP].dbo.[PSE_Dim_Product_Groups_GSC] '''

stock_history_query = '''
    SELECT SO_Code, 
        LTRIM(RTRIM(Part_Ref)) as Part_Ref, 
        Part_Desc, 
        Quantity, 
        PVP_1, 
        Sales_Price, 
        [Date]
    FROM [DMS_MLG_01].[dbo].[DMS_CRP_04_SM_Parts_Stock]
      where  SO_Code in ({}) and NL_Company = '14' 
      and [Date] >= '{}' '''

regex_dict = {
    'bmw_part_ref_format': r'BM\d{2}\.\d{2}\.\d{1}\.\d{3}.\d{3}'
}

truncate_table_query = '''
    DELETE 
    FROM [BI_MLG].[dbo].[{}]
    WHERE PSE_Code = '{}' '''

bmw_original_oil_words = ['óleo', 'oleo', 'oil', 'óleos', 'oleos', 'oils']


column_sql_renaming = {
    'Part_Ref': 'Part_Ref',
    'Cost': 'Cost',
    'PVP': 'PVP',
    'Margin': 'Margin',
    'Last Stock': 'Last_Stock',
    'Last Stock Value': 'Last_Stock_Value',
    'Last Year Sales': 'Last_Year_Sales',
    'Last Year Sales Mean': 'Last_Year_Sales_Mean',
    'DaysToSell_1_Part_v2_mean': 'Days_To_Sell_Mean',
    'DaysToSell_1_Part_v2_median': 'Days_To_Sell_Median',
    'Group': 'Part_Ref_Group_Desc',
    'PSE_Code': 'PSE_Code',
}

columns_sql_solver_solution = [
    'Part_Ref', 'Part_Ref_Group_Desc', 'Quantity', 'Days_To_Sell', 'Above_Goal_Flag', 'Cost', 'PVP',
]

SPG_Parts_Stock_cols = '''        
       [NL_Company]
      ,[SO_Code]
      ,[Franchise_Code]
      ,[Product_Group]
      ,[Part_Ref]
      ,[Part_Desc]
      ,[Analysis_Code]
      ,[Model_Code]
      ,[Discount_Code]
      ,[PL_Account]
      ,[Created_Date]
      ,[Stock_Created_Date]
      ,[Last_Buy_Date]
      ,[Last_Sell_Date]
      ,[Last_Movement_Date]
      ,[Stock_Balance]
      ,[Quantity]
      ,[Reserved_Quantity]
      ,[Ordered_Quantity]
      ,[Average_Cost]
      ,[Supplier_Cost]
      ,[PVP_1]
      ,[Avg_Month_Demand]
      ,[Standard_Cost]
      ,[Sales_Price]
      ,[Last_Buy_Company_Date]
      ,[Last_Sell_Company_Date]
      ,[Parts_Included]'''

SPG_Parts_Stock_Warehouses_cols = '''
       [NL_Company]
      ,[SO_Code]
      ,[Franchise_Code]
      ,[Product_Group]
      ,[Part_Ref]
      ,[Part_Desc]
      ,[Analysis_Code]
      ,[Model_Code]
      ,[Discount_Code]
      ,[PL_Account]
      ,[Created_Date]
      ,[Last_Buy_Date]
      ,[Last_Sell_Date]
      ,[Last_Movement_Date]
      ,[Stock_Balance]
      ,[Quantity]
      ,[Average_Cost]
      ,[PVP_1]
      ,[Avg_Month_Demand]
      ,[Standard_Cost]
      ,[Sales_Price]
      ,[Warehouse_Code]
      ,[Warehouse]
      ,[Last_Buy_Company_Date]
      ,[Last_Sell_Company_Date]'''

ATL_Parts_Stock_cols = '''
       [SO_Code]
      ,[Franchise_code]
      ,[Product_Group]
      ,[Part_Ref]
      ,[Part_Desc]
      ,[Alternative_Part_Desc_1]
      ,[Alternative_Part_Desc_2]
      ,[Analysis_Code]
      ,[Model_Code]
      ,[Discount_Code]
      ,[Line_Class]
      ,[Sales_Code]
      ,[Quantity]
      ,[Reserved_Quantity]
      ,[Ordered_Quantity]
      ,[Average_Cost]
      ,[PVP_1]
      ,[PVP_2]
      ,[PVP_3]
      ,[Created_Date]
      ,[Last_Buy_Date]
      ,[Last_Sell_Date]
      ,[Last_Movement_Date]
      ,[Stock_Balance]
      ,[PL_Account]
      ,[Avg_Month_Demand]
      ,[ReOrder_Category]
      ,[Standard_Cost]
      ,[Discount_Group_Code]
      ,[Sales_Price]
      ,[NStock_Item]
      ,[Part_Location_1]
      ,[Part_Location_2]
      ,[Part_Location_3]
      ,[Part_Location_4]
      ,[Part_Location_5]
      ,[Part_Location_6]
      ,[Part_Location_7]
      ,[Part_Location_8]
      ,[Part_Location_9]
      ,[Part_LocationDesc_1]
      ,[Part_LocationDesc_2]
      ,[Part_LocationDesc_3]
      ,[Part_LocationDesc_4]
      ,[Part_LocationDesc_5]
      ,[Part_LocationDesc_6]
      ,[Part_LocationDesc_7]
      ,[Part_LocationDesc_8]
      ,[Part_LocationDesc_9]
      ,[Quantity_Location_1]
      ,[Quantity_Location_2]
      ,[Quantity_Location_3]
      ,[Quantity_Location_4]
      ,[Quantity_Location_5]
      ,[Quantity_Location_6]
      ,[Quantity_Location_7]
      ,[Quantity_Location_8]
      ,[Quantity_Location_9]
      ,[Surcharge]
      ,[Pending_Quantity]
      ,[Supplier_Cost]
      ,[Parts_Included]'''

