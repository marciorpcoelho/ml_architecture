import os
from py_dotenv import read_dotenv
dotenv_path = 'info.env'
read_dotenv(dotenv_path)

DSN_MLG = os.getenv('DSN_MLG')
DSN_PRD = os.getenv('DSN_Prd')
UID = os.getenv('UID')
PWD = os.getenv('PWD')

sql_info = {
    'database': 'BI_CRP',
    'database_histy': 'BI_DW_History',
    'database_final': 'BI_MLG',
    'sales_table': '',
    'purchases_table': '',
    'stock_table': '',
    'final_table': '',
}

project_id = 2259
pse_code = '0B'
urgent_purchases_flags = [4, 5]

log_files = {
    'full_log': 'logs/apv_baviera_2259.txt'
}


bmw_ta_mapping = {
    'BMW_Bonus_Group_1': ['1', '2'],  # Peças + Óleos
    'BMW_Bonus_Group_2': ['-'],  # Vendas de 4º Nível - No idea on this one...
    'BMW_Bonus_Group_3': ['Chemical'],  # Químicos
    'BMW_Bonus_Group_4': ['3', '5', '7'],  # Acessórios + Jantes + Lifestyle
    'BMW_Bonus_Group_5': ['8'],  # Pneus
    'Outros': ['4', '6', '9'],
}

# The cost goals for group 1 and 2, as well as the sale goals for groups 3, 4 and 5 are taken from the provided files.
# The remaining sale/cost goals are fluctuations of 5% from the previous values
# There currently are no goals for MINI groups
group_goals = {
    'BMW_Bonus_Group_1': [252051, 264654],  # Cost, Sales
    'BMW_Bonus_Group_2': [56055, 58858],  # Cost, Sales
    'BMW_Bonus_Group_3': [5075, 5329],  # Cost, Sales
    'BMW_Bonus_Group_4': [9539, 10016],  # Cost, Sales
    'BMW_Bonus_Group_5': [8675, 9109],  # Cost, Sales
    'MINI_Bonus_Group_1': [0, 0],
    'MINI_Bonus_Group_2': [0, 0],
    'MINI_Bonus_Group_3': [0, 0],
    'MINI_Bonus_Group_4': [0, 0],
}

mini_ta_mapping = {
    'MINI_Bonus_Group_1': ['1', '2'],  # Peças + Óleos
    'MINI_Bonus_Group_2': ['-'],  # MINI Regeneration - No idea on this one...
    'MINI_Bonus_Group_3': ['3', '5', '7'],  # Acessórios + Jantes + Lifestyle
    'MINI_Bonus_Group_4': ['8'],  # Pneus
    'Outros': ['4', '6', '9'],
}


sales_query = '''
    Select Sales.SLR_Document_Date,  
        Sales.Movement_Date,  
        Sales.SLR_Document, 
        Sales.Part_Ref,  
        Sales.Part_Desc, 
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
    WHERE Client_Id = 3  and Parts_Included = 1 AND NLR_Code = '701' and PSE_Code = '{}' and LEFT(Part_Ref, 2) in ('MN', 'BM')  
    GROUP BY Sales.SLR_Document_Date,  
        Sales.Movement_Date,  
        Sales.SLR_Document, 
        Sales.Part_Ref,  
        Sales.Part_Desc, 
        Sales.Client_Id, 
        Sales.NLR_Code, 
        Sales.PSE_Code, 
        Sales.WIP_Number, 
        Sales.WIP_Date_Created   
    union all  
    Select Sales.SLR_Document_Date,  
        Sales.Movement_Date,  
        Sales.SLR_Document, 
        Sales.Part_Ref,  
        Sales.Part_Desc,  
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
    WHERE Client_Id = 3  and Parts_Included = 1 AND NLR_Code = '701' and PSE_Code = '{}' and LEFT(Part_Ref, 2) in ('MN', 'BM') 
    GROUP BY Sales.SLR_Document_Date,  
        Sales.Movement_Date,  
        Sales.SLR_Document, 
        Sales.Part_Ref,  
        Sales.Part_Desc, 
        Sales.Client_Id, 
        Sales.NLR_Code, 
        Sales.PSE_Code, 
        Sales.WIP_Number,  
        Sales.WIP_Date_Created '''.format(pse_code, pse_code)

purchases_query = '''
    SELECT Movement_Date,  
        PLR_Document,  
        Quantity,  
        Cost_Value, 
        Part_Ref, 
        Part_Desc, 
        Product_Group_DW,  
        Order_Type_DW,  
        WIP_Number, 
        WIP_Date_Created  
    FROM [BI_CRP].[dbo].[PSE_parts_purchase] WITH (NOLOCK)  
    WHERE NLR_Code = '701'  AND Parts_Included=1 and PSE_Code = '{}' and LEFT(Part_Ref, 2) in ('MN', 'BM')  
    union all  
    SELECT Movement_Date,  
        PLR_Document, 
        Quantity,  
        Cost_Value, 
        Part_Ref, 
        Part_Desc, 
        Product_Group_DW,  
        Order_Type_DW,  
        WIP_Number, 
        WIP_Date_Created  
    FROM [BI_DW_History].[dbo].[PSE_parts_purchase] WITH (NOLOCK)  
    WHERE NLR_Code = '701' AND Parts_Included=1 and PSE_Code = '{}' and LEFT(Part_Ref, 2) in ('MN', 'BM') '''.format(pse_code, pse_code)

stock_query = '''
    SELECT Part_Ref, 
        Part_Desc, 
        Product_Group_DW, 
        Quantity, 
        PSE_Code, 
        Record_Date, 
        Stock_Age_Days as Last_Enter,  
        Stock_Age2_Days as Last_Exit  
    FROM [BI_CRP].[dbo].[PSE_Fact_BI_Parts_Stock_Month] WITH (NOLOCK)  
    WHERE NLR_Code = '701' and Parts_Included=1 and PSE_Code = '{}' and LEFT(Part_Ref, 2) in ('MN', 'BM')  and Warehouse_Code <> -1  
    union all  
    SELECT Part_Ref, 
        Part_Desc, 
        Product_Group_DW, Quantity, 
        PSE_Code, 
        Record_Date, 
        Stock_Age_Days as Last_Enter, 
        Stock_Age2_Days as Last_Exit  
    FROM [BI_DW_History].[dbo].[PSE_Fact_BI_Parts_Stock_Month] WITH (NOLOCK)  
    WHERE NLR_Code = '701' and Parts_Included=1 and PSE_Code = '{}' and LEFT(Part_Ref, 2) in ('MN', 'BM')  and Warehouse_Code <> -1 '''.format(pse_code, pse_code)

reg_query = '''
    SELECT Movement_Date,  
        Cost_Value,  
        SLR_Document, 
        Record_Date,  
        WIP_Date_Created,  
        Quantity,Part_Ref,  
        Part_Desc  
    FROM [BI_CRP].[dbo].[PSE_Parts_Adjustments] WITH (NOLOCK)  
    WHERE NLR_Code = '701' and PSE_Code = '{}' and LEFT(Part_Ref, 2) in ('MN', 'BM')'''.format(pse_code)

reg_autoline_clients = '''
    SELECT *  
    FROM [BI_CRP].dbo.[PSE_Mapping_Adjustments_SLR_Accounts] WITH (NOLOCK)  
    WHERE nlr_code = '701' '''
