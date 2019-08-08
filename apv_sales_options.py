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
