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
elif 'posix' in os.name:
    DSN = os.getenv('DSN_Prd_Linux')
    DSN_SRV3_PRD = os.getenv('DSN_SRV3_Prd_Linux')
    DSN_MLG_PRD = os.getenv('DSN_MLG_Prd_Linux')
UID = os.getenv('UID')
PWD = os.getenv('PWD')

project_id = 2406
update_frequency_days = 0
api_backend_loc = 'optimizations/vhe_hyundai_honda/'
# stock_days_threshold = 150  # DaysInStock_Global
stock_days_threshold = [90, 120, 150, 180, 270, 365]
margin_threshold = "nan"  # Currently there is no threshold;

models = ['rf', 'lgb', 'xgb', 'ridge', 'll_cv', 'elastic_cv', 'svr']
target = 'DaysInStock_Global'
configuration_parameters = ['PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc', 'PT_PDB_Exterior_Color_Desc', 'PT_PDB_Interior_Color_Desc']
# configuration_parameters = ['PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc', 'PT_PDB_Painting_Type_Desc', 'PT_PDB_Bodywork_Desc']
client_lvl_cols = ['Customer_Group_Desc', 'NDB_VATGroup_Desc', 'VAT_Number_Display', 'NDB_Contract_Dealer_Desc', 'NDB_VHE_PerformGroup_Desc', 'NDB_VHE_Team_Desc', 'Customer_Display']
client_lvl_cols_renamed = ['Tipo Cliente', 'Agrupamento NIF', 'NIF - Nome', 'Contrato Concessionário', 'Agrupamento Performance', 'Equipa de Vendas', 'Cliente Morada']
range_dates = ['PDB_Start_Order_Date', 'PDB_End_Order_Date']

metric, metric_threshold = 'R2', 0.50  # The metric to compare on the final models and the minimum threshold to consider;
k, gridsearch_score = 10, 'neg_mean_squared_error'  # Stratified Cross-Validation number of Folds and the Metric on which to optimize GridSearchCV
gamas_match_temp_file = base_path + '/dbs/gamas_match_{}.xlsx'
documentation_url_solver_app = 'https://gruposalvadorcaetano.sharepoint.com/:b:/s/rigor/6825_DGAA/ETn-fkuxzHVJj1L5KdQrUeUBMWxtiPU8wkEH9CxXRxsoNg?e=Ja90Mb'
documentation_url_gamas_match_app = 'https://gruposalvadorcaetano.sharepoint.com/:b:/s/rigor/6825_DGAA/EUnbjdE09-lBtoa-0BKWWToBtdfHCux_E3nAmdstl8lRxw?e=80NEuN'

sql_info = {
    'database_source': 'BI_DTR',
    'database_final': 'BI_MLG',
    'product_db': 'VHE_Dim_VehicleData_DTR',
    'sales': 'VHE_Fact_BI_Sales_DTR',
    'stock': 'VHE_Fact_BI_Stock_DTR',
    'final_table': 'VHE_Fact_MLG_Sales_DTR',
    'optimization_solution_table': 'VHE_Fact_PA_OrderOptimization_Solver_Optimization',
    'commercial_version_matching': 'VHE_MapDMS_Vehicle_Commercial_Versions_DTR',
    'proposals_table': 'VHE_Fact_DW_HPK_Proposals_DTR',
    'proposals_view': 'View_VHE_Fact_PA_OrderOptimization_HPK_Proposals_Old',
    'stock_view': 'View_VHE_Fact_PA_OrderOptimization_Stock_Old',
    'unit_count_number_history': 'LOG_Project_Units_Count_History',
    'new_score_streamlit_view': 'VHE_Fact_PA_OrderOptimization_Streamlit',
    'sales_plan_aux': 'VHE_Setup_Sales_Plan_Aux_DTR',
}

log_files = {
    'full_log': 'logs/optionals_hyundai.txt'
}

score_weights = {
    'Avg_DaysInStock_Global_normalized': 0.18,
    'TotalGrossMarginPerc_normalized': 0.05,
    'MarginRatio_normalized': 0.14,
    'Sum_Qty_CHS_normalized': 0.18,
    'Proposals_VDC_normalized': 0.11,
    'Stock_OC_Diff_normalized': 0.17,
    'NEDC_normalized': 0.11,
}

cols_to_normalize = [
    'Avg_DaysInStock_Global',
    'TotalGrossMarginPerc',
    'MarginRatio',
    'Sum_Qty_CHS',
    'Proposals_VDC',
    'Stock_OC_Diff',
    'NEDC'
]

# Reversed columns normalization (the greater the normalized value, the worst)
reverse_normalization_cols = [
    'Avg_DaysInStock_Global',
    'NEDC'
]


column_translate_dict = {
    'PT_PDB_Model_Desc': 'Modelo',
    'PT_PDB_Engine_Desc': 'Motorização',
    'PT_PDB_Transmission_Type_Desc': 'Transmissão',
    'PT_PDB_Version_Desc': 'Versão',
    'PT_PDB_Exterior_Color_Desc': 'Cor Exterior',
    'PT_PDB_Interior_Color_Desc': 'Cor Interior',
    'Customer_Group_Desc': 'Tipo Cliente',
    'NDB_VATGroup_Desc': 'Agrupamento NIF',
    'VAT_Number_Display': 'NIF - Nome',
    'NDB_Contract_Dealer_Desc': 'Contrato Concessionário',
    'NDB_VHE_PerformGroup_Desc': 'Agrupamento Performance',
    'NDB_VHE_Team_Desc': 'Equipa de Vendas',
    'Customer_Display': 'Cliente Morada',
    'Date': 'Data',
    'Quantity_Sold': '#Veículos Vendidos',
    'Average_Score_Euros': 'Score (€)',
    'Measure_9': 'Custo Base',
    'Measure_10': 'Custo Base - Outros',
    'number_prev_sales': '#Vendas Anteriores',
    'Quantity:': 'Sug.Encomenda',
    'Proposals_Count': 'Propostas Entregues',
    'Stock_Count': 'Em Stock',
    'Proposals_Count_VDC': 'Propostas Entregues',
    'Stock_Count_VDC': 'Em Stock',
    'Avg_DaysInStock_Global': 'Médias Dias em Stock',
    'Avg_DaysInStock_Global_normalized': 'Médias Dias em Stock (score)',
    'Sum_Qty_CHS_normalized': '#Veículos Vendidos (score)',
    'Proposals_VDC': '#Propostas',
    'Proposals_VDC_normalized': '#Propostas (score)',
    'Margin_HP': 'Margem HP',
    'TotalGrossMarginPerc': 'Margem Global',
    'TotalGrossMarginPerc_normalized': 'Margem Global (score)',
    'MarginRatio': 'Rácio Margem (HP/Global)',
    'MarginRatio_normalized': 'Rácio Margem (score)',
    'OC': 'Obj. Cobertura Stock (unidades)',
    'Stock_VDC': '#Stock',
    'Stock_OC_Diff': 'O.C. vs Stock',
    'Stock_OC_Diff_normalized': 'O.C. vs Stock (score)',
    'NEDC': 'Co2 (NEDC)',
    'NEDC_normalized': 'Co2 (NEDC) (score)',
    "PDB_Start_Order_Date": "Início Gama",
    "PDB_End_Order_Date": "Final Gama",
    "Chassis_Number": "Chassis",
    "Registration_Number": "Matrícula",
    "DaysInStock_Global": "Dias em Stock",
    "TotalGrossMargin": "Margem Global",
    "Created_Time": "Dt.Criação",
    "Dealer_Desc": "Concessão",
    "Team_Desc": "Equipa de Vendas",
    "Location_Desc": "Localização",
    "Production_Date": "Dt.Produção",
    "Purchase_Date": "Dt.Compra",
    "Ship_Arrival_Date": "Dt.Chegada Embarcação",
    "Sales_Plan_Period": "Período Plano de Vendas",
    "WLTP_CO2": "Co2 (WLTP)",
    "NEDC_CO2": "Co2 (NEDC)",
    "Quantity": "Quantidade",
    'Max_Qty_Per_Sales_Plan_Period': 'Quantidade Máx por Período',
    'Model_Code': 'Model Code',
    'PT_Stock_Status_Level_1_Desc': 'Estado (1º Nível)',
    'PT_Stock_Status_Desc': 'Estado (2º Nível)',
    'SLR_Document_Date_CHS': 'Dt.Venda',
    'Quantity_CHS': 'Quantidade'
}

col_color_dict = {
    "Sug.Encomenda": 'Beige',
    "Motorização": 'FloralWhite',
    "Transmissão": 'FloralWhite',
    "Versão": 'FloralWhite',
    "Cor Exterior": 'FloralWhite',
    "Cor Interior": 'FloralWhite',
    "Médias Dias em Stock": 'LightGray',
    "Médias Dias em Stock (score)": 'LightGray',
    "#Veículos Vendidos": 'Lavender',
    "#Veículos Vendidos (score)": 'Lavender',
    "#Propostas": 'LightGrey',
    "#Propostas (score)": 'LightGrey',
    "Margem HP": 'Lavender',
    "Margem Global": 'Lavender',
    "Margem Global (score)": 'Lavender',
    "Rácio Margem (HP/Global)": 'LightSlateGray',
    "Rácio Margem (score)": 'LightSlateGray',
    "Obj. Cobertura Stock (unidades)": 'LightSteelBlue',
    "#Stock": 'LightSteelBlue',
    "O.C. vs Stock": 'LightSteelBlue',
    "O.C. vs Stock (score)": 'LightSteelBlue',
    "Co2 (NEDC)": 'SlateGrey',
    'Co2 (NEDC) (score)': 'SlateGrey',
    "Score": 'LightBlue',
}

col_decimals_place_dict = {
    "Sug.Encomenda": '{:.0f}',
    "Médias Dias em Stock": '{:.0f}',
    "Médias Dias em Stock (score)": '{:.2f}',
    "#Veículos Vendidos": '{:.0f}',
    "#Veículos Vendidos (score)": '{:.2f}',
    "#Propostas": '{:.0f}',
    "#Propostas (score)": '{:.2f}',
    "Margem HP": '{:.1%}',
    "Margem Global": '{:.1%}',
    "Margem Global (score)": '{:.2f}',
    "Rácio Margem (HP/Global)": '{:.2f}',
    "Rácio Margem (score)": '{:.2f}',
    "Obj. Cobertura Stock (unidades)": '{:.1f}',
    "#Stock": '{:.0f}',
    "O.C. vs Stock": '{:.0f}',
    "O.C. vs Stock (score)": '{:.2f}',
    "Co2 (NEDC)": '{:.1f}',
    'Co2 (NEDC) (score)': '{:.2f}',
    "Score": '{:.3f}',
}

nlr_code_desc = {
    'Hyundai': 702,
    'Honda': 706
}

sales_query_filtered = '''
        SELECT *
        FROM [BI_DTR].dbo.[VHE_Fact_BI_Sales_DTR] WITH (NOLOCK)
        where Registration_Flag = '1' and Chassis_Flag = '1' and VehicleData_Code <> '1'
            and Sales_Type_Code_DMS in ('RAC', 'STOCK', 'UVENDA', 'VENDA') and Sales_Type_Dealer_Code <> 'Demo'
        UNION ALL
        SELECT *
        FROM [BI_DTR_History].dbo.[VHE_Fact_BI_Sales_DTR] WITH (NOLOCK) 
        where Registration_Flag = '1' and Chassis_Flag = '1' and VehicleData_Code <> '1'
            and Sales_Type_Code_DMS in ('RAC', 'STOCK', 'UVENDA', 'VENDA') and Sales_Type_Dealer_Code <> 'Demo' '''


sales_query = '''
        SELECT [Client_Id]
      ,[NLR_Code]
      ,[Environment]
      ,[DMS_Type_Code]
      ,[Value_Type_Code]
      ,[Record_Type]
      ,[Vehicle_ID]
      ,[SLR_Document]
      ,[SLR_Document_Account]
      ,[VHE_Type_Orig]
      ,[VHE_Type]
      ,[VHE_Type_Detail]
      ,[Chassis_Number]
      ,[Registration_Number]
      ,[NLR_Posting_Date]
      ,[SLR_Document_Category]
      ,[Chassis_Flag]
      ,[SLR_Document_Date_CHS]
      ,[SLR_Document_Period_CHS]
      ,[SLR_Document_Year_CHS]
      ,[SLR_Document_CHS]
      ,[SLR_Document_Type_CHS]
      ,[SLR_Account_CHS]
      ,[SLR_Account_CHS_Key]
      ,[Quantity_CHS]
      ,[Registration_Flag]
      ,[Analysis_Date_RGN]
      ,[Analysis_Period_RGN]
      ,[Analysis_Year_RGN]
      ,[SLR_Document_Date_RGN]
      ,[SLR_Document_RGN]
      ,[SLR_Document_Type_RGN]
      ,[SLR_Account_RGN]
      ,[SLR_Account_RGN_Key]
      ,[Quantity_RGN]
      ,[Product_Code]
      ,[Sales_Type_Code_DMS]
      ,[Sales_Type_Code]
      ,[Location_Code]
      ,[VehicleData_Key]
      ,[VehicleData_Code]
      ,[Vehicle_Code]
      ,[PDB_Vehicle_Type_Code_DMS]
      ,[Vehicle_Type_Code]
      ,[PDB_Fuel_Type_Code_DMS]
      ,[Fuel_Type_Code]
      ,[PDB_Transmission_Type_Code_DMS]
      ,[Transmission_Type_Code]
      ,[Vehicle_Area_Code]
      ,[Dispatch_Type_Code]
      ,[Sales_Status_Code]
      ,[Ship_Arrival_Date]
      ,[Registration_Request_Date]
      ,[Registration_Date]
      ,[DaysInStock_Distributor]
      ,[Stock_Age_Distributor_Code]
      ,[DaysInStock_Dealer]
      ,[Stock_Age_Dealer_Code]
      ,[DaysInStock_Global]
      ,[Stock_Age_Global_Code]
      ,[Immobilized_Number]
      ,[SLR_Account_Dealer_Code]
      ,[Salesman_Dealer_Code]
      ,[Sales_Type_Dealer_Code]
      ,[Measure_1]
      ,[Measure_2]
      ,[Measure_3]
      ,[Measure_4]
      ,[Measure_5]
      ,[Measure_6]
      ,[Measure_7]
      ,[Measure_8]
      ,[Measure_9]
      ,[Measure_10]
      ,[Measure_11]
      ,[Measure_12]
      ,[Measure_13]
      ,[Measure_14]
      ,[Measure_15]
      ,[Measure_16]
      ,[Measure_17]
      ,[Measure_18]
      ,[Measure_19]
      ,[Measure_20]
      ,[Measure_21]
      ,[Measure_22]
      ,[Measure_23]
      ,[Measure_24]
      ,[Measure_25]
      ,[Measure_26]
      ,[Measure_27]
      ,[Measure_28]
      ,[Measure_29]
      ,[Measure_30]
      ,[Measure_31]
      ,[Measure_32]
      ,[Measure_33]
      ,[Measure_34]
      ,[Measure_35]
      ,[Measure_36]
      ,[Measure_37]
      ,[Measure_38]
      ,[Measure_39]
      ,[Measure_40]
      ,[Measure_41]
      ,[Measure_42]
      ,[Measure_43]
      ,[Measure_44]
      ,[Measure_45]
      ,[Measure_46]
      ,[Currency_Rate]
      ,[Currency_Rate2]
      ,[Currency_Rate3]
      ,[Currency_Rate4]
      ,[Currency_Rate5]
      ,[Currency_Rate6]
      ,[Currency_Rate7]
      ,[Currency_Rate8]
      ,[Currency_Rate9]
      ,[Currency_Rate10]
      ,[Currency_Rate11]
      ,[Currency_Rate12]
      ,[Currency_Rate13]
      ,[Currency_Rate14]
      ,[Currency_Rate15]
      ,[Record_Date]
        FROM [BI_DTR].dbo.[VHE_Fact_BI_Sales_DTR] WITH (NOLOCK) 
        UNION ALL 
        SELECT [Client_Id]
      ,[NLR_Code]
      ,[Environment]
      ,[DMS_Type_Code]
      ,[Value_Type_Code]
      ,[Record_Type]
      ,[Vehicle_ID]
      ,[SLR_Document]
      ,[SLR_Document_Account]
      ,[VHE_Type_Orig]
      ,[VHE_Type]
      ,[VHE_Type_Detail]
      ,[Chassis_Number]
      ,[Registration_Number]
      ,[NLR_Posting_Date]
      ,[SLR_Document_Category]
      ,[Chassis_Flag]
      ,[SLR_Document_Date_CHS]
      ,[SLR_Document_Period_CHS]
      ,[SLR_Document_Year_CHS]
      ,[SLR_Document_CHS]
      ,[SLR_Document_Type_CHS]
      ,[SLR_Account_CHS]
      ,[SLR_Account_CHS_Key]
      ,[Quantity_CHS]
      ,[Registration_Flag]
      ,[Analysis_Date_RGN]
      ,[Analysis_Period_RGN]
      ,[Analysis_Year_RGN]
      ,[SLR_Document_Date_RGN]
      ,[SLR_Document_RGN]
      ,[SLR_Document_Type_RGN]
      ,[SLR_Account_RGN]
      ,[SLR_Account_RGN_Key]
      ,[Quantity_RGN]
      ,[Product_Code]
      ,[Sales_Type_Code_DMS]
      ,[Sales_Type_Code]
      ,[Location_Code]
      ,[VehicleData_Key]
      ,[VehicleData_Code]
      ,[Vehicle_Code]
      ,[PDB_Vehicle_Type_Code_DMS]
      ,[Vehicle_Type_Code]
      ,[PDB_Fuel_Type_Code_DMS]
      ,[Fuel_Type_Code]
      ,[PDB_Transmission_Type_Code_DMS]
      ,[Transmission_Type_Code]
      ,[Vehicle_Area_Code]
      ,[Dispatch_Type_Code]
      ,[Sales_Status_Code]
      ,[Ship_Arrival_Date]
      ,[Registration_Request_Date]
      ,[Registration_Date]
      ,[DaysInStock_Distributor]
      ,[Stock_Age_Distributor_Code]
      ,[DaysInStock_Dealer]
      ,[Stock_Age_Dealer_Code]
      ,[DaysInStock_Global]
      ,[Stock_Age_Global_Code]
      ,[Immobilized_Number]
      ,[SLR_Account_Dealer_Code]
      ,[Salesman_Dealer_Code]
      ,[Sales_Type_Dealer_Code]
      ,[Measure_1]
      ,[Measure_2]
      ,[Measure_3]
      ,[Measure_4]
      ,[Measure_5]
      ,[Measure_6]
      ,[Measure_7]
      ,[Measure_8]
      ,[Measure_9]
      ,[Measure_10]
      ,[Measure_11]
      ,[Measure_12]
      ,[Measure_13]
      ,[Measure_14]
      ,[Measure_15]
      ,[Measure_16]
      ,[Measure_17]
      ,[Measure_18]
      ,[Measure_19]
      ,[Measure_20]
      ,[Measure_21]
      ,[Measure_22]
      ,[Measure_23]
      ,[Measure_24]
      ,[Measure_25]
      ,[Measure_26]
      ,[Measure_27]
      ,[Measure_28]
      ,[Measure_29]
      ,[Measure_30]
      ,[Measure_31]
      ,[Measure_32]
      ,[Measure_33]
      ,[Measure_34]
      ,[Measure_35]
      ,[Measure_36]
      ,[Measure_37]
      ,[Measure_38]
      ,[Measure_39]
      ,[Measure_40]
      ,[Measure_41]
      ,[Measure_42]
      ,[Measure_43]
      ,[Measure_44]
      ,[Measure_45]
      ,[Measure_46]
      ,[Currency_Rate]
      ,[Currency_Rate2]
      ,[Currency_Rate3]
      ,[Currency_Rate4]
      ,[Currency_Rate5]
      ,[Currency_Rate6]
      ,[Currency_Rate7]
      ,[Currency_Rate8]
      ,[Currency_Rate9]
      ,[Currency_Rate10]
      ,[Currency_Rate11]
      ,[Currency_Rate12]
      ,[Currency_Rate13]
      ,[Currency_Rate14]
      ,[Currency_Rate15]
      ,[Record_Date]
        FROM [BI_DTR_History].dbo.[VHE_Fact_BI_Sales_DTR] WITH (NOLOCK)'''

stock_query = '''
        select *
        from [BI_DTR].dbo.[VHE_Fact_BI_Stock_DTR] WITH (NOLOCK)'''

product_db_query = '''
        SELECT [VehicleData_Code]
      ,[VehicleData_Key]
      ,[PDB_Franchise_Code]
      ,[Factory_Model_Code]
      ,[Factory_Vehicle_Option_Code]
      ,[Factory_Exterior_Color_Code]
      ,[Factory_Interior_Color_Code]
      ,[Local_Vehicle_Option_Code]
      ,[PDB_Model_Key]
      ,[PDB_Serie_Key]
      ,[PDB_Bodywork_Key]
      ,[PDB_Version_Key]
      ,[PDB_Engine_Key]
      ,[Local_Model_Code]
      ,[PDB_Serie_Code]
      ,[PDB_Model_Code]
      ,[PDB_Version_Code]
      ,[PDB_Exterior_Color_Code]
      ,[PDB_Interior_Color_Code]
      ,[PDB_Painting_Type_Code]
      ,[PDB_Bodywork_Code]
      ,[PDB_Engine_Code]
      ,[PDB_Transmission_Type_Code]
      ,[PDB_Fuel_Type_Code]
      ,[PDB_Vehicle_Type_Code]
      ,[PDB_Commercial_Version_Code]
      ,[PT_PDB_Franchise_Desc]
      ,[PT_PDB_Model_Desc]
      ,[PT_PDB_Serie_Desc]
      ,[PT_PDB_Bodywork_Desc]
      ,[PT_PDB_Version_Desc]
      ,[PT_PDB_Version_Desc_New]
      ,[PT_PDB_Engine_Desc]
      ,[PT_PDB_Engine_Desc_New]
      ,[PT_PDB_Exterior_Color_Desc]
      ,[PT_PDB_Interior_Color_Desc]
      ,[PT_PDB_Painting_Type_Desc]
      ,[PT_PDB_Transmission_Type_Desc]
      ,[PT_PDB_Fuel_Type_Desc]
      ,[PT_PDB_Vehicle_Type_Desc]
      ,[PT_PDB_Commercial_Version_Desc]
      ,[PT_PDB_Commercial_Version_Desc_New]
      ,[PDB_Total_Tara]
      ,[PDB_Displacement]
      ,[PDB_Combined_CO2]
      ,[PDB_Combined_Fuel_Consumption]
      ,[PDB_Tires_Number]
      ,[PDB_Engine_Oil]
      ,[PDB_GearBox_Oil]
      ,[Record_Date]
      ,[Last_Modified_Date]
      ,[PDB_Start_Order_Date]
      ,[PDB_End_Order_Date]
        FROM [BI_DTR].dbo.[VHE_Dim_VehicleData_DTR] WITH (NOLOCK)'''

dealers_query = '''
        select "SLR_Dim_Dealers_DTR_VHE"."Client_Id" AS "Client_Id", "SLR_Dim_Dealers_DTR_VHE"."Environment"
        AS "Environment", "SLR_Dim_Dealers_DTR_VHE"."NDB_Dealer_Code" AS "NDB_Dealer_Code",
        "SLR_Dim_Dealers_DTR_VHE"."SLR_Account" AS "SLR_Account", "SLR_Dim_Dealers_DTR_VHE"."Customer_Display"
        AS "Customer_Display", "SLR_Dim_Dealers_DTR_VHE"."NDB_VATGroup_Code" AS "NDB_VATGroup_Code",
        "SLR_Dim_Dealers_DTR_VHE"."NDB_VATGroup_Desc" AS "NDB_VATGroup_Desc", "SLR_Dim_Dealers_DTR_VHE"."NDB_VAT_Number_Code"
        AS "NDB_VAT_Number_Code", "SLR_Dim_Dealers_DTR_VHE"."NDB_VAT_Number" AS "NDB_VAT_Number",
        "SLR_Dim_Dealers_DTR_VHE"."NDB_Contract_Dealer_Code" AS "NDB_Contract_Dealer_Code",
        "SLR_Dim_Dealers_DTR_VHE"."NDB_Contract_Dealer_Desc" AS "NDB_Contract_Dealer_Desc",
        "SLR_Dim_Dealers_DTR_VHE"."NDB_VHE_PerformGroup_Code" AS "NDB_VHE_PerformGroup_Code",
        "SLR_Dim_Dealers_DTR_VHE"."NDB_VHE_PerformGroup_Desc" AS "NDB_VHE_PerformGroup_Desc",
        "SLR_Dim_Dealers_DTR_VHE"."NDB_PSE_PerformGroup_Code" AS "NDB_PSE_PerformGroup_Code",
        "SLR_Dim_Dealers_DTR_VHE"."NDB_PSE_PerformGroup_Desc" AS "NDB_PSE_PerformGroup_Desc",
        "SLR_Dim_Dealers_DTR_VHE"."NDB_VHE_AreaManager_Code" AS "NDB_VHE_AreaManager_Code",
        "SLR_Dim_Dealers_DTR_VHE"."NDB_VHE_AreaManager_Desc" AS "NDB_VHE_AreaManager_Desc",
        "SLR_Dim_Dealers_DTR_VHE"."NDB_PSE_AreaManager_Code" AS "NDB_PSE_AreaManager_Code",
        "SLR_Dim_Dealers_DTR_VHE"."NDB_PSE_AreaManager_Desc" AS "NDB_PSE_AreaManager_Desc",
        "SLR_Dim_Dealers_DTR_VHE"."NDB_VHE_Team_Code" AS "NDB_VHE_Team_Code", "SLR_Dim_Dealers_DTR_VHE"."NDB_VHE_Team_Desc"
        AS "NDB_VHE_Team_Desc", "SLR_Dim_Dealers_DTR_VHE"."NDB_PSE_Team_Code" AS "NDB_PSE_Team_Code",
        "SLR_Dim_Dealers_DTR_VHE"."NDB_PSE_Team_Desc" AS "NDB_PSE_Team_Desc", "SLR_Dim_Dealers_DTR_VHE"."NDB_Headquarters"
        AS "NDB_Headquarters", "SLR_Dim_Dealers_DTR_VHE"."Record_Date" AS "Record_Date",
        "SLR_Dim_Dealers_DTR_VHE"."Last_Modified_Date" AS "Last_Modified_Date", "SLR_Dim_Dealers_DTR_VHE"."Customer_Group_Code"
        AS "Customer_Group_Code", "SLR_Dim_Dealers_DTR_VHE"."VAT_Number_Display" AS "VAT_Number_Display",
        "SLR_Dim_Dealers_DTR_VHE"."SLR_Account_Key" AS "SLR_Account_CHS_Key", "SLR_Dim_Dealers_DTR_VHE"."SLR_AccountGroup_Code"
        AS "SLR_AccountGroup_Code", "SLR_Dim_Dealers_DTR_VHE"."NDB_Dealer_Desc" AS "NDB_Dealer_Desc",
        "SLR_Dim_Dealers_DTR_VHE"."SLR_Account_Invoice" AS "SLR_Account_Invoice"
         from "SLR_Dim_Dealers_DTR"
        "SLR_Dim_Dealers_DTR_VHE"
         where "SLR_Dim_Dealers_DTR_VHE"."Record_Type" <> 3 '''

customer_group_query = '''
        select "SLR_Dim_Customer_Groups_DTR"."Customer_Group_Code" AS "Customer_Group_Code",
        "SLR_Dim_Customer_Groups_DTR"."EN_Customer_Group_Desc" AS "Customer_Group_Desc",
        "SLR_Dim_Customer_Groups_DTR"."Customer_Group_Display_Order" AS "Customer_Group_Display_Order"
        from "BI_DTR"."dbo"."SLR_Dim_Customer_Groups_DTR" "SLR_Dim_Customer_Groups_DTR"  WITH (NOLOCK)'''


date_columns = ['NLR_Posting_Date', 'SLR_Document_Date_CHS', 'Analysis_Date_RGN', 'SLR_Document_Date_RGN', 'Record_Date', 'Registration_Request_Date']


# Motorização
motor_translation = {
    '1.0i/g': ['1.0 lpgi'],
    '1.0i': ['1.0 t-gdi', '1.0i', '1.0l', '1.0 mpi'],
    '1.1d': ['1.1 crdi'],
    '1.2i': ['1.2i', '1.2 mpi'],
    '1.3i': ['1.3l'],
    '1.4d': ['1.4 crdi'],
    '1.4i': ['1.4 t-gdi'],
    '1.5i': ['1.5l'],
    # '1.6': [],
    '1.6d': ['1.6l', '1.6 crdi'],
    '1.6i': ['1.6 t-gdi', '1.6 gdi'],
    '1.7d': ['1.7 crdi'],
    '2.0d': ['2.0 crdi'],
    '2.0i': ['2.0l', '2.0 t-gdi'],
    '2.2d': ['2.2 crdi'],
    '2.5d': ['2.5 crdi'],
    'eletrico': ['motor elétrico'],
    'NÃO_PARAMETRIZADOS': [],
}

# v1
motor_grouping = {
    '1.0': ['1.0i', '1.0i/g'],
    '1.1/1.2': ['1.1d', '1.2i'],
    '1.3/1.4/1.5': ['1.3i', '1.4i', '1.4d', '1.5i'],
    '1.6/1.7': ['1.6i', '1.6d', '1.7d'],
    '2.0+': ['2.0i', '2.0d', '2.2d', '2.5d'],
    'Elétrico': ['eletrico'],
    'Outros': [],
}


# Transmissão
transmission_translation = {
    'Manual': ['manual 6 velocidades', 'manual 5 velocidades', 'mt'],
    'Auto': ['at', 's/info', 'caixa automática 4 velocidades', 'caixa automática 6 velocidades', 'caixa automática 8 velocidades'],
    'CVT': ['cvt'],
    'DCT': ['dct', 'automática de dupla embraiagem de 6 velocidades (6 dct)', 'automática de dupla embraiagem de 7 velocidades (7 dct)'],
}

# v1
transmission_grouping = {
    'Manual': ['Manual'],
    'Auto/CVT/DCT': ['Auto', 'CVT', 'DCT'],
}

# Versão
# version_translation = {
#     'Access': ['access', 'access plus', 'access my17'],
#     'Comfort': ['comfort ', 'comfort', 'comfort + connect navi ', 'comfort', 'van 3 lugares', 'comfort my19', 'comfort navi', 'blue comfort my17', 'blue comfort hp my17', 'comfort + navi', 'comfort + connect navi', 'blue comfort', 'comfort my19\'5', 'comfort my20', 'blue comfort hp my16'],
#     'Creative': ['creative plus'],
#     'Dynamic': ['dynamic', 'dynamic + connect navi'],
#     'Elegance': ['elegance navi', '1.5 i-vtec turbo cvt elegance navi', '1.6 i-dtec turbo elegance navi', 'elegance ', 'elegance + connect navi ', 'elegance plus + connect n', 'elegance', 'elegance + connect navi', '1.5 i-vtec turbo elegance'],
#     'EV': ['ev'],
#     'Executive': ['executive ', 'executive', 'executive premium', '1.5 i-vtec turbo executive', '1.5 i-vtec turbo cvt executive', '1.6 i-dtec turbo executive', 'executive', 'executive my19', 'executive my20', 'executive my19\'5'],
#     'GO': ['go', 'go+', 'go!', 'go!+'],
#     'HEV': ['hev'],
#     'Launch': ['launch edition'],
#     'Lifestyle': ['lifestyle', 'lifestyle + navi', 'lifestyle + connect navi'],
#     'Performance': ['performance pack'],
#     'PHEV': ['phev'],
#     'Premium': ['premium', 'premium my19', 'premium my19 + pack pele', 'premium my20', 'premium + pack pele + pack style my19\'5', 'premium + pack pele + style plus my19\'5', 'premium + pack pele my19\'5', 'premium my19\'5'],
#     'Prestige': ['prestige'],
#     'Pro': ['pro edition'],
#     'Sport': ['sport plus', 'sport', 'turbo sport'],
#     'Style': ['style', 'comfort my18', 'style my18', 'style plus my18', 'style+', 'blue style hp my17', 'blue style', 'style my19', 'style plus my19'],
#     'Type R': ['gt pack', 'gt'],
#     'Trend': ['trend', 'trend '],
#     'X-Road': ['x-road navi'],
#     'Teclife': ['teclife'],
#     'Turbo': ['turbo'],
#     'MY18': ['my18'],
#     'LED': ['led'],
#     'Panorama': ['panorama'],
#     'N': ['250cv', 'n-line my19\'5', 'n-line'],  # Represents Hyundai i30 N
#     'NÃO_PARAMETRIZADOS': ['dynamic + connect navi ', 'auto ribeiro', 'teclife', 'van 6 lugares', 'style + navi']
# }

version_translation = {
    'Access': ['access my17'],
    'Comfort': ['comfort ' 'comfort my18', 'comfort', 'comfort', 'comfort my19', 'blue comfort my17', 'blue comfort hp my17', 'blue comfort', 'comfort my19\'5', 'comfort my20', 'blue comfort hp my16'],
    'Elegance': ['elegance ', 'elegance'],
    'EV': ['ev'],
    'Executive': ['executive ', 'executive', 'executive my19', 'executive my20', 'executive my19\'5'],
    'HEV': ['hev'],
    'Lifestyle': ['lifestyle'],
    'PHEV': ['phev'],
    'Premium': ['premium', 'premium my19', 'premium my20', 'premium my19\'5'],
    'Prestige': ['prestige'],
    'Style': ['style', 'style my18', 'blue style hp my17', 'blue style', 'style my19'],
    'Trend': ['trend', 'trend '],
    'Teclife': ['teclife'],
    'Turbo': ['turbo'],
    'MY18': ['my18'],
    'LED': ['led'],
    'Panorama': ['panorama'],
    'N': ['250cv', 'n-line my19\'5', 'n-line'],  # Represents Hyundai i30 N
    'NÃO_PARAMETRIZADOS': ['dynamic + connect navi ', 'auto ribeiro', 'teclife', 'van 6 lugares', 'style + navi']
}

# v1
version_grouping = {
    'Premium': ['Premium'],
    'Comfort': ['Comfort'],
    'Style': ['Style'],
    'Access': ['Access'],
    'Elegance': ['Elegance'],
    'Executive': ['Executive'],
    'EV/HEV/PHEV': ['HEV', 'EV', 'PHEV'],
    'GO/Sport': ['GO', 'Sport'],
    'Outros': ['Launch', 'Type R', 'Lifestyle', 'Creative', 'Performance', 'Trend', 'Pro', 'Prestige', 'Dynamic', 'X-Road']
}

# Cor Exterior
ext_color_translation = {
    'Amarelo': ['acid yellow', 'acid yellow (tt)', 'ral1016'],
    'Azul': ['p crystal blue m.', 'slate blue (teto preto)', 'midnight blue beam m', 'midnight blue beam m.', 'aqua turquoise (teto preto)', 'cosmic blue m.', 'obsidian blue p.', 'stormy sea', 'ocean view', 'aqua sparkling', 'clean slate', 'intense blue', 'brilliant sporty blue m.', 'morpho blue p.', 'stargazing blue', 'champion blue', 'ceramic blue', 'stellar blue', 'blue lagoon', 'performance blue', 'morning blue', 'ara blue', 'marina blue', 'ceramic blue (tt)', 'blue lagoon (tt)', 'skyride blue m.',
             'twilight blue m.', 'surf blue', 'taffeta white iii'],
    'Branco': ['polar white (teto vermelho)', 'polar white  (teto preto)', 'taffeta white', 'platinum white p', 'platinum white p.', 'white orchid p.', 'polar white', 'white sand', 'creamy white', 'chalk white', 'pure white', 'white crystal', 'white cream', 'chalk white (tt)', 'championship white', 'psunlight whitepearl'],
    'Castanho': ['brass (teto preto)', 'iced coffee', 'moon rock', 'golden brown m.', 'cashmere brown', 'tan brown', 'demitasse brown', 'premium agate brown p.'],
    'Cinzento': ['shining gray m', 'urban titanium m.', 'velvet dune', 'velvet dune (tt)', 'dark knight (tt)', 'wild explorer', 'rain forest', 'magnetic force', 'olivine grey', 'dark knight', 'star dust', 'polished metal m.', 'shining grey m.', 'modern steel m.', 'micron grey', 'galactic grey', 'iron gray', 'galactic grey (tt)', 'sonic grey p.', 'shadow grey', 'stone gray'],
    'Laranja': ['tangerine comet (tt)', 'tangerine comet', 'sunset orange ii'],
    'Prateado': ['fluidic metal', 'star dust (teto vermelho)', 'sleek silver (teto preto)', 'lunar silver m.', 'platinum silver', 'sleek silver', 'lake silver', 'aurora silver', 'titanium silver', 'platinum silve', 'typhoon silver', 'lake silver (tt)', 'alabaster silver m.', 'tinted silver m.', 'platinum gray m'],
    'Preto': ['electric shadow', 'phantom black (teto vermelho)', 'midnight burgundy p.', 'crystal black p.', 'ruse black m.', 'phantom black', ' black2'],
    'Vermelho': ['dragon red', 'dragon red (teto preto)', 'premium crystal red m.', 'ral3000', 'rallye red', 'milano red', 'fiery red', 'passion red', 'tomato red', 'pulse red', 'engine red', 'magma red', 'pulse red (tt)', 'passion red p.'],
    'NÃO_PARAMETRIZADOS': [],
}

ext_color_grouping = {
    'Branco': ['Branco'],
    'Cinzento': ['Cinzento'],
    'Prateado': ['Prateado'],
    'Preto': ['Preto'],
    'Vermelho/Azul': ['Vermelho', 'Azul'],
    'Castanho/Laranja/Amarelo': ['Castanho', 'Amarelo', 'Laranja'],
}

int_color_translation = {
    'Azul': ['blue', 'blue point'],
    'Bege': ['beige', 'sahara beige', 'dark beige', 'elegant beige'],
    'Bege/Preto': ['beige + black'],
    'Branco': ['ivory'],
    'Castanho': ['brilliant brown'],
    'Cinzento': ['gray2', 'blue grey', 'grey', 'pele sintética cinza', 'dark grey', 'grey/blue'],
    'Laranja': ['orange'],
    'Preto': ['black', 'black 2', 'black 3', 'black3', 'neutral black', 'black/darred 4', ' black2'],
    'Preto/Castanho': ['black / brown'],
    'Preto/Cinzento': ['black/charcoal'],
    'Preto/Laranja': ['black/orange'],
    'Vermelho': ['red', 'red point', 'black + red point', 'lava stone'],
    'NÃO_PARAMETRIZADOS': [],
}

int_color_grouping = {
    'Interior Standard': ['Preto'],
    'Interior Customizado': ['Cinzento', 'Vermelho', 'Preto/Laranja', 'Azul', 'Bege', 'Castanho', 'Preto/Castanho', 'Laranja', 'Branco', 'Preto/Cinzento']
}

classification_models = {
    'dt': [tree.DecisionTreeClassifier, [{'min_samples_leaf': [3, 5, 7, 9, 10, 15, 20, 30], 'max_depth': [3, 5, 6], 'class_weight': ['balanced']}]],
    'rf': [RandomForestClassifier, [{'n_estimators': [10, 25, 50, 100, 200, 500, 1000], 'max_depth': [5, 10, 20], 'class_weight': ['balanced']}]],
    'lr': [linear_model.LogisticRegression, [{'C': np.logspace(-2, 2, 20), 'solver': ['liblinear'], 'max_iter': [1000]}]],
    'knn': [KNeighborsClassifier, [{'n_neighbors': np.arange(1, 50, 1)}]],
    'svm': [SVC, [{'C': np.logspace(-2, 2, 10)}]],
    'ab': [AdaBoostClassifier, [{'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}]],
    'gc': [GradientBoostingClassifier, [{'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}]],
    'xgb': [xgb.XGBClassifier, [{'objective': ['binary:logistic'], 'booster': ['gbtree'], 'max_depth': [5, 10, 20, 50, 100]}]],  # ToDo: need to add L1 (reg_alpha) and L2 (reg_lambda) regularization to counter the overfitting
    'lgb': [lgb.LGBMClassifier, [{'num_leaves': [15, 31, 50], 'n_estimators': [50, 100, 200], 'objective': ['multiclass']}]],
    'bayes': [GaussianNB],  # ToDo: Need to create an exception for this model
    'ann': [MLPClassifier, [{'activation': ['identity', 'logistic', 'tanh', 'relu'], 'hidden_layer_sizes': (100, 100), 'solver': ['sgd'], 'max_iter': [1000]}]],
    'voting': [VotingClassifier, [{'voting': ['soft']}]]
}

regression_models_standard = {
    'rf': [RandomForestRegressor, [{'max_depth': [3, 5, 7, 9, 11, 13], 'n_estimators': [50, 100, 200, 250, 500]}]],
    'lgb': [lgb.LGBMRegressor, [{'num_leaves': [5, 10, 15, 20, 25, 30, 35, 50, 100], 'max_depth': [3, 5, 7, 9, 11, 13], 'n_estimators': [50, 100, 200, 250, 500]}]],
    # 'lgb': [lgb.LGBMRegressor, [{'num_leaves': [5, 10], 'n_estimators': [50]}]],
    # 'lgb': [lgb.LGBMRegressor, [{'num_leaves': [5, 10, 15, 20, 25, 30, 35, 50, 100], 'max_bin': [50, 100, 200, 500], 'max_depth': [3, 5, 7, 9, 11, 13], 'n_estimators': [x for x in range(50, 5001, 50)], 'min_data_in_leaf': [10, 20, 50, 100, 200, 300, 500]}]],
    'xgb': [xgb.XGBRegressor, [{'objective': ['reg:squarederror'], 'max_depth': [3, 5, 7, 9, 11, 13], 'n_estimators': [50, 100, 200, 250, 500]}]],
    'lasso_cv': [LassoCV, [{'eps': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10], 'max_iter': [1000, 2000, 5000], 'tol': [0.0001, 0.001, 0.01, 0.1], 'cv': [5]}]],
    'ridge': [Ridge, [{'alpha': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]}]],
    'll_cv': [LassoLarsCV, [{'max_iter': [15, 20, 25, 50, 100, 250, 500, 1000], 'eps': [1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1], 'cv': [5]}]],
    # 'll_cv': [LassoLarsCV, [{'max_iter': [15, 20], 'cv': [5]}]],
    'elastic_cv': [ElasticNetCV, [{'eps': [1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17], 'cv': [5]}]],
    'svr': [SVR, [{'kernel': ['linear', 'rbf'], 'gamma': ['auto', 'scale']}]],
}

regression_models = {
    'rf': [RandomForestRegressor, [{'max_depth': [11, 13, 15, 17, 19, 21, 23, 25, 27, 29], 'n_estimators': [50, 100, 200, 250, 500, 1000, 2000, 3000, 5000, 10000]}]],
    'lgb': [lgb.LGBMRegressor, [{'num_leaves': [15, 20, 25, 30, 35, 50, 75, 100], 'max_depth': [11, 13, 15, 17, 19], 'n_estimators': [50, 100, 200, 250, 500, 1000, 1500, 2000, 3000, 5000, 10000]}]],
    'xgb': [xgb.XGBRegressor, [{'objective': ['reg:squarederror'], 'max_depth': [7, 9, 11, 13, 15, 17], 'n_estimators': [50, 100, 200, 250, 500, 1000, 2000, 3000, 5000, 10000]}]],
}

sql_columns_vhe_fact_bi = [
    'NLR_Code', 'Environment', 'Value_Type_Code', 'Chassis_Number', 'Registration_Number', 'NLR_Posting_Date', 'SLR_Document_Date_CHS', 'SLR_Account_CHS_Key', 'SLR_Document_Date_RGN',
    'Product_Code', 'Sales_Type_Code', 'Sales_Type_Code_DMS', 'Location_Code', 'VehicleData_Code', 'Vehicle_Type_Code', 'Fuel_Type_Code', 'Transmission_Type_Code', 'Vehicle_Area_Code', 'Dispatch_Type_Code',
    'Sales_Status_Code', 'Ship_Arrival_Date', 'Registration_Request_Date', 'Registration_Date', 'DaysInStock_Distributor', 'DaysInStock_Dealer', 'DaysInStock_Global', 'SLR_Account_Dealer_Code',
    'Sales_Type_Dealer_Code', 'Measure_1', 'Measure_2', 'Measure_3', 'Measure_4', 'Measure_5', 'Measure_6', 'Measure_7', 'Measure_8', 'Measure_9', 'Measure_10', 'Measure_11', 'Measure_12', 'Measure_13',
    'Measure_14', 'Measure_15', 'Measure_16', 'Measure_17', 'Measure_18', 'Measure_19', 'Measure_20', 'Measure_21', 'Measure_22', 'Measure_23', 'Measure_24', 'Measure_25', 'Measure_26', 'Measure_27',
    'Measure_28', 'Measure_29', 'Measure_30', 'Measure_31', 'Measure_32', 'Measure_33', 'Measure_34', 'Measure_35', 'Measure_36', 'Measure_37', 'Measure_38', 'Measure_39', 'Measure_40', 'Measure_41',
    'Measure_42', 'Measure_43', 'Measure_44', 'Measure_45', 'Measure_46', 'NDB_VATGroup_Desc', 'VAT_Number_Display', 'NDB_Contract_Dealer_Desc', 'NDB_VHE_PerformGroup_Desc', 'NDB_VHE_Team_Desc',
    'Customer_Display', 'Customer_Group_Code', 'Customer_Group_Desc', 'No_Registration_Number_Flag', 'Registration_Number_No_SLR_Document_RGN_Flag', 'SLR_Document_RGN_Flag', 'Undefined_VHE_Status',
    'prev_sales_check', 'number_prev_sales', 'PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc',
    'PT_PDB_Exterior_Color_Desc', 'PT_PDB_Interior_Color_Desc', 'ML_VehicleData_Code', 'Fixed_Margin_II', 'PDB_Start_Order_Date', 'PDB_End_Order_Date', 'NDB_Dealer_Code'
]


# Validation Queries:
stock_validation_query = '''
WITH CTE_AUX AS
    (
        SELECT MAX(Stock_Month) AS Stock_Month
            ,  MAX(Stock_Day) AS Stock_Day
        FROM dbo.VHE_Fact_BI_Stock_DTR
    ),
STOCK_CTE AS
(
    SELECT PDB.PT_PDB_Model_Desc
        ,  PDB.PT_PDB_Engine_Desc
        ,  PDB.PT_PDB_Transmission_Type_Desc
        ,  PDB.PT_PDB_Version_Desc
        ,  PDB.PT_PDB_Exterior_Color_Desc
        ,  PDB.PT_PDB_Interior_Color_Desc
        ,  SStatus.PT_Stock_Status_Level_1_Desc
        ,  SStatus.PT_Stock_Status_Desc
        ,  FactBI.Chassis_Number
        ,  FactBI.Registration_Number
        ,  FactBI.Location_Desc
        ,  FactBI.Production_Date
        ,  FactBI.Purchase_Date
        ,  FactBI.Ship_Arrival_Date
    FROM            dbo.VHE_Fact_BI_Stock_DTR AS FactBI
    INNER JOIN      CTE_Aux AS CTE_Aux_1 ON CTE_Aux_1.Stock_Month = FactBI.Stock_Month
            AND CTE_Aux_1.Stock_Day = FactBI.Stock_Day
    LEFT  JOIN dbo.SLR_Dim_Dealers_DTR AS Dealers ON FactBI.SLR_Account_Key = Dealers.SLR_Account_Key
        AND (LEFT(Dealers.NDB_Dealer_Code, 3) NOT IN ('706', '702'))
    INNER JOIN dbo.VHE_Dim_VehicleData_DTR AS PDB ON PDB.VehicleData_Code = FactBI.VehicleData_Code
    LEFT JOIN dbo.VHE_Dim_Stock_Status_DTR AS SStatus ON SStatus.Stock_Status_Code = FactBI.Stock_Status_Code
    WHERE (1 = 1)
        AND FactBI.Stock_Included = 1
    AND PDB.PDB_Start_Order_Date IS NOT NULL -- Critério Gama Válida
    AND (
        PDB.PDB_End_Order_Date IS NULL
        OR PDB.PDB_End_Order_Date >= GETDATE()) -- Critério Gama Viva      
    UNION ALL
    SELECT PDB.PT_PDB_Model_Desc
            ,  PDB.PT_PDB_Engine_Desc_New
            ,  PDB.PT_PDB_Transmission_Type_Desc
            ,  PDB.PT_PDB_Version_Desc_New
            ,  PDB.PT_PDB_Exterior_Color_Desc
            ,  PDB.PT_PDB_Interior_Color_Desc
            ,  SStatus.PT_Stock_Status_Level_1_Desc
            ,  SStatus.PT_Stock_Status_Desc
            ,  FactBI.Chassis_Number
            ,  FactBI.Registration_Number
            ,  FactBI.Location_Desc
            ,  FactBI.Production_Date
            ,  FactBI.Purchase_Date
            ,  FactBI.Ship_Arrival_Date
        FROM            dbo.VHE_Fact_BI_Stock_DTR AS FactBI
        INNER JOIN      CTE_Aux AS CTE_Aux_1 ON CTE_Aux_1.Stock_Month = FactBI.Stock_Month
                AND CTE_Aux_1.Stock_Day = FactBI.Stock_Day
        LEFT OUTER JOIN dbo.SLR_Dim_Dealers_DTR AS Dealers ON FactBI.SLR_Account_Key = Dealers.SLR_Account_Key
                AND (LEFT(Dealers.NDB_Dealer_Code, 3) NOT IN ('706', '702'))
        INNER JOIN  dbo.VHE_Dim_VehicleData_DTR AS PDB ON PDB.VehicleData_Code = FactBI.VehicleData_Code
        LEFT JOIN dbo.VHE_Dim_Stock_Status_DTR AS SStatus ON SStatus.Stock_Status_Code = FactBI.Stock_Status_Code
        WHERE (1 = 1)
        AND FactBI.Stock_Included = 1
        AND PDB.PDB_Start_Order_Date IS NOT NULL --critério Gama válida
        AND PDB.PDB_End_Order_Date < GETDATE() -- Critério Gama Morta
        AND PDB.PT_PDB_Engine_Desc_New IS NOT NULL --se for gama morta tem que ter correspondencia nas colunas New
)
SELECT
    PT_Stock_Status_Level_1_Desc
    ,  PT_Stock_Status_Desc
    ,  Chassis_Number
    ,  Registration_Number
    ,  Location_Desc
    ,  Production_Date
    ,  Purchase_Date
    ,  Ship_Arrival_Date
FROM      Stock_cte AS Stock_cte
    LEFT JOIN VHE_MapDMS_Transmission_DTR AS Transmission_Map ON Transmission_Map.Original_Value = Stock_cte.PT_PDB_Transmission_Type_Desc
    LEFT JOIN VHE_MapDMS_Ext_Color_DTR AS Ext_Color_Map ON Ext_Color_Map.Original_Value = Stock_cte.PT_PDB_Exterior_Color_Desc
    LEFT JOIN VHE_MapDMS_Int_Color_DTR AS Int_Color_Map ON Int_Color_Map.Original_Value = Stock_cte.PT_PDB_Interior_Color_Desc
WHERE 1=1        
        and Stock_cte.PT_PDB_Model_Desc = '{}'
        and Stock_cte.PT_PDB_Engine_Desc = '{}'
        and Stock_cte.PT_PDB_Version_Desc = '{}'
        and Transmission_Map.Mapped_Value = '{}'
        and Ext_Color_Map.Mapped_Value = '{}'
        and Int_Color_Map.Mapped_Value = '{}'
        ORDER BY PT_Stock_Status_Level_1_Desc,  PT_Stock_Status_Desc
'''

proposals_validation_query = '''
    with proposals_middle as (
        SELECT 
                PDB.PT_PDB_Model_Desc
            ,  PDB.PT_PDB_Engine_Desc
            ,  Transmission_Map.Mapped_Value as PT_PDB_Transmission_Type_Desc
            ,  PDB.PT_PDB_Version_Desc
            ,  Ext_Color_Map.Mapped_Value as PT_PDB_Exterior_Color_Desc
            ,  Int_Color_Map.Mapped_Value as PT_PDB_Interior_Color_Desc
            ,  Proposals.Factory_Model_Code as Model_Code
            ,  Proposals.Factory_Vehicle_Option_Code as OCN
            ,  Proposals.Created_Time
            ,  Proposals.Dealer_Desc
            ,  Proposals.Record_Date
        FROM      dbo.VHE_Fact_DW_HPK_Proposals_DTR AS Proposals
        LEFT JOIN VHE_Dim_VehicleData_DTR AS PDB ON PDB.VehicleData_Code = Proposals.VehicleData_Code
        LEFT JOIN VHE_MapDMS_Transmission_DTR AS Transmission_Map ON Transmission_Map.Original_Value = PDB.PT_PDB_Transmission_Type_Desc
        LEFT JOIN VHE_MapDMS_Ext_Color_DTR AS Ext_Color_Map ON Ext_Color_Map.Original_Value = PDB.PT_PDB_Exterior_Color_Desc
        LEFT JOIN VHE_MapDMS_Int_Color_DTR AS Int_Color_Map ON Int_Color_Map.Original_Value = PDB.PT_PDB_Interior_Color_Desc
        WHERE (1 = 1)
        AND PDB.PDB_Start_Order_Date IS NOT NULL -- Critério Gama Válida
        AND (
            PDB.PDB_End_Order_Date IS NULL
            OR PDB.PDB_End_Order_Date >= GETDATE()) -- Critério Gama Viva
        AND (Proposals.VehicleData_Code <> 1)
        AND (Proposals.Proposal_Stage = 'Entregue')
        AND ((CONVERT(date, Created_Time, 105)) BETWEEN DATEADD(MONTH, - 3, GETDATE()) AND GETDATE())
        UNION ALL
        SELECT 
                PDB.PT_PDB_Model_Desc
            ,  PDB.PT_PDB_Engine_Desc
            ,  Transmission_Map.Mapped_Value as PT_PDB_Transmission_Type_Desc
            ,  PDB.PT_PDB_Version_Desc
            ,  Ext_Color_Map.Mapped_Value as PT_PDB_Exterior_Color_Desc
            ,  Int_Color_Map.Mapped_Value as PT_PDB_Interior_Color_Desc
            ,  Proposals.Factory_Model_Code as Model_Code
            ,  Proposals.Factory_Vehicle_Option_Code as OCN
            ,  Proposals.Created_Time
            ,  Proposals.Dealer_Desc
            ,  Proposals.Record_Date
        FROM      dbo.VHE_Fact_DW_HPK_Proposals_DTR AS Proposals
        LEFT JOIN VHE_Dim_VehicleData_DTR AS PDB ON PDB.VehicleData_Code = Proposals.VehicleData_Code
        LEFT JOIN VHE_MapDMS_Transmission_DTR AS Transmission_Map ON Transmission_Map.Original_Value = PDB.PT_PDB_Transmission_Type_Desc
        LEFT JOIN VHE_MapDMS_Ext_Color_DTR AS Ext_Color_Map ON Ext_Color_Map.Original_Value = PDB.PT_PDB_Exterior_Color_Desc
        LEFT JOIN VHE_MapDMS_Int_Color_DTR AS Int_Color_Map ON Int_Color_Map.Original_Value = PDB.PT_PDB_Interior_Color_Desc
        WHERE (1 = 1)
        AND PDB.PDB_Start_Order_Date IS NOT NULL --Critério Gama Válida
        AND PDB.PDB_End_Order_Date < GETDATE() -- Critério Gama Morta
        AND PDB.PT_PDB_Engine_Desc_New IS NOT NULL --se for gama morta tem que ter correspondencia nas colunas New
        AND (Proposals.VehicleData_Code <> 1)
        AND (Proposals.Proposal_Stage = 'Entregue')
        AND ((CONVERT(date, Created_Time, 105)) BETWEEN DATEADD(MONTH, - 3, GETDATE()) AND GETDATE())
    )
    SELECT 
           proposals_middle.Model_Code
        ,  proposals_middle.OCN
        ,  proposals_middle.Created_Time
        ,  proposals_middle.Dealer_Desc
        ,  proposals_middle.Record_Date
    FROM proposals_middle
    WHERE 1=1
      and proposals_middle.PT_PDB_Model_Desc = '{}'
      and proposals_middle.PT_PDB_Engine_Desc = '{}'
      and proposals_middle.PT_PDB_Version_Desc = '{}'
      and proposals_middle.PT_PDB_Transmission_Type_Desc = '{}'
      and proposals_middle.PT_PDB_Exterior_Color_Desc = '{}'
      and proposals_middle.PT_PDB_Interior_Color_Desc = '{}'
'''

sales_validation_query = '''
    WITH DEALERS AS
    (
        SELECT SLR_Dim_Dealers_DTR_VHE.Client_Id AS Client_Id
            ,  SLR_Dim_Dealers_DTR_VHE.Environment AS Environment
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_Dealer_Code AS NDB_Dealer_Code
            ,  SLR_Dim_Dealers_DTR_VHE.SLR_Account AS SLR_Account
            ,  SLR_Dim_Dealers_DTR_VHE.Customer_Display AS Customer_Display
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_VATGroup_Code AS NDB_VATGroup_Code
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_VATGroup_Desc AS NDB_VATGroup_Desc
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_VAT_Number_Code AS NDB_VAT_Number_Code
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_VAT_Number AS NDB_VAT_Number
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_Contract_Dealer_Code AS NDB_Contract_Dealer_Code
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_Contract_Dealer_Desc AS NDB_Contract_Dealer_Desc
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_VHE_PerformGroup_Code AS NDB_VHE_PerformGroup_Code
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_VHE_PerformGroup_Desc AS NDB_VHE_PerformGroup_Desc
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_PSE_PerformGroup_Code AS NDB_PSE_PerformGroup_Code
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_PSE_PerformGroup_Desc AS NDB_PSE_PerformGroup_Desc
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_VHE_AreaManager_Code AS NDB_VHE_AreaManager_Code
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_VHE_AreaManager_Desc AS NDB_VHE_AreaManager_Desc
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_PSE_AreaManager_Code AS NDB_PSE_AreaManager_Code
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_PSE_AreaManager_Desc AS NDB_PSE_AreaManager_Desc
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_VHE_Team_Code AS NDB_VHE_Team_Code
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_VHE_Team_Desc AS NDB_VHE_Team_Desc
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_PSE_Team_Code AS NDB_PSE_Team_Code
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_PSE_Team_Desc AS NDB_PSE_Team_Desc
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_Headquarters AS NDB_Headquarters
            ,  SLR_Dim_Dealers_DTR_VHE.Record_Date AS Record_Date
            ,  SLR_Dim_Dealers_DTR_VHE.Last_Modified_Date AS Last_Modified_Date
            ,  SLR_Dim_Dealers_DTR_VHE.Customer_Group_Code AS Customer_Group_Code
            ,  SLR_Dim_Dealers_DTR_VHE.VAT_Number_Display AS VAT_Number_Display
            ,  SLR_Dim_Dealers_DTR_VHE.SLR_Account_Key AS SLR_Account_CHS_Key
            ,  SLR_Dim_Dealers_DTR_VHE.SLR_AccountGroup_Code AS SLR_AccountGroup_Code
            ,  SLR_Dim_Dealers_DTR_VHE.NDB_Dealer_Desc AS NDB_Dealer_Desc
            ,  SLR_Dim_Dealers_DTR_VHE.SLR_Account_Invoice AS SLR_Account_Invoice
        FROM dbo.SLR_Dim_Dealers_DTR AS SLR_Dim_Dealers_DTR_VHE
        WHERE SLR_Dim_Dealers_DTR_VHE.Record_Type <> 3
    ),
    CUSTOMER_GROUP AS
    (
        SELECT SLR_Dim_Customer_Groups_DTR.Customer_Group_Code AS Customer_Group_Code
            ,  SLR_Dim_Customer_Groups_DTR.EN_Customer_Group_Desc AS Customer_Group_Desc
            ,  SLR_Dim_Customer_Groups_DTR.Customer_Group_Display_Order AS Customer_Group_Display_Order
        FROM dbo.SLR_Dim_Customer_Groups_DTR SLR_Dim_Customer_Groups_DTR WITH (NOLOCK)
    )
    SELECT Sales.[Factory_Model_Code] as Model_Code
          ,Sales.[Local_Vehicle_Option_Code] as OCN
          ,Sales.[PDB_Start_Order_Date]
          ,Sales.[PDB_End_Order_Date]
          ,Sales.[Chassis_Number]
          ,Sales.[Registration_Number]
          --,Sales.[NDB_Dealer_Code] --join with dealers table
          ,DEALERS.NDB_VATGroup_Desc as NDB_VATGroup_Desc
          --,Sales.[Customer_Group_Code] -- join with customer table
          ,CUSTOMER_GROUP.Customer_Group_Desc as Customer_Group_Desc
          ,Sales.[DaysInStock_Global]
          ,Sales.[TotalGrossMargin]
          ,Sales.[TotalGrossMarginPerc]
          ,Sales.[SLR_Document_Date_CHS]
          ,Sales.[Quantity_CHS]
    FROM [BI_DTR].[dbo].[View_VHE_Fact_PA_OrderOptimization_Sales] as Sales
    LEFT JOIN  Dealers ON Dealers.SLR_Account_CHS_Key = Sales.SLR_Account_CHS_Key
    LEFT JOIN  Customer_Group ON Customer_Group.Customer_Group_Code = Dealers.Customer_Group_Code
        WHERE 1=1
      and Sales.PT_PDB_Model_Desc = '{}'
      and Sales.PT_PDB_Engine_Desc = '{}'
      and Sales.PT_PDB_Version_Desc = '{}'
      and Sales.PT_PDB_Transmission_Type_Desc = '{}'
      and Sales.PT_PDB_Exterior_Color_Desc = '{}'
      and Sales.PT_PDB_Interior_Color_Desc = '{}'
      and Quantity_CHS > 0
  '''

sales_plan_validation_query_step_1 = '''
    SELECT DISTINCT
        Sales_Plan.Factory_Model_Code as Model_Code
        ,  Sales_Plan.Local_Vehicle_Option_Code as OCN
        ,  Sales_Plan.Sales_Plan_Period
        ,  Sales_Plan.WLTP_CO2
        ,  Sales_Plan.NEDC_CO2
        ,  Sales_Plan.Quantity
        ,  Sales_Plan.Record_Date
    FROM  dbo.VHE_Setup_Sales_Plan_DTR AS Sales_Plan
    LEFT OUTER JOIN dbo.VHE_Dim_VehicleData_DTR AS PDB ON PDB.Factory_Model_Code = Sales_Plan.Factory_Model_Code AND PDB.Local_Vehicle_Option_Code = Sales_Plan.Local_Vehicle_Option_Code				
    LEFT JOIN VHE_MapDMS_Transmission_DTR AS Transmission_Map ON Transmission_Map.Original_Value = PDB.PT_PDB_Transmission_Type_Desc
    LEFT JOIN VHE_MapDMS_Ext_Color_DTR AS Ext_Color_Map ON Ext_Color_Map.Original_Value = PDB.PT_PDB_Exterior_Color_Desc
    LEFT JOIN VHE_MapDMS_Int_Color_DTR AS Int_Color_Map ON Int_Color_Map.Original_Value = PDB.PT_PDB_Interior_Color_Desc
    WHERE (1 = 1)
    AND (Sales_Plan.Sales_Plan_Period BETWEEN YEAR(GETDATE()) * 100 + MONTH(GETDATE()) AND YEAR(DATEADD(month, 4, GETDATE())) * 100 + MONTH(DATEADD(month, 4, GETDATE())))
    AND Sales_Plan.Factory_Model_Code <> '1'
      and PDB.PT_PDB_Model_Desc = '{}'
      and PDB.PT_PDB_Engine_Desc = '{}'
      and PDB.PT_PDB_Version_Desc = '{}'
      and Transmission_Map.Mapped_Value = '{}'
      and Ext_Color_Map.Mapped_Value = '{}'
      and Int_Color_Map.Mapped_Value = '{}'
    ORDER BY Sales_Plan.Factory_Model_Code, Sales_Plan.Local_Vehicle_Option_Code, Sales_Plan_Period
'''

sales_plan_validation_query_step_2 = '''
    WITH SALES_PLAN_TEMP AS
    (
        SELECT PDB.PT_PDB_Model_Desc
            ,  PDB.PT_PDB_Engine_Desc
            ,  PDB.PT_PDB_Transmission_Type_Desc
            ,  PDB.PT_PDB_Version_Desc
            ,  PDB.PT_PDB_Exterior_Color_Desc
            ,  PDB.PT_PDB_Interior_Color_Desc
            ,  MAX(Sales_Plan.WLTP_CO2) AS WLTP_CO2
            ,  MAX(Sales_Plan.NEDC_CO2) AS NEDC_CO2
            ,  SUM(Sales_Plan.Quantity) AS Max_Qty_Per_Sales_Plan_Period
        FROM            dbo.VHE_Setup_Sales_Plan_DTR AS Sales_Plan
        LEFT OUTER JOIN dbo.VHE_Dim_VehicleData_DTR AS PDB ON PDB.Factory_Model_Code = Sales_Plan.Factory_Model_Code
                AND PDB.Local_Vehicle_Option_Code = Sales_Plan.Local_Vehicle_Option_Code				
        WHERE (1 = 1)
        AND (Sales_Plan.Sales_Plan_Period BETWEEN YEAR(GETDATE()) * 100 + MONTH(GETDATE()) AND YEAR(DATEADD(month, 4, GETDATE())) * 100 + MONTH(DATEADD(month, 4, GETDATE())))
        AND Sales_Plan.Factory_Model_Code <> '1' --
        GROUP BY PDB.PT_PDB_Model_Desc
            , PDB.PT_PDB_Engine_Desc
            , PDB.PT_PDB_Transmission_Type_Desc
            , PDB.PT_PDB_Version_Desc
            , PDB.PT_PDB_Exterior_Color_Desc
            , PDB.PT_PDB_Interior_Color_Desc
            , Sales_Plan.Sales_Plan_Period
    )
    SELECT DISTINCT
          Sales_Plan.WLTP_CO2
        ,  Sales_Plan.NEDC_CO2
        ,  Sales_Plan.Max_Qty_Per_Sales_Plan_Period
    FROM      Sales_Plan_Temp AS Sales_Plan
    LEFT JOIN VHE_MapDMS_Transmission_DTR AS Transmission_Map ON Transmission_Map.Original_Value = Sales_Plan.PT_PDB_Transmission_Type_Desc
    LEFT JOIN VHE_MapDMS_Ext_Color_DTR AS Ext_Color_Map ON Ext_Color_Map.Original_Value = Sales_Plan.PT_PDB_Exterior_Color_Desc
    LEFT JOIN VHE_MapDMS_Int_Color_DTR AS Int_Color_Map ON Int_Color_Map.Original_Value = Sales_Plan.PT_PDB_Interior_Color_Desc
    WHERE 1=1
      and Sales_Plan.PT_PDB_Model_Desc = '{}'
      and Sales_Plan.PT_PDB_Engine_Desc = '{}'
      and Sales_Plan.PT_PDB_Version_Desc = '{}'
      and Transmission_Map.Mapped_Value = '{}'
      and Ext_Color_Map.Mapped_Value = '{}'
      and Int_Color_Map.Mapped_Value = '{}'
'''


sales_plan_validation_query_step_3 = '''
    SELECT 
          [WLTP_CO2]
          ,[NEDC_CO2]
          ,[OC]
      FROM [BI_DTR].[dbo].[View_VHE_Fact_PA_OrderOptimization_Sales_Plan] as PDB
      WHERE 1=1
        and PDB.PT_PDB_Model_Desc = '{}'
      and PDB.PT_PDB_Engine_Desc = '{}'
      and PDB.PT_PDB_Version_Desc = '{}'
      and PDB.[PT_PDB_Transmission_Type_Desc] = '{}'
      and PDB.[PT_PDB_Exterior_Color_Desc] = '{}'
      and PDB.[PT_PDB_Interior_Color_Desc] = '{}'
'''

proposals_max_date_query = '''
    SELECT MAX([Record_Date])
    FROM [BI_DTR].[dbo].[VHE_Fact_DW_HPK_Proposals_DTR]
'''
