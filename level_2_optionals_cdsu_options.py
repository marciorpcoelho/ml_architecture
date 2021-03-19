import os
from py_dotenv import read_dotenv
dotenv_path = '/info.env'
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
read_dotenv(base_path + dotenv_path)

# Options:
margin_threshold = 3.5
stock_days_threshold = 45
update_frequency_days = 0
selected_configuration_parameters = ['Motor', 'Caixa Auto', 'Cor_Exterior', 'Jantes', 'Modelo', 'Sensores Est. Tras.', 'Sensores Est. Front.', 'Tipo_Interior', 'Versao', 'Combustível', 'Câmara Traseira']
# Full: ['7_Lug', 'Alarme', 'AC Auto', 'Barras_Tej', 'Caixa Auto', 'Cor_Exterior', 'Cor_Interior', 'Farois_LED', 'Farois_Xenon', 'Jantes', 'Modelo', 'Navegação', 'Prot.Solar', 'Sensores', 'Teto_Abrir', 'Tipo_Interior', 'Versao']

if 'nt' in os.name:
    DSN_SRV3_PRD = os.getenv('DSN_SRV3_Prd')
    DSN_MLG_PRD = os.getenv('DSN_MLG_Prd')
    DSN_MLG_DEV = os.getenv('DSN_MLG_Dev')
elif 'posix' in os.name:
    DSN_SRV3_PRD = os.getenv('DSN_SRV3_Prd_Linux')
    DSN_MLG_PRD = os.getenv('DSN_MLG_Prd_Linux')
UID = os.getenv('UID')
PWD = os.getenv('PWD')
configuration_parameters_full = ['Motor', 'Caixa Auto', 'Cor_Exterior', 'Jantes', 'Modelo', 'Sensores Est. Tras.', 'Sensores Est. Front.', 'Tipo_Interior', 'Versao', 'Combustível', 'Câmara Traseira']
api_backend_loc = 'optimizations/vhe_baviera/'

# Dictionaries:
sql_info = {
    'database': 'BI_MLG',
    'database_final': 'BI_MLG',
    'initial_table': 'VHE_Fact_DW_SalesNew_WithSpecs',
    'unit_count_number_history': 'LOG_Project_Units_Count_History',
    'checkpoint_b_table': 'VHE_Fact_DW_Checkpoint_B_OrderOptimization_RTL',
    'feature_contribution': 'VHE_Fact_Feature_Contribution',
    'final_table': 'VHE_Fact_PA_OrderOptimization_RTL',
    'model_mapping': ['VHE_MapBI_Model_Fase2'],
    'mappings': ['VHE_MapBI_Rims_Size', 'VHE_MapBI_Sales_Place', 'VHE_MapBI_Sales_Place_v2', 'VHE_MapBI_Model', 'VHE_MapBI_Version', 'VHE_MapBI_Interior_Type', 'VHE_MapBI_Color_Ext', 'VHE_MapBI_Color_Int', 'VHE_MapBI_Motor_Desc'],
    'mappings_temp': ['VHE_MapBI_Sales_Place', 'VHE_MapBI_Sales_Place_v2', 'VHE_MapBI_Sales_Place_Fase2'],  # When no training is needed in this project
    'optimization_solution_table': 'VHE_Fact_PA_OrderOptimization_Solver_Optimization',
}

project_id = 2775
nlr_code = '4R0'

regex_dict = {
    # 'motorization_value': r'(\d{1}\.{1}\d{1}\s{0,1}[a-zA-Z]{1,3}|\d{1}\.{1}\d{1})'
    'motorization_value': r'(\d{1}\.{1}\d{1})',
    'interior_type_value': r'(couro|tecido)',
    'version_type': r'(confortline|hybrid|stream|rline|life|elegance|business|design|sport|style|trendline|highline|alltrack)',
    'rims_size': r'\d{2}',
}

# New Cor_Exterior
color_ext_dict = {
}

colors_pt = ['petróleo', 'oceano', 'cobre', 'branco', 'azul', 'verde', 'tartufo', 'vermelho', 'antracite/vermelho', 'anthtacite/preto', 'preto/laranja/preto/lara', 'prata/cinza', 'cinza', 'preto/silver', 'cinzento', 'prateado', 'prata', 'amarelo',
             'laranja', 'castanho', 'dourado', 'antracit', 'antracite/preto', 'antracite/cinza/preto', 'branco/outras', 'antracito', 'antracite', 'antracite/vermelho/preto', 'oyster/preto', 'prata/preto/preto', 'âmbar/preto/pr',
             'bege', 'terra', 'preto/laranja', 'cognac/preto', 'bronze', 'beige', 'beje', 'veneto/preto', 'zagora/preto', 'mokka/preto', 'taupe/preto', 'sonoma/preto', 'preto/preto', 'preto/laranja/preto', 'preto/vermelho', 'preto']
colors_en = ['black', 'havanna', 'merino', 'walnut', 'chocolate', 'nevada', 'moonstone', 'anthracite/silver', 'white', 'coffee', 'blue', 'red', 'grey', 'silver', 'orange', 'green', 'bluestone', 'aqua', 'burgundy', 'anthrazit',
             'truffle', 'brown', 'oyster', 'tobacco', 'jatoba', 'storm', 'champagne', 'cedar', 'silverstone', 'chestnut', 'kaschmirsilber', 'oak', 'mokka', 'sunstone', 'topaz']

colors_to_replace_dict = {'petróleo': 'preto', 'cobre': 'castanho', 'oceano': 'azul', 'black': 'preto', 'preto/silver': 'preto/prateado', 'tartufo': 'truffle', 'preto/laranja/preto/lara': 'preto/laranja', 'white': 'branco', 'blue': 'azul', 'red': 'vermelho', 'grey': 'cinzento', 'silver': 'prateado', 'orange': 'laranja', 'green': 'verde', 'anthrazit': 'antracite', 'antracit': 'antracite', 'brown': 'castanho', 'antracito': 'antracite', 'âmbar/preto/pr': 'ambar/preto/preto', 'beige': 'bege', 'kaschmirsilber': 'cashmere', 'beje': 'bege'}

# Cor_Interior single parametrization table:
color_int_dict = {
    'preto': ['nappa_antracite', 'vernasca_anthracite/preto', 'merino_preto', 'nevada_preto', 'merino_preto/preto', 'nevada_preto/preto', 'vernasca_preta', 'vernasca_preto/com',
              'vernasca_preto/preto', 'preto', 'dakota_preto/preto', 'dakota_preto/vermelho/preto', 'dakota_preto/laranja/preto', 'dakota_preto/oyster', 'dakota_preto/debroado', 'dakota_preto/azul/preto', 'dakota_preto',
              'dakota_preta', 'dakota_black/contrast', 'nappa_preto', 'merino_night/preto/pret'],
    'castanho/mocha': ['vernasca_castanhas/preto', 'merino_tartufo/preto', 'merino_tartufo/preto/preto', 'merino_coffee/preto', 'mocha', 'dakota_mocha/preto', 'dakota_mocha/preto/mocha', 'dakota_mocha', 'nappa_mocha', 'nevada_mocha',
                       'castanho', 'merino_castanho', 'nevada_terra', 'nevada_brown', 'vernasca_mocha', 'vernasca_mocha/preto', 'vernasca_cognac', 'vernasca_cognac/preto', 'nappa_castanho', 'nappa_cognac/preto',
                       'dakota_castanho', 'dakota_conhaque', 'dakota_conhaque/castanho/preto', 'dakota_conhaque/castanho/preto/conhaque', 'dakota_cognac/preto', 'dakota_brown', 'dakota_terra', 'vernasca_coffee/preto'],
    'bege/oyster/branco': ['merino_branco/azul', 'merino_bege', 'dakota_bege', 'nappa_bege', 'vernasca_canberra', 'vernasca_bege', 'nevada_bege', 'bege', 'dakota_oyster', 'dakota_oyster/oyster', 'dakota_oyster/cinza', 'vernasca_oyster',
                           'nevada_oyster', 'nevada_oyster/leather', 'oyster', 'nevada_branco', 'merino_branco', 'dakota_oyster/preto', 'dakota_ivory/preto', 'dakota_ivory', 'dakota_branco', 'dakota_white',
                           'nappa_white', 'nappa_ivory', 'nappa_ivory/branco', 'vernasca_branco'],
    'outros': ['amarelo', 'vermelho', 'merino_vermelho', 'dakota_coral', 'dakota_azul', 'vernasca_azuis/preto', 'merino_laranja', 'merino_orange', 'merino_silverstone', 'merino_taupe/preto', 'cinzento', 'others', '0'],
}

# New Jantes
jantes_dict = {
    '16': ['16'],
    '17': ['17'],
    '18': ['18'],
    'stand/19/20': ['standard', '19', '20']
}

sales_place_dict = {
}


model_dict = {
}

versao_dict = {
    'advantage': ['advantage'],
    'sport': ['line_sport'],
    'base': ['base'],
    'luxury': ['line_luxury'],
    'xline/urban': ['xline', 'line_urban'],
    # 'urban/desportiva': ['line_urban', 'desportiva_m', 'pack_desportivo_m']
    'desportiva': ['desportiva_m', 'pack_desportivo_m'],
}

# v2
tipo_int_dict = {
    'tecido': ['tecido'],
    'pele': ['pele'],
    'combinação/interior_m': ['combinação', 'tecido_micro', '0']
}

# Motor v1
motor_dict_v1 = {
}


sql_to_code_renaming = {
    'VHE_Number': 'Nº Stock',
    'Colour_Ext_Desc': 'Cor',
    'Colour_Ext_Code': 'Colour_Ext_Code',
    'Colour_Int_Desc': 'Interior',
    'Model_Desc': 'Modelo',
    'Version_Desc': 'Versão',
    'Version_Code': 'Version_Code',
    'Optional_Desc': 'Opcional',
    'Site_Desc': 'Local da Venda',
    # 'Order_Type_Desc': 'Tipo Encomenda',
    'Purchase_Date': 'Data Compra',
    'Sell_Date': 'Data Venda',
    'Margin': 'Margem',
    'Estimated_Cost': 'Custo',
    'Registration_Number': 'Registration_Number',
    'Franchise_Code': 'Franchise_Code',
    'Fuel_Type_Desc': 'Combustível'
}

column_sql_renaming = {
        'Jantes': 'Rims_Size',
        'Caixa Auto': 'Auto_Trans',
        'Combustível': 'Fuel_Type',
        'Câmara Traseira': 'Rear_Cam',
        'Sensores Est. Tras.': 'Park_Front_Sens',
        'Sensores Est. Front.': 'Park_Rear_Sens',
        'Cor_Exterior': 'Colour_Ext',
        'Modelo': 'Model_Code',
        'Motor': 'Motor_Desc',
        'Local da Venda': 'Sales_Place',
        'Margem': 'Margin',
        'margem_percentagem': 'Margin_Percentage',
        'price_total': 'Sell_Value',
        'Data Venda': 'Sell_Date',
        'Data Compra': 'Purchase_Date',
        'buy_day': 'Purchase_Day',
        'buy_month': 'Purchase_Month',
        'buy_year': 'Purchase_Year',
        'score_euros': 'Score_Euros',
        'stock_days': 'Stock_Days',
        'days_stock_price': 'Stock_Days_Price',
        'proba_0': 'Probability_0',
        'proba_1': 'Probability_1',
        'score_class_gt': 'Score_Class_GT',
        'score_class_pred': 'Score_Class_Pred',
        'AC Auto': 'AC_Auto', 'Alarme': 'Alarm',
        'Tipo_Interior': 'Interior_Type',
        'Versao': 'Version',
        'Nº Stock': 'VHE_Number',
        'average_percentage_margin': 'Average_Margin_Percentage',
        'average_percentage_margin_local': 'Average_Margin_Percentage_Local',
        'average_percentage_margin_local_v2': 'Average_Margin_Percentage_Local_v2',
        'average_percentage_margin_local_Fase2_level_1': 'Average_Margin_Percentage_Local_Fase2_Level_1',
        'average_percentage_margin_local_Fase2_level_2': 'Average_Margin_Percentage_Local_Fase2_Level_2',
        'average_score_euros': 'Average_Score_Euros',
        'average_score_euros_local': 'Average_Score_Euros_Local',
        'average_score_euros_local_v2': 'Average_Score_Euros_Local_v2',
        'average_score_euros_local_Fase2_level_1': 'Average_Score_Euros_Local_Fase2_Level_1',
        'average_score_euros_local_Fase2_level_2': 'Average_Score_Euros_Local_Fase2_Level_2',
        'average_stock_days': 'Average_Stock_Days',
        'average_stock_days_local': 'Average_Stock_Days_Local',
        'average_stock_days_local_v2': 'Average_Stock_Days_Local_v2',
        'average_stock_days_local_Fase2_level_1': 'Average_Stock_Days_Local_Fase2_Level_1',
        'average_stock_days_local_Fase2_level_2': 'Average_Stock_Days_Local_Fase2_Level_2',
        'average_score': 'Average_Score_Class_GT',
        'average_score_local': 'Average_Score_Class_GT_Local',
        'average_score_local_v2': 'Average_Score_Class_GT_Local_v2',
        'average_score_local_Fase2_level_1': 'Average_Score_Class_GT_Fase2_Level_1',
        'average_score_local_Fase2_level_2': 'Average_Score_Class_GT_Fase2_Level_2',
        'average_score_pred': 'Average_Score_Class_Pred',
        'average_score_pred_local': 'Average_Score_Class_Pred_Local',
        'average_score_pred_local_v2': 'Average_Score_Class_Pred_Local_v2',
        'nr_cars_sold': 'Number_Cars_Sold',
        'nr_cars_sold_local': 'Number_Cars_Sold_Local',
        'nr_cars_sold_local_v2': 'Number_Cars_Sold_Local_v2',
        'nr_cars_sold_local_Fase2_level_1': 'Number_Cars_Sold_Local_Fase2_Level_1',
        'nr_cars_sold_local_Fase2_level_2': 'Number_Cars_Sold_Local_Fase2_Level_2',
}


columns_for_sql = ['Auto_Trans', 'Rear_Cam', 'Fuel_Type', 'Rear_Front_Sens', 'Navigation', 'Park_Front_Sens', 'Rims_Size', 'Colour_Ext', 'Sales_Place',
                   'Sales_Place_v2', 'Model_Code', 'Margin', 'Margin_Percentage',
                   'Stock_Days_Price', 'Score_Euros', 'Stock_Days', 'Sell_Value', 'Probability_0', 'Probability_1', 'Score_Class_GT',
                   'Score_Class_Pred', 'Sell_Date',
                   'Interior_Type', 'Version', 'Motor_Desc', 'Average_Margin_Percentage', 'Average_Score_Euros',
                   'Average_Stock_Days', 'Average_Score_Class_GT', 'Average_Score_Class_Pred', 'Number_Cars_Sold', 'Number_Cars_Sold_Local', 'Number_Cars_Sold_Local_v2',
                   'Average_Margin_Percentage_Local', 'Average_Margin_Percentage_Local_v2', 'Average_Score_Euros_Local', 'Average_Score_Euros_Local_v2',
                   'Average_Stock_Days_Local', 'Average_Stock_Days_Local_v2', 'Average_Score_Class_GT_Local', 'Average_Score_Class_GT_Local_v2',
                   'Average_Score_Class_Pred_Local', 'Average_Score_Class_Pred_Local_v2', 'Registration_Number', 'VHE_Number', 'Sales_Place_Fase2_Level_1', 'Sales_Place_Fase2_Level_2',
                   'Average_Margin_Percentage_Local_Fase2_Level_1', 'Average_Margin_Percentage_Local_Fase2_Level_2', 'Average_Score_Euros_Local_Fase2_Level_1', 'Average_Score_Euros_Local_Fase2_Level_2',
                   'Average_Stock_Days_Local_Fase2_Level_1', 'Average_Stock_Days_Local_Fase2_Level_2', 'Number_Cars_Sold_Local_Fase2_Level_1', 'Number_Cars_Sold_Local_Fase2_Level_2']

columns_for_sql_temp = ['Auto_Trans', 'Park_Front_Sens', 'Rims_Size', 'Colour_Ext', 'Sales_Place',
                        'Model_Code', 'Margin', 'Margin_Percentage',
                        'Stock_Days_Price', 'Score_Euros', 'Stock_Days', 'Sell_Value',
                        'Sell_Date',
                        'Interior_Type', 'Version', 'Motor_Desc', 'Average_Margin_Percentage', 'Average_Score_Euros',
                        'Average_Stock_Days', 'Number_Cars_Sold', 'Number_Cars_Sold_Local',
                        'Average_Margin_Percentage_Local', 'Average_Score_Euros_Local',
                        'Average_Stock_Days_Local', 'Registration_Number', 'VHE_Number',
                        'Average_Score_Class_GT', 'Average_Score_Class_GT_Local', 'Score_Class_GT', 'Rear_Cam', 'Fuel_Type', 'Rear_Front_Sens']

column_checkpoint_sql_renaming = {
    'NLR_Code': 'NLR_Code',
    'Jantes': 'Rims_Size',
    'Caixa Auto': 'Auto_Trans',
    'Combustível': 'Fuel_Type',
    'Câmara Traseira': 'Rear_Cam',
    'Sensores Est. Tras.': 'Park_Front_Sens',
    'Sensores Est. Front.': 'Park_Rear_Sens',
    'Cor_Exterior': 'Colour_Ext',
    'Modelo': 'Model_Code',
    'Local da Venda': 'Sales_Place',
    'Margem': 'Margin',
    'margem_percentagem': 'Margin_Percentage',
    'price_total': 'Sell_Value',
    'Data Compra': 'Purchase_Date',
    'Data Venda': 'Sell_Date',
    'score_euros': 'Score_Euros',
    'stock_days': 'Stock_Days',
    'days_stock_price': 'Stock_Days_Price',
    'Tipo_Interior': 'Interior_Type',
    'Versao': 'Version',
    'Motor': 'Motor_Desc',
    'Nº Stock': 'VHE_Number',
    'score_class_gt': 'Score_Class_GT',
    'Registration_Number': 'Registration_Number',
    'nr_cars_sold': 'Number_Cars_Sold',
    'average_percentage_margin': 'Average_Margin_Percentage',
    'average_stock_days': 'Average_Stock_Days',
    'average_score': 'Average_Score_Class_GT',
    'average_score_euros': 'Average_Score_Euros',
    'nr_cars_sold_local': 'Number_Cars_Sold_Local',
    'average_percentage_margin_local': 'Average_Margin_Percentage_Local',
    'average_stock_days_local': 'Average_Stock_Days_Local',
    'average_score_local': 'Average_Score_Class_GT_Local',
    'average_score_euros_local': 'Average_Score_Euros_Local',
}

log_files = {
    'full_log': 'logs/optionals_cdsu.txt'
}

col_color_dict = {
    'Quantidade': 'Beige',
    'Motorização': 'FloralWhite',
    'Caixa Auto.': 'FloralWhite',
    'Cor Exterior': 'FloralWhite',
    'Tam. Jantes': 'FloralWhite',
    'Jantes': 'FloralWhite',
    'Sensores Est. Tras.': 'FloralWhite',
    'Sensores Est. Front.': 'FloralWhite',
    'Barras Tej.': 'FloralWhite',
    'Tipo Interior': 'FloralWhite',
    'Combustível': 'FloralWhite',
    'Câmara Traseira': 'FloralWhite',
    'Versão': 'FloralWhite',
    '#Vendas Local': 'Lavender',
    '#Vendas Global': 'LightGrey',
    'Score (€)': 'LightBlue'
}

col_decimals_place_dict = {
    'Quantidade': '{:.0f}',
    'Score (€)': '{:.3f}',
    '#Vendas Local': '{:.0f}',
    '#Vendas Global': '{:.0f}',
}
