import numpy as np
import os
from sklearn import tree, linear_model, neighbors, svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from py_dotenv import read_dotenv
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
dotenv_path = 'info.env'
read_dotenv(dotenv_path)

# Options:
margin_threshold = 3.5
stock_days_threshold = 45
update_frequency_days = 15

DSN = os.getenv('DSN')
UID = os.getenv('UID')
PWD = os.getenv('PWD')

# Dictionaries:
sql_info = {
    'database': 'BI_MLG',
    'checkpoint_b_table': 'VHE_Fact_Checkpoint_B_OrderOptimization',
    'feature_contribution': 'VHE_Fact_Feature_Contribution',
    'performance_running_time': 'LOG_Performance_Running_Time',
    'performance_algorithm_results': 'LOG_Performance_Algorithms_Results',
    'final_table': 'VHE_Fact_DW_OrderOptimization',
}

# Old Cor_Exterior
color_ext_dict = {
    'preto': ['preto'],
    'cinzento': ['cinzento', 'prateado', 'prata', 'cinza'],
    'branco': ['branco'],
    'azul': ['azul', 'bluestone'],
    'verde': ['verde'],
    'vermelho/laranja': ['vermelho', 'laranja'],
    'burgundy': ['burgundy'],
    'castanho': ['castanho', 'terra', 'jatoba'],
    'outros': ['aqua', 'storm', 'cedar', 'bronze', 'chestnut', 'havanna', 'cashmere', 'champagne', 'dourado', 'amarelo', 'bege', 'silverstone', 'moonstone']
}

# # New Cor_Exterior
color_ext_dict_layer_1 = {
    'preto': ['preto'],
    'branco': ['branco'],
    'cinzento': ['cinzento', 'prateado', 'prata', 'cinza', 'bluestone'],
    'azul': ['azul'],
    'outros': ['undefined', 'castanho', 'terra', 'jatoba', 'burgundy', 'aqua', 'storm', 'cedar', 'bronze', 'chestnut', 'havanna', 'cashmere', 'champagne', 'dourado', 'amarelo', 'bege', 'silverstone', 'moonstone', 'verde', 'vermelho', 'laranja']
}


# New Cor_Exterior (v2)
# color_ext_dict_layer_1 = {
#     'preto': ['preto'],
#     'branco': ['branco'],
#     'cinzento': ['cinzento', 'prateado', 'prata', 'cinza', 'bluestone'],
#     'azul/outros': ['azul', 'castanho', 'terra', 'jatoba', 'burgundy', 'aqua', 'storm', 'cedar', 'bronze', 'chestnut', 'havanna', 'cashmere', 'champagne', 'dourado', 'amarelo', 'bege', 'silverstone', 'moonstone', 'verde', 'vermelho', 'laranja']
# }

# Old Cor_Interior
color_int_dict = {
    'preto': ['preto', 'prata/preto/preto', 'veneto/preto', 'preto/preto', 'ambar/preto/preto'],
    'antracite': ['antracite', 'antracite/cinza/preto', 'antracite/preto', 'antracite/vermelho/preto', 'antracite/vermelho', 'anthtacite/preto', 'anthracite/silver'],
    'castanho': ['castanho', 'oak', 'terra', 'mokka', 'vernasca'],
    'others': ['champagne', 'branco', 'oyster', 'prata/cinza', 'bege', 'oyster/preto', 'azul', 'cinzento', 'truffle', 'burgundy', 'zagora/preto', 'sonoma/preto', 'laranja', 'taupe/preto', 'vermelho', 'silverstone', 'nevada', 'cognac/preto', 'preto/laranja', 'preto/prateado']
}

dakota_colors = ['oyster/cinza', 'black/contrast', 'preto/preto', 'preto/vermelho/preto', 'preto/oyster', 'preto/debroado', 'preto/azul/preto', 'oyster/preto', 'ivory/preto', 'ivory', 'coral', 'preto', 'preta', 'branco', 'branca', 'bege', 'veneto/preto', 'oyster', 'oyster/oyster', 'castanho', 'terra', 'conhaque', 'conhaque/castanho/preto', 'conhaque/castanho/preto/conhaque', 'cognac/preto', 'brown', 'azul', 'mocha/preto', 'mocha/preto/mocha', 'mocha']
nappa_colors = ['preto', 'white', 'ivory', 'ivory/branco', 'bege', 'mocha', 'castanho', 'cognac/preto', 'antracite']
vernasca_colors = ['canberra', 'preta', 'preto/com', 'preto/preto', 'oyster', 'mocha', 'mocha/preto', 'cognac', 'azuis/preto', 'anthracite/preto']
nevada_colors = ['terra', 'brown', 'preto/preto', 'bege', 'oyster', 'oyster/leather', 'preto', 'branco', 'mocha']
merino_colors = ['preto', 'bege', 'castanho', 'silverstone', 'preto/preto', 'branco', 'laranja', 'taupe/preto', 'vermelho']


# New Cor_Interior v2
color_int_dict_layer_1 = {
    'preto': ['nappa_antracite', 'vernasca_anthracite/preto', 'merino_preto', 'nevada_preto', 'merino_preto/preto', 'nevada_preto/preto', 'vernasca_preta', 'vernasca_preto/com', 'vernasca_preto/preto', 'preto', 'dakota_preto/preto', 'dakota_preto/vermelho/preto', 'dakota_preto/oyster', 'dakota_preto/debroado', 'dakota_preto/azul/preto', 'dakota_preto', 'dakota_preta', 'dakota_black/contrast', 'nappa_preto'],
    'branco': ['nevada_branco', 'merino_branco', 'dakota_oyster/preto', 'dakota_ivory/preto', 'dakota_ivory', 'dakota_branco', 'dakota_white', 'nappa_white', 'nappa_ivory', 'nappa_ivory/branco'],
    'vermelho': ['vermelho', 'merino_vermelho', 'dakota_coral'],
    'bege': ['merino_bege', 'dakota_bege', 'nappa_bege', 'vernasca_canberra', 'nevada_bege', 'bege'],
    'oyster': ['dakota_oyster', 'dakota_oyster/oyster', 'dakota_oyster/cinza', 'vernasca_oyster', 'nevada_oyster', 'nevada_oyster/leather', 'oyster'],
    'castanho': ['castanho', 'merino_castanho', 'nevada_terra', 'nevada_brown', 'vernasca_mocha', 'vernasca_mocha/preto', 'vernasca_cognac', 'nappa_castanho', 'nappa_cognac/preto', 'dakota_castanho', 'dakota_conhaque', 'dakota_conhaque/castanho/preto', 'dakota_conhaque/castanho/preto/conhaque', 'dakota_cognac/preto', 'dakota_brown', 'dakota_terra'],
    'azul': ['dakota_azul', 'vernasca_azuis/preto'],
    'mocha': ['dakota_mocha/preto', 'dakota_mocha/preto/mocha', 'dakota_mocha', 'nappa_mocha', 'nevada_mocha'],
    'cinzento': ['merino_silverstone', 'merino_taupe/preto', 'cinzento'],
    'laranja': ['merino_laranja'],
    'amarelo': ['amarelo'],
    'mini/mota': ['mini/mota'],
    'others': ['0', 0]
}

color_int_dict_layer_2 = {
    'preto': ['preto'],
    'castanho/mocha': ['castanho', 'mocha'],
    'bege/oyster/branco': ['bege', 'oyster', 'branco'],
    'outros': ['amarelo', 'vermelho', 'azul', 'laranja', 'cinzento', 'others'],
}

# Old Jantes
# jantes_dict = {
#     'standard': ['standard', '15', '16'],
#     '17': ['17'],
#     '18': ['18'],
#     '19/20': ['19', '20']
# }

# New Jantes
jantes_dict = {
    '16': ['16'],
    '17': ['17'],
    '18': ['18'],
    'stand/19/20': ['standard', '19', '20']
}

sales_place_dict = {
    'centro': ['DCV - Coimbrões', 'DCC - Aveiro'],
    'norte': ['DCC - Feira', 'DCG - Gaia', 'DCN-Porto', 'DCN-Porto Mini', 'DCG - Gaia Mini', 'DCN-Porto Usados', 'DCG - Gaia Usados', 'DCC - Feira Usados', 'DCC - Aveiro Usados', 'DCC - Viseu Usados'],
    'sul': ['DCS-Expo Frotas Busi', 'DCS-V Especiais BMW', 'DCS-V Especiais MINI', 'DCS-Expo Frotas Flee', 'DCS-Cascais', 'DCS-Parque Nações', 'DCS-Parque Nações Mi', 'DCS-24 Jul BMW Usad', 'DCS-Cascais Usados', 'DCS-24 Jul MINI Usad', 'DCS-Lisboa Usados'],
    'algarve': ['DCA - Faro', 'DCA - Portimão', 'DCA - Mini Faro', 'DCA -Portimão Usados'],
    'motorcycles': ['DCA - Motos Faro', 'DCS- Vendas Motas', 'DCC - Motos Aveiro']
}

model_dict = {
    's2_gran': ['S2 Gran Tourer'],
    's2_active': ['S2 Active Tourer'],
    's3_touring': ['S3 Touring'],
    's3_berlina': ['S3 Berlina'],
    's4_gran': ['S4 Gran Coupé'],
    's5_touring': ['S5 Touring'],
    's5_lim_ber': ['S5 Limousine', 'S5 Berlina'],
    's1': ['S1 3p', 'S1 5p'],
    'x1': ['X1'],
    'x3': ['X3 SUV'],
    'mini_club': ['MINI CLUBMAN'],
    'mini_cabrio': ['MINI CABRIO'],
    'mini_country': ['MINI COUNTRYMAN'],
    'mini': ['MINI 5p', 'MINI 3p'],
    'motos': ['Série C', 'Série F', 'Série K', 'Série R'],
    'outros': ['S2 Cabrio', 'S2 Coupé', 'S3 Gran Turismo', 'S4 Coupé', 'S4 Cabrio', 'S5 Gran Turismo', 'S6 Cabrio', 'S6 Gran Turismo', 'S6 Gran Coupe', 'S6 Coupé', 'S7 Berlina', 'S7 L Berlina', 'X2 SAC', 'X4 SUV', 'X5 SUV', 'X5 M', 'X6', 'X6 M', 'Z4 Roadster', 'M2 Coupé', 'M3 Berlina', 'M4 Cabrio', 'M4 Coupé', 'M5 Berlina', 'S6 Gran Turismo', 'S6 Cabrio', 'S6 Coupé', 'S6 Gran Coupe', 'S7 Berlina', 'S7 L Berlina']
}


versao_dict = {
    'advantage': ['advantage'],
    'sport': ['line_sport'],
    'base': ['base'],
    'luxury': ['line_luxury'],
    'xline': ['xline'],
    'urban/desportiva': ['line_urban', 'desportiva_m', 'pack_desportivo_m']
}

# v2
tipo_int_dict = {
    'tecido': ['tecido'],
    'pele': ['pele'],
    'combinação/interior_m': ['combinação', 'tecido_micro', 0, '0']
}

classification_models = {
    'dt': [tree.DecisionTreeClassifier, [{'min_samples_leaf': [3, 5, 7, 9, 10, 15, 20, 30], 'max_depth': [3, 5, 6], 'class_weight': ['balanced']}]],
    'rf': [RandomForestClassifier, [{'n_estimators': [10, 25, 50, 100], 'max_depth': [5, 10, 20], 'class_weight': ['balanced']}]],
    'lr': [linear_model.LogisticRegression, [{'C': np.logspace(-2, 2, 20)}]],
    'knn': [neighbors.KNeighborsClassifier, [{'n_neighbors': np.arange(1, 50, 1)}]],
    'svm': [svm.SVC, [{'C': np.logspace(-2, 2, 10)}]],
    'ab': [AdaBoostClassifier, [{'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}]],
    'gc': [GradientBoostingClassifier, [{'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}]],
    'bayes': [GaussianNB],  # ToDo: Need to create an exception for this model
    'ann': [MLPClassifier, [{'activation': ['identity', 'logistic', 'tanh', 'relu'], 'alpha': [1e-5], 'solver': ['sgd']}]],
    'voting': [VotingClassifier, [{'voting': ['soft']}]]
}

column_sql_renaming = {
        'Jantes_new': 'Rims_Size',
        'Caixa Auto': 'Auto_Trans',
        'Navegação': 'Navigation',
        'Sensores': 'Park_Front_Sens',
        'Cor_Interior_new': 'Colour_Int',
        'Cor_Exterior_new': 'Colour_Ext',
        'Modelo_new': 'Model_Code',
        'Local da Venda_new': 'Sales_Place',
        'Margem': 'Margin', 'margem_percentagem':
        'Margin_Percentage',
        'price_total': 'Sell_Value',
        'Data Venda': 'Sell_Date',
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
        '7_Lug': 'Seven_Seats',
        'AC Auto': 'AC_Auto', 'Alarme': 'Alarm',
        'Barras_Tej': 'Roof_Bars',
        'Teto_Abrir': 'Open_Roof',
        'Farois_LED': 'LED_Lights',
        'Farois_Xenon': 'Xenon_Lights',
        'Prot.Solar': 'Solar_Protection',
        'Tipo_Interior_new': 'Interior_Type',
        'Versao_new': 'Version',
        'average_percentage_margin': 'Average_Margin_Percentage',
        'average_percentage_margin_local': 'Average_Margin_Percentage_Local',
        'average_score_euros': 'Average_Score_Euros',
        'average_score_euros_local': 'Average_Score_Euros_Local',
        'average_stock_days': 'Average_Stock_Days',
        'average_stock_days_local': 'Average_Stock_Days_Local',
        'average_score': 'Average_Score_Class_GT',
        'average_score_local': 'Average_Score_Class_GT_Local',
        'average_score_pred': 'Average_Score_Class_Pred',
        'average_score_pred_local': 'Average_Score_Class_Pred_Local',
        'nr_cars_sold': 'Number_Cars_Sold',
        'nr_cars_sold_local': 'Number_Cars_Sold_Local'
}

columns_for_sql = ['Auto_Trans', 'Navigation', 'Park_Front_Sens', 'Rims_Size', 'Colour_Int', 'Colour_Ext', 'Sales_Place',
                   'Model_Code', 'Purchase_Day', 'Purchase_Month', 'Purchase_Year', 'Margin', 'Margin_Percentage',
                   'Stock_Days_Price', 'Score_Euros', 'Stock_Days', 'Sell_Value', 'Probability_0', 'Probability_1', 'Score_Class_GT',
                   'Score_Class_Pred', 'Sell_Date', 'Seven_Seats', 'AC_Auto', 'Alarm', 'Roof_Bars', 'Open_Roof', 'LED_Lights',
                   'Xenon_Lights', 'Solar_Protection', 'Interior_Type', 'Version', 'Average_Margin_Percentage', 'Average_Score_Euros',
                   'Average_Stock_Days', 'Average_Score_Class_GT', 'Average_Score_Class_Pred', 'Number_Cars_Sold', 'Number_Cars_Sold_Local',
                   'Average_Margin_Percentage_Local', 'Average_Score_Euros_Local', 'Average_Stock_Days_Local', 'Average_Score_Class_GT_Local',
                   'Average_Score_Class_Pred_Local']

column_performance_sql_renaming = {
    'start_section_a': 'Section_A_Start',
    'start_section_b': 'Section_B_Start',
    'checkpoint_b1': 'Checkpoint_B1',
    'start_section_c': 'Section_C_Start',
    'start_section_d': 'Section_D_Start',
    'start_section_e': 'Section_E_Start',
    'end_section_a': 'Section_A_End',
    'end_section_b': 'Section_B_End',
    'end_section_c': 'Section_C_End',
    'end_section_d': 'Section_D_End',
    'end_section_e': 'Section_E_End',
    'start_modelo': 'Model_Code_Start',
    'end_modelo': 'Model_Code_End',
    'start_nav_all': 'Navigation_Start',
    'end_nav_all': 'Navigation_End',
    'start_barras_all': 'Roof_Bars_Start',
    'end_barras_all': 'Roof_Bars_End',
    'start_alarme_all': 'Alarm_Start',
    'end_alarme_all': 'Alarm_End',
    'start_7_lug_all': 'Seven_Seats_Start',
    'end_7_lug_all': 'Seven_Seats_End',
    'start_prot_all': 'Solar_Protection_Start',
    'end_prot_all': 'Solar_Protection_End',
    'start_ac_all': 'AC_Auto_Start',
    'end_ac_all': 'AC_Auto_End',
    'start_teto_all': 'Open_Roof_Start',
    'end_teto_all': 'Open_Roof_End',
    'duration_versao_all': 'Version_Duration',
    'duration_trans_all': 'Auto_Trans_Duration',
    'duration_sens_all': 'Park_Front_Sens_Duration',
    'duration_jantes_all': 'Rims_Size_Duration',
    'duration_farois_all': 'Lights_Duration',
    'start_cor_ext_all': 'Colour_Ext_Start',
    'end_cor_ext_all': 'Colour_Ext_End',
    'start_cor_int_all': 'Colour_Int_Start',
    'end_cor_int_all': 'Colour_Int_End',
    'start_int_type_all': 'Interior_Type_Start',
    'end_int_type_all': 'Interior_Type_End',
    'start_standard': 'Standard_Start',
    'end_standard': 'Standard_End'
}

column_checkpoint_sql_renaming = {
    'Jantes_new': 'Rims_Size',
    'Caixa Auto': 'Auto_Trans',
    'Navegação': 'Navigation',
    'Sensores': 'Park_Front_Sens',
    'Cor_Interior_new': 'Colour_Int',
    'Cor_Exterior_new': 'Colour_Ext',
    'Modelo_new': 'Model_Code',
    'Local da Venda_new': 'Sales_Place',
    'Margem': 'Margin',
    'margem_percentagem': 'Margin_Percentage',
    'price_total': 'Sell_Value',
    'Data Compra': 'Buy_Date',
    'Data Venda': 'Sell_Date',
    'buy_day': 'Purchase_Day',
    'buy_month': 'Purchase_Month',
    'buy_year': 'Purchase_Year',
    'score_euros': 'Score_Euros',
    'stock_days': 'Stock_Days',
    'days_stock_price': 'Stock_Days_Price',
    '7_Lug': 'Seven_Seats',
    'AC Auto': 'AC_Auto',
    'Alarme': 'Alarm',
    'Barras_Tej': 'Roof_Bars',
    'Teto_Abrir': 'Open_Roof',
    'Farois_LED': 'LED_Lights',
    'Farois_Xenon': 'Xenon_Lights',
    'Prot.Solar': 'Solar_Protection',
    'Tipo_Interior_new': 'Interior_Type',
    'Versao_new': 'Version',
    'Nº Stock': 'Vehicle_Number',
    'average_score_dynamic': 'Average_Score_Dynamic',
    'average_score_dynamic_std': 'Average_Score_Dynamic_STD',
    'average_score_global': 'Average_Score_Global',
    'last_margin': 'Margin_Last',
    'last_score': 'Score_Last',
    'last_stock_days': 'Stock_Days_Last',
    'margin_class': 'Margin_Class',
    'max_score_global': 'Max_Score_Global',
    'median_score_global': 'Median_Score_Global',
    'min_score_global': 'Min_Score_Global',
    'new_score': 'New_Score',
    'number_prev_sales': 'Number_Previous_Sales',
    'prev_average_score_dynamic': 'Prev_Average_Score_Dynamic',
    'prev_average_score_dynamic_std': 'Prev_Average_Score_Dynamic_STD',
    'prev_sales_check': 'Previous_Sales_Check',
    'q1_score_global': 'Q1_Score_Global',
    'q3_score_global': 'Q3_Score_Global',
    'stock_days_class': 'Stock_Days_Class',
}



