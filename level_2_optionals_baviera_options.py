import numpy as np
import os
from sklearn import tree, linear_model, neighbors, svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from py_dotenv import read_dotenv
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost as xgb
dotenv_path = 'info.env'
read_dotenv(dotenv_path)

# Options:
margin_threshold = 3.5
stock_days_threshold = 45
update_frequency_days = 0
metric, metric_threshold = 'ROC_Curve', 0.70  # The metric to compare on the final models and the minimum threshold to consider;
k, gridsearch_score = 10, 'recall'  # Stratified Cross-Validation number of Folds and the Metric on which to optimize GridSearchCV
selected_configuration_parameters = ['Motor', 'Alarme', 'AC Auto', 'Barras_Tej', 'Caixa Auto', 'Cor_Exterior', 'Cor_Interior', 'Farois_LED', 'Farois_Xenon', 'Jantes', 'Modelo', 'Navegação', 'Sensores', 'Teto_Abrir', 'Tipo_Interior', 'Versao']
# Full: ['7_Lug', 'Alarme', 'AC Auto', 'Barras_Tej', 'Caixa Auto', 'Cor_Exterior', 'Cor_Interior', 'Farois_LED', 'Farois_Xenon', 'Jantes', 'Modelo', 'Navegação', 'Prot.Solar', 'Sensores', 'Teto_Abrir', 'Tipo_Interior', 'Versao']

DSN_MLG = os.getenv('DSN_MLG')
UID = os.getenv('UID')
PWD = os.getenv('PWD')
# EMAIL = os.getenv('EMAIL')
# EMAIL_PASS = os.getenv('EMAIL_PASS')
configuration_parameters_full = ['Motor', 'Alarme', 'AC Auto', 'Barras_Tej', 'Caixa Auto', 'Cor_Exterior', 'Cor_Interior', 'Farois_LED', 'Farois_Xenon', 'Jantes', 'Modelo', 'Navegação', 'Sensores', 'Teto_Abrir', 'Tipo_Interior', 'Versao']

# Dictionaries:
sql_info = {
    'database': 'BI_MLG',
    'database_final': 'BI_MLG',
    # 'log_record': 'LOG_OrderOptimization',
    'initial_table': 'VHE_Fact_DW_SalesNew_WithSpecs',
    'checkpoint_b_table': 'VHE_Fact_Checkpoint_B_OrderOptimization',
    'feature_contribution': 'VHE_Fact_Feature_Contribution',
    # 'performance_running_time': 'LOG_Performance_Running_Time',
    # 'performance_algorithm_results': 'LOG_Performance_Algorithms_Results',
    'final_table': 'VHE_Fact_BI_OrderOptimization',
    # 'error_log': 'LOG_Performance_Errors',
    # 'warning_log': 'LOG_Performance_Warnings',
    # 'model_choices': 'LOG_Performance_Model_Choices',
    # 'mail_users': 'LOG_MailUsers',
}

project_id = 2162

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
    'centro': ['DCV - Coimbrões', 'DCC - Aveiro', 'DCC - Aveiro Usados', 'DCC - Viseu Usados',  'DCV - Viseu Usados'],
    'norte': ['DCC - Feira', 'DCG - Gaia', 'DCN-Porto', 'DCN-Porto Mini', 'DCG - Gaia Mini', 'DCN-Porto Usados', 'DCG - Gaia Usados', 'DCC - Feira Usados', 'DCN-Maia'],
    'sul': ['DCS-Expo Frotas Busi', 'DCS-V Especiais BMW', 'DCS-V Especiais MINI', 'DCS-Expo Frotas Flee', 'DCS-Cascais', 'DCS-Parque Nações', 'DCS-Parque Nações Mi', 'DCS-24 Jul BMW Usad', 'DCS-Cascais Usados', 'DCS-24 Jul MINI Usad', 'DCS-Lisboa Usados'],
    'algarve': ['DCA - Faro', 'DCA - Portimão', 'DCA - Mini Faro', 'DCA -Portimão Usados'],
    'motorcycles': ['DCA - Motos Faro', 'DCS- Vendas Motas', 'DCC - Motos Aveiro']
}

sales_place_dict_v2 = {'porto': ['DCN-Porto', 'DCN-Porto Mini', 'DCN-Porto Usados',  'DCN-Maia'],
                       'gaia': ['DCC - Feira', 'DCG - Gaia', 'DCG - Gaia Mini', 'DCG - Gaia Usados', 'DCC - Feira Usados'],
                       'aveiro': ['DCV - Coimbrões', 'DCC - Aveiro', 'DCC - Aveiro Usados', 'DCC - Viseu Usados',  'DCV - Viseu Usados'],
                       'lisboa': ['DCS-Expo Frotas Busi', 'DCS-V Especiais BMW', 'DCS-V Especiais MINI', 'DCS-Expo Frotas Flee', 'DCS-Cascais', 'DCS-Parque Nações', 'DCS-Parque Nações Mi', 'DCS-24 Jul BMW Usad', 'DCS-Cascais Usados', 'DCS-24 Jul MINI Usad', 'DCS-Lisboa Usados'],
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
    'outros': ['S8 Coupe', 'S2 Cabrio', 'S2 Coupé', 'S3 Gran Turismo', 'S4 Coupé', 'S4 Cabrio', 'S5 Gran Turismo', 'S6 Cabrio', 'S6 Gran Turismo', 'S6 Gran Coupe', 'S6 Coupé', 'S7 Berlina', 'S7 L Berlina', 'X2 SAC', 'X4 SUV', 'X5 SUV', 'X5 M', 'X6', 'X6 M', 'Z4 Roadster', 'M2 Coupé', 'M3 Berlina', 'M4 Cabrio', 'M4 Coupé', 'M5 Berlina', 'S6 Gran Turismo', 'S6 Cabrio', 'S6 Coupé', 'S6 Gran Coupe', 'S7 Berlina', 'S7 L Berlina']
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
    'combinação/interior_m': ['combinação', 'tecido_micro', 0, '0']
}

# Motor v1
motor_dict_v1 = {
    '16d': ['114d', '116d', '214d', '216d', '316d', 'sdrive16d'],
    '16i': ['114i', '116i', '216i'],
    '18d': ['118d', '218d', '318d', '418d', '518d', 'sdrive18d', 'xdrive18d'],
    '18i': ['118i', '218i', 'sdrive18i'],
    '20d': ['120d', '220d', '320d', '420d', '520d', '620d', 'sdrive20d', 'xdrive20d', 'sdrive'],
    '20i': ['125i', '320i', '420i', '520i', 'sdrive20i', 'xdrive20i'],
    '25d': ['125d', '225d', '325d', '425d', '525d', '725d', 'xdrive25d', 'sdrive25d'],
    '25i': ['xdrive25i'],
    '28i': ['228i', '428i', '528i'],
    '30d': ['330d', '430d', '530d', '630d', '730d', '730ld', 'xdrive30d', 'xdrive'],
    '30i': ['m235i', '230i', '330i', '430i', '530i', 'xdrive30i'],
    '35d': ['335d', '435d', '535d', 'xdrive35d'],
    '35i': ['335i', '435i', 'xdrive35i'],
    '40d': ['540d', '640d', '740ld', '740d', 'xdrive40d', '840d'],
    '40i': ['m140i', '340i', '440i', '540i', '640i', '740i', '740li', 'm40i'],
    '50d': ['m550d', '750d', '750ld', 'm50d'],
    '50i': ['750li', 'm850i'],
    'Híbrido': ['225xe', '330e', 'activehybrid', '530e', '740e', '740le', 'xdrive40e'],
    'M2': ['m2'],
    'M3': ['m3'],
    'M4': ['m4'],
    'M5': ['m5'],
    'M6': ['m6'],
    'M': ['m'],
}


# Motor v2
motor_dict_v2 = {
    '1.5d': ['114d', '116d', '214d', '216d', '316d', 'sdrive16d'],
    '1.5i': ['114i', '116i', '118i', '216i', '218i', 'sdrive18i'],
    '2.0d': ['118d', '120d', '125d', '218d', '220d', '225d', '318d', '320d', '325d', '418d', '420d', '425d', '518d', '520d', '525d', '620d', '725d', 'sdrive', 'sdrive18d', 'sdrive20d', 'sdrive25d', 'xdrive18d', 'xdrive20d', 'xdrive25d', 'xdrive25d'],
    '2.0i': ['125i', '228i', '230i', '320i', '330i', '420i', '428i', '430i', '520i', '528i', '530i', 'sdrive20i', 'xdrive20i', 'xdrive25i', 'xdrive30i'],
    '3.0d': ['330d', '335d', '430d', '435d', '530d', '535d', '540d', '630d', '640d', '730d', '730ld', '740d', '740ld', '750d', '750ld', 'm550d', 'xdrive30d', 'xdrive35d', 'm50d', 'xdrive', 'xdrive40d', '840d'],
    '3.0i': ['335i', '340i', '435i', '440i', '540i', '640i', '740i', '740li', 'm140i', 'm2', 'm3', 'm4', 'm40i', 'xdrive35i', 'm235i'],
    '4.0i': ['750li', 'm5', 'm6', 'm', 'm850i'],
    'Híbrido': ['225xe', '330e', '530e', '740e', '740le', 'activehybrid', 'xdrive40e'],
}

# motor_dict_v2_layer_2 = {
#     '1.5': ['1.5d', '1.5i'],
#     '2.0': ['2.0d', '2.0i'],
#     '3.0': ['3.0d', '3.0i'],
#     'Hybrid/4.0': ['Híbrido', '4.0i'],
# }

motor_dict_v2_layer_2 = {
    'gasoline': ['1.5d', '2.0d', '3.0d'],
    'diesel': ['1.5i', '2.0i', '3.0i', '4.0i'],
    'hybrid': ['Híbrido'],
}


classification_models = {
    'dt': [tree.DecisionTreeClassifier, [{'min_samples_leaf': [3, 5, 7, 9, 10, 15, 20, 30], 'max_depth': [3, 5, 6], 'class_weight': ['balanced']}]],
    'rf': [RandomForestClassifier, [{'n_estimators': [10, 25, 50, 100, 200, 500, 1000], 'max_depth': [5, 10, 20], 'class_weight': ['balanced']}]],
    'lr': [linear_model.LogisticRegression, [{'C': np.logspace(-2, 2, 20), 'solver': ['liblinear']}]],
    'knn': [neighbors.KNeighborsClassifier, [{'n_neighbors': np.arange(1, 50, 1)}]],
    'svm': [svm.SVC, [{'C': np.logspace(-2, 2, 10)}]],
    'ab': [AdaBoostClassifier, [{'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}]],
    'gc': [GradientBoostingClassifier, [{'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}]],
    # 'xgb': [xgb.XGBClassifier, [{'objective': ['binary:logistic'], 'booster': ['gbtree'], 'max_depth': [5, 10, 20, 50], 'reg_lambda': [0.0001, 0.001, 0.01, 0.1, 1]}]],
    'xgb': [xgb.XGBClassifier, [{'objective': ['binary:logistic'], 'booster': ['gbtree'], 'max_depth': [5, 10, 20, 50, 100]}]],  # ToDo: need to add L1 (reg_alpha) and L2 (reg_lambda) regularization to counter the overfitting
    'bayes': [GaussianNB],  # ToDo: Need to create an exception for this model
    'ann': [MLPClassifier, [{'activation': ['identity', 'logistic', 'tanh', 'relu'], 'hidden_layer_sizes': (100, 100), 'solver': ['sgd'], 'max_iter': [1000]}]],
    'voting': [VotingClassifier, [{'voting': ['soft']}]]
}

dict_models_name_conversion = {
    'dt': ['Decision Tree'],
    'rf': ['Random Forest'],
    'lr': ['Logistic Regression'],
    'knn': ['KNN'],
    'svm': ['SVM'],
    'ab': ['Adaboost'],
    'xgb': ['XGBoost'],
    'gc': ['Gradient'],
    'bayes': ['Bayesian'],
    'ann': ['ANN'],
    'voting': ['Voting']
}

sql_to_code_renaming = {
    'VHE_Number': 'Nº Stock',
    'Prov_Desc': 'Prov',
    'Colour_Ext_Desc': 'Cor',
    'Colour_Int_Desc': 'Interior',
    'Model_Desc': 'Modelo',
    'Version_Desc': 'Versão',
    'Optional_Desc': 'Opcional',
    'Sell_Location_Desc': 'Local da Venda',
    'Order_Type_Desc': 'Tipo Encomenda',
    'Purchase_Date': 'Data Compra',
    'Sell_Date': 'Data Venda',
    'Margin': 'Margem',
    'Estimated_Cost': 'Custo',
    'Registration_Number': 'Registration_Number',
    'Franchise_Code': 'Franchise_Code',
}

column_sql_renaming = {
        'Jantes': 'Rims_Size',
        'Caixa Auto': 'Auto_Trans',
        'Navegação': 'Navigation',
        'Sensores': 'Park_Front_Sens',
        'Cor_Interior': 'Colour_Int',
        'Cor_Exterior': 'Colour_Ext',
        'Modelo': 'Model_Code',
        'Motor': 'Motor_Desc',
        'Local da Venda': 'Sales_Place',
        'Local da Venda_v2': 'Sales_Place_v2',
        'Margem': 'Margin',
        'margem_percentagem': 'Margin_Percentage',
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
        # '7_Lug': 'Seven_Seats',
        'AC Auto': 'AC_Auto', 'Alarme': 'Alarm',
        'Barras_Tej': 'Roof_Bars',
        'Teto_Abrir': 'Open_Roof',
        'Farois_LED': 'LED_Lights',
        'Farois_Xenon': 'Xenon_Lights',
        # 'Prot.Solar': 'Solar_Protection',
        'Tipo_Interior': 'Interior_Type',
        'Versao': 'Version',
        'Nº Stock': 'VHE_Number',
        'average_percentage_margin': 'Average_Margin_Percentage',
        'average_percentage_margin_local': 'Average_Margin_Percentage_Local',
        'average_percentage_margin_local_v2': 'Average_Margin_Percentage_Local_v2',
        'average_score_euros': 'Average_Score_Euros',
        'average_score_euros_local': 'Average_Score_Euros_Local',
        'average_score_euros_local_v2': 'Average_Score_Euros_Local_v2',
        'average_stock_days': 'Average_Stock_Days',
        'average_stock_days_local': 'Average_Stock_Days_Local',
        'average_stock_days_local_v2': 'Average_Stock_Days_Local_v2',
        'average_score': 'Average_Score_Class_GT',
        'average_score_local': 'Average_Score_Class_GT_Local',
        'average_score_local_v2': 'Average_Score_Class_GT_Local_v2',
        'average_score_pred': 'Average_Score_Class_Pred',
        'average_score_pred_local': 'Average_Score_Class_Pred_Local',
        'average_score_pred_local_v2': 'Average_Score_Class_Pred_Local_v2',
        'nr_cars_sold': 'Number_Cars_Sold',
        'nr_cars_sold_local': 'Number_Cars_Sold_Local',
        'nr_cars_sold_local_v2': 'Number_Cars_Sold_Local_v2',
}

columns_for_sql = ['Auto_Trans', 'Navigation', 'Park_Front_Sens', 'Rims_Size', 'Colour_Int', 'Colour_Ext', 'Sales_Place',
                   'Sales_Place_v2', 'Model_Code', 'Margin', 'Margin_Percentage',
                   'Stock_Days_Price', 'Score_Euros', 'Stock_Days', 'Sell_Value', 'Probability_0', 'Probability_1', 'Score_Class_GT',
                   'Score_Class_Pred', 'Sell_Date', 'AC_Auto', 'Alarm', 'Roof_Bars', 'Open_Roof', 'LED_Lights',
                   'Xenon_Lights', 'Interior_Type', 'Version', 'Motor_Desc', 'Average_Margin_Percentage', 'Average_Score_Euros',
                   'Average_Stock_Days', 'Average_Score_Class_GT', 'Average_Score_Class_Pred', 'Number_Cars_Sold', 'Number_Cars_Sold_Local', 'Number_Cars_Sold_Local_v2',
                   'Average_Margin_Percentage_Local', 'Average_Margin_Percentage_Local_v2', 'Average_Score_Euros_Local', 'Average_Score_Euros_Local_v2',
                   'Average_Stock_Days_Local', 'Average_Stock_Days_Local_v2', 'Average_Score_Class_GT_Local', 'Average_Score_Class_GT_Local_v2',
                   'Average_Score_Class_Pred_Local', 'Average_Score_Class_Pred_Local_v2', 'Registration_Number', 'VHE_Number']

column_performance_sql_renaming = {
    'start_section_a': 'Section_A_Start',
    'start_section_b': 'Section_B_Start',
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
    'start_motor': 'Motor_Desc_Start',
    'end_motor': 'Motor_Desc_End',
    'nav_all': 'Navigation_Duration',
    'barras_all': 'Roof_Bars_Duration',
    'alarme_all': 'Alarm_Duration',
    'seven_lug_all': 'Seven_Seats_Duration',
    'prot_all': 'Solar_Protection_Duration',
    'ac_all': 'AC_Auto_Duration',
    'teto_all': 'Open_Roof_Duration',
    'cor_ext_all': 'Colour_Ext_Duration',
    'cor_int_all': 'Colour_Int_Duration',
    'int_type_all': 'Interior_Type_Duration',
    'versao_all': 'Version_Duration',
    'trans_all': 'Auto_Trans_Duration',
    'sens_all': 'Park_Front_Sens_Duration',
    'jantes_all': 'Rims_Size_Duration',
    'farois_all': 'Lights_Duration',
    'start_standard': 'Standard_Start',
    'end_standard': 'Standard_End'
}

column_checkpoint_sql_renaming = {
    'Jantes': 'Rims_Size',
    'Caixa Auto': 'Auto_Trans',
    'Navegação': 'Navigation',
    'Sensores': 'Park_Front_Sens',
    'Cor_Interior': 'Colour_Int',
    'Cor_Exterior': 'Colour_Ext',
    'Modelo': 'Model_Code',
    'Local da Venda': 'Sales_Place',
    'Local da Venda_v2': 'Sales_Place_v2',
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
    # '7_Lug': 'Seven_Seats',
    'AC Auto': 'AC_Auto',
    'Alarme': 'Alarm',
    'Barras_Tej': 'Roof_Bars',
    'Teto_Abrir': 'Open_Roof',
    'Farois_LED': 'LED_Lights',
    'Farois_Xenon': 'Xenon_Lights',
    # 'Prot.Solar': 'Solar_Protection',
    'Tipo_Interior': 'Interior_Type',
    'Versao': 'Version',
    'Motor': 'Motor_Desc',
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
    'Registration_Number': 'Registration_Number',
}

log_files = {
    'full_log': 'logs/optionals_baviera.txt'
}
