import os
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from py_dotenv import read_dotenv
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree, linear_model, neighbors, svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LassoCV, Ridge, LassoLarsCV, ElasticNetCV
from sklearn.svm import SVR


dotenv_path = 'info.env'
read_dotenv(dotenv_path)

DSN_PRD = os.getenv('DSN_Prd')
DSN_MLG = os.getenv('DSN_MLG')
UID = os.getenv('UID')
PWD = os.getenv('PWD')

project_id = 2406
update_frequency_days = 0
# stock_days_threshold = 150  # DaysInStock_Global
stock_days_threshold = [90, 120, 150, 180, 270, 365]
margin_threshold = "nan"  # Currently there is no threshold;

metric, metric_threshold = 'R2', 0.50  # The metric to compare on the final models and the minimum threshold to consider;
k, gridsearch_score = 10, 'neg_mean_squared_error'  # Stratified Cross-Validation number of Folds and the Metric on which to optimize GridSearchCV


sql_info = {
    'database_source': 'BI_DTR',
    'database_final': 'BI_MLG',
    'product_db': 'VHE_Dim_VehicleData_DTR',
    'sales': 'VHE_Fact_BI_Sales_DTR',
    'stock': 'VHE_Fact_BI_Stock_DTR',
    'final_table': 'VHE_Fact_BI_Sales_DTR_Temp',
    'feature_contribution': 'VHE_Fact_Feature_Contribution',
}

log_files = {
    'full_log': 'logs/optionals_hyundai.txt'
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
        from [BI_DTR].dbo.[VHE_Fact_BI_Stock_DTR] WITH (NOLOCK)'''

product_db_query = '''
        SELECT *
        FROM [BI_DTR].dbo.[VHE_Dim_VehicleData_DTR] WITH (NOLOCK)
        UNION ALL
        SELECT *
        FROM [BI_DW_History].dbo.[VHE_Dim_VehicleData_DTR] WITH (NOLOCK)'''

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

# Modelo
# v1
# model_grouping = {
#     'i20': [],
#     'kauai': [],
#     'tucson': [],
#     'i10': [],
#     'hr-v': [],
#     'civic': [],
# }

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
version_translation = {
    'Access': ['access', 'access plus', 'access my17'],
    'Comfort': ['comfort ', 'comfort', 'comfort + connect navi ', 'comfort', 'van 3 lugares', 'comfort my19', 'comfort navi', 'blue comfort my17', 'blue comfort hp my17', 'comfort + navi', 'comfort + connect navi', 'blue comfort'],
    'Creative': ['creative plus'],
    'Dynamic': ['dynamic', 'dynamic + connect navi'],
    'Elegance': ['elegance navi', '1.5 i-vtec turbo cvt elegance navi', '1.6 i-dtec turbo elegance navi', 'elegance ', 'elegance + connect navi ', 'elegance plus + connect n', 'elegance', 'elegance + connect navi', '1.5 i-vtec turbo elegance'],
    'EV': ['ev'],
    'Executive': ['executive ', 'executive', 'executive premium', '1.5 i-vtec turbo executive', '1.5 i-vtec turbo cvt executive', '1.6 i-dtec turbo executive', 'executive', 'executive my19'],
    'GO': ['go', 'go+', 'go!', 'go!+'],
    'HEV': ['hev'],
    'Launch': ['launch edition'],
    'Lifestyle': ['lifestyle', 'lifestyle + navi', 'lifestyle + connect navi'],
    'Performance': ['performance pack'],
    'PHEV': ['phev'],
    'Premium': ['premium', 'premium my19', 'premium my19 + pack pele'],
    'Prestige': ['prestige'],
    'Pro': ['pro edition'],
    'Sport': ['sport plus', 'sport'],
    'Style': ['style', 'comfort my18', 'style my18', 'style plus my18', 'style+', 'blue style hp my17', 'blue style', 'style my19'],
    'Type R': ['gt pack', 'gt'],
    'Trend': ['trend', 'trend '],
    'X-Road': ['x-road navi'],
    'Teclife': ['teclife'],
    'Turbo': ['turbo'],
    'MY18': ['my18'],
    'LED': ['led'],
    'Panorama': ['panorama'],
    'N': ['250cv'],  # Represents Hyundai i30 N
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
    'Azul': ['stormy sea', 'ocean view', 'aqua sparkling', 'clean slate', 'intense blue', 'brilliant sporty blue m.', 'morpho blue p.', 'stargazing blue', 'champion blue', 'ceramic blue', 'stellar blue', 'blue lagoon', 'performance blue', 'morning blue', 'ara blue', 'marina blue', 'ceramic blue (tt)', 'blue lagoon (tt)', 'skyride blue m.', 'twilight blue m.'],
    'Branco': ['taffeta white', 'platinum white p.', 'white orchid p.', 'polar white', 'white sand', 'creamy white', 'chalk white', 'pure white', 'white crystal', 'white cream', 'chalk white (tt)', 'championship white'],
    'Castanho': ['iced coffee', 'moon rock', 'golden brown m.', 'cashmere brown', 'tan brown', 'demitasse brown', 'premium agate brown p.'],
    'Cinzento': ['urban titanium m.', 'velvet dune', 'velvet dune (tt)', 'dark knight (tt)', 'wild explorer', 'rain forest', 'magnetic force', 'olivine grey', 'dark knight', 'star dust', 'polished metal m.', 'shining grey m.', 'modern steel m.', 'micron grey', 'galactic grey', 'iron gray', 'galactic grey (tt)', 'sonic grey p.', 'shadow grey', 'stone gray'],
    'Laranja': ['tangerine comet (tt)', 'tangerine comet', 'sunset orange ii'],
    'Prateado': ['lunar silver m.', 'platinum silver', 'sleek silver', 'lake silver', 'aurora silver', 'titanium silver', 'platinum silve', 'typhoon silver', 'lake silver (tt)', 'alabaster silver m.', 'tinted silver m.'],
    'Preto': ['midnight burgundy p.', 'crystal black p.', 'ruse black m.', 'phantom black'],
    'Vermelho': ['ral3000', 'rallye red', 'milano red', 'fiery red', 'passion red', 'tomato red', 'pulse red', 'engine red', 'magma red', 'pulse red (tt)', 'passion red p.'],
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
    'Cinzento': ['blue grey', 'lava stone', 'grey', 'pele sintética cinza', 'dark grey'],
    'Laranja': ['orange'],
    'Preto': ['black', 'black 2', 'black 3', 'neutral black'],
    'Preto/Castanho': ['black / brown'],
    'Preto/Cinzento': ['black/charcoal'],
    'Preto/Laranja': ['black/orange'],
    'Vermelho': ['red', 'red point', 'black + red point'],
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
    'knn': [neighbors.KNeighborsClassifier, [{'n_neighbors': np.arange(1, 50, 1)}]],
    'svm': [svm.SVC, [{'C': np.logspace(-2, 2, 10)}]],
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


