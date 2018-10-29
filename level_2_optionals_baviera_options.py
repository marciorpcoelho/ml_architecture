import numpy as np
from sklearn import tree, linear_model, neighbors, svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier


# Dictionaries:
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
    'outros': ['castanho', 'terra', 'jatoba', 'burgundy', 'aqua', 'storm', 'cedar', 'bronze', 'chestnut', 'havanna', 'cashmere', 'champagne', 'dourado', 'amarelo', 'bege', 'silverstone', 'moonstone', 'verde', 'vermelho', 'laranja']
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


# # New Cor_Interior
# color_int_dict_layer_1 = {
#     'preto': ['preto'],
#     'dakota_preto': ['dakota_preto/preto', 'dakota_preto/vermelho/preto', 'dakota_preto/oyster', 'dakota_preto/debroado', 'dakota_preto/azul/preto', 'dakota_preto', 'dakota_preta', 'dakota_black/contrast'],
#     'dakota_branco': ['dakota_oyster/preto', 'dakota_ivory/preto', 'dakota_ivory', 'dakota_branco', 'dakota_white'],
#     'dakota_vermelho': ['dakota_coral'],
#     'dakota_bege': ['dakota_bege'],
#     'dakota_oyster': ['dakota_oyster', 'dakota_oyster/oyster', 'dakota_oyster/cinza'],
#     'dakota_castanho': ['dakota_castanho', 'dakota_conhaque', 'dakota_conhaque/castanho/preto', 'dakota_conhaque/castanho/preto/conhaque', 'dakota_cognac/preto', 'dakota_brown', 'dakota_terra'],
#     'dakota_azul': ['dakota_azul'],
#     'dakota_mocha_castanho': ['dakota_mocha/preto', 'dakota_mocha/preto/mocha', 'dakota_mocha'],
#     'nappa_preto': ['nappa_preto'],
#     'nappa_branco': ['nappa_white', 'nappa_ivory', 'nappa_ivory/branco'],
#     'nappa_bege': ['nappa_bege'],
#     'nappa_mocha': ['nappa_mocha'],
#     'nappa_castanho': ['nappa_castanho', 'nappa_cognac/preto'],
#     'vernasca_bege': ['vernasca_canberra'],
#     'vernasca_preto': ['vernasca_preta', 'vernasca_preto/com', 'vernasca_preto/preto'],
#     'vernasca_oyster': ['vernasca_oyster'],
#     'vernasca_castanho': ['vernasca_mocha', 'vernasca_mocha/preto', 'vernasca_cognac'],
#     'others': [0],
# }

# New Cor_Interior v2
color_int_dict_layer_1 = {
    'preto': ['nappa_antracite', 'vernasca_anthracite/preto', 'merino_preto', 'nevada_preto', 'merino_preto/preto', 'nevada_preto/preto', 'vernasca_preta', 'vernasca_preto/com', 'vernasca_preto/preto', 'preto', 'dakota_preto/preto', 'dakota_preto/vermelho/preto', 'dakota_preto/oyster', 'dakota_preto/debroado', 'dakota_preto/azul/preto', 'dakota_preto', 'dakota_preta', 'dakota_black/contrast', 'nappa_preto'],
    'branco': ['nevada_branco', 'merino_branco', 'dakota_oyster/preto', 'dakota_ivory/preto', 'dakota_ivory', 'dakota_branco', 'dakota_white', 'nappa_white', 'nappa_ivory', 'nappa_ivory/branco'],
    'vermelho': ['merino_vermelho', 'dakota_coral'],
    'bege': ['merino_bege', 'dakota_bege', 'nappa_bege', 'vernasca_canberra', 'nevada_bege'],
    'oyster': ['dakota_oyster', 'dakota_oyster/oyster', 'dakota_oyster/cinza', 'vernasca_oyster', 'nevada_oyster', 'nevada_oyster/leather', 'oyster'],
    'castanho': ['castanho', 'merino_castanho', 'nevada_terra', 'nevada_brown', 'vernasca_mocha', 'vernasca_mocha/preto', 'vernasca_cognac', 'nappa_castanho', 'nappa_cognac/preto', 'dakota_castanho', 'dakota_conhaque', 'dakota_conhaque/castanho/preto', 'dakota_conhaque/castanho/preto/conhaque', 'dakota_cognac/preto', 'dakota_brown', 'dakota_terra'],
    'azul': ['dakota_azul', 'vernasca_azuis/preto'],
    'mocha': ['dakota_mocha/preto', 'dakota_mocha/preto/mocha', 'dakota_mocha', 'nappa_mocha', 'nevada_mocha'],
    'cinzento': ['merino_silverstone', 'merino_taupe/preto', 'cinzento'],
    'laranja': ['merino_laranja'],
    'amarelo': ['amarelo'],
    'mini/mota': ['mini/mota'],
    'others': ['0']
}

color_int_dict_layer_2 = {
    'preto': ['preto'],
    'castanho/mocha': ['castanho', 'mocha'],
    'bege/oyster/branco': ['bege', 'oyster', 'branco'],
    'cinzento': ['cinzento'],
    'outros': ['amarelo', 'vermelho', 'azul', 'laranja', 'others'],
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

# tipo_int_dict = {
#     'tecido': ['tecido'],
#     'pele': ['pele'],
#     'combinação': ['combinação'],
#     'interior_m': ['tecido_micro', 0]
# }

# v2
tipo_int_dict = {
    'tecido': ['tecido'],
    'pele': ['pele'],
    'combinação/interior_m': ['combinação', 'tecido_micro', 0]
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
    'voting': [VotingClassifier, [{'voting': ['soft']}]]  # ToDo: Need to create code for this model
}

