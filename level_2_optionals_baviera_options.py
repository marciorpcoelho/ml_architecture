import numpy as np
from sklearn import tree, linear_model, neighbors, svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier


# Dictionaries:
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

color_int_dict = {
    'preto': ['preto', 'prata/preto/preto', 'veneto/preto', 'preto/preto', 'ambar/preto/preto'],
    'antracite': ['antracite', 'antracite/cinza/preto', 'antracite/preto', 'antracite/vermelho/preto', 'antracite/vermelho', 'anthtacite/preto', 'anthracite/silver'],
    'castanho': ['castanho', 'oak', 'terra', 'mokka', 'vernasca'],
    'others': ['champagne', 'branco', 'oyster', 'prata/cinza', 'bege', 'oyster/preto', 'azul', 'cinzento', 'truffle', 'burgundy', 'zagora/preto', 'sonoma/preto', 'laranja', 'taupe/preto', 'vermelho', 'silverstone', 'nevada', 'cognac/preto', 'preto/laranja', 'preto/prateado']
}

jantes_dict = {
    'standard': ['standard', '15', '16'],
    '17': ['17'],
    '18': ['18'],
    '19/20': ['19', '20']
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

