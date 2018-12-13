import nltk
import sys
import logging
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy import stats
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from level_2_optionals_baviera_options import dakota_colors, vernasca_colors, nappa_colors, nevada_colors, merino_colors, pool_workers
from level_2_optionals_baviera_performance_report_info import performance_info_append
warnings.simplefilter('ignore', FutureWarning)

# Globals Definition
MEAN_TOTAL_PRICE = 0  # optionals_baviera
STD_TOTAL_PRICE = 0  # optionals_baviera
my_dpi = 96

# logging.basicConfig(level=logging.WARN, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S @ %d/%m/%y', filename='logs/optionals_baviera.txt', filemode='a')

# List of Functions available:

# Generic:
# lowercase_column_convertion - Converts specified column's name to lowercase
# remove_columns - Removes specified columns from db
# remove_rows - Removes specified rows from db, from the index of the rows to remove
# string_replacer - Replaces specified strings (from dict)
# date_cols - Creates new columns (day, month, year) from dateformat columns
# duplicate_removal - Removes duplicate rows, based on a subset column
# reindex - Creates a new index for the data frame
# new_column_creation - Creates new columns with values equal to 0

# Project Specific Functions:
# options_scraping - Scrapes the "Options" field from baviera sales, checking for specific words in order to fill the following fields - Navegação, Caixa Automática, Sensores Dianteiros, Cor Interior and Cor Exterior
# color_replacement - Replaces and corrects some specified colors from Cor Exterior and Cor Interior
# score_calculation - Calculates new metrics (Score) based on the stock days and margin of a sale


def lowercase_column_convertion(df, columns):

    for column in columns:
        df.loc[:, column] = df[column].str.lower()

    return df


def remove_columns(df, columns):

    for column in columns:
        try:
            df.drop([column], axis=1, inplace=True)
        except KeyError:
            continue

    return df


def remove_rows(df, rows):

    for condition in rows:
        df.drop(condition, axis=0, inplace=True)

    return df


def string_replacer(df, dictionary):

    for key in dictionary.keys():
        df.loc[:, key[0]] = df[key[0]].str.replace(key[1], dictionary[key])
    return df


def date_cols(df, dictionary):
    for key in dictionary.keys():
        df.loc[:, key + 'day'] = df[dictionary[key]].dt.day
        df.loc[:, key + 'month'] = df[dictionary[key]].dt.month
        df.loc[:, key + 'year'] = df[dictionary[key]].dt.year

    return df


def null_analysis(df):
    # Displays the number and percentage of null values in the DF

    tab_info = pd.DataFrame(df.dtypes).T.rename(index={0: 'column type'})
    tab_info = tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0: '#null:'}))
    tab_info = tab_info.append(pd.DataFrame(df.isnull().sum() / df.shape[0] * 100).T.rename(index={0: '%null:'}))

    print(tab_info)


def inf_analysis(df):
    # Displays the number and percentage of null values in the DF

    tab_info = pd.DataFrame(df.dtypes).T.rename(index={0: 'column type'})
    tab_info = tab_info.append(pd.DataFrame((df == np.inf).astype(int).sum(axis=0)).T.rename(index={0: '#inf:'}))
    tab_info = tab_info.append(pd.DataFrame((df == -np.inf).astype(int).sum(axis=0)).T.rename(index={0: '#-inf:'}))
    tab_info = tab_info.append(pd.DataFrame((df == np.inf).astype(int).sum() / df.shape[0] * 100).T.rename(index={0: '%inf:'}))
    tab_info = tab_info.append(pd.DataFrame((df == -np.inf).astype(int).sum() / df.shape[0] * 100).T.rename(index={0: '%-inf:'}))

    print(tab_info)


def zero_analysis(df):
    # Displays the number and percentage of null values in the DF

    tab_info = pd.DataFrame(df.dtypes).T.rename(index={0: 'column type'})
    tab_info = tab_info.append(pd.DataFrame((df == 0).astype(int).sum(axis=0)).T.rename(index={0: '#zero:'}))
    tab_info = tab_info.append(pd.DataFrame((df == 0).astype(int).sum(axis=0) / df.shape[0] * 100).T.rename(index={0: '%zero:'}))

    print(tab_info)


def value_count_histogram(df, columns, tag, output_dir='output/'):
    for column in columns:
        plt.subplots(figsize=(1000 / my_dpi, 600 / my_dpi), dpi=my_dpi)
        # df.loc[df[column] == 0, column] = '0'
        # df.loc[df[column] == 1, column] = '1'

        df_column_as_str = df[column].astype(str)
        counts = df_column_as_str.value_counts().values
        values = df_column_as_str.value_counts().index
        rects = plt.bar(values, counts)

        # plt.tight_layout()
        plt.xlabel('Values')
        plt.xticks(rotation=30)
        plt.ylabel('Counts')
        plt.title('Distribution for column - ' + column)
        bar_plot_auto_label(rects)
        save_fig(str(column) + '_' + tag, output_dir)
        # plt.show()


def bar_plot_auto_label(rects):

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%d' % int(height), ha='center', va='bottom')


def save_fig(name, save_dir='output/'):
    # Saves plot in at least two formats, png and pdf
    plt.savefig(save_dir + str(name) + '.pdf')
    plt.savefig(save_dir + str(name) + '.png')


# def options_scraping(df):
#     start_stanrd = time.time()
#     print('options scraping standard')
#
#     colors_pt = ['preto', 'branco', 'azul', 'verde', 'tartufo', 'vermelho', 'antracite/vermelho', 'anthtacite/preto', 'preto/laranja/preto/lara', 'prata/cinza', 'cinza', 'preto/silver', 'cinzento', 'prateado', 'prata', 'amarelo', 'laranja', 'castanho', 'dourado', 'antracit', 'antracite/preto', 'antracite/cinza/preto', 'branco/outras', 'antracito', 'antracite', 'antracite/vermelho/preto', 'oyster/preto', 'prata/preto/preto', 'âmbar/preto/pr', 'bege', 'terra', 'preto/laranja', 'cognac/preto', 'bronze', 'beige', 'beje', 'veneto/preto', 'zagora/preto', 'mokka/preto', 'taupe/preto', 'sonoma/preto', 'preto/preto', 'preto/laranja/preto', 'preto/vermelho']
#     colors_en = ['black', 'havanna', 'merino', 'walnut', 'chocolate', 'nevada', 'moonstone', 'anthracite/silver', 'white', 'coffee', 'blue', 'red', 'grey', 'silver', 'orange', 'green', 'bluestone', 'aqua', 'burgundy', 'anthrazit', 'truffle', 'brown', 'oyster', 'tobacco', 'jatoba', 'storm', 'champagne', 'cedar', 'silverstone', 'chestnut', 'kaschmirsilber', 'oak', 'mokka']
#
#     df = remove_rows(df, [df[df.Modelo.str.contains('Série')].index, df[df.Modelo.str.contains('Z4')].index, df[df.Modelo.str.contains('MINI')].index, df[df['Prov'] == 'Demonstração'].index, df[df['Prov'] == 'Em utilização'].index])
#
#     df_grouped = df.groupby('Nº Stock')
#     start_nav_all, end_nav_all = [], []
#     start_barras_all, end_barras_all = [], []
#     start_alarme_all, end_alarme_all = [], []
#     start_7_lug_all, end_7_lug_all = [], []
#     start_prot_all, end_prot_all = [], []
#     start_ac_all, end_ac_all = [], []
#     start_teto_all, end_teto_all = [], []
#     duration_sens_all, duration_trans_all, duration_versao_all, duration_farois_all, duration_jantes_all = [], [], [], [], []
#     start_cor_ext_all, end_cor_ext_all = [], []
#     start_cor_int_all, end_cor_int_all = [], []
#     start_int_type_all, end_int_type_all = [], []
#
#     # Modelo
#     performance_info_append(time.time(), 'start_modelo')
#     unique_models = df['Modelo'].unique()
#     for model in unique_models:
#         if 'Série' not in model:
#             tokenized_modelo = nltk.word_tokenize(model)
#             df.loc[df['Modelo'] == model, 'Modelo'] = ' '.join(tokenized_modelo[:-3])
#     performance_info_append(time.time(), 'end_modelo')
#
#     for key, group in df_grouped:
#         duration_sens, duration_trans, duration_versao, duration_farois, duration_jantes = 0, 0, 0, 0, 0
#         line_modelo = group['Modelo'].head(1).values[0]
#         tokenized_modelo = nltk.word_tokenize(line_modelo)
#         optionals = set(group['Opcional'])
#
#         # Navegação
#         start_nav_all.append(time.time())
#         if len([x for x in optionals if 'navegação' in x]):
#             df.loc[df['Nº Stock'] == key, 'Navegação'] = 1
#         end_nav_all.append(time.time())
#
#         # Barras Tejadilho
#         start_barras_all.append(time.time())
#         if len([x for x in optionals if 'barras' in x]):
#             df.loc[df['Nº Stock'] == key, 'Barras_Tej'] = 1
#         end_barras_all.append(time.time())
#
#         # Alarme
#         start_alarme_all.append(time.time())
#         if len([x for x in optionals if 'alarme' in x]):
#             df.loc[df['Nº Stock'] == key, 'Alarme'] = 1
#         end_alarme_all.append(time.time())
#
#         # 7 Lugares
#         start_7_lug_all.append(time.time())
#         if len([x for x in optionals if 'terceira' in x]):
#             df.loc[df['Nº Stock'] == key, '7_Lug'] = 1
#         end_7_lug_all.append(time.time())
#
#         # Vidros com Proteção Solar
#         start_prot_all.append(time.time())
#         if len([x for x in optionals if 'proteção' in x and 'solar' in x]):
#             df.loc[df['Nº Stock'] == key, 'Prot.Solar'] = 1
#         end_prot_all.append(time.time())
#
#         # AC Auto
#         start_ac_all.append(time.time())
#         if len([x for x in optionals if 'ar' in x and 'condicionado' in x]):
#             df.loc[df['Nº Stock'] == key, 'AC Auto'] = 1
#         end_ac_all.append(time.time())
#
#         # Teto Abrir
#         start_teto_all.append(time.time())
#         if len([x for x in optionals if 'teto' in x and 'abrir' in x]):
#             df.loc[df['Nº Stock'] == key, 'Teto_Abrir'] = 1
#         end_teto_all.append(time.time())
#
#         # Sensor/Transmissão/Versão/Jantes
#         for line_options in group['Opcional']:
#             tokenized_options = nltk.word_tokenize(line_options)
#
#             start = time.time()
#             if 'pdc-sensores' in tokenized_options:
#                 for word in tokenized_options:
#                     if 'diant' in word:
#                         df.loc[df['Nº Stock'] == key, 'Sensores'] = 1
#             duration = time.time() - start
#             duration_sens += duration
#
#             start = time.time()
#             if 'transmissão' in tokenized_options or 'caixa' in tokenized_options:
#                 for word in tokenized_options:
#                     if 'auto' in word:
#                         df.loc[df['Nº Stock'] == key, 'Caixa Auto'] = 1
#             duration = time.time() - start
#             duration_trans += duration
#
#             # Versão
#             start = time.time()
#             if 'advantage' in tokenized_options:
#                 df.loc[df['Nº Stock'] == key, 'Versao'] = 'advantage'
#             elif 'versão' in tokenized_options or 'bmw' in tokenized_options:
#                 if 'line' in tokenized_options and 'sport' in tokenized_options:
#                     df.loc[df['Nº Stock'] == key, 'Versao'] = 'line_sport'
#                 if 'line' in tokenized_options and 'urban' in tokenized_options:
#                     df.loc[df['Nº Stock'] == key, 'Versao'] = 'line_urban'
#                 if 'desportiva' in tokenized_options and 'm' in tokenized_options:
#                     df.loc[df['Nº Stock'] == key, 'Versao'] = 'desportiva_m'
#                 if 'line' in tokenized_options and 'luxury' in tokenized_options:
#                     df.loc[df['Nº Stock'] == key, 'Versao'] = 'line_luxury'
#             if 'pack' in tokenized_options and 'desportivo' in tokenized_options and 'm' in tokenized_options:
#                 if 'S1' in tokenized_modelo:
#                     df.loc[df['Nº Stock'] == key, 'Versao'] = 'desportiva_m'
#                 elif 'S5' in tokenized_modelo or 'S3' in tokenized_modelo or 'S2' in tokenized_modelo:
#                     df.loc[df['Nº Stock'] == key, 'Versao'] = 'pack_desportivo_m'
#             if 'bmw' in tokenized_options and 'modern' in tokenized_options:  # no need to search for string line, there are no bmw modern without line;
#                 if 'S5' in tokenized_modelo:
#                     df.loc[df['Nº Stock'] == key, 'Versao'] = 'line_luxury'
#                 else:
#                     df.loc[df['Nº Stock'] == key, 'Versao'] = 'line_urban'
#             if 'xline' in tokenized_options:
#                 df.loc[df['Nº Stock'] == key, 'Versao'] = 'xline'
#             duration = time.time() - start
#             duration_versao += duration
#
#             # Faróis
#             start = time.time()
#             if "xénon" in tokenized_options or 'bixénon' in tokenized_options:
#                 df.loc[df['Nº Stock'] == key, 'Farois_Xenon'] = 1
#             elif "luzes" in tokenized_options and "led" in tokenized_options and 'nevoeiro' not in tokenized_options or 'luzes' in tokenized_options and 'adaptativas' in tokenized_options and 'led' in tokenized_options or 'faróis' in tokenized_options and 'led' in tokenized_options and 'nevoeiro' not in tokenized_options:
#                 df.loc[df['Nº Stock'] == key, 'Farois_LED'] = 1
#             duration = time.time() - start
#             duration_farois += duration
#
#             # Jantes
#             start = time.time()
#             for value in range(15, 21):
#                 if str(value) in tokenized_options:
#                     jantes_size = [str(value)] * group.shape[0]
#                     df.loc[df['Nº Stock'] == key, 'Jantes'] = jantes_size
#             duration = time.time() - start
#             duration_jantes += duration
#
#         durations = [duration_farois, duration_jantes, duration_sens, duration_trans, duration_versao]
#         durations_all = [duration_versao_all, duration_trans_all, duration_sens_all, duration_jantes_all, duration_farois_all]
#         [duration_all.append(duration) for (duration_all, duration) in zip(durations_all, durations)]
#
#         # Cor Exterior
#         start_cor_ext_all.append(time.time())
#         line_color = group['Cor'].head(1).values[0]
#         tokenized_color = nltk.word_tokenize(line_color)
#         color = [x for x in colors_pt if x in tokenized_color]
#         if not color:
#             color = [x for x in colors_en if x in tokenized_color]
#         if not color:
#             if tokenized_color == ['pintura', 'bmw', 'individual'] or tokenized_color == ['hp', 'motorsport', ':', 'branco/azul/vermelho', '``', 'racing', "''"] or tokenized_color == ['p0b58']:
#                 color = ['undefined']
#             else:
#                 sys.exit('Error: Color Not Found')
#         if len(color) > 1:  # Fixes cases such as 'white silver'
#             color = [color[0]]
#         color = color * group.shape[0]
#         try:
#             df.loc[df['Nº Stock'] == key, 'Cor_Exterior'] = color
#         except ValueError:
#             print(color)
#         end_cor_ext_all.append(time.time())
#
#         # Cor Interior
#         start_cor_int_all.append(time.time())
#         line_interior = group['Interior'].head(1).values[0]
#         tokenized_interior = nltk.word_tokenize(line_interior)
#
#         if 'dakota' in tokenized_interior:
#             color_int = [x for x in tokenized_interior if x in dakota_colors]
#             if color_int:
#                 df.loc[df['Nº Stock'] == key, 'Cor_Interior'] = 'dakota_' + color_int[0]
#         elif 'nappa' in tokenized_interior:
#             color_int = [x for x in tokenized_interior if x in nappa_colors]
#             if color_int:
#                 df.loc[df['Nº Stock'] == key, 'Cor_Interior'] = 'nappa_' + color_int[0]
#         elif 'vernasca' in tokenized_interior:
#             color_int = [x for x in tokenized_interior if x in vernasca_colors]
#             if color_int:
#                 df.loc[df['Nº Stock'] == key, 'Cor_Interior'] = 'vernasca_' + color_int[0]
#         elif 'nevada' in tokenized_interior:
#             color_int = [x for x in tokenized_interior if x in nevada_colors]
#             if color_int:
#                 df.loc[df['Nº Stock'] == key, 'Cor_Interior'] = 'nevada_' + color_int[0]
#         elif 'merino' in tokenized_interior:
#             color_int = [x for x in tokenized_interior if x in merino_colors]
#             if color_int:
#                 df.loc[df['Nº Stock'] == key, 'Cor_Interior'] = 'merino_' + color_int[0]
#         else:
#             if 'antraci' in tokenized_interior or 'antracit' in tokenized_interior or 'anthracite/silver' in tokenized_interior or 'preto/laranja' in tokenized_interior or 'preto/silver' in tokenized_interior or 'preto/preto' in tokenized_interior or 'confort' in tokenized_interior or 'standard' in tokenized_interior or 'preto' in tokenized_interior or 'antracite' in tokenized_interior or 'antracite/laranja' in tokenized_interior or 'antracite/preto' in tokenized_interior or 'antracite/cinza/preto' in tokenized_interior or 'antracite/vermelho/preto' in tokenized_interior or 'antracite/vermelho' in tokenized_interior or 'interiores' in tokenized_interior:
#                 df.loc[df['Nº Stock'] == key, 'Cor_Interior'] = 'preto'
#             elif 'oyster/preto' in tokenized_interior:
#                 df.loc[df['Nº Stock'] == key, 'Cor_Interior'] = 'oyster'
#             elif 'platinu' in tokenized_interior or 'grey' in tokenized_interior or 'prata/preto/preto' in tokenized_interior or 'prata/cinza' in tokenized_interior:
#                 df.loc[df['Nº Stock'] == key, 'Cor_Interior'] = 'cinzento'
#             elif 'castanho' in tokenized_interior or 'walnut' in tokenized_interior:
#                 df.loc[df['Nº Stock'] == key, 'Cor_Interior'] = 'castanho'
#             elif 'âmbar/preto/pr' in tokenized_interior:
#                 df.loc[df['Nº Stock'] == key, 'Cor_Interior'] = 'amarelo'
#             elif 'champagne' in tokenized_interior:
#                 df.loc[df['Nº Stock'] == key, 'Cor_Interior'] = 'bege'
#             elif 'crimson' in tokenized_interior:
#                 df.loc[df['Nº Stock'] == key, 'Cor_Interior'] = 'vermelho'
#         end_cor_int_all.append(time.time())
#
#         # Tipo Interior
#         start_int_type_all.append(time.time())
#         if 'comb' in tokenized_interior or 'combin' in tokenized_interior or 'combinação' in tokenized_interior or 'tecido/pele' in tokenized_interior:
#             df.loc[df['Nº Stock'] == key, 'Tipo_Interior'] = 'combinação'
#         elif 'hexagon\'' in tokenized_interior or 'hexagon/alcantara' in tokenized_interior:
#             df.loc[df['Nº Stock'] == key, 'Tipo_Interior'] = 'tecido_micro'
#         elif 'tecido' in tokenized_interior or 'cloth' in tokenized_interior:
#             df.loc[df['Nº Stock'] == key, 'Tipo_Interior'] = 'tecido'
#         elif 'pele' in tokenized_interior or 'leather' in tokenized_interior or 'dakota\'' in tokenized_interior or 'couro' in tokenized_interior:
#             df.loc[df['Nº Stock'] == key, 'Tipo_Interior'] = 'pele'
#         end_int_type_all.append(time.time())
#
#     starts_ends = [start_nav_all, end_nav_all, start_barras_all, end_barras_all, start_alarme_all, end_alarme_all, start_7_lug_all, end_7_lug_all, start_prot_all, end_prot_all, start_ac_all, end_ac_all,
#                    start_teto_all, end_teto_all, duration_versao_all, duration_trans_all, duration_sens_all, duration_jantes_all, duration_farois_all, start_cor_ext_all, end_cor_ext_all, start_cor_int_all, end_cor_int_all, start_int_type_all, end_int_type_all]
#     tags = ['start_nav_all', 'end_nav_all', 'start_barras_all', 'end_barras_all', 'start_alarme_all', 'end_alarme_all', 'start_7_lug_all', 'end_7_lug_all', 'start_prot_all', 'end_prot_all', 'start_ac_all', 'end_ac_all',
#             'start_teto_all', 'end_teto_all', 'duration_versao_all', 'duration_trans_all', 'duration_sens_all', 'duration_jantes_all', 'duration_farois_all', 'start_cor_ext_all', 'end_cor_ext_all', 'start_cor_int_all', 'end_cor_int_all', 'start_int_type_all', 'end_int_type_all']
#     [performance_info_append(start_end, tag) for (start_end, tag) in zip(starts_ends, tags)]
#
#     # ToDo: move the following code to it's own function?
#     # Standard Equipment
#     performance_info_append(time.time(), 'start_standard')
#     criteria_model_s1 = df['Modelo'].str.contains('S1')
#     criteria_model_s2 = df['Modelo'].str.contains('S2')
#     criteria_model_s3 = df['Modelo'].str.contains('S3')
#     criteria_model_s4 = df['Modelo'].str.contains('S4')
#     criteria_model_s5 = df['Modelo'].str.contains('S5')
#     criteria_model_x1 = df['Modelo'].str.contains('X1')
#     criteria_model_x3 = df['Modelo'].str.contains('X3')
#     criteria_jantes_0 = df['Jantes'] == 0
#     criteria_farois_led_0 = df['Farois_LED'] == 0
#     criteria_buy_year_ge_2017 = pd.to_datetime(df['Data Compra'].values).year >= 2017
#     criteria_buy_year_lt_2017 = pd.to_datetime(df['Data Compra'].values).year < 2017
#     criteria_buy_year_ge_2016 = pd.to_datetime(df['Data Compra'].values).year >= 2016
#
#     df.loc[df[criteria_model_s1 & criteria_jantes_0].index, 'Jantes'] = '16'
#     df.loc[df[criteria_model_s2 & criteria_jantes_0].index, 'Jantes'] = '16'
#     df.loc[df[criteria_model_s3 & criteria_jantes_0].index, 'Jantes'] = '16'
#     df.loc[df[criteria_model_s3 & criteria_buy_year_ge_2017].index, 'Farois_LED'] = 1
#     df.loc[df[criteria_model_s4 & criteria_jantes_0].index, 'Jantes'] = '17'
#     df.loc[df[criteria_model_s4 & criteria_buy_year_ge_2016].index, 'Sensores'] = 1
#     df.loc[df[criteria_model_s4 & criteria_buy_year_ge_2017].index, 'Farois_LED'] = 1
#     df.loc[df[criteria_model_s4 & criteria_buy_year_lt_2017 & criteria_farois_led_0].index, 'Farois_Xenon'] = 1
#     df.loc[df[criteria_model_s5 & criteria_jantes_0].index, 'Jantes'] = '17'
#     df.loc[df[criteria_model_s5 & criteria_buy_year_ge_2017].index, 'Farois_LED'] = 1
#     df.loc[df[criteria_model_s5 & criteria_buy_year_ge_2017].index, 'Alarme'] = 1
#     df.loc[df[criteria_model_s5 & criteria_buy_year_lt_2017 & criteria_farois_led_0].index, 'Farois_Xenon'] = 1
#     df.loc[df[criteria_model_s5].index, 'Sensores'] = 1
#     df.loc[df[criteria_model_s5].index, 'Navegação'] = 1
#     df.loc[df[criteria_model_s5].index, 'Caixa Auto'] = 1
#     df.loc[df[criteria_model_x1 & criteria_jantes_0].index, 'Jantes'] = '17'
#     df.loc[df[criteria_model_x3 & criteria_jantes_0].index, 'Jantes'] = '18'
#     df.loc[df[criteria_model_x3].index, 'Sensores'] = 1
#     df.loc[df[criteria_model_x3].index, 'Caixa Auto'] = 1
#     df.loc[df['Versao'] == 0, 'Versao'] = 'base'
#     df.loc[df['Jantes'] == 0, 'Jantes'] = 'standard'
#     performance_info_append(time.time(), 'end_standard')
#
#     return df


def options_scraping(df):
    print('before removing Motos, Z4, MINI and Prov = Demo & Utilização', df.shape)
    df = remove_rows(df, [df[df.Modelo.str.contains('Série')].index, df[df.Modelo.str.contains('Z4')].index, df[df.Modelo.str.contains('MINI')].index, df[df['Prov'] == 'Demonstração'].index, df[df['Prov'] == 'Em utilização'].index])
    print('after removing Motos, Z4, MINI and Prov = Demo & Utilização', df.shape)

    df_grouped = df.groupby('Nº Stock')
    start_nav_all, end_nav_all = [], []
    start_barras_all, end_barras_all = [], []
    # start_alarme_all, end_alarme_all = [], []
    # start_7_lug_all, end_7_lug_all = [], []
    # start_prot_all, end_prot_all = [], []
    # start_ac_all, end_ac_all = [], []
    # start_teto_all, end_teto_all = [], []
    duration_sens_all, duration_trans_all, duration_versao_all, duration_farois_all, duration_jantes_all = [], [], [], [], []
    # start_cor_ext_all, end_cor_ext_all = [], []
    # start_cor_int_all, end_cor_int_all = [], []
    # start_int_type_all, end_int_type_all = [], []

    # Modelo
    performance_info_append(time.time(), 'start_modelo')
    unique_models = df['Modelo'].unique()
    for model in unique_models:
        if 'Série' not in model:
            tokenized_modelo = nltk.word_tokenize(model)
            df.loc[df['Modelo'] == model, 'Modelo'] = ' '.join(tokenized_modelo[:-3])
    performance_info_append(time.time(), 'end_modelo')

    workers = pool_workers
    pool = Pool(processes=workers)
    results = pool.map(options_scraping_per_line, [(key, group) for (key, group) in df_grouped])
    pool.close()
    df = pd.concat([result for result in results])
    # [start_nav_all.append(result[1]) for result in results]
    # [end_nav_all.append(result[2]) for result in results]
    # [start_barras_all.append(result[3]) for result in results]
    # [end_barras_all.append(result[4]) for result in results]
    # [duration_sens_all.append(result[5]) for result in results]
    # [duration_trans_all.append(result[6]) for result in results]

    # starts_ends = [start_nav_all, end_nav_all, start_barras_all, end_barras_all, duration_sens_all, duration_trans_all]
    # tags = ['start_nav_all', 'end_nav_all', 'start_barras_all', 'end_barras_all', 'duration_sens_all', 'duration_trans_all']
    # starts_ends = [start_nav_all, end_nav_all, start_barras_all, end_barras_all, start_alarme_all, end_alarme_all, start_7_lug_all, end_7_lug_all, start_prot_all, end_prot_all, start_ac_all, end_ac_all,
    #                start_teto_all, end_teto_all, duration_versao_all, duration_trans_all, duration_sens_all, duration_jantes_all, duration_farois_all, start_cor_ext_all, end_cor_ext_all, start_cor_int_all, end_cor_int_all, start_int_type_all, end_int_type_all]
    # tags = ['start_nav_all', 'end_nav_all', 'start_barras_all', 'end_barras_all', 'start_alarme_all', 'end_alarme_all', 'start_7_lug_all', 'end_7_lug_all', 'start_prot_all', 'end_prot_all', 'start_ac_all', 'end_ac_all',
    #         'start_teto_all', 'end_teto_all', 'duration_versao_all', 'duration_trans_all', 'duration_sens_all', 'duration_jantes_all', 'duration_farois_all', 'start_cor_ext_all', 'end_cor_ext_all', 'start_cor_int_all', 'end_cor_int_all', 'start_int_type_all', 'end_int_type_all']
    # [performance_info_append(start_end, tag) for (start_end, tag) in zip(starts_ends, tags)]

    # ToDo: move the following code to it's own function?
    # Standard Equipment
    performance_info_append(time.time(), 'start_standard')
    criteria_model_s1 = df['Modelo'].str.contains('S1')
    criteria_model_s2 = df['Modelo'].str.contains('S2')
    criteria_model_s3 = df['Modelo'].str.contains('S3')
    criteria_model_s4 = df['Modelo'].str.contains('S4')
    criteria_model_s5 = df['Modelo'].str.contains('S5')
    criteria_model_x1 = df['Modelo'].str.contains('X1')
    criteria_model_x3 = df['Modelo'].str.contains('X3')
    criteria_jantes_0 = df['Jantes'] == 0
    criteria_farois_led_0 = df['Farois_LED'] == 0
    criteria_buy_year_ge_2017 = pd.to_datetime(df['Data Compra'].values).year >= 2017
    criteria_buy_year_lt_2017 = pd.to_datetime(df['Data Compra'].values).year < 2017
    criteria_buy_year_ge_2016 = pd.to_datetime(df['Data Compra'].values).year >= 2016

    df.loc[df[criteria_model_s1 & criteria_jantes_0].index, 'Jantes'] = '16'
    df.loc[df[criteria_model_s2 & criteria_jantes_0].index, 'Jantes'] = '16'
    df.loc[df[criteria_model_s3 & criteria_jantes_0].index, 'Jantes'] = '16'
    df.loc[df[criteria_model_s3 & criteria_buy_year_ge_2017].index, 'Farois_LED'] = 1
    df.loc[df[criteria_model_s4 & criteria_jantes_0].index, 'Jantes'] = '17'
    df.loc[df[criteria_model_s4 & criteria_buy_year_ge_2016].index, 'Sensores'] = 1
    df.loc[df[criteria_model_s4 & criteria_buy_year_ge_2017].index, 'Farois_LED'] = 1
    df.loc[df[criteria_model_s4 & criteria_buy_year_lt_2017 & criteria_farois_led_0].index, 'Farois_Xenon'] = 1
    df.loc[df[criteria_model_s5 & criteria_jantes_0].index, 'Jantes'] = '17'
    df.loc[df[criteria_model_s5 & criteria_buy_year_ge_2017].index, 'Farois_LED'] = 1
    df.loc[df[criteria_model_s5 & criteria_buy_year_ge_2017].index, 'Alarme'] = 1
    df.loc[df[criteria_model_s5 & criteria_buy_year_lt_2017 & criteria_farois_led_0].index, 'Farois_Xenon'] = 1
    df.loc[df[criteria_model_s5].index, 'Sensores'] = 1
    df.loc[df[criteria_model_s5].index, 'Navegação'] = 1
    df.loc[df[criteria_model_s5].index, 'Caixa Auto'] = 1
    df.loc[df[criteria_model_x1 & criteria_jantes_0].index, 'Jantes'] = '17'
    df.loc[df[criteria_model_x3 & criteria_jantes_0].index, 'Jantes'] = '18'
    df.loc[df[criteria_model_x3].index, 'Sensores'] = 1
    df.loc[df[criteria_model_x3].index, 'Caixa Auto'] = 1
    df.loc[df['Versao'] == 0, 'Versao'] = 'base'
    df.loc[df['Jantes'] == 0, 'Jantes'] = 'standard'
    performance_info_append(time.time(), 'end_standard')

    return df


def options_scraping_per_line(args):
    key, group = args
    colors_pt = ['preto', 'branco', 'azul', 'verde', 'tartufo', 'vermelho', 'antracite/vermelho', 'anthtacite/preto', 'preto/laranja/preto/lara', 'prata/cinza', 'cinza', 'preto/silver', 'cinzento', 'prateado', 'prata', 'amarelo', 'laranja', 'castanho', 'dourado', 'antracit', 'antracite/preto', 'antracite/cinza/preto', 'branco/outras', 'antracito', 'antracite', 'antracite/vermelho/preto', 'oyster/preto', 'prata/preto/preto', 'âmbar/preto/pr', 'bege', 'terra', 'preto/laranja', 'cognac/preto',
                 'bronze', 'beige', 'beje', 'veneto/preto', 'zagora/preto', 'mokka/preto', 'taupe/preto', 'sonoma/preto', 'preto/preto', 'preto/laranja/preto', 'preto/vermelho']
    colors_en = ['black', 'havanna', 'merino', 'walnut', 'chocolate', 'nevada', 'moonstone', 'anthracite/silver', 'white', 'coffee', 'blue', 'red', 'grey', 'silver', 'orange', 'green', 'bluestone', 'aqua', 'burgundy', 'anthrazit', 'truffle', 'brown', 'oyster', 'tobacco', 'jatoba', 'storm', 'champagne', 'cedar', 'silverstone', 'chestnut', 'kaschmirsilber', 'oak', 'mokka']

    # duration_sens, duration_trans, duration_versao, duration_farois, duration_jantes = 0, 0, 0, 0, 0
    line_modelo = group['Modelo'].head(1).values[0]
    tokenized_modelo = nltk.word_tokenize(line_modelo)
    optionals = set(group['Opcional'])

    # start_nav_all, end_nav_all = [], []
    # start_barras_all, end_barras_all = [], []
    #     start_alarme_all, end_alarme_all = [], []
    #     start_7_lug_all, end_7_lug_all = [], []
    #     start_prot_all, end_prot_all = [], []
    #     start_ac_all, end_ac_all = [], []
    #     start_teto_all, end_teto_all = [], []
    # duration_sens_group, duration_trans_group, duration_versao_group, duration_farois_group, duration_jantes_group = [], [], [], [], []
    #     start_cor_ext_all, end_cor_ext_all = [], []
    #     start_cor_int_all, end_cor_int_all = [], []
    #     start_int_type_all, end_int_type_all = [], []

    # Navegação
    # start_nav_all.append(time.time())
    # start_nav = time.time()
    if len([x for x in optionals if 'navegação' in x]):
        group.loc[group['Nº Stock'] == key, 'Navegação'] = 1
    # end_nav = time.time()

    # Barras Tejadilho
    # start_barras = time.time()
    if len([x for x in optionals if 'barras' in x]):
        group.loc[group['Nº Stock'] == key, 'Barras_Tej'] = 1
    # end_barras = time.time()

    # Alarme
    # start_alarme_all.append(time.time())
    if len([x for x in optionals if 'alarme' in x]):
        group.loc[group['Nº Stock'] == key, 'Alarme'] = 1
    # end_alarme_all.append(time.time())

    # 7 Lugares
    # start_7_lug_all.append(time.time())
    if len([x for x in optionals if 'terceira' in x]):
        group.loc[group['Nº Stock'] == key, '7_Lug'] = 1
    # end_7_lug_all.append(time.time())

    # Vidros com Proteção Solar
    # start_prot_all.append(time.time())
    if len([x for x in optionals if 'proteção' in x and 'solar' in x]):
        group.loc[group['Nº Stock'] == key, 'Prot.Solar'] = 1
    # end_prot_all.append(time.time())

    # AC Auto
    # start_ac_all.append(time.time())
    if len([x for x in optionals if 'ar' in x and 'condicionado' in x]):
        group.loc[group['Nº Stock'] == key, 'AC Auto'] = 1
    # end_ac_all.append(time.time())

    # Teto Abrir
    # start_teto_all.append(time.time())
    if len([x for x in optionals if 'teto' in x and 'abrir' in x]):
        group.loc[group['Nº Stock'] == key, 'Teto_Abrir'] = 1
    # end_teto_all.append(time.time())

    # Sensor/Transmissão/Versão/Jantes
    for line_options in group['Opcional']:
        tokenized_options = nltk.word_tokenize(line_options)

        # start = time.time()
        if 'pdc-sensores' in tokenized_options:
            for word in tokenized_options:
                if 'diant' in word:
                    group.loc[group['Nº Stock'] == key, 'Sensores'] = 1
        # duration = time.time() - start
        # duration_sens += duration

        # start = time.time()
        if 'transmissão' in tokenized_options or 'caixa' in tokenized_options:
            for word in tokenized_options:
                if 'auto' in word:
                    group.loc[group['Nº Stock'] == key, 'Caixa Auto'] = 1
        # duration = time.time() - start
        # duration_trans += duration

        # Versão
        # start = time.time()
        if 'advantage' in tokenized_options:
            group.loc[group['Nº Stock'] == key, 'Versao'] = 'advantage'
        elif 'versão' in tokenized_options or 'bmw' in tokenized_options:
            if 'line' in tokenized_options and 'sport' in tokenized_options:
                group.loc[group['Nº Stock'] == key, 'Versao'] = 'line_sport'
            if 'line' in tokenized_options and 'urban' in tokenized_options:
                group.loc[group['Nº Stock'] == key, 'Versao'] = 'line_urban'
            if 'desportiva' in tokenized_options and 'm' in tokenized_options:
                group.loc[group['Nº Stock'] == key, 'Versao'] = 'desportiva_m'
            if 'line' in tokenized_options and 'luxury' in tokenized_options:
                group.loc[group['Nº Stock'] == key, 'Versao'] = 'line_luxury'
        if 'pack' in tokenized_options and 'desportivo' in tokenized_options and 'm' in tokenized_options:
            if 'S1' in tokenized_modelo:
                group.loc[group['Nº Stock'] == key, 'Versao'] = 'desportiva_m'
            elif 'S5' in tokenized_modelo or 'S3' in tokenized_modelo or 'S2' in tokenized_modelo:
                group.loc[group['Nº Stock'] == key, 'Versao'] = 'pack_desportivo_m'
        if 'bmw' in tokenized_options and 'modern' in tokenized_options:  # no need to search for string line, there are no bmw modern without line;
            if 'S5' in tokenized_modelo:
                group.loc[group['Nº Stock'] == key, 'Versao'] = 'line_luxury'
            else:
                group.loc[group['Nº Stock'] == key, 'Versao'] = 'line_urban'
        if 'xline' in tokenized_options:
            group.loc[group['Nº Stock'] == key, 'Versao'] = 'xline'
        # duration = time.time() - start
        # duration_versao += duration

        # Faróis
        # start = time.time()
        if "xénon" in tokenized_options or 'bixénon' in tokenized_options:
            group.loc[group['Nº Stock'] == key, 'Farois_Xenon'] = 1
        elif "luzes" in tokenized_options and "led" in tokenized_options and 'nevoeiro' not in tokenized_options or 'luzes' in tokenized_options and 'adaptativas' in tokenized_options and 'led' in tokenized_options or 'faróis' in tokenized_options and 'led' in tokenized_options and 'nevoeiro' not in tokenized_options:
            group.loc[group['Nº Stock'] == key, 'Farois_LED'] = 1
        # duration = time.time() - start
        # duration_farois += duration

        # Jantes
        start = time.time()
        for value in range(15, 21):
            if str(value) in tokenized_options:
                jantes_size = [str(value)] * group.shape[0]
                group.loc[group['Nº Stock'] == key, 'Jantes'] = jantes_size
        # duration = time.time() - start
        # duration_jantes += duration

    # durations = [duration_sens, duration_trans]
    # durations_all = [duration_sens_group, duration_trans_group]
    # durations = [duration_farois, duration_jantes, duration_sens, duration_trans, duration_versao]
    # durations_all = [duration_versao_all, duration_trans_all, duration_sens_all, duration_jantes_all, duration_farois_all]
    # [duration_all.append(duration) for (duration_all, duration) in zip(durations_all, durations)]

    # Cor Exterior
    # start_cor_ext_all.append(time.time())
    line_color = group['Cor'].head(1).values[0]
    tokenized_color = nltk.word_tokenize(line_color)
    color = [x for x in colors_pt if x in tokenized_color]
    if not color:
        color = [x for x in colors_en if x in tokenized_color]
    if not color:
        if tokenized_color == ['pintura', 'bmw', 'individual'] or tokenized_color == ['hp', 'motorsport', ':', 'branco/azul/vermelho', '``', 'racing', "''"] or tokenized_color == ['p0b58']:
            color = ['undefined']
        else:
            sys.exit('Error: Color Not Found')
    if len(color) > 1:  # Fixes cases such as 'white silver'
        color = [color[0]]
    color = color * group.shape[0]
    try:
        group.loc[group['Nº Stock'] == key, 'Cor_Exterior'] = color
    except ValueError:
        print(color)
    # end_cor_ext_all.append(time.time())

    # Cor Interior
    # start_cor_int_all.append(time.time())
    line_interior = group['Interior'].head(1).values[0]
    tokenized_interior = nltk.word_tokenize(line_interior)

    if 'dakota' in tokenized_interior:
        color_int = [x for x in tokenized_interior if x in dakota_colors]
        if color_int:
            group.loc[group['Nº Stock'] == key, 'Cor_Interior'] = 'dakota_' + color_int[0]
    elif 'nappa' in tokenized_interior:
        color_int = [x for x in tokenized_interior if x in nappa_colors]
        if color_int:
            group.loc[group['Nº Stock'] == key, 'Cor_Interior'] = 'nappa_' + color_int[0]
    elif 'vernasca' in tokenized_interior:
        color_int = [x for x in tokenized_interior if x in vernasca_colors]
        if color_int:
            group.loc[group['Nº Stock'] == key, 'Cor_Interior'] = 'vernasca_' + color_int[0]
    elif 'nevada' in tokenized_interior:
        color_int = [x for x in tokenized_interior if x in nevada_colors]
        if color_int:
            group.loc[group['Nº Stock'] == key, 'Cor_Interior'] = 'nevada_' + color_int[0]
    elif 'merino' in tokenized_interior:
        color_int = [x for x in tokenized_interior if x in merino_colors]
        if color_int:
            group.loc[group['Nº Stock'] == key, 'Cor_Interior'] = 'merino_' + color_int[0]
    else:
        if 'antraci' in tokenized_interior or 'antracit' in tokenized_interior or 'anthracite/silver' in tokenized_interior or 'preto/laranja' in tokenized_interior or 'preto/silver' in tokenized_interior or 'preto/preto' in tokenized_interior or 'confort' in tokenized_interior or 'standard' in tokenized_interior or 'preto' in tokenized_interior or 'antracite' in tokenized_interior or 'antracite/laranja' in tokenized_interior or 'antracite/preto' in tokenized_interior or 'antracite/cinza/preto' in tokenized_interior or 'antracite/vermelho/preto' in tokenized_interior or 'antracite/vermelho' in tokenized_interior or 'interiores' in tokenized_interior:
            group.loc[group['Nº Stock'] == key, 'Cor_Interior'] = 'preto'
        elif 'oyster/preto' in tokenized_interior:
            group.loc[group['Nº Stock'] == key, 'Cor_Interior'] = 'oyster'
        elif 'platinu' in tokenized_interior or 'grey' in tokenized_interior or 'prata/preto/preto' in tokenized_interior or 'prata/cinza' in tokenized_interior:
            group.loc[group['Nº Stock'] == key, 'Cor_Interior'] = 'cinzento'
        elif 'castanho' in tokenized_interior or 'walnut' in tokenized_interior:
            group.loc[group['Nº Stock'] == key, 'Cor_Interior'] = 'castanho'
        elif 'âmbar/preto/pr' in tokenized_interior:
            group.loc[group['Nº Stock'] == key, 'Cor_Interior'] = 'amarelo'
        elif 'champagne' in tokenized_interior:
            group.loc[group['Nº Stock'] == key, 'Cor_Interior'] = 'bege'
        elif 'crimson' in tokenized_interior:
            group.loc[group['Nº Stock'] == key, 'Cor_Interior'] = 'vermelho'
    # end_cor_int_all.append(time.time())

    # Tipo Interior
    # start_int_type_all.append(time.time())
    if 'comb' in tokenized_interior or 'combin' in tokenized_interior or 'combinação' in tokenized_interior or 'tecido/pele' in tokenized_interior:
        group.loc[group['Nº Stock'] == key, 'Tipo_Interior'] = 'combinação'
    elif 'hexagon\'' in tokenized_interior or 'hexagon/alcantara' in tokenized_interior:
        group.loc[group['Nº Stock'] == key, 'Tipo_Interior'] = 'tecido_micro'
    elif 'tecido' in tokenized_interior or 'cloth' in tokenized_interior:
        group.loc[group['Nº Stock'] == key, 'Tipo_Interior'] = 'tecido'
    elif 'pele' in tokenized_interior or 'leather' in tokenized_interior or 'dakota\'' in tokenized_interior or 'couro' in tokenized_interior:
        group.loc[group['Nº Stock'] == key, 'Tipo_Interior'] = 'pele'
    # end_int_type_all.append(time.time())

    # return group, start_nav, end_barras, start_barras, end_nav, duration_sens_group, duration_trans_group
    return group


def column_rename(df, cols_to_replace, new_cols_names):
    for column in cols_to_replace:
        df.rename(index=str, columns={cols_to_replace[cols_to_replace.index(column)]: new_cols_names[cols_to_replace.index(column)]}, inplace=True)
    return df


def col_group(df, columns_to_replace, dictionaries):
    for dictionary in dictionaries:
        for key in dictionary.keys():
            df.loc[df[columns_to_replace[dictionaries.index(dictionary)]].isin(dictionary[key]), columns_to_replace[dictionaries.index(dictionary)] + '_new'] = key
        if df[columns_to_replace[dictionaries.index(dictionary)] + '_new'].isnull().values.any():
            # logging.WARNING('NaNs detected on column')
            variable = df.loc[df[columns_to_replace[dictionaries.index(dictionary)] + '_new'].isnull(), columns_to_replace[dictionaries.index(dictionary)]].unique()
            logging.warning('Column Grouping - NaNs detected in: {}'.format(columns_to_replace[dictionaries.index(dictionary)] + '_new'))
            logging.warning('Value(s) not grouped: {}'.format(variable))
        df.drop(columns_to_replace[dictionaries.index(dictionary)], axis=1, inplace=True)
    return df


def total_price(df):
    df['price_total'] = df['Custo'].groupby(df['Nº Stock']).transform('sum')

    return df


def remove_zero_price_total_vhe(df):
    # df.groupby(['case', 'cluster']).filter(lambda x: len(x) > 1)
    df = df.groupby(['Nº Stock']).filter(lambda x: x.price_total != 0)

    return df


def margin_calculation(df):
    df['margem_percentagem'] = (df['Margem'] / df['price_total']) * 100

    return df


def prov_replacement(df):
    df.loc[df['Prov'] == 'Viaturas Km 0', 'Prov'] = 'Novos'
    df.rename({'Prov': 'Prov_new'}, axis=1, inplace=True)

    return df


def color_replacement(df):
    color_types = ['Cor_Exterior']
    colors_to_replace = {'black': 'preto', 'preto/silver': 'preto/prateado', 'tartufo': 'truffle', 'preto/laranja/preto/lara': 'preto/laranja', 'white': 'branco', 'blue': 'azul', 'red': 'vermelho', 'grey': 'cinzento', 'silver': 'prateado', 'orange': 'laranja', 'green': 'verde', 'anthrazit': 'antracite', 'antracit': 'antracite', 'brown': 'castanho', 'antracito': 'antracite', 'âmbar/preto/pr': 'ambar/preto/preto', 'beige': 'bege', 'kaschmirsilber': 'cashmere', 'beje': 'bege'}

    for color_type in color_types:
        df[color_type] = df[color_type].replace(colors_to_replace)
        df.drop(df[df[color_type] == 0].index, axis=0, inplace=True)

    return df


def score_calculation(df, stockdays_threshold, margin_threshold):
    df['stock_days'] = (df['Data Venda'] - df['Data Compra']).dt.days
    df.loc[df['stock_days'].lt(0), 'stock_days'] = 0

    df['stock_days_class'] = 0
    df.loc[df['stock_days'] <= stockdays_threshold, 'stock_days_class'] = 1
    df['margin_class'] = 0
    df.loc[df['margem_percentagem'] >= margin_threshold, 'margin_class'] = 1

    df['new_score'] = 0
    df.loc[(df['stock_days_class'] == 1) & (df['margin_class'] == 1), 'new_score'] = 1

    df['days_stock_price'] = (0.05/360) * df['price_total'] * df['stock_days']
    df['score_euros'] = df['Margem'] - df['days_stock_price']

    return df


def new_features_optionals_baviera(df, sel_cols):
    df_grouped = df.sort_values(by=['Data Venda']).groupby(sel_cols)
    df = df_grouped.apply(previous_sales_info_optionals_baviera)

    return df.fillna(0)


def previous_sales_info_optionals_baviera(x):

    prev_scores, i = [], 0
    if len(x) > 1:
        x['prev_sales_check'] = [0] + [1] * (len(x) - 1)
        x['number_prev_sales'] = list(range(len(x)))
        x['last_score'] = x['new_score'].shift(1)
        x['last_margin'] = x['margem_percentagem'].shift(1)
        x['last_stock_days'] = x['stock_days'].shift(1)

        for key, row in x.iterrows():
            prev_scores.append(x.loc[x.index == key, 'new_score'].values.tolist()[0])
            x.loc[x.index == key, 'average_score_dynamic'] = np.mean(prev_scores)
            if i == 0:
                x.loc[x.index == key, 'average_score_dynamic'] = 0
                i += 1

        x['average_score_dynamic_std'] = np.std(x['average_score_dynamic'])
        x['prev_average_score_dynamic'] = x['average_score_dynamic'].shift(1)  # New column - This one and following new column boost Adaboost for more than 15% in ROC Area
        x['prev_average_score_dynamic_std'] = x['average_score_dynamic_std'].shift(1)  # New column

        x['average_score_global'] = x['new_score'].mean()
        x['min_score_global'] = x['new_score'].min()
        x['max_score_global'] = x['new_score'].max()
        x['q3_score_global'] = x['new_score'].quantile(0.75)
        x['median_score_global'] = x['new_score'].median()
        x['q1_score_global'] = x['new_score'].quantile(0.25)

    elif len(x) == 0:
        x['prev_sales_check'] = 0
        x['number_prev_sales'] = 0
        x['last_score'] = 0
        # The reason I'm not filling all the other columns with zeros for the the len(x) == 0 case, is because I have a fillna(0) at the end of the function that called this one.

    return x


def z_scores_function(df, cols_to_normalize):
    for column in cols_to_normalize:
        df[column] = stats.zscore(df[column])

    return df


def global_variables_saving(df, project):
    if project == 'optionals_baviera':
        global MEAN_TOTAL_PRICE
        MEAN_TOTAL_PRICE = np.mean(df['price_total'])
        global STD_TOTAL_PRICE
        STD_TOTAL_PRICE = np.std(df['price_total'])


def feature_selection(df, features, target_variable, feature_count, criteria=f_classif):

    sel_features = []
    for feature in features:
        for feature_2 in list(df):
            if feature in feature_2:
                sel_features.append(feature_2)

    selector = SelectKBest(criteria, k=feature_count).fit(df[sel_features], df[target_variable])
    idxs_selected = selector.get_support(indices=True)

    features_new = list(list(sel_features)[i] for i in idxs_selected)

    removed_features = [x for x in sel_features if x not in list(features_new)]

    return features_new, removed_features


def df_copy(df):

    copy = df.copy(deep=True)

    return df, copy


def dataset_split(df, target, oversample=0):
    df_train, df_test = train_test_split(df, stratify=df[target], random_state=2)  # This ensures that the classes are evenly distributed by train/test datasets; Default split is 0.75/0.25 train/test

    df_train_y = df_train[target]
    df_train_x = df_train.drop(target, axis=1)

    df_test_y = df_test[target]
    df_test_x = df_test.drop(target, axis=1)

    if oversample:
        print('Oversampling small classes...')
        df_train_x, df_train_y = oversample_data(df_train_x, df_train_y)

    return df_train_x, df_train_y, df_test_x, df_test_y


def oversample_data(train_x, train_y):

    train_x['oversample_flag'] = range(train_x.shape[0])
    train_x['original_index'] = train_x.index
    # print(train_x.shape, '\n', train_y['new_score'].value_counts())

    print(train_x, train_y)

    ros = RandomOverSampler(random_state=42)
    train_x_resampled, train_y_resampled = ros.fit_sample(train_x, train_y.values.ravel())

    train_x_resampled = pd.DataFrame(np.atleast_2d(train_x_resampled), columns=list(train_x))
    train_y_resampled = pd.Series(train_y_resampled)
    for column in list(train_x_resampled):
        if train_x_resampled[column].dtype != train_x[column].dtype:
            print('Problem found with dtypes, fixing it...', )
            dtype_checkup(train_x_resampled, train_x)
        break

    return train_x_resampled, train_y_resampled


def dtype_checkup(train_x_resampled, train_x):
    for column in list(train_x):
        train_x_resampled[column] = train_x_resampled[column].astype(train_x[column].dtype)


def ohe(df, cols):

    for column in cols:
        uniques = df[column].unique()
        for value in uniques:
            new_column = column + '_' + str(value)
            df[new_column] = 0
            df.loc[df[column] == value, new_column] = 1
        df.drop(column, axis=1, inplace=True)

    return df


def duplicate_removal(df, subset_col):
    df.drop_duplicates(subset=subset_col, inplace=True)

    return df


def reindex(df):
    df.index = range(df.shape[0])

    return df


def new_column_creation(df, columns):

    for column in columns:
        df.loc[:, column] = 0

    return df



