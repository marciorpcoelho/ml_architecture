import os
import re
import nltk
import time
import string
import warnings
import operator
import itertools
import unidecode
import numpy as np
import pandas as pd
import datetime
from scipy import stats
from langdetect import detect
import matplotlib.pyplot as plt
from Levenshtein import distance
from multiprocessing import Pool
from difflib import SequenceMatcher
from dateutil.relativedelta import relativedelta
from nltk.stem.snowball import SnowballStemmer
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler

import modules.level_0_performance_report as level_0_performance_report
import modules.level_1_e_deployment as level_1_e_deployment

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))

warnings.simplefilter('ignore', FutureWarning)

# Globals Definition
MEAN_TOTAL_PRICE = 0  # optionals_baviera
STD_TOTAL_PRICE = 0  # optionals_baviera
my_dpi = 96

# List of Functions available:

# Generic:
# lowercase_column_convertion - Converts specified column's name to lowercase
# remove_columns - Removes specified columns from db
# remove_rows - Removes specified rows from db, from the index of the rows to remove
# string_replacer - Replaces specified strings (from dict)
# date_cols - Creates new columns (day, month, year) from dateformat columns
# duplicate_removal - Removes duplicate rows, based on a subset column
# reindex - Creates a new index for the data frame
# new_column_creation - Creates new columns with values equal to a chosen value


# Project Specific Functions:
# options_scraping - Scrapes the "Options" field from baviera sales, checking for specific words in order to fill the following fields - Navegação, Caixa Automática, Sensores Dianteiros, Cor Interior and Cor Exterior
# color_replacement - Replaces and corrects some specified colors from Cor Exterior and Cor Interior
# score_calculation - Calculates new metrics (Score) based on the stock days and margin of a sale


def lowercase_column_conversion(df, columns):
    df[columns] = df[columns].apply(lambda x: x.str.lower())

    return df


def trim_columns(df, columns):
    df[columns] = df[columns].apply(lambda x: x.str.strip())

    return df


def remove_columns(df, columns, project_id):

    for column in columns:
        try:
            df.drop([column], axis=1, inplace=True)
        except KeyError:
            level_0_performance_report.log_record('Aviso de remoção de coluna - A coluna {} não foi encontrada.'.format(column), project_id, flag=1)
            continue

    return df


def remove_rows(df, rows, project_id, warning=0):

    if not warning:
        for condition in rows:
            df.drop(condition, axis=0, inplace=True)

        return df
    else:
        filtered_df = pd.DataFrame()
        start_size = df.shape[0]
        for condition in rows:
            filtered_df = df.drop(condition, axis=0)
        end_size = filtered_df.shape[0]
        if start_size - end_size:
            level_0_performance_report.log_record('Existem veículos com informação em falta que foram removidos - {}.'.format(df[df.index.isin(rows[0].values)]['Nº Stock'].unique()), project_id, flag=1)

        return filtered_df


def string_replacer(df, dictionary):

    for key in dictionary.keys():
        df.loc[:, key[0]] = df[key[0]].str.replace(r'\b{}\b|\b{}'.format(key[1], key[1]), dictionary[key], regex=True)
    return df


def date_cols(df, dictionary):
    for key in dictionary.keys():
        df.loc[:, key + 'day'] = df[dictionary[key]].dt.day
        df.loc[:, key + 'month'] = df[dictionary[key]].dt.month
        df.loc[:, key + 'year'] = df[dictionary[key]].dt.year

    return df


def null_analysis(df):
    # Displays the number and percentage of null values in the DF

    if df.shape[0]:
        tab_info = pd.DataFrame(df.dtypes).T.rename(index={0: 'column type'})
        tab_info = tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0: '#null:'}))
        tab_info = tab_info.append(pd.DataFrame(df.isnull().sum() / df.shape[0] * 100).T.rename(index={0: '%null:'}))

        print(tab_info)

        if not tab_info.loc['#null:', :].sum(axis=0):
            print('No nulls detected!')
    else:
        print('Dataframe is empty!')


def inf_analysis(df):
    # Displays the number and percentage of infinite values in the DF

    if df.shape[0]:
        tab_info = pd.DataFrame(df.dtypes).T.rename(index={0: 'column type'})
        tab_info = tab_info.append(pd.DataFrame((df == np.inf).astype(int).sum(axis=0)).T.rename(index={0: '#inf:'}))
        tab_info = tab_info.append(pd.DataFrame((df == -np.inf).astype(int).sum(axis=0)).T.rename(index={0: '#-inf:'}))
        tab_info = tab_info.append(pd.DataFrame((df == np.inf).astype(int).sum() / df.shape[0] * 100).T.rename(index={0: '%inf:'}))
        tab_info = tab_info.append(pd.DataFrame((df == -np.inf).astype(int).sum() / df.shape[0] * 100).T.rename(index={0: '%-inf:'}))

        print(tab_info)

        if not tab_info.loc['#inf:', :].sum(axis=0) + tab_info.loc['#-inf:', :].sum(axis=0):
            print('No infinities detected!')
    else:
        print('Dataframe is empty!')


def zero_analysis(df):
    # Displays the number and percentage of zero values in the DF
    if df.shape[0]:
        tab_info = pd.DataFrame(df.dtypes).T.rename(index={0: 'column type'})
        tab_info = tab_info.append(pd.DataFrame((df == 0).astype(int).sum(axis=0)).T.rename(index={0: '#zero:'}))
        tab_info = tab_info.append(pd.DataFrame((df == 0).astype(int).sum(axis=0) / df.shape[0] * 100).T.rename(index={0: '%zero:'}))

        print(tab_info)

        if not tab_info.loc['#zero:', :].sum(axis=0):
            print('No zeros detected!')
    else:
        print('Dataframe is empty!')


def value_count_histogram(df, columns, tag, output_dir=base_path + '/plots/'):
    for column in columns:
        plt.subplots(figsize=(1000 / my_dpi, 600 / my_dpi), dpi=my_dpi)

        if column != 'DaysInStock_Global':
            df_column_as_str = df[column].astype(str)
            counts = df_column_as_str.value_counts().values
            values = df_column_as_str.value_counts().index
            rects = plt.bar(values, counts, label='#Different Values: {}'.format(len(counts)))

            # plt.tight_layout()
            plt.xlabel('Values')
            plt.xticks(rotation=30)
            plt.ylabel('Counts')
            plt.title('Distribution for column - {}. Total Count = {}'.format(column, sum(counts)))
            plt.legend()
            bar_plot_auto_label(rects)
            save_fig(str(column) + '_' + tag, output_dir)
            # plt.show()

        else:
            n, bin_edge = np.histogram(df[column].values)
            n = n * 100. / n.sum()
            bincenters = 0.5 * (bin_edge[1:] + bin_edge[:-1])
            plt.plot(bincenters, n, '-', label='Target Class')

            plt.xlabel('Days In Stock')
            plt.ylabel('Relative Frequency (%)')
            plt.title('Distribution for column - {}'.format(column))
            plt.grid()
            save_fig(str(column) + '_' + tag + '_plot', output_dir)
            plt.clf()

            ser = pd.Series(df[column].values)
            cum_dist = np.linspace(0., 1., len(ser))
            ser_cdf = pd.Series(cum_dist, index=ser.sort_values())
            plt.plot(ser_cdf, label='Target Class')
            plt.xlabel('Days In Stock')
            plt.ylabel('CDF')
            plt.title('Distribution for column - {}'.format(column))
            plt.grid()
            plt.tight_layout()
            save_fig(str(column) + '_' + tag + '_cdf', output_dir)
            plt.clf()


def bar_plot_auto_label(rects):

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%d' % int(height), ha='center', va='bottom')


def save_fig(name, save_dir=base_path + '/output/'):
    # Saves plot in at least two formats, png and pdf
    plt.savefig(save_dir + str(name) + '.pdf')
    plt.savefig(save_dir + str(name) + '.png')


def options_scraping_v2(df, options_file, model_mapping={}, model_training_check=0):

    if options_file.project_id == 2162:  # Baviera
        from level_2_optionals_baviera_options import colors_pt, colors_en, dakota_colors, vernasca_colors, nappa_colors, nevada_colors, merino_colors
        colors_list = [colors_pt, colors_en, dakota_colors, vernasca_colors, nappa_colors, nevada_colors, merino_colors]
        df_grouped = df.groupby('Nº Stock')

        options_scraping_model(df, options_file.project_id, model_mapping=model_mapping, model_training_check=model_training_check)
        options_scraping_motorization(df, options_file)

        pool = Pool(processes=level_0_performance_report.pool_workers_count)
        results = pool.map(options_scraping_per_group, [(key, group, colors_list, options_file.project_id) for (key, group) in df_grouped])
        pool.close()
        df = pd.concat([result[0] for result in results if result is not None])

        baviera_standard_equipment(df)

    elif options_file.project_id == 2775:  # CDSU
        options_scraping_model(df, options_file.project_id, approach='first_word')
        options_scraping_motorization(df, options_file, fillna_value='Sem Info Motorização', regex_approach=1)
        options_scraping_interior_type(df, options_file, fillna_value='Tecido', regex_approach=1)
        options_scraping_version(df, options_file, fillna_value='Sem Info Versão', regex_approach=1)
        # Combustível columns is handled as is;

        from level_2_optionals_cdsu_options import colors_pt, colors_en
        colors_list = [colors_pt, colors_en]

        pool = Pool(processes=level_0_performance_report.pool_workers_count)
        results = pool.map(options_scraping_per_group_cdsu, [(key, group, colors_list, options_file.project_id, options_file.regex_dict) for (key, group) in df.groupby('Nº Stock')])
        pool.close()
        df = pd.concat([result for result in results if result is not None])

    return df


def options_scraping_model(df, project_id, approach='', model_mapping={}, model_training_check=0):
    level_0_performance_report.performance_info_append(time.time(), 'Model_Code_Start')

    if approach != 'first_word':
        if model_training_check:
            unique_models = df['Modelo'].unique()
            for model in unique_models:
                tokenized_modelo = nltk.word_tokenize(model)
                df.loc[df['Modelo'] == model, 'Modelo'] = ' '.join(tokenized_modelo[:-3])
        elif not model_training_check:
            unique_version_code = df['Version_Code'].unique()
            for version_code in unique_version_code:
                found_flag = 0
                for key in model_mapping.keys():
                    if version_code in model_mapping[key]:
                        df.loc[df['Version_Code'] == version_code, 'Modelo'] = key
                        found_flag = 1
                        break
                if not found_flag:
                    level_0_performance_report.log_record('Não foi encontrada a parametrização para o seguinte código de versão: {}.'.format(version_code), project_id, flag=1)
    elif approach == 'first_word':
        unique_models = df['Modelo'].unique()
        for model in unique_models:
            tokenized_modelo = nltk.word_tokenize(model)
            first_word_model = tokenized_modelo[0]
            if first_word_model.lower() == 'novo':
                df.loc[df['Modelo'] == model, 'Modelo'] = tokenized_modelo[1]
            else:
                df.loc[df['Modelo'] == model, 'Modelo'] = tokenized_modelo[0]

    level_0_performance_report.performance_info_append(time.time(), 'Model_Code_End')

    return df


def options_scraping_motorization(df, options_file, fillna_value=None, regex_approach=0):
    level_0_performance_report.performance_info_append(time.time(), 'Motor_Desc_Start')

    if regex_approach:
        string_regex_extraction(df, 'Versão', 'Motor', options_file.regex_dict['motorization_value'], fillna_value)
    else:
        unique_versions = df['Versão'].unique()
        for version in unique_versions:
            mask_version = df['Versão'] == version
            if 'x1 ' in version or 'x2 ' in version or 'x3 ' in version or 'x4 ' in version or 'x5 ' in version or 'x6 ' in version or 'x7 ' in version:  # The extra free space in the X models is because there are references next to the version description that match the searching criteria. Ex: 420D Coupé (4X31) matches when searched by X3
                df.loc[mask_version, 'Motor'] = [x.split(' ')[1] for x in df[mask_version]['Versão']]
            else:
                df.loc[mask_version, 'Motor'] = [x.split(' ')[0] for x in df[mask_version]['Versão']]

    level_0_performance_report.performance_info_append(time.time(), 'Motor_Desc_End')
    return df


def options_scraping_interior_type(df, options_file, fillna_value=None, regex_approach=1):
    level_0_performance_report.performance_info_append(time.time(), 'Int_Type_Start')

    if regex_approach:
        string_regex_extraction(df, 'Interior', 'Tipo_Interior', options_file.regex_dict['interior_type_value'], fillna_value)

    level_0_performance_report.performance_info_append(time.time(), 'Int_Type_End')
    return df


def options_scraping_version(df, options_file, fillna_value=None, regex_approach=1):
    level_0_performance_report.performance_info_append(time.time(), 'Version_Start')

    if regex_approach:
        string_regex_extraction(df, 'Versão', 'Versao', options_file.regex_dict['version_type'], fillna_value)

    level_0_performance_report.performance_info_append(time.time(), 'Version_End')


def string_regex_extraction(df, orig_col, new_col, regex_rule, fillna_value=None):

    if fillna_value:
        df[new_col] = df[orig_col].str.extract(regex_rule).fillna(fillna_value)
        # When str.extract matches nothing, it returns NaN
    else:
        df[new_col] = df[orig_col].str.extract(regex_rule)

    return df


def options_scraping(df, options_file, model_mapping={}, model_training_check=0):
    from level_2_optionals_baviera_options import colors_pt, colors_en, dakota_colors, vernasca_colors, nappa_colors, nevada_colors, merino_colors
    colors_list = [colors_pt, colors_en, dakota_colors, vernasca_colors, nappa_colors, nevada_colors, merino_colors]

    project_id = options_file.project_id

    df_grouped = df.groupby('Nº Stock')

    # Modelo
    if len(list(model_mapping.keys())):
        unique_version_code = df['Version_Code'].unique()
        for version_code in unique_version_code:
            found_flag = 0
            for key in model_mapping.keys():
                if version_code in model_mapping[key]:
                    df.loc[df['Version_Code'] == version_code, 'Modelo'] = key
                    found_flag = 1
                    break
            if not found_flag:
                level_0_performance_report.log_record('Não foi encontrada a parametrização para o seguinte código de versão: {}.'.format(version_code), project_id, flag=1)
    else:
        unique_models = df['Modelo'].unique()
        for model in unique_models:
            min_sell_date_model = df.loc[df['Modelo'] == model, 'Data Venda'].min()
            # print(model, min_sell_date_model, datetime.datetime.strptime('2020-10-01', "%Y-%m-%d"))
            if model in ['Serie 4', 'X6', 'X3']:
                tokenized_modelo = [model]
            if min_sell_date_model < datetime.datetime.strptime('2020-10-01', "%Y-%m-%d"):
                if model.startswith('BMW'):
                    new_model = model[4:].capitalize()
                    tokenized_modelo = nltk.word_tokenize(new_model)[:-3]
                else:
                    tokenized_modelo = nltk.word_tokenize(model)[:-3]
            elif min_sell_date_model >= datetime.datetime.strptime('2020-10-01', "%Y-%m-%d"):  # Spiga Migration changed the way Model_Desc is made. Serie 5 Touring to (G31) Serie 5 Touring
                if model.startswith('BMW'):
                    new_model = model[4:].capitalize()
                    tokenized_modelo = nltk.word_tokenize(new_model)[3::]
                else:
                    tokenized_modelo = nltk.word_tokenize(model)[3::]

            new_model = re.sub(r'^(s)(?=[0-9])', 'Serie ', ' '.join(tokenized_modelo), flags=re.IGNORECASE)
            df.loc[df['Modelo'] == model, 'Modelo'] = new_model
    level_0_performance_report.performance_info_append(time.time(), 'Model_Code_End')

    # Motorização
    level_0_performance_report.performance_info_append(time.time(), 'Motor_Desc_Start')
    unique_versions = df['Versão'].unique()
    for version in unique_versions:
        mask_version = df['Versão'] == version
        if 'x1 ' in version or 'x2 ' in version or 'x3 ' in version or 'x4 ' in version or 'x5 ' in version or 'x6 ' in version or 'x7 ' in version:  # The extra free space in the X models is because there are references next to the version description that match the searching criteria. Ex: 420D Coupé (4X31) matches when searched by X3
            df.loc[mask_version, 'Motor'] = [x.split(' ')[1] for x in df[mask_version]['Versão']]
        else:
            df.loc[mask_version, 'Motor'] = [x.split(' ')[0] for x in df[mask_version]['Versão']]
    level_0_performance_report.performance_info_append(time.time(), 'Motor_Desc_End')

    pool = Pool(processes=level_0_performance_report.pool_workers_count)
    results = pool.map(options_scraping_per_group, [(key, group, colors_list, project_id) for (key, group) in df_grouped])
    pool.close()
    df = pd.concat([result for result in results if result is not None])

    return df


def baviera_standard_equipment(df):

    # ToDo: move the following code to it's own function?
    # Standard Equipment
    level_0_performance_report.performance_info_append(time.time(), 'Standard_Start')
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
    # df.loc[df[criteria_model_s4 & criteria_buy_year_lt_2017 & criteria_farois_led_0].index, 'Farois_Xenon'] = 1
    df.loc[df[criteria_model_s5 & criteria_jantes_0].index, 'Jantes'] = '17'
    df.loc[df[criteria_model_s5 & criteria_buy_year_ge_2017].index, 'Farois_LED'] = 1
    df.loc[df[criteria_model_s5 & criteria_buy_year_ge_2017].index, 'Alarme'] = 1
    # df.loc[df[criteria_model_s5 & criteria_buy_year_lt_2017 & criteria_farois_led_0].index, 'Farois_Xenon'] = 1
    df.loc[df[criteria_model_s5].index, 'Sensores'] = 1
    df.loc[df[criteria_model_s5].index, 'Navegação'] = 1
    df.loc[df[criteria_model_s5].index, 'Caixa Auto'] = 1
    df.loc[df[criteria_model_x1 & criteria_jantes_0].index, 'Jantes'] = '17'
    df.loc[df[criteria_model_x3 & criteria_jantes_0].index, 'Jantes'] = '18'
    df.loc[df[criteria_model_x3].index, 'Sensores'] = 1
    df.loc[df[criteria_model_x3].index, 'Caixa Auto'] = 1
    df.loc[df['Versao'] == 0, 'Versao'] = 'base'
    df.loc[df['Jantes'] == 0, 'Jantes'] = 'standard'
    level_0_performance_report.performance_info_append(time.time(), 'Standard_End')

    return df


def options_scraping_per_group(args):
    key, group, colors_list, project_id = args
    colors_pt, colors_en, dakota_colors, vernasca_colors, nappa_colors, nevada_colors, merino_colors = colors_list[0], colors_list[1], colors_list[2], colors_list[3], colors_list[4], colors_list[5], colors_list[6]

    line_modelo = group['Modelo'].head(1).values[0]
    tokenized_modelo = nltk.word_tokenize(line_modelo)
    optionals = set(group['Opcional'])

    # Navegação
    if len([x for x in optionals if 'navegação' in x]):
        group['Navegação'] = 1

    # Barras Tejadilho
    if len([x for x in optionals if 'barras' in x]):
        group['Barras_Tej'] = 1

    # Alarme
    if len([x for x in optionals if 'alarme' in x]):
        group['Alarme'] = 1

    # AC Auto
    if len([x for x in optionals if 'ar' in x and 'condicionado' in x and 'automático' in x]):
        group['AC Auto'] = 1

    # Teto Abrir
    if len([x for x in optionals if 'teto' in x and 'abrir' in x]):
        group['Teto_Abrir'] = 1

    # Sensor/Transmissão/Versão/Jantes
    jantes_size = [0]
    for line_options in group['Opcional']:
        tokenized_options = nltk.word_tokenize(line_options)

        if 'pdc-sensores' in tokenized_options:
            for word in tokenized_options:
                if 'diant' in word:
                    group['Sensores'] = 1

        if 'transmissão' in tokenized_options or 'caixa' in tokenized_options:
            for word in tokenized_options:
                if 'auto' in word:
                    group['Caixa Auto'] = 1

        # Versão
        if 'advantage' in tokenized_options:
            group['Versao'] = 'advantage'
        elif 'versão' in tokenized_options or 'bmw' in tokenized_options:
            if 'line' in tokenized_options and 'sport' in tokenized_options:
                group['Versao'] = 'line sport'
            if 'line' in tokenized_options and 'urban' in tokenized_options:
                group['Versao'] = 'line urban'
            if 'desportiva' in tokenized_options and 'm' in tokenized_options:
                group['Versao'] = 'desportiva m'
            if 'line' in tokenized_options and 'luxury' in tokenized_options:
                group['Versao'] = 'line luxury'
        if 'pack' in tokenized_options and 'desportivo' in tokenized_options and 'm' in tokenized_options:
            if 'S1' in tokenized_modelo:
                group['Versao'] = 'desportiva m'
            elif 'S5' in tokenized_modelo or 'S3' in tokenized_modelo or 'S2' in tokenized_modelo:
                group['Versao'] = 'pack desportivo m'
        if 'bmw' in tokenized_options and 'modern' in tokenized_options:  # no need to search for string line, there are no bmw modern without line;
            if 'S5' in tokenized_modelo:
                group['Versao'] = 'line luxury'
            else:
                group['Versao'] = 'line urban'
        if 'xline' in tokenized_options:
            group['Versao'] = 'xline'

        # Faróis
        # if "xénon" in tokenized_options or 'bixénon' in tokenized_options:
        #     group['Farois_Xenon'] = 1
        if "luzes" in tokenized_options and "led" in tokenized_options and 'nevoeiro' not in tokenized_options or 'luzes' in tokenized_options and 'adaptativas' in tokenized_options and 'led' in tokenized_options or 'faróis' in tokenized_options and 'led' in tokenized_options and 'nevoeiro' not in tokenized_options:
            group['Farois_LED'] = 1

        # Jantes
        for value in range(15, 21):
            if str(value) in tokenized_options and value > int(jantes_size[0]):
                jantes_size = [str(value)] * group.shape[0]
                group['Jantes'] = jantes_size

    # Cor Exterior
    line_color = group['Cor'].head(1).values[0]
    tokenized_color = nltk.word_tokenize(line_color)
    color = [x for x in colors_pt if x in tokenized_color]
    if not color:
        color = [x for x in colors_en if x in tokenized_color]
    if not color:
        if tokenized_color == ['pintura', 'bmw', 'individual'] or tokenized_color == ['hp', 'motorsport', ':', 'branco/azul/vermelho', '``', 'racing', "''"] or tokenized_color == ['p0b58'] or tokenized_color == [' ']:
            color = ['undefined']
            level_0_performance_report.log_record('Cor exterior não encontrada {} para o veículo {}.'.format(tokenized_color, key), project_id, flag=1)
        elif tokenized_color == ['verm', 'tk', 'mmm']:
            color = ['vermelho']
        else:
            line_color_ext_code = group['Colour_Ext_Code'].head(1).values[0]
            if line_color_ext_code == 'P0X13':
                color = ['castanho']
            elif len(tokenized_color) == 0:
                level_0_performance_report.log_record('Não foi encontrada a cor exterior do veículo {} com a seguinte descrição de cor exterior: \'{}\'.'.format(key, line_color), project_id, flag=1)
                return
            else:
                level_0_performance_report.log_record('Cor exterior não encontrada {} para o veículo {}.'.format(tokenized_color, key), project_id, flag=1)
                color = np.nan
                pass
                # raise ValueError('Color Ext Not Found: {} in Vehicle {}'.format(tokenized_color, key))
    try:
        if len(color) > 1:  # Fixes cases such as 'white silver'
            color = [color[0]]
        color = color * group.shape[0]
    except TypeError:
        pass
    try:
        group['Cor_Exterior'] = color
    except ValueError:
        print(color)

    # Cor Interior
    line_interior = group['Interior'].head(1).values[0]
    tokenized_interior = nltk.word_tokenize(line_interior)

    if 'dakota' in tokenized_interior:
        color_int = [x for x in tokenized_interior if x in dakota_colors]
        if color_int:
            group['Cor_Interior'] = 'dakota_' + color_int[0]
    elif 'nappa' in tokenized_interior:
        color_int = [x for x in tokenized_interior if x in nappa_colors]
        if color_int:
            group['Cor_Interior'] = 'nappa_' + color_int[0]
    elif 'vernasca' in tokenized_interior:
        color_int = [x for x in tokenized_interior if x in vernasca_colors]
        if color_int:
            group['Cor_Interior'] = 'vernasca_' + color_int[0]
    elif 'nevada' in tokenized_interior:
        color_int = [x for x in tokenized_interior if x in nevada_colors]
        if color_int:
            group['Cor_Interior'] = 'nevada_' + color_int[0]
    elif 'merino' in tokenized_interior:
        color_int = [x for x in tokenized_interior if x in merino_colors]
        if color_int:
            group['Cor_Interior'] = 'merino_' + color_int[0]
    else:
        if 'antraci' in tokenized_interior or 'antracit' in tokenized_interior or 'anthracite/silver' in tokenized_interior or 'preto/laranja' in tokenized_interior or 'preto/silver' in tokenized_interior or 'preto/preto' in tokenized_interior or 'confort' in tokenized_interior or 'standard' in tokenized_interior or 'preto' in tokenized_interior or 'antracite' in tokenized_interior or 'antracite/laranja' in tokenized_interior or 'antracite/preto' in tokenized_interior or 'antracite/cinza/preto' in tokenized_interior or 'antracite/vermelho/preto' in tokenized_interior or 'antracite/vermelho' in tokenized_interior or 'interiores' in tokenized_interior:
            group['Cor_Interior'] = 'preto'
        elif 'oyster/preto' in tokenized_interior:
            group['Cor_Interior'] = 'oyster'
        elif 'platinu' in tokenized_interior or 'grey' in tokenized_interior or 'prata/preto/preto' in tokenized_interior or 'prata/cinza' in tokenized_interior:
            group['Cor_Interior'] = 'cinzento'
        elif 'castanho' in tokenized_interior or 'walnut' in tokenized_interior:
            group['Cor_Interior'] = 'castanho'
        elif 'âmbar/preto/pr' in tokenized_interior:
            group['Cor_Interior'] = 'amarelo'
        elif 'champagne' in tokenized_interior:
            group['Cor_Interior'] = 'bege'
        elif 'crimson' in tokenized_interior:
            group['Cor_Interior'] = 'vermelho'
        else:
            group['Cor_Interior'] = '0'
            # level_0_performance_report.log_record('Cor Interior não encontrada: \'{}\' para o veículo {}.'.format(tokenized_color, key), project_id, flag=1)

    # Tipo Interior
    if 'comb' in tokenized_interior or 'combin' in tokenized_interior or 'combinação' in tokenized_interior or 'tecido/pele' in tokenized_interior:
        group['Tipo_Interior'] = 'combinação'
    elif 'hexagon\'' in tokenized_interior or 'hexagon/alcantara' in tokenized_interior:
        group['Tipo_Interior'] = 'tecido_micro'
    elif 'tecido' in tokenized_interior or 'cloth' in tokenized_interior:
        group['Tipo_Interior'] = 'tecido'
    elif 'pele' in tokenized_interior or 'leather' in tokenized_interior or 'dakota\'' in tokenized_interior or 'couro' in tokenized_interior:
        group['Tipo_Interior'] = 'pele'
    else:
        group['Tipo_Interior'] = '0'

    return group


def options_scraping_per_group_cdsu(args):
    key, group, colors_list, project_id, regex_dict = args
    colors_pt, colors_en = colors_list[0], colors_list[1]

    # Sensor/Transmissão/Versão/Jantes
    for line_options in group['Opcional']:
        tokenized_options = nltk.word_tokenize(str(line_options))

        if 'caixa' in tokenized_options:
            for word in tokenized_options:
                if 'aut' in word:
                    group['Caixa Auto'] = 1
        if 'dsg' in tokenized_options:
            group['Caixa Auto'] = 1

        if 'sensores' in tokenized_options:
            if 'estacionamento' in tokenized_options or 'estcmt' in tokenized_options:
                if 'dianteiros' in tokenized_options or 'frt' in tokenized_options:
                    group['Sensores Est. Front.'] = 1
                if 'traseiros' in tokenized_options or 'tras' in tokenized_options:
                    group['Sensores Est. Tras.'] = 1

        if 'camara' in tokenized_options:
            group['Câmara Traseira'] = 1

        # Jantes
        if 'sobresselente' not in tokenized_options and 'sobressalente' not in tokenized_options:
            if 'jantes' in tokenized_options \
                    or 'jante' in tokenized_options \
                    or 'jantes' in tokenized_options  \
                    or 'jante' in tokenized_options \
                    or '"woodstack"' in tokenized_options:
                match = re.search(regex_dict['rims_size'], line_options)
                if match:
                    group['Jantes'] = re.search(regex_dict['rims_size'], line_options).group()

    if group['Jantes'].head(1).values[0] == 0:
        group['Jantes'] = 'Standard'

    # Cor Exterior
    line_color = group['Cor'].head(1).values[0]
    tokenized_color = nltk.word_tokenize(line_color)
    color = [x for x in colors_pt if x in tokenized_color]
    if not color:
        color = [x for x in colors_en if x in tokenized_color]
    if not color:
        if tokenized_color == ['pintura', 'bmw', 'individual'] or tokenized_color == ['hp', 'motorsport', ':', 'branco/azul/vermelho', '``', 'racing', "''"] or tokenized_color == ['p0b58'] or tokenized_color == [' ']:
            color = ['undefined']
            level_0_performance_report.log_record('1 - Cor exterior não encontrada {} para o veículo {}.'.format(tokenized_color, key), project_id, flag=1)
        elif tokenized_color == ['verm', 'tk', 'mmm']:
            color = ['vermelho']
        elif tokenized_color == ['pintura', 'metalizada']:
            color = ['cinzento']
        else:
            line_color_ext_code = group['Colour_Ext_Code'].head(1).values[0]
            if line_color_ext_code == 'P0X13':
                color = ['castanho']
            elif len(tokenized_color) == 0:
                level_0_performance_report.log_record('Não foi encontrada a cor exterior do veículo {} com a seguinte descrição de cor exterior: \'{}\'.'.format(key, line_color), project_id, flag=1)
                return
            else:
                level_0_performance_report.log_record('2 - Cor exterior não encontrada {} para o veículo {}.'.format(tokenized_color, key), project_id, flag=1)
                color = np.nan
                pass
                # raise ValueError('Color Ext Not Found: {} in Vehicle {}'.format(tokenized_color, key))
    try:
        if len(color) > 1:  # Fixes cases such as 'white silver'
            color = [color[0]]
        color = color * group.shape[0]
    except TypeError:
        pass
    try:
        group['Cor_Exterior'] = color
    except ValueError:
        print(color)

    return group


def pandas_regex_extraction_v2(df, pattern_dict, sel_col_to_search, sel_col_to_flag):

    for key, value in pattern_dict.items():
        find_words = []
        non_find_words = []

        words = value[0::2]
        word_flags = value[1::2]
        for word, word_flag in zip(words, word_flags):
            if word_flag:
                find_words.append(word)
            else:
                non_find_words.append(word)

        print('non_find_words', non_find_words)
        non_find_words_pattern = r'^(?:(?!{}).)*$'.format('|'.join(non_find_words))
        print('non_find_words_pattern: {}'.format(non_find_words_pattern))

        print('find_words', find_words)
        all_sel_text_list_combinations = list(itertools.permutations(find_words))  # Creates all combinations with the selected words, taking order into account
        print(all_sel_text_list_combinations)

        find_words_pattern = '|'.join(['.*{}.*'.format(x) for x in ['\\b.*\\b'.join(x) for x in all_sel_text_list_combinations]])
        print('find_words_pattern: {}'.format(find_words_pattern))

        df.loc[(df[sel_col_to_search].str.contains(non_find_words_pattern, regex=True)) & (df[sel_col_to_search].str.contains(find_words_pattern, regex=True)), sel_col_to_flag] = 1
        print(df)


def datasets_dictionary_function(train_x, train_y, test_x, test_y, train_x_oversampled=pd.DataFrame(), train_y_oversampled=pd.Series()):

    if train_x_oversampled.shape[0]:
        dataset_dict = {
            'train_x': train_x_oversampled[[col for col in list(train_x) if col not in ['oversample_flag', 'original_index']]],
            'train_y': train_y_oversampled,
            'test_x': test_x,
            'test_y': test_y,
            'train_x_original': train_x,
            'train_y_original': train_y,
            'train_x_oversampled_original': train_x_oversampled,
        }
    # The column removal in train_x is due to oversampling. When oversample is on, additional (columns oversample_flag and original_index) are created in order to control which rows are
    # from the original dataset and which are oversampled. But those two columns aren't added to the test_x dataset (no oversample in the test dataset) and therefore train and test datasets
    # won't match, hence why i need to remove the two extra columns;

    else:
        dataset_dict = {
            'train_x': train_x,
            'train_y': train_y,
            'test_x': test_x,
            'test_y': test_y
        }

    return dataset_dict


def column_rename(df, cols_to_replace, new_cols_names):
    for column in cols_to_replace:
        df.rename(index=str, columns={cols_to_replace[cols_to_replace.index(column)]: new_cols_names[cols_to_replace.index(column)]}, inplace=True)

    return df


def constant_columns_removal(df, project_id, value=None):
    features_removed, list_after = [], []

    list_before = list(df)
    if value is None:
        df = df.loc[:, df.apply(pd.Series.nunique) != 1]

        list_after = list(df)

        features_removed = [item for item in list_before if item not in list_after]
    elif value is not None:
        for column in list(df):
            if df[column].nunique() == 1 and df[column].unique() == value:
                print('constant column {}'.format(column))
                features_removed.append(column)
            else:
                list_after.append(column)

    if len(features_removed):
        columns_string = level_1_e_deployment.sql_string_preparation_v2(features_removed)
        level_0_performance_report.log_record('A(s) seguinte(s) coluna(s) sem variação de valores foram removida(s): ' + str(columns_string), project_id, flag=1)

    return df[list_after]


def col_group(df, columns_to_replace, dictionaries, project_id):
    non_parametrized_data_flag = 0

    for dictionary in dictionaries:
        column = columns_to_replace[dictionaries.index(dictionary)]
        try:
            for key in dictionary.keys():
                df.loc[df[column].isin(dictionary[key]), column + '_new'] = key
            if df[column + '_new'].isnull().values.any():
                non_parametrized_data_flag = 1
                variable = df.loc[df[column + '_new'].isnull(), column].unique()
                if project_id == 2162:
                    level_0_performance_report.log_record('Aviso no Agrupamento de Colunas  - NaNs detetados em: {}_new, valor(es) não agrupados: {} nos veículos(s) com VHE_Number(s): {}'.format(columns_to_replace[dictionaries.index(dictionary)], variable, df[df[column + '_new'].isnull()]['Nº Stock'].unique()), project_id, flag=1)
                elif project_id == 2406:
                    level_0_performance_report.log_record('Aviso no Agrupamento de Colunas  - NaNs detetados em: {}_new, valor(es) não agrupados: {} nos veículos(s) com VehicleData_Code(s): {}'.format(columns_to_replace[dictionaries.index(dictionary)], variable, df[df[column + '_new'].isnull()]['VehicleData_Code'].unique()), project_id, flag=1)
                elif project_id == 2259:
                    level_0_performance_report.log_record('Aviso no Agrupamento de Colunas  - NaNs detetados em: {}_new, valor(es) não agrupados: {} nos veículos(s) com Group(s): {}'.format(columns_to_replace[dictionaries.index(dictionary)], variable, df[df[column + '_new'].isnull()]['Group'].unique()), project_id, flag=1)
                else:
                    null_analysis(df)
            df.drop(column, axis=1, inplace=True)
            df.rename(index=int, columns={column + '_new': column}, inplace=True)
        except KeyError:
            level_0_performance_report.log_record('Aviso no Agrupamento de Colunas - Coluna {} não encontrada.'.format(column), project_id, flag=1)

    if non_parametrized_data_flag:
        raise ValueError('Existem valores não parametrizados. Por favor corrigir.')

    return df


def sell_place_parametrization(df, original_sale_place_column, new_sale_place_column, mapping, project_id):

    sell_districts = list(mapping.values())
    sel_districts_flat = [x for sublist in sell_districts for x in sublist]

    unique_sale_places = df[original_sale_place_column].unique()

    for sale_place in unique_sale_places:
        sale_place_level_2 = [x for x in sel_districts_flat if x in sale_place]
        if len(sale_place_level_2):
            df.loc[df[original_sale_place_column] == sale_place, new_sale_place_column + '_level_2'] = sale_place_level_2[0]
        else:
            if sale_place[0:3] == 'DCS':  # This is for the cases where the description has no Local. Currently, all of them are from Lisboa (DCS)
                df.loc[df[original_sale_place_column] == sale_place, new_sale_place_column + '_level_2'] = 'Lisboa'
            else:
                level_0_performance_report.log_record('Não foi encontrada a parametrização para o seguinte local de venda: {}.'.format(sale_place), project_id, flag=1)

    df = new_column_creation(df, [new_sale_place_column + '_level_1'], df[new_sale_place_column + '_level_2'])
    df = col_group(df, [new_sale_place_column + '_level_1'], [mapping], project_id)

    return df


def total_price(df):
    df['price_total'] = df['Custo'].groupby(df['Nº Stock']).transform('sum')

    return df


def remove_zero_price_total_vhe(df, project_id):
    old_count = df['Nº Stock'].nunique()
    remove_rows(df, [df.loc[df.price_total == 0, :].index], project_id)
    new_count = df['Nº Stock'].nunique()

    removed_vehicles = old_count - new_count
    if removed_vehicles:
        level_0_performance_report.log_record('Foram removidos {} veículos com custo total de 0.'.format(old_count - new_count), project_id, flag=1)
    return df


def margin_calculation(df):
    df['margem_percentagem'] = (df['Margem'].round(2) / df['price_total'].round(2)) * 100
    df['margem_percentagem'] = df['margem_percentagem'].round(4)

    return df


def prov_replacement(df):
    df.loc[df['Prov'] == 'Viaturas Km 0', 'Prov'] = 'Novos'
    # df.rename({'Prov': 'Prov_new'}, axis=1, inplace=True)

    return df


def color_replacement(df, color_replacement_dict, project_id):
    # Project_Id = 2162

    color_types = ['Cor_Exterior']

    try:
        for color_type in color_types:
            df[color_type] = df[color_type].replace(color_replacement_dict)
            df.drop(df[df[color_type] == 0].index, axis=0, inplace=True)
    except TypeError:
        level_0_performance_report.log_record('Color Ext Not Found', project_id, flag=1)

    return df


def score_calculation(df, stockdays_threshold, margin_threshold, project_id):
    if project_id == 2162 or project_id == 2775:
        df['stock_days'] = (df['Data Venda'] - df['Data Compra']).dt.days
        df.loc[df['stock_days'].lt(0), 'stock_days'] = 0

        df['stock_days_class'] = 0
        df.loc[df['stock_days'] <= stockdays_threshold[0], 'stock_days_class'] = 1
        df['margin_class'] = 0
        df.loc[df['margem_percentagem'] >= margin_threshold, 'margin_class'] = 1

        df['new_score'] = 0
        df.loc[(df['stock_days_class'] == 1) & (df['margin_class'] == 1), 'new_score'] = 1

        df['days_stock_price'] = (0.05/360) * df['price_total'] * df['stock_days']
        df['score_euros'] = df['Margem'] - df['days_stock_price']

    elif project_id == 2406:
        # Binary Approach
        # df['stock_days_class'] = 0
        # df.loc[df['DaysInStock_Global'] <= stockdays_threshold, 'stock_days_class'] = 1
        #
        # df['target_class'] = 0
        # df.loc[(df['stock_days_class'] == 1), 'target_class'] = 1

        # MultiCLass Approach v1
        df['target_class'] = 0
        number_of_classes = len(stockdays_threshold)

        df.loc[df['DaysInStock_Global'] <= stockdays_threshold[0], 'target_class'] = 0
        df.loc[(df['DaysInStock_Global'] > 90) & (df['DaysInStock_Global'] <= 120), 'target_class'] = 1
        df.loc[(df['DaysInStock_Global'] > 120) & (df['DaysInStock_Global'] <= 150), 'target_class'] = 2
        df.loc[(df['DaysInStock_Global'] > 150) & (df['DaysInStock_Global'] <= 180), 'target_class'] = 3
        df.loc[(df['DaysInStock_Global'] > 180) & (df['DaysInStock_Global'] <= 270), 'target_class'] = 4
        df.loc[(df['DaysInStock_Global'] > 270) & (df['DaysInStock_Global'] <= 365), 'target_class'] = 5
        df.loc[df['DaysInStock_Global'] > stockdays_threshold[number_of_classes - 1], 'target_class'] = number_of_classes

    return df


def value_substitution(df, non_null_column=None, null_column=None):
    if non_null_column is not None and null_column is not None:
        df.loc[df[null_column].isnull() & ~df[non_null_column].isnull(), null_column] = df.loc[df[null_column].isnull() & ~df[non_null_column].isnull(), non_null_column]

    return df


def new_features(df, sel_cols, project_id):

    if project_id == 2162 or project_id == 2775:
        df.dropna(inplace=True)  # This is here for the cases where a value is not grouped. So it doesn't stop the code. Warning will still be uploaded to SQL.
        df_grouped = df.sort_values(by=['Data Venda']).groupby(sel_cols)
        print('Number of Configurations: {}'.format(len(df_grouped)))
        df = df_grouped.apply(previous_sales_info_order_optimization_projects)

        return df.fillna(0)

    if project_id == 2406:
        df.dropna(subset=['Registration_Request_Date'], inplace=True)
        df_grouped = df.sort_values(by='Registration_Request_Date').groupby(sel_cols)
        pool = Pool(processes=level_0_performance_report.pool_workers_count)
        results = pool.map(additional_info_optimization_hyundai, [(key, group) for (key, group) in df_grouped])
        pool.close()
        df = pd.concat([result for result in results if result is not None])
        # df = df_grouped.apply(additional_info_optimization_hyundai)

        return df


def additional_info_optimization_hyundai(args):
    # Project_ID = 2406

    key, x = args

    if len(x) > 1:
        x['prev_sales_check'] = [0] + [1] * (len(x) - 1)
        x['number_prev_sales'] = list(range(len(x)))

        for key, row in x.iterrows():
            last_date = row['Registration_Request_Date']

            month_minus_1 = last_date - relativedelta(months=1)
            month_minus_2 = last_date - relativedelta(months=2)
            month_minus_3 = last_date - relativedelta(months=3)
            month_minus_4 = last_date - relativedelta(months=4)
            month_minus_5 = last_date - relativedelta(months=5)
            month_minus_6 = last_date - relativedelta(months=6)

            x.loc[x.index == key, 'sales_month-1'] = x[(x['Registration_Request_Date'] > month_minus_1) & (x['Registration_Request_Date'] < last_date)].shape[0]
            x.loc[x.index == key, 'sales_month-2'] = x[(x['Registration_Request_Date'] > month_minus_2) & (x['Registration_Request_Date'] < month_minus_1)].shape[0]
            x.loc[x.index == key, 'sales_month-3'] = x[(x['Registration_Request_Date'] > month_minus_3) & (x['Registration_Request_Date'] < month_minus_2)].shape[0]
            x.loc[x.index == key, 'sales_month-4'] = x[(x['Registration_Request_Date'] > month_minus_4) & (x['Registration_Request_Date'] < month_minus_3)].shape[0]
            x.loc[x.index == key, 'sales_month-5'] = x[(x['Registration_Request_Date'] > month_minus_5) & (x['Registration_Request_Date'] < month_minus_4)].shape[0]
            x.loc[x.index == key, 'sales_month-6'] = x[(x['Registration_Request_Date'] > month_minus_6) & (x['Registration_Request_Date'] < month_minus_5)].shape[0]

            x.loc[x.index == key, 'sales_month-1_cum'] = x[(x['Registration_Request_Date'] > month_minus_1)].shape[0]
            x.loc[x.index == key, 'sales_month-2_cum'] = x[(x['Registration_Request_Date'] > month_minus_2)].shape[0]
            x.loc[x.index == key, 'sales_month-3_cum'] = x[(x['Registration_Request_Date'] > month_minus_3)].shape[0]
            x.loc[x.index == key, 'sales_month-4_cum'] = x[(x['Registration_Request_Date'] > month_minus_4)].shape[0]
            x.loc[x.index == key, 'sales_month-5_cum'] = x[(x['Registration_Request_Date'] > month_minus_5)].shape[0]
            x.loc[x.index == key, 'sales_month-6_cum'] = x[(x['Registration_Request_Date'] > month_minus_6)].shape[0]

    elif len(x) <= 1:
        x['prev_sales_check'] = 0
        x['number_prev_sales'] = 0
        x['sales_month-1'], x['sales_month-2'], x['sales_month-3'], x['sales_month-4'], x['sales_month-5'], x['sales_month-6'] = 0, 0, 0, 0, 0, 0
        x['sales_month-1_cum'], x['sales_month-2_cum'], x['sales_month-3_cum'], x['sales_month-4_cum'], x['sales_month-5_cum'], x['sales_month-6_cum'] = 0, 0, 0, 0, 0, 0

    return x


def previous_sales_info_order_optimization_projects(x):
    # Project_ID = 2162

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


def global_variables_saving(df, project_id):
    if project_id == 2162:
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


def dataset_split(df, target, oversample=0, objective='classification'):
    if objective == 'classification':
        df_train, df_test = train_test_split(df.copy(), stratify=df[target], random_state=2)  # This ensures that the classes are evenly distributed by train/test datasets; Default split is 0.75/0.25 train/test

        df_train_y = df_train[target]
        df_train_x = df_train.drop(target, axis=1)

        df_test_y = df_test[target]
        df_test_x = df_test.drop(target, axis=1)

    elif objective == 'regression':
        df_train_x, df_test_x, df_train_y, df_test_y = train_test_split(df.copy(), df[target], test_size=0.2, random_state=5)

        df_train_x.drop(target, axis=1, inplace=True)
        df_test_x.drop(target, axis=1, inplace=True)

    if oversample:
        print('Oversampling small classes...')
        df_train_x_oversampled, df_train_y_oversampled = oversample_data(df_train_x, df_train_y)
        datasets = datasets_dictionary_function(df_train_x, df_train_y, df_test_x, df_test_y, df_train_x_oversampled, df_train_y_oversampled)

        return datasets

    datasets = datasets_dictionary_function(df_train_x, df_train_y, df_test_x, df_test_y)

    return datasets


def oversample_data(train_x, train_y):

    train_x['oversample_flag'] = range(train_x.shape[0])
    train_x['original_index'] = pd.to_numeric(train_x.index)

    ros = RandomOverSampler(random_state=42)
    train_x_resampled, train_y_resampled = ros.fit_sample(train_x, train_y.values.ravel())

    train_x_resampled = pd.DataFrame(np.atleast_2d(train_x_resampled), columns=list(train_x))
    train_y_resampled = pd.Series(train_y_resampled)
    for column in list(train_x_resampled):
        if train_x_resampled[column].dtype != train_x[column].dtype:
            print('Problem found with dtypes, fixing it...')
            dtype_checkup(train_x_resampled, train_x)
        break

    return train_x_resampled, train_y_resampled


def dtype_checkup(train_x_resampled, train_x):
    for column in list(train_x):
        train_x_resampled[column] = train_x_resampled[column].astype(train_x[column].dtype)


def ohe(df, cols, project_id):

    for column in cols:
        try:
            uniques = df[column].unique()
            for value in uniques:
                new_column = column + '_' + str(value)
                df[new_column] = 0
                df.loc[df[column] == value, new_column] = 1
            df.drop(column, axis=1, inplace=True)
        except KeyError:
            level_0_performance_report.log_record('Aviso de conversão OHE de coluna - A coluna {} não foi encontrada.'.format(column), project_id, flag=1)
            continue

    return df


def duplicate_removal(df, subset_col):
    df.drop_duplicates(subset=subset_col, inplace=True)

    return df


def reindex(df):
    df.index = range(df.shape[0])

    return df


def new_column_creation(df, columns, value):

    for column in columns:
        df.loc[:, column] = value

    return df


def language_detection(df, column_to_detect, new_column):

    rows = []
    for key, row in df.iterrows():
        try:
            rows.append(detect(row[column_to_detect]))
        except:
            rows.append('Undefined')
            continue

    df[new_column] = rows
    return df


# Converts a column of a data frame with only strings to a list with all the unique strings
def string_to_list(df, column):

    lower_case_strings = lowercase_column_conversion(df, columns=column)[column[0]].dropna(axis=0).values
    strings = ' '.join(lower_case_strings).split()

    strings = unidecode_function(strings)

    return list(np.unique(strings))


def unidecode_function(strings):
    decoded_strings = []

    if type(strings) == list:
        for single_string in strings:
            new_string = unidecode.unidecode(single_string)
            decoded_strings.append(new_string)
        return decoded_strings
    else:
        decoded_string = unidecode.unidecode(strings)
        decoded_string_step_2 = string_punctuation_removal(decoded_string)
        return decoded_string_step_2


def df_join_function(df_a, df_b, **kwargs):

    df_a = df_a.join(df_b, **kwargs)

    return df_a


def null_handling(df, handling_approach):

    if type(handling_approach) == dict and len(handling_approach.keys()) == 1:
        column_to_fix = list(handling_approach.keys())[0]
        value_to_replace_by = handling_approach[column_to_fix]
        df.loc[df[column_to_fix].isnull(), column_to_fix] = value_to_replace_by

    return df


def literal_removal(df, column):
    # Removes newlines, returns, tabs and unicode character;

    df.loc[~df[column].isnull(), column] = df[~df[column].isnull()][column].map(lambda s: s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\\u', ' '))

    return df


def value_replacement(df, replace_approach):

    # ToDo: there might not be a need for the type checkup
    if type(replace_approach) == dict and len(replace_approach.keys()) == 2:
        col1, col2 = list(replace_approach.keys())[0], list(replace_approach.keys())[1]
        values1, values2 = replace_approach[col1], replace_approach[col2]

        for value in values1:
            df.loc[df[col1] == value, col2] = values2[values1.index(value)]

    elif type(replace_approach) == dict and len(replace_approach.keys()) == 1:
        col1 = list(replace_approach.keys())[0]
        regex = replace_approach[col1]

        df.loc[~df[col1].isnull(), col1] = df[~df[col1].isnull()][col1].replace(regex, np.nan, regex=True)

    return df


def summary_description_null_checkup(df):
    # Cleans requests which have the Summary and Description null
    df = df[(~df['Summary'].isnull()) & (~df['Description'].isnull())]

    return df


def similar_words_handling(df, keywords_df, similar_word_dict):
    _, df_top_words_pt = top_words_processing(df[df['Language'] == 'pt'], description_col='Description')

    # keywords_initial = [x for x in keywords_df['Keywords_PT'] if not x.startswith('User') and not x.startswith('Forced')]
    # keywords_split = [x.split(';') for x in keywords_initial]
    # keywords_v2 = [item for sublist in keywords_split for item in sublist]
    # keywords_split_v2 = [x.split(' ') for x in keywords_v2]
    # keywords_v3 = [item for sublist in keywords_split_v2 for item in sublist if len(item) > 2]
    # keywords = np.unique(keywords_v3)
    #
    # similar_word_dict = {}
    # levenshtein_similar_word_dict = {}
    # used_words = []
    #
    # for keyword in keywords:
    #     for word in df_top_words_pt:
    #         result = similar(keyword, word)
    #
    #         if len(word) > 2:
    #             result_3 = levenshtein_dist(keyword, word)
    #             if result_3 == 1:
    #
    #                 if keyword in levenshtein_similar_word_dict.keys():
    #                     if word not in levenshtein_similar_word_dict[keyword]:
    #                         levenshtein_similar_word_dict[keyword].append(word)
    #                 else:
    #                     levenshtein_similar_word_dict[keyword] = [word]
    #
    #         if len(keyword) >= 4 and len(keyword) >= len(word):
    #             if len(keyword) > 5:
    #                 if 0.85 <= result < 1 and word not in used_words:
    #                     print('Original Word: {}, Similar Word: {}, Similarity Ratio: {:.2f}'.format(keyword, word, result))
    #                     used_words.append(word)
    #
    #                     if keyword in similar_word_dict.keys():
    #                         if word not in similar_word_dict[keyword]:
    #                             similar_word_dict[keyword].append(word)
    #                     else:
    #                         similar_word_dict[keyword] = [word]
    #
    #             elif len(keyword) <= 5:
    #                 if 0.85 <= result < 1 and word not in used_words:
    #                     print('Original Word: {}, Similar Word: {}, Similarity Ratio: {:.2f}'.format(keyword, word, result))
    #                     used_words.append(word)
    #
    #                     if keyword in similar_word_dict.keys():
    #                         if word not in similar_word_dict[keyword]:
    #                             similar_word_dict[keyword].append(word)
    #                     else:
    #                         similar_word_dict[keyword] = [word]
    #
    # print(similar_word_dict)
    # print(levenshtein_similar_word_dict)

    for keyword, similar_words in zip(similar_word_dict.keys(), similar_word_dict.values()):
        pattern = '\\b' + '\\b|\\b'.join(sorted(re.escape(k) for k in similar_words)) + '\\b'
        df.loc[df['Language'] == 'pt', 'Description'] = df.loc[df['Language'] == 'pt', 'Description'].str.replace(pattern, keyword, regex=True)

    return df


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def levenshtein_dist(a, b):
    return distance(a, b)


def text_preprocess(df, unique_clients_decoded, options_file):
    df['StemmedDescription'] = str()
    stemmer_pt = SnowballStemmer('porter')
    # stemmer_es = SnowballStemmer('spanish')

    # Punctuation Removal
    for key, row in df.iterrows():
        description, stemmed_words = row['Description'], []

        description = string_digit_removal(description)
        description = string_punctuation_removal(description)
        description = unidecode_function(description)

        try:
            tokenized = nltk.tokenize.word_tokenize(description)
            for word in tokenized:
                if word in ['\'\'', '``', '“', '”', '', '\'', ',']:
                    continue
                else:
                    stemmed_word = stemmer_pt.stem(word)
                    if len(stemmed_word) >= 2:
                        stemmed_words.append(stemmed_word)
                    else:
                        continue

        except TypeError:
            pass
        df.at[key, 'StemmedDescription'] = ' '.join([x for x in stemmed_words])
    return df


def stop_words_removal(x, stop_words_list):
    new_string = ' '.join([x for x in nltk.tokenize.word_tokenize(x) if x not in stop_words_list])

    return new_string


def abbreviations_correction(string_to_correct, abbreviations_dict):
    tokenized_string_to_correct = nltk.tokenize.word_tokenize(string_to_correct)

    string_corrected = ' '.join([abbreviations_dict[x] if x in abbreviations_dict.keys() else x for x in tokenized_string_to_correct])

    return string_corrected


def string_punctuation_removal(string_to_process):
    punctuation_remover = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    processed_string = str(string_to_process).translate(punctuation_remover)

    return processed_string.strip()


def string_digit_removal(string_to_process):

    digit_remover = str.maketrans('', '', string.digits)

    processed_string = str(string_to_process).translate(digit_remover)

    return processed_string


def word_frequency(df, unit_col, description_col, threshold=0):
    word_dict = {}
    word_dict_unit = {}
    word_dict_unit_count = {}

    for unit, row in df.iterrows():
        try:
            description = nltk.tokenize.word_tokenize(row[description_col])
            for word in description:
                if word in word_dict.keys():
                    word_dict[word] += 1
                    if row[unit_col] not in word_dict_unit[word]:
                        word_dict_unit[word].append(row[unit_col])
                else:
                    word_dict[word] = 1
                    word_dict_unit[word] = [row[unit_col]]

        except TypeError:
            pass

    for key in word_dict_unit.keys():
        word_dict_unit_count[key] = len(word_dict_unit[key])

    # The following two lines convert the dictionaries to lists with ascending order of the values
    # sorted_word_dict = sorted(word_dict.items(), key=operator.itemgetter(1))
    # sorted_word_dict_unit = sorted(word_dict_unit_count.items(), key=operator.itemgetter(1))

    filtered_dict = {k: v for k, v in word_dict_unit_count.items() if v > threshold}
    filtered_dict_units = {k: v for k, v in word_dict_unit_count.items() if v > threshold}

    # word_histogram(sorted(filtered_dict.items(), key=operator.itemgetter(1)))
    # cdf(sorted(filtered_dict.items(), key=operator.itemgetter(1)), '#units')

    return filtered_dict, filtered_dict_units


def words_dataframe_creation(df, top_words_dict, unit_col, description_col):
    start = time.time()

    words_list = sorted(top_words_dict.items(), key=operator.itemgetter(1))
    df_total = pd.DataFrame(index=range(df.shape[0]))

    unique_stemmed_descriptions_non_nan = df[~df[description_col].isnull()].apply(lambda row: nltk.word_tokenize(row[description_col]), axis=1)

    unique_units = df.dropna(axis=0, subset=[description_col])[unit_col]
    cleaned_df = df.dropna(axis=0, subset=[description_col])

    pool = Pool(processes=level_0_performance_report.pool_workers_count)
    results = pool.map(keyword_detection, [(key, occurrence, unique_stemmed_descriptions_non_nan) for (key, occurrence) in words_list])
    pool.close()
    df_total = df_total.join([result for result in results])

    df_total.index = unique_units
    print('Elapsed time is: {:.3f}'.format(time.time() - start))
    return df_total, cleaned_df


def keyword_detection(args):
    key, occurrence, unique_stemmed_descriptions_non_nan = args
    x = pd.DataFrame()

    result = map(lambda y: int(key in y), unique_stemmed_descriptions_non_nan)
    x.loc[:, key] = list(result)

    return x


def object_column_removal(df):

    g = df.columns.to_series().groupby(df.dtypes).groups
    dtype_dict = {k.name: v for k, v in g.items()}

    object_columns = list(dtype_dict['object'].values)
    non_object_columns = list(dtype_dict['int64'].values) + list(dtype_dict['float64'].values)

    df_inter = df[non_object_columns].dropna(axis=0)

    # df_inter.dropna(axis=0, inplace=True)
    print('Categorical Columns: {}'.format(object_columns))
    return df_inter, object_columns, non_object_columns


def close_and_resolve_date_replacements(x):

    if len(x) > 1 and len(x) > sum(x['Assignee_Date'].isnull()) >= 1:
        x.dropna(subset=['Assignee_Date'], axis=0, inplace=True)

    if len(x) > 1 and len(x) > sum(x['Close_Date'].isnull()) >= 1:
        x.dropna(subset=['Close_Date'], axis=0, inplace=True)

    if len(x) > 1 and len(x) > sum(x['Resolve_Date'].isnull()) >= 1:
        x.dropna(subset=['Resolve_Date'], axis=0, inplace=True)

    return x


def min_max_scaling(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    df_scaled = pd.DataFrame(df_scaled, columns=list(df))

    return df_scaled, scaler


def min_max_scaling_reverse(df, scaler):
    reversed_df = scaler.inverse_transform(df)

    return reversed_df


def data_type_conversion(df, type_string):

    df = df.astype(type_string)
    # for column in columns:
    #     try:
    #         df[column] = pd.to_numeric(df[column])
    #     except:
    #         raise Exception

    return df


def threshold_grouping(x, column, value, threshold=0):

    if x.shape[0] < threshold:
        x[column] = value

    return x


def top_words_processing(df_facts, description_col):
    time_tag_date, _ = level_1_e_deployment.time_tags(format_date="%Y_%m_%d")

    try:
        df_cleaned = pd.read_csv(base_path + '/output/df_cleaned_' + str(time_tag_date) + '_' + str(description_col) + '.csv', index_col=0)
        df_top_words = pd.read_csv(base_path + '/output/df_top_words_' + str(time_tag_date) + '_' + str(description_col) + '.csv', index_col=0)
    except FileNotFoundError:
        top_words_frequency, top_words_ticket_frequency = word_frequency(df_facts, unit_col='Request_Num', description_col=description_col)
        df_top_words, df_cleaned = words_dataframe_creation(df_facts, top_words_ticket_frequency, unit_col='Request_Num', description_col=description_col)

        # df_top_words.to_csv(base_path + '/output/df_top_words_' + str(time_tag_date) + '_' + str(description_col) + '.csv')
        # df_cleaned.to_csv(base_path + '/output/df_cleaned_' + str(time_tag_date) + '_' + str(description_col) + '.csv')

    return df_cleaned, df_top_words


def apv_dataset_treatment(df_sales, pse_code, urgent_purchases_flags, project_id):
    current_date, _ = level_1_e_deployment.time_tags(format_date='%Y%m%d')
    sales_cols_to_keep = ['Movement_Date', 'WIP_Number', 'SLR_Document', 'WIP_Date_Created', 'SLR_Document_Date', 'Part_Ref', 'PVP_1', 'Cost_Sale_1', 'Qty_Sold_sum_wip', 'Qty_Sold_sum_slr', 'Qty_Sold_sum_mov', 'Product_Group', 'Part_Desc']

    sales_file_name = base_path + '/dbs/df_sales_cleaned_' + str(pse_code) + '_' + str(current_date)

    try:
        df_sales = pd.read_csv(sales_file_name + '.csv', index_col=0, parse_dates=['Movement_Date', 'WIP_Date_Created', 'SLR_Document_Date'], usecols=sales_cols_to_keep).sort_values(by='Movement_Date')
        # print('{} file found.'.format(sales_file_name))
    except FileNotFoundError:
        print('{} file not found, processing...'.format(sales_file_name))

        df_sales = df_sales[df_sales['Qty_Sold'] != 0]
        df_sales = data_processing_negative_values(df_sales, sales_flag=1)
        df_sales.to_csv(base_path + '/dbs/df_sales_processed_' + str(pse_code) + '_' + str(current_date) + '.csv')

        df_sales['PVP_1'] = df_sales['PVP'] / df_sales['Qty_Sold']
        df_sales['Cost_Sale_1'] = df_sales['Cost_Sale'] / df_sales['Qty_Sold']

        df_sales.drop(['PVP', 'Sale_Value', 'Gross_Margin', 'Cost_Sale'], axis=1, inplace=True)

        df_sales_grouped_slr = df_sales.groupby(['SLR_Document_Date', 'Part_Ref'])  # Old Approach, using SLR_Document_Date
        df_sales_grouped_wip = df_sales.groupby(['WIP_Date_Created', 'Part_Ref'])  # Old approach, where WIP_Date_Created is used instead of the SLR_Document_Date
        df_sales_grouped_mov = df_sales.groupby(['Movement_Date', 'Part_Ref'])  # New Approach, using Movement_Date

        df_sales['Qty_Sold_sum_wip'] = df_sales_grouped_wip['Qty_Sold'].transform('sum')
        df_sales['Qty_Sold_sum_slr'] = df_sales_grouped_slr['Qty_Sold'].transform('sum')
        df_sales['Qty_Sold_sum_mov'] = df_sales_grouped_mov['Qty_Sold'].transform('sum')

        df_sales.drop('Qty_Sold', axis=1, inplace=True)

        df_sales = remove_columns(df_sales, [x for x in list(df_sales) if x not in sales_cols_to_keep], project_id)

        df_sales.sort_index(inplace=True)

        df_sales.to_csv(sales_file_name + '.csv')

    return df_sales


def data_processing_negative_values(df, sales_flag=0, purchases_flag=0):
    start = time.time()

    # print('number of wips', len(df.groupby('WIP_Number')))
    pool = Pool(processes=int(level_0_performance_report.pool_workers_count))
    results = pool.map(matching_negative_row_removal_2, [(y[0], y[1], sales_flag, purchases_flag) for y in df.groupby('WIP_Number')])
    pool.close()
    gt_treated = pd.concat([result for result in results if result is not None])

    print('Sales Negative Values Processing - Elapsed Time: {:.2f}'.format(time.time() - start))
    return gt_treated


def matching_negative_row_removal_2(args):
    key, group, sales_flag, purchases_flag = args
    negative_rows = pd.DataFrame()

    matching_positive_rows = pd.DataFrame()

    if sales_flag:
        negative_rows = group[group['Qty_Sold'] < 0]

        if negative_rows.shape[0]:
            for key, row in negative_rows.iterrows():
                matching_positive_row = group[(group['Movement_Date'] == row['Movement_Date']) & (group['Qty_Sold'] == row['Qty_Sold'] * -1) & (group['Sale_Value'] == row['Sale_Value'] * -1) & (group['Cost_Sale'] == row['Cost_Sale'] * -1) & (group['Gross_Margin'] == row['Gross_Margin'] * -1)]

                # Control Prints
                # if matching_positive_row.shape[0]:
                #     if group['WIP_Number'].unique() == 23468:
                #         if row['Part_Ref'] == 'BM83.21.0.406.573':
                #             print('negative row: \n {}'.format(row))
                #         if matching_positive_row[matching_positive_row['Part_Ref'] == 'BM83.21.0.406.573'].shape[0]:
                #             print('matching_positive_row: \n {}'.format(matching_positive_row[matching_positive_row['Part_Ref'] == 'BM83.21.0.406.573']))

                if matching_positive_row.shape[0] > 1:
                    matched_positive_row_idxs = list(matching_positive_row.sort_values(by='Movement_Date').index)
                    # sel_row = matching_positive_row[matching_positive_row.index == matching_positive_row['Movement_Date'].idxmax()]

                    added, j = 0, 0
                    while not added:
                        try:
                            idx = matched_positive_row_idxs[j]
                            if idx not in matching_positive_rows.index:
                                matching_positive_rows = pd.concat([matching_positive_rows, matching_positive_row[matching_positive_row.index == idx]])
                                added = 1
                        except IndexError:
                            # Reached the end of the matched rows and all have already been added
                            added = 1
                        j += 1

                    # Control Prints
                    # if group['WIP_Number'].unique() == 23468:
                    #     if row['Part_Ref'] == 'BM83.21.0.406.573':
                    #         if sel_row.shape[0]:
                    #             print('Row selected: \n', sel_row)
                    #
                    # if group['WIP_Number'].unique() == 23468:
                    #     if row['Part_Ref'] == 'BM83.21.0.406.573':
                    #         print('matching_positive_rows that will be removed \n{}'.format(matching_positive_rows))
                else:
                    matching_positive_rows = pd.concat([matching_positive_rows, matching_positive_row.head(1)])

    elif purchases_flag:
        negative_rows = group[group['Quantity'] < 0]
        if negative_rows.shape[0]:
            for key, row in negative_rows.iterrows():
                matching_positive_row = group[(group['Quantity'] == abs(row['Quantity'])) & (group['Cost_Value'] == abs(row['Cost_Value'])) & (group['Part_Ref'] == row['Part_Ref']) & (group['WIP_Number'] == row['WIP_Number'])]

                if matching_positive_row.shape[0] > 1:
                    matching_positive_rows = pd.concat([matching_positive_rows, matching_positive_row[matching_positive_row.index == matching_positive_row['Movement_Date'].idxmax()]])
                else:
                    matching_positive_rows = pd.concat([matching_positive_rows, matching_positive_row.head(1)])

    if negative_rows.shape[0]:
        group.drop(negative_rows.index, axis=0, inplace=True)
        group.drop(matching_positive_rows.index, axis=0, inplace=True)
        # Note: Sometimes, identical negative rows with only Part_Ref different will match with the same row with positive values. This is okay as when I remove the matched rows from the
        # original group I remove by index, so double matched rows make no problem whatsoever

    return group


def purchases_na_fill(df_grouped):
    start = time.time()

    pool = Pool(processes=int(level_0_performance_report.pool_workers_count))
    results = pool.map(na_group_fill, [(z[0], z[1]) for z in df_grouped])
    pool.close()
    df_filled = pd.concat([result for result in results if result is not None])

    print('Purchases NaN Fill - Elapsed Time: {:.2f}'.format(time.time() - start))
    return df_filled


def na_group_fill(args):
    _, group = args

    group[['Qty_Purchased_urgent_sum', 'Qty_Purchased_non_urgent_sum']] = group[['Qty_Purchased_urgent_sum', 'Qty_Purchased_non_urgent_sum']].fillna(method='ffill').fillna(method='bfill').fillna(0)

    return group


def na_fill_hyundai(df_grouped):
    # project_id = 2406
    start = time.time()

    pool = Pool(processes=4)
    results = pool.map(na_group_fill_hyundai, [(z[0], z[1]) for z in df_grouped])
    pool.close()
    df_filled = pd.concat([result for result in results if result is not None])

    print('Elapsed Time: {:.2f}'.format(time.time() - start))
    return df_filled


def na_group_fill_hyundai(args):
    # project_id = 2406
    key, group = args
    slr_document_date_chs_min_idx, slr_document_date_chs_min = 0, 0

    cols_to_fill = ['Quantity_CHS']
    measure_cols = ['Measure_2', 'Measure_3', 'Measure_4', 'Measure_5', 'Measure_6', 'Measure_7', 'Measure_9', 'Measure_10', 'Measure_11', 'Measure_12']
    support_measure_cols = ['Measure_13', 'Measure_14', 'Measure_15']

    if group[measure_cols].sum(axis=0).sum(axis=0) == 0:
        # print('inside case 0 - ', key, '\n', group)
        return None

    if group.shape[0] == 1 and group['Quantity_CHS'].values == 0:
        # print('inside case 1', key, '\n', group)
        return None

    if sum(group['SLR_Document_Date_CHS'].isnull()) == group['SLR_Document_Date_CHS'].shape[0]:  # Sem data de fatura de chassis
        # print('inside case 2', key, '\n', group)
        # print('No SLR_Document_Date_CHS: \n', group)
        return None

    if group['SLR_Document_Date_CHS'].nunique() > 1:
        # print('inside case 3', key, '\n', group)
        slr_document_date_chs_min = group['SLR_Document_Date_CHS'].min()
        # print('min slr_document_date_chs found:', slr_document_date_chs_min)
        try:
            slr_document_date_chs_min_idx = group[(group['SLR_Document_Date_CHS'] == slr_document_date_chs_min) & (group['Quantity_CHS'] == 1)].index.values[0]
            # print('case 3 - min date found which follows the requirements', key, '\n', group)
            # print('the line found was: \n', group[(group['SLR_Document_Date_CHS'] == slr_document_date_chs_min) & (group['Quantity_CHS'] == 1)][['Measure_2', 'Measure_3', 'Measure_4', 'Measure_5', 'Measure_6', 'Measure_7', 'Measure_8', 'Measure_9', 'Measure_10', 'Measure_11', 'Measure_12']])
        except IndexError:
            slr_document_date_chs_min_idx = group['SLR_Document_Date_CHS'].idxmin()
            # print('case 3 - min date not found')
    elif group['SLR_Document_Date_CHS'].nunique() == 1:
        # print('inside case 4', key, '\n', group)
        slr_document_date_chs_min = group['SLR_Document_Date_CHS'].min()
        slr_document_date_chs_min_idx = group['SLR_Document_Date_CHS'].idxmin()

    check_for_registration_number = group['Registration_Number'].nunique()
    check_for_slr_document_chs = group['SLR_Document_CHS'].nunique()
    check_for_slr_document_rgn = group['SLR_Document_RGN'].nunique()

    group.loc[:, cols_to_fill] = group[group.index == slr_document_date_chs_min_idx][cols_to_fill].head(1).values[0][0]

    exception_check = group[(group['SLR_Document_Date_CHS'] == slr_document_date_chs_min) | (group['SLR_Document_Date_CHS'].isnull())]['Measure_2'].sum(axis=0)
    second_exception_check = group[(group['SLR_Document_Date_CHS'] == slr_document_date_chs_min) | (group['Quantity_CHS'] == 1)]['Measure_2'].sum(axis=0)

    for col_o in measure_cols:
        if exception_check:
            group[col_o] = group[(group['SLR_Document_Date_CHS'] == slr_document_date_chs_min) | (group['SLR_Document_Date_CHS'].isnull())][col_o].sum(axis=0)
            # print('passed exception')
        else:
            if not second_exception_check:
                group[col_o] = group[col_o].sum(axis=0)
            else:
                group[col_o] = group[(group['SLR_Document_Date_CHS'] == slr_document_date_chs_min) & (group['Quantity_CHS'] == 1)][col_o].sum(axis=0)
            # print('caught by exception')

    for col_s in support_measure_cols:
        group[col_s] = group[(group['SLR_Document_Date_CHS'] == slr_document_date_chs_min)][col_s].sum(axis=0)

    group['SLR_Document_Date_CHS'] = group['SLR_Document_Date_CHS'].min()
    group['SLR_Document_Date_RGN'] = group['SLR_Document_Date_RGN'].min()

    [group[x].fillna(method='bfill', inplace=True) for x in cols_to_fill]

    if not check_for_registration_number and check_for_slr_document_chs:
        group['No_Registration_Number_Flag'] = 1
    elif check_for_registration_number and check_for_slr_document_chs and not check_for_slr_document_rgn:
        group['Registration_Number_No_SLR_Document_RGN_Flag'] = 1
    elif check_for_slr_document_chs and check_for_slr_document_rgn:
        group['SLR_Document_RGN_Flag'] = 1
    else:
        group['Undefined_VHE_Status'] = 1

    return group.head(1)


def measures_calculation_hyundai(df):
    # project_id = 2406
    measure_2, measure_3, measure_4 = df['Measure_2'], df['Measure_3'], df['Measure_4']
    measure_5, measure_6, measure_7 = df['Measure_5'], df['Measure_6'], df['Measure_7']
    measure_9, measure_10, measure_11, measure_12 = df['Measure_9'], df['Measure_10'], df['Measure_11'], df['Measure_12']
    measure_13, measure_14, measure_15 = df['Measure_13'], df['Measure_14'], df['Measure_15']
    measure_17, measure_18, measure_19, measure_20 = df['Measure_17'], df['Measure_18'], df['Measure_19'], df['Measure_20']
    measure_40, measure_41, measure_42, measure_43 = df['Measure_40'], df['Measure_41'], df['Measure_42'], df['Measure_43']
    measure_21, measure_22, measure_23, measure_24, measure_25 = df['Measure_21'], df['Measure_22'], df['Measure_23'], df['Measure_24'], df['Measure_25']
    measure_26, measure_27, measure_28, measure_29, measure_30 = df['Measure_26'], df['Measure_27'], df['Measure_28'], df['Measure_29'], df['Measure_30']

    df['Total_Sales'] = measure_2 + measure_3 + measure_4
    df['Total_Discount'] = measure_5 + measure_6 + measure_7
    df['Total_Discount_%'] = df['Total_Discount'] / df['Total_Sales']
    df['Total_Net_Sales'] = (measure_2 - measure_5) + (measure_3 - measure_6) + (measure_4 - measure_7)
    df['Total_Cost'] = measure_9 + measure_10 + measure_11 + measure_12
    df['Fixed_Margin_I'] = (((measure_2 - measure_5) + (measure_3 - measure_6)) + (measure_4 - measure_7)) + ((measure_9 + measure_10) + (measure_11 + measure_12)) + ((measure_13 + measure_14) + measure_15)
    df['Fixed_Margin_I_%'] = df['Fixed_Margin_I'] / df['Total_Net_Sales']
    df['Quality_Margin'] = measure_17 + measure_18 + measure_19 + measure_20 + measure_40 + measure_41 + measure_42
    df['Total_Network_Support'] = df['Quality_Margin'] + measure_21 + measure_22 + measure_23 + measure_24 + measure_25 + measure_26 + measure_27 + measure_28 + measure_29 + measure_30 + measure_40 + measure_41 + measure_42 + measure_43
    df['Fixed_Margin_II'] = df['Fixed_Margin_I'] + df['Total_Network_Support']
    df['Fixed_Margin_II_%'] = df['Fixed_Margin_II'] / df['Total_Net_Sales']
    df['HME_Support'] = measure_13 + measure_14 + measure_15

    return df


def parameter_processing_hyundai(df_sales, options_file, description_cols):
    # Project_ID = 2406

    # Modelo
    df_sales.loc[:, 'PT_PDB_Model_Desc'] = df_sales['PT_PDB_Model_Desc'].str.split().str[0]

    return df_sales


def pandas_object_columns_categorical_conversion_auto(df):

    for c in df.columns:
        col_type = df[c].dtype
        if col_type == 'object':
            df[c] = df[c].astype('category')

    return df


def pandas_object_columns_categorical_conversion(df, columns, project_id):

    for column in columns:
        try:
            df[column] = df[column].astype('category')
        except KeyError:
            level_0_performance_report.log_record('Aviso de conversão de coluna - A coluna {} não foi encontrada.'.format(column), project_id, flag=1)
            continue

    return df


def numerical_columns_detection(df):
    numerical_cols = [x for x in list(df) if df[x].dtypes == 'int64' or df[x].dtypes == 'float64']

    return numerical_cols


def skewness_reduction(df, target):
    # This function tries to reduce the skewness of each column, using log transformations (can only apply to non-neg valued columns)

    numerical_cols = numerical_columns_detection(df)

    try:
        numerical_cols.remove(target)
    except KeyError:
        pass

    for feature in numerical_cols:
        if df[feature].sum() == df[feature].abs().sum():  # All values are positive
            original_skew = df[feature].skew()
            normalized_col = np.log1p(df[feature])
            normalized_skew = stats.skew(normalized_col)

            if abs(normalized_skew) < abs(original_skew):
                df[feature] = normalized_col

    return df


def robust_scaler_function(df, target):
    # This function applies RobustScaler which is a normalizer robust to outliers

    robust_normalizer = RobustScaler()
    numerical_cols = numerical_columns_detection(df)

    try:
        numerical_cols.remove(target)
    except KeyError:
        pass

    df[numerical_cols] = robust_normalizer.fit_transform(df[numerical_cols])

    return df


def boolean_replacement(df, boolean_cols):

    if list(df[boolean_cols[0]].unique()) == [1, 0] or list(df[boolean_cols[0]].unique()) == [0, 1]:
        for column in boolean_cols:
            df[column] = pd.Series(np.where(df[column].values == 1, 'Sim', 'Não'), df.index)
    else:
        for column in boolean_cols:
            df[column] = pd.Series(np.where(df[column].values == "Sim", 1, 0), df.index)

    return df


def master_file_processing(master_files_to_convert):
    # PRJ-2610
    # This function's goal is to take raw txt master files and identify each column and convert the file to csv;
    # For each file, this function needs: column delimiter positions, column's names and 2 flags on whether to ignore the first/last row;

    for master_file_loc in master_files_to_convert.keys():
        master_file_info = master_files_to_convert[master_file_loc]
        master_file_spiga_flag = master_file_info[0]
        master_file_positions = master_file_info[1]
        master_file_col_names = master_file_info[2]
        header_flag = master_file_info[3]
        tail_flag = master_file_info[4]

        if not master_file_spiga_flag:
            fields_dict = {key: [] for key in master_file_col_names}

            # f = open(master_file_loc + '.txt', 'r')
            f = open(master_file_loc + '.txt', 'r', encoding='latin-1')  # ToDo need to handle this problem, as mercedes' master file needs this encoding

            if header_flag:
                lines = f.readlines()[1:]
            elif tail_flag:
                lines = f.readlines()[:-1]
            elif header_flag and tail_flag:
                lines = f.readlines()[1:-1]
            else:
                lines = f.readlines()

            result = pd.DataFrame(columns=master_file_col_names)
            for x in lines:

                for initial_field_pos, end_field_pos, field_name in zip(master_file_positions[:-1], master_file_positions[1:], master_file_col_names):
                    fields_dict[field_name].append(x[initial_field_pos:end_field_pos].strip())

            for field_name in master_file_col_names:
                result[field_name] = fields_dict[field_name]

        elif master_file_spiga_flag:
            result = pd.read_csv(master_file_loc, delimiter=';', header=None, delim_whitespace=False, usecols=[0, 1], names=master_file_col_names)

        result.to_csv(master_file_loc + '.csv')

    return


def regex_string_replacement(string_to_process, regex_rule, replacement=''):
    regex = re.compile(regex_rule)

    processed_string = regex.sub(replacement, str(string_to_process))

    return processed_string


def string_volkswagen_preparation(string_to_process):
    # 807434 7L6 DB41 -> 7L6 807434 DB41  (MF -> Stock)
    # 103801 06L -> 06L 103801 (MF -> Stock)
    # 141153 02T F -> 02T 141153 F (MF -> Stock)

    tag_ref = string_to_process[0:3]
    main_ref = string_to_process[3:9]
    leftover_ref = string_to_process[9:]

    new_ref_reordered = main_ref + tag_ref + leftover_ref

    return new_ref_reordered


def brand_code_removal(string_to_process, dms_codes):
    # PRJ 2610
    dms_codes.sort(key=len, reverse=True)  # I need to sort by length, from larger to smaller to avoid substrings of other dms codes. Ex: FI and FIA. FIA has to be searched first, otherwise FI will remove only FI and leave an erroneous A;

    regex_code = r'^' + '|^'.join(dms_codes)

    processed_string = regex_string_replacement(string_to_process, regex_code)

    return processed_string


def duplicate_test(df, col):
    duplicated_rows = df[df.duplicated(subset=col, keep=False)]

    if duplicated_rows.shape[0]:
        print('Possible duplicates found! Here\'s a sample: {}: \n'.format(duplicated_rows.shape[0]), duplicated_rows.sort_values(by=col).head(20))

    return


def lemmatisation(x, lemmatizer):
    return ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(x)])


def stemming(x, stemmer):
    return ' '.join([stemmer.stem(w) for w in nltk.word_tokenize(x)])


def update_new_gamas(df, df_pdb):
    # PRJ-2406
    gama_viva_mask_matchup = df_pdb['PT_PDB_Commercial_Version_Desc_New'].notnull()

    df_pdb_sel = df_pdb.loc[gama_viva_mask_matchup]

    for key, row in df_pdb_sel.iterrows():
        df.loc[df['VehicleData_Code'] == row['VehicleData_Code'], 'PT_PDB_Version_Desc'] = row['PT_PDB_Version_Desc_New']
        df.loc[df['VehicleData_Code'] == row['VehicleData_Code'], 'PT_PDB_Engine_Desc'] = row['PT_PDB_Engine_Desc_New']

    return df


def feat_eng(df_in):
    # PRJ - 2527

    df = df_in.copy()

    df['Power_Weight_Ratio'] = df['Power_kW'] / df['Weight_Empty']
    df['Contract_km'] = df['Contract_km'] / 1000

    # contract start month
    df['contract_start_month'] = df['contract_start_date'].str[5:7]

    # create additional column, representing accident vs no accident
    df['target_accident'] = 0
    df.loc[~df.target.isna(), 'target_accident'] = 1

    # change target column name, representing the cost
    df['target_cost'] = df.target
    df = df.drop(['target'], axis=1)
    df['target_cost'] = df['target_cost'].fillna(0)

    values = {
        'Mean_repair_value_cust_full': 0,
        'Sum_contrat_km_full': 0,
        'Sum_repair_value_full': 0,
        'Mean_accident_rel_date_cust_full': 0,
        'Mean_contract_duration_cust_full': 0,
        'Mean_monthly_repair_cost_cust_full': 0,
        'Mean_repair_value_cust_full.1': 0,
        'Mean_repair_value_cust_1year': 0,
        'Mean_accident_rel_date_cust_1year': 0,
        'Mean_contract_duration_cust_1year': 0,
        'Mean_monthly_repair_cost_cust_1year': 0,
        'Mean_repair_value_cust_1year.1': 0,
        'Mean_repair_value_cust_5year': 0,
        'Mean_accident_rel_date_cust_5year': 0,
        'Mean_contract_duration_cust_5year': 0,
        'Mean_monthly_repair_cost_cust_5year': 0,
        'Mean_repair_value_cust_5year.1': 0,
    }
    df = df.fillna(value=values)

    df.loc[df['LL'].str.startswith('€50.000.000'), 'LL'] = '€50.000.000'
    df['AR'] = df['AR'].str.extract(r'^(.+%)')
    df.loc[df['FI'].str.startswith('Até €1.000/Ano'), 'FI'] = 'Até €1.000/Ano'

    values = {
        'LL': '0',
        'AR': '0',
        'FI': '0'
    }
    df = df.fillna(value=values)

    columns_to_drop = [
        'contract_customer',
        'contract_contract',
        'Vehicle_No',
        'Accident_No',
        'contract_start_date',
        'contract_end_date'
    ]

    df = df.drop(columns_to_drop, axis=1)

    return df


def apply_ohenc(col, df_apply_in, enc):
    # PRJ - 2527

    import pandas as pd

    df_apply = df_apply_in.copy()

    #process test df
    df_apply = pd.concat([
        df_apply,
        pd.DataFrame(
            enc.transform(df_apply[[col]]).toarray(),
            columns=col + '_' + enc.get_feature_names())
    ], axis=1).drop([col], axis=1)

    return df_apply
