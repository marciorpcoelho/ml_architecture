import nltk
import sys
import logging
import numpy as np
from scipy import stats

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
        df.drop([column], axis=1, inplace=True)

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
    # df.loc[:, 'buy_day'] = df['Data Compra'].dt.day
    # df.loc[:, 'buy_month'] = df['Data Compra'].dt.month
    # df.loc[:, 'buy_year'] = df['Data Compra'].dt.year
    # df.loc[:, 'sell_day'] = df['Data Venda'].dt.day
    # df.loc[:, 'sell_month'] = df['Data Venda'].dt.month
    # df.loc[:, 'sell_year'] = df['Data Venda'].dt.year

    for key in dictionary.keys():
        df.loc[:, key + 'day'] = df[dictionary[key]].dt.day
        df.loc[:, key + 'month'] = df[dictionary[key]].dt.month
        df.loc[:, key + 'year'] = df[dictionary[key]].dt.year

    return df


def options_scraping(df):
    colors_pt = ['preto', 'branco', 'azul', 'verde', 'tartufo', 'vermelho', 'antracite/vermelho', 'dacota', 'anthtacite/preto', 'preto/laranja/preto/lara', 'prata/cinza', 'cinza', 'preto/silver', 'cinzento', 'prateado', 'prata', 'amarelo', 'laranja', 'castanho', 'dourado', 'antracit', 'antracite/preto', 'antracite/cinza/preto', 'branco/outras', 'antracito', 'dakota', 'antracite', 'antracite/vermelho/preto', 'oyster/preto', 'prata/preto/preto', 'âmbar/preto/pr', 'bege', 'terra', 'preto/laranja', 'cognac/preto', 'bronze', 'beige', 'beje', 'veneto/preto', 'zagora/preto', 'mokka/preto', 'taupe/preto', 'sonoma/preto', 'preto/preto', 'preto/laranja/preto']
    colors_en = ['black', 'havanna', 'merino', 'vernasca', 'walnut', 'chocolate', 'nevada', 'moonstone', 'anthracite/silver', 'white', 'coffee', 'blue', 'red', 'grey', 'silver', 'orange', 'green', 'bluestone', 'aqua', 'burgundy', 'anthrazit', 'truffle', 'brown', 'oyster', 'tobacco', 'jatoba', 'storm', 'champagne', 'cedar', 'silverstone', 'chestnut', 'kaschmirsilber', 'oak', 'mokka']

    df_grouped = df.groupby('Nº Stock')
    for key, group in df_grouped:
        ### Navegação/Sensor/Transmissão
        for line_options in group['Opcional']:
            tokenized_options = nltk.word_tokenize(line_options)
            if 'navegação' in tokenized_options:
                df.loc[df['Nº Stock'] == key, 'Navegação'] = 1

            if 'pdc-sensores' in tokenized_options:
                for word in tokenized_options:
                    if 'diant' in word:
                        df.loc[df['Nº Stock'] == key, 'Sensores'] = 1

            if 'transmissão' in tokenized_options or 'caixa' in tokenized_options:
                for word in tokenized_options:
                    if 'auto' in word:
                        df.loc[df['Nº Stock'] == key, 'Caixa Auto'] = 1

        ### Cor Exterior
        line_color = group['Cor'].head(1).values[0]
        tokenized_color = nltk.word_tokenize(line_color)
        color = [x for x in colors_pt if x in tokenized_color]
        if not color:
            color = [x for x in colors_en if x in tokenized_color]
        if not color:
            if tokenized_color == ['pintura', 'bmw', 'individual'] or tokenized_color == ['hp', 'motorsport', ':', 'branco/azul/vermelho', '``', 'racing', "''"] or tokenized_color == ['p0b58']:
                continue
            else:
                # print(tokenized_color)
                sys.exit('Error: Color Not Found')
        if len(color) > 1:  # Fixes cases such as 'white silver'
            color = [color[0]]
        color = color * group.shape[0]
        df.loc[df['Nº Stock'] == key, 'Cor_Exterior'] = color

        ### Cor Interior
        line_interior = group['Interior'].head(1).values[0]
        tokenized_interior = nltk.word_tokenize(line_interior)
        # print(tokenized_interior)

        color_interior = [x for x in colors_pt if x in tokenized_interior]

        if not color_interior:
            color_interior = [x for x in colors_en if x in tokenized_interior]

        if 'truffle' in tokenized_interior:
            color_interior = ['truffle']

        if 'banco' in tokenized_interior and 'standard' in tokenized_interior:
            color_interior = ['preto']

        # if not color_interior:
        #     print(tokenized_interior)
        #     sys.exit('No Interior Color')

        if len(color_interior) > 1:
            # print('Too Many Colors:', tokenized_interior, color_interior)

            if 'nevada' in tokenized_interior:
                color_interior = ['nevada']
            if 'preto' in tokenized_interior and 'antracite' in tokenized_interior:
                color_interior = ['preto/antracite']

            # if 'beje' in tokenized_interior and 'havanna' in tokenized_interior:
            #     color_interior = ['beje']
            # if 'bege' in tokenized_interior and 'veneto' in tokenized_interior:
            #     color_interior = ['bege']
            # if 'bege' in tokenized_interior and 'preto' in tokenized_interior:
            #     color_interior = ['bege']
            # if 'bege' in tokenized_interior and 'sonoma/preto' in tokenized_interior:
            #     color_interior = ['bege']
            # if 'bege' in tokenized_interior and 'zagora/preto' in tokenized_interior:
            #     color_interior = ['bege']

            if 'beje' in tokenized_interior or 'bege' in tokenized_interior:
                if 'havanna' in tokenized_interior or 'veneto' in tokenized_interior or 'preto' in tokenized_interior or 'sonoma/preto' in tokenized_interior or 'zagora/preto' in tokenized_interior:
                    color_interior = ['bege']

            if 'dacota' in tokenized_interior and 'bege' in tokenized_interior:
                color_interior = ['dakota']
            if 'dakota' in tokenized_interior:
                color_interior = ['dakota']

            if 'vernasca' in tokenized_interior and 'anthtacite/preto' in tokenized_interior:
                color_interior = ['vernasca']
            # print('Changed to:', color_interior)

        color_interior = color_interior * group.shape[0]
        try:
            df.loc[df['Nº Stock'] == key, 'Cor_Interior'] = color_interior
        except ValueError:
            # print(key, '\n', color_interior)
            continue

        ### Jantes
        for line_options in group['Opcional']:
            tokenized_jantes = nltk.word_tokenize(line_options)
            for value in range(15, 21):
                if str(value) in tokenized_jantes:
                    jantes_size = [str(value)] * group.shape[0]
                    df.loc[df['Nº Stock'] == key, 'Jantes'] = jantes_size

        ## Modelo
        line_modelo = group['Modelo'].head(1).values[0]
        tokenized_modelo = nltk.word_tokenize(line_modelo)
        if tokenized_modelo[0] == 'Série':
            df.loc[df['Modelo'] == line_modelo, 'Modelo'] = ' '.join(tokenized_modelo[:2])
        else:
            df.loc[df['Modelo'] == line_modelo, 'Modelo'] = ' '.join(tokenized_modelo[:-3])

    df.loc[df['Jantes'] == 0, 'Jantes'] = 'standard'
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
    return df

# df.loc[df['B'].isin(teste), 'B_new'] = 2


def total_price(df):
    df['price_total'] = df['Custo'].groupby(df['Nº Stock']).transform('sum')

    return df


def margin_calculation(df):
    df['margem_percentagem'] = (df['Margem'] / df['price_total']) * 100

    return df


def color_replacement(df):
    color_types = ['Cor_Interior', 'Cor_Exterior']
    colors_to_replace = {'black': 'preto', 'preto/silver': 'preto/prateado', 'tartufo': 'truffle', 'preto/laranja/preto/lara': 'preto/laranja', 'dacota': 'dakota', 'white': 'branco', 'blue': 'azul', 'red': 'vermelho', 'grey': 'cinzento', 'silver': 'prateado', 'orange': 'laranja', 'green': 'verde', 'anthrazit': 'antracite', 'antracit': 'antracite', 'brown': 'castanho', 'antracito': 'antracite', 'âmbar/preto/pr': 'ambar/preto/preto', 'beige': 'bege', 'kaschmirsilber': 'cashmere', 'beje': 'bege'}

    unknown_ext_colors = df[df['Cor_Exterior'] == 0]['Cor'].unique()
    unknown_int_colors = df[df['Cor_Interior'] == 0]['Interior'].unique()
    print('Unknown Exterior Colors:', unknown_ext_colors, ', Removed', df[df['Cor_Exterior'] == 0].shape[0], 'lines in total, corresponding to ', df[df['Cor_Exterior'] == 0]['Nº Stock'].nunique(), 'vehicles')  # 49 lines removed, 3 vehicles
    print('Unknown Interior Colors:', unknown_int_colors, ', Removed', df[df['Cor_Interior'] == 0].shape[0], 'lines in total, corresponding to ', df[df['Cor_Interior'] == 0]['Nº Stock'].nunique(), 'vehicles')  # 2120 lines removed, 464 vehicles

    for color_type in color_types:
        df[color_type] = df[color_type].replace(colors_to_replace)
        df.drop(df[df[color_type] == 0].index, axis=0, inplace=True)

    return df


def score_calculation(df, stockdays_threshold, margin_threshold):
    df['stock_days'] = (df['Data Venda'] - df['Data Compra']).dt.days
    df['stock_days_norm'] = (df['stock_days'] - df['stock_days'].min()) / (df['stock_days'].max() - df['stock_days'].min())
    df['inv_stock_days_norm'] = 1 - df['stock_days_norm']

    df['margem_percentagem_norm'] = (df['margem_percentagem'] - df['margem_percentagem'].min()) / (df['margem_percentagem'].max() - df['margem_percentagem'].min())
    df['score'] = df['inv_stock_days_norm'] * df['margem_percentagem_norm']

    df['stock_days_class'] = 0
    df.loc[df['stock_days'] <= stockdays_threshold, 'stock_days_class'] = 1
    df['margin_class'] = 0
    df.loc[df['margem_percentagem'] >= margin_threshold, 'margin_class'] = 1

    df['new_score'] = 0
    df.loc[(df['stock_days_class'] == 1) & (df['margin_class'] == 1), 'new_score'] = 1

    df.drop(['stock_days_norm', 'inv_stock_days_norm', 'margem_percentagem_norm'], axis=1, inplace=True)

    return df


def new_features_optionals_baviera(df, sel_cols):
    df_grouped = df.sort_values(by=['Data Venda']).groupby(sel_cols)
    df = df_grouped.apply(previous_sales_info_optionals_baviera)
    remove_columns(df, ['sell_day', 'sell_month', 'sell_year'])

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



