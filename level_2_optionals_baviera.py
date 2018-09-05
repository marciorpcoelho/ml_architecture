import time
import pandas as pd
from level_1_a_data_acquisition import read_csv
from level_1_b_data_processing import lowercase_column_convertion, remove_rows, remove_columns, string_replacer, date_cols, options_scraping, color_replacement, score_calculation, duplicate_removal
from level_1_e_deployment import save_csv
pd.set_option('display.expand_frame_repr', False)


def main():
    input_file = 'db/' + 'ENCOMENDA.csv'
    output_file = 'output/' + 'db_full_baviera.csv'

    df = data_acquistion(input_file)
    data_processing(df)
    data_modelling()
    model_evaluation()
    deployment()


def data_acquistion(input_file):
    print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Started Step A...')

    df = read_csv(input_file, delimiter=';', parse_dates=['Data Compra', 'Data Venda'], infer_datetime_format=True, decimal=',')

    print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Ended Step A.')

    return df


def data_processing(df):
    print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Started Step B...')

    df = lowercase_column_convertion(df, ['Opcional', 'Cor', 'Interior'])
    df = remove_rows(df, df.loc[df['Opcional'] == 'preço de venda', :].index)

    dict_strings_to_replace = {('Modelo', ' - não utilizar'): '', ('Interior', '|'): '/', ('Cor', '|'): '', ('Interior', 'ind.'): '', ('Interior', ']'): '/', ('Interior', '.'): ' ', ('Interior', '\'merino\''): 'merino', ('Interior', '\' merino\''): 'merino', ('Interior', '\'vernasca\''): 'vernasca'}
    df = string_replacer(df, dict_strings_to_replace)
    df = remove_columns(df, ['CdInt', 'CdCor'])  # Columns that has missing values which are needed
    df.dropna(axis=0, inplace=True)  # Removes all remaining NA's.

    df.loc[:, 'Navegação'], df.loc[:, 'Sensores'], df.loc[:, 'Cor_Interior'], df.loc[:, 'Caixa Auto'], df.loc[:, 'Cor_Exterior'], df.loc[:, 'Jantes'] = 0, 0, 0, 0, 0, 0  # New Columns

    dict_cols_to_take_date = {'buy_': 'Data Compra', 'sell_': 'Data Venda'}
    df = date_cols(df, dict_cols_to_take_date)

    df = options_scraping(df)  # Need to recheck this
    df = color_replacement(df)  # Need to recheck this
    df = score_calculation(df)  # Need to recheck this
    df = duplicate_removal(df)  # Need to recheck this

    print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Finished Step B.')


def data_modelling():
    print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Started Step C...')

    print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Finished Step C.')


def model_evaluation():
    print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Started Step D...')

    print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Finished Step D.')


def deployment():
    print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Started Step E...')

    print(time.strftime("%H:%M:%S @ %d/%m/%y"), '- Finished Step E.')


if __name__ == '__main__':
    main()
