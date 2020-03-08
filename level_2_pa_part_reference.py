import sys
import time
import logging
import numpy as np
import pandas as pd
from traceback import format_exc
import level_2_pa_part_reference_options as options_file
from modules.level_1_a_data_acquisition import read_csv, sql_retrieve_df_specified_query
from modules.level_1_b_data_processing import lowercase_column_conversion
from modules.level_1_e_deployment import save_csv
local_flag = 1


def main():

    df_afr_current_stock, df_crp_current_stock, df_ibe_current_stock, df_ca_current_stock, dim_product_group, dim_clients = data_acquisition(['dbs/df_afr_current_stock.csv', 'dbs/df_crp_current_stock.csv', 'dbs/df_ibe_current_stock.csv', 'dbs/df_ca_current_stock.csv', 'dbs/dim_product_group.csv', 'dbs/dim_clients.csv'], local=local_flag)
    data_processing(df_afr_current_stock, df_crp_current_stock, df_ibe_current_stock, df_ca_current_stock, dim_product_group, dim_clients, local=0)


def data_acquisition(input_files, local=0):

    if local:
        df_afr_current_stock = read_csv(input_files[0], index_col=0)
        df_crp_current_stock = read_csv(input_files[1], index_col=0)
        df_ibe_current_stock = read_csv(input_files[2], index_col=0)
        df_ca_current_stock = read_csv(input_files[3], index_col=0)
        dim_product_group = read_csv(input_files[4], index_col=0)
        dim_clients = read_csv(input_files[5], index_col=0)
    elif not local:
        df_afr_current_stock = sql_retrieve_df_specified_query(options_file.DSN, options_file.sql_info['database_crp'],  options_file, options_file.afr_current_stock_query)
        df_crp_current_stock = sql_retrieve_df_specified_query(options_file.DSN, options_file.sql_info['database_crp'], options_file, options_file.crp_current_stock_query)
        df_ibe_current_stock = sql_retrieve_df_specified_query(options_file.DSN, options_file.sql_info['database_ibe'], options_file, options_file.ibe_current_stock_query)
        df_ca_current_stock = sql_retrieve_df_specified_query(options_file.DSN, options_file.sql_info['database_ca'], options_file, options_file.ca_current_stock_query)
        dim_product_group = sql_retrieve_df_specified_query(options_file.DSN, options_file.sql_info['database_afr'], options_file, options_file.dim_product_group_query)
        dim_clients = sql_retrieve_df_specified_query(options_file.DSN, options_file.sql_info['database_gsc'], options_file, options_file.dim_clients_query)

        save_csv([df_afr_current_stock, df_crp_current_stock, df_ibe_current_stock, df_ca_current_stock, dim_product_group, dim_clients],
                 ['dbs/df_afr_current_stock', 'dbs/df_crp_current_stock', 'dbs/df_ibe_current_stock', 'dbs/df_ca_current_stock', 'dbs/dim_product_group', 'dbs/dim_clients'])

    return df_afr_current_stock, df_crp_current_stock, df_ibe_current_stock, df_ca_current_stock, dim_product_group, dim_clients


def data_processing(df_afr_current_stock, df_crp_current_stock, df_ibe_current_stock, df_ca_current_stock, dim_product_group, dim_clients, local=0):

    if local:
        df_afr_current_stock_unique = read_csv('dbs/df_afr_current_stock_unique.csv', index_col=0)
        df_crp_current_stock_unique = read_csv('dbs/df_crp_current_stock_unique.csv', index_col=0)
        df_ibe_current_stock_unique = read_csv('dbs/df_ibe_current_stock_unique.csv', index_col=0)
        df_ca_current_stock_unique = read_csv('dbs/df_ca_current_stock_unique.csv', index_col=0)
    elif not local:
        df_afr_current_stock = lowercase_column_conversion(df_afr_current_stock, ['Part_Desc'])  # Lowercases the strings of these columns
        df_crp_current_stock = lowercase_column_conversion(df_crp_current_stock, ['Part_Desc'])  # Lowercases the strings of these columns
        df_ibe_current_stock = lowercase_column_conversion(df_ibe_current_stock, ['Part_Desc'])  # Lowercases the strings of these columns
        df_ca_current_stock = lowercase_column_conversion(df_ca_current_stock, ['Part_Desc'])  # Lowercases the strings of these columns

        df_afr_current_stock_unique, df_ca_current_stock_unique, df_crp_current_stock_unique, df_ibe_current_stock_unique = description_merge_per_reference([df_afr_current_stock.dropna(subset=['Part_Desc']), df_ca_current_stock.dropna(subset=['Part_Desc']), df_crp_current_stock.dropna(subset=['Part_Desc']), df_ibe_current_stock.dropna(subset=['Part_Desc'])], ['dbs/df_afr_current_stock_unique', 'dbs/df_ca_current_stock_unique', 'dbs/df_crp_current_stock_unique', 'dbs/df_ibe_current_stock_unique'])

    for df in [df_afr_current_stock_unique, df_ca_current_stock_unique, df_crp_current_stock_unique, df_ibe_current_stock_unique]:
        df['word_count'] = df['Part_Desc'].apply(lambda x: len(str(x).split(" ")))
        # print(df.word_count.describe())

        try:
            freq = pd.Series(' '.join(df['Part_Desc'])
                         .split()
                         ).value_counts()[:20]
            print(freq)
        except TypeError:
            print(df.iloc[56998])


def description_merge_per_reference(dfs, dfs_locations):
    new_dfs = []
    start_out = time.time()
    print('a - here')
    for original_df, new_df_loc in zip(dfs, dfs_locations):
        duplicated_refs = list(original_df.groupby('Part_Ref')['Part_Ref'].agg('count').where(lambda x: x > 1).dropna().index)

        selected_references_df = original_df.loc[original_df['Part_Ref'].isin(duplicated_refs), :]
        remaining_references_df = original_df.loc[~original_df['Part_Ref'].isin(duplicated_refs), :]

        selected_references_df = selected_references_df.groupby('Part_Ref').apply(repeated_references_description_concatenation).drop_duplicates()

        final_df = remaining_references_df.append(selected_references_df)

        new_dfs.append(final_df)
        print('d - elapsed time: {:.3f}'.format(time.time() - start_out))

    save_csv(new_dfs, dfs_locations)
    return new_dfs[0], new_dfs[1], new_dfs[2], new_dfs[3]
    # If after joining all descriptions for the same Reference, there are still NaN descriptions, drop the reference;


def repeated_references_description_concatenation(x):
    concatenated_description = ' '.join(list(x['Part_Desc']))
    concatenated_description_unique_words = ' '.join(unique_list_creation(concatenated_description.split()))

    x['Part_Desc'] = concatenated_description_unique_words

    return x


def unique_list_creation(old_list):
    new_list = []
    [new_list.append(x) for x in old_list if x not in new_list]
    return new_list


if __name__ == '__main__':
    main()
