import re
import sys
import time
import logging
import numpy as np
import pandas as pd
from traceback import format_exc
import level_2_pa_part_reference_options as options_file
from level_2_pa_part_reference_options import regex_dict
from modules.level_1_a_data_acquisition import read_csv, sql_retrieve_df_specified_query
from modules.level_1_b_data_processing import lowercase_column_conversion, literal_removal, string_punctuation_removal, stop_words_removal
from modules.level_1_e_deployment import save_csv
local_flag = 1
current_platforms = ['BI_AFR', 'BI_CRP', 'BI_IBE', 'BI_CA']
sel_month = '202002'
master_file_processing_flag = 0


def main():

    if master_file_processing_flag:
        master_file_processing()
        master_file_reference_match()

    platforms_stock, dim_product_group, dim_clients = data_acquisition(current_platforms, 'dbs/dim_product_group_section_A.csv', 'dbs/dim_clients_section_A.csv')
    data_processing(platforms_stock, dim_product_group, dim_clients)


def master_file_processing():
    # This function's goal is to take raw txt master files and identify each column and convert the file to csv;
    # For each file, this function needs: column delimiter positions, column's names and 2 flags on whether to ignore the first/last row;

    for master_file_loc in options_file.master_files_to_convert.keys():

        master_file_info = options_file.master_files_to_convert[master_file_loc]
        master_file_positions = master_file_info[0]
        master_file_col_names = master_file_info[1]
        header_flag = master_file_info[2]
        tail_flag = master_file_info[3]

        fields_dict = {key: [] for key in master_file_col_names}

        f = open(master_file_loc + '.txt', 'r')

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

        result.to_csv(master_file_loc + '.csv')

    vag_merge()  # ToDo: need to replace this by the corresponding function in Section B, if not, create a new one;


def vag_merge():
    df = pd.read_csv('dbs/Master_Files/VAG_TPCNCAVW.csv', index_col=0)
    df2 = pd.read_csv('dbs/Master_Files/VAG_TPCNCSK.csv', index_col=0)

    df3 = pd.concat([df, df2])

    df3.to_csv('dbs/Master_Files/VAG_2.csv')


def master_file_reference_match():
    all_platforms = ['BI_AFR', 'BI_CRP', 'BI_IBE', 'BI_CA']  # This will probably be a global variable
    current_stock_master_file = pd.concat([pd.read_csv('dbs/df_{}_current_stock_unique_202002_section_B_step_2.csv'.format(platform), index_col=False) for platform in all_platforms])

    for master_file_loc in options_file.master_files_converted:
        # print('master_file_loc: {}'.format(master_file_loc))

        master_file_brands = options_file.master_files_and_brand[master_file_loc]
        master_file = read_csv(master_file_loc, index_col=0, dtype={'Part_Ref': str})
        # print(master_file.head())
        # master_file = master_file.loc[master_file['Part_Ref'].isin(['13481236', '13481237', '13481277'])]

        master_file_refs = master_file['Part_Ref'].unique()  # ToDo HERE
        # print('Master File References Count: {}'.format(len(master_file_refs)))

        for master_file_brand in master_file_brands:
            dms_codes_list = dms_codes_retrieval(all_platforms, master_file_brand)
            current_stock_master_file_filtered = current_stock_master_file.loc[current_stock_master_file['Franchise_Code'].isin(dms_codes_list)]
            # current_stock_master_file_filtered = current_stock_master_file.loc[current_stock_master_file['Franchise_Code'].isin(['VAG'])]  # This is for VAG file testing, where instead of filtering by all the selected DMS Codes, I only select VAG

            if current_stock_master_file_filtered.shape[0]:
                if master_file_brand == 'fiat':  # Fiat
                    print('### FIAT - {} Refs ###'.format(current_stock_master_file_filtered.shape[0]))
                    matched_dfs = []
                    # Plain Match
                    matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file, 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                    matched_dfs.append(matched_refs_df)
                    non_matched_dfs = non_matched_refs_df

                    if non_matched_refs_df.shape[0]:
                        # Erroneous References:
                        non_matched_refs_df['Part_Ref'] = non_matched_refs_df['Part_Ref'].apply(fiat_error_handling)

                        non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(dms_code_removal, args=(dms_codes_list,)).apply(zero_at_beginning_removal)
                        master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(zero_at_beginning_removal)

                        # current_stock_master_file_filtered_step_2 = pd.merge(current_stock_master_file_filtered_step_2, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], left_on='Part_Ref_Step_2', right_on='Part_Ref_Step_2', how='left', suffixes=('', '_temp')).drop(['Part_Ref_Step_2'], axis=1).rename(columns={'Part_Desc_PT': 'Part_Desc_MF'})
                        matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file, 'DMS Code Retrieval', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                        matched_dfs.append(matched_refs_df_step_2)
                        non_matched_dfs = non_matched_refs_df_step_2

                    matched_df_total = pd.concat(matched_dfs)
                    print('{} - Matched: {}'.format(master_file_brand, matched_df_total.shape[0]))
                    print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs.shape[0]))
                    non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                if master_file_brand == 'nissan':
                    print('### NISSAN - {} Refs ###'.format(current_stock_master_file_filtered.shape[0]))
                    matched_dfs = []
                    # Plain Match
                    matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file, 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                    matched_dfs.append(matched_refs_df)
                    non_matched_dfs = non_matched_refs_df

                    if non_matched_refs_df.shape[0]:
                        # Removal of DMS_Code in the Reference:
                        non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(dms_code_removal, args=(dms_codes_list,)).apply(zero_at_beginning_removal)
                        master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(zero_at_beginning_removal)

                        matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'DMS Code Retrieval', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                        matched_dfs.append(matched_refs_df_step_2)
                        non_matched_dfs = non_matched_refs_df_step_2

                        if non_matched_refs_df_step_2.shape[0]:
                            # Previous References Match
                            matched_refs_df_step_3, non_matched_refs_df_step_3 = references_merge(non_matched_refs_df_step_2, master_file[['Supersession Leader', 'Part_Desc_PT']], 'Previous References Match', left_key='Part_Ref', right_key='Supersession Leader')
                            matched_dfs.append(matched_refs_df_step_3)
                            non_matched_dfs = non_matched_refs_df_step_3

                            if non_matched_refs_df_step_3.shape[0]:
                                # Error Handling
                                non_matched_refs_df_step_3['Part_Ref_Step_4'] = non_matched_refs_df_step_3['Part_Ref'].apply(hifen_removal).apply(last_dot_removal)
                                master_file['Part_Ref_Step_4'] = master_file['Part_Ref'].apply(hifen_removal).apply(last_dot_removal)

                                matched_refs_df_step_4, non_matched_refs_df_step_4 = references_merge(non_matched_refs_df_step_3, master_file[['Part_Ref_Step_4', 'Part_Desc_PT']], 'Erroneous References', left_key='Part_Ref_Step_4', right_key='Part_Ref_Step_4')
                                matched_dfs.append(matched_refs_df_step_4)
                                non_matched_dfs = non_matched_refs_df_step_4

                    matched_df_total = pd.concat(matched_dfs)
                    print('{} - Matched: {}'.format(master_file_brand, matched_df_total.shape[0]))
                    print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs.shape[0]))
                    non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                if master_file_brand == 'seat':
                    print('### SEAT - {} Refs ###'.format(current_stock_master_file_filtered.shape[0]))
                    matched_dfs = []
                    # Plain Match
                    matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file, 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                    matched_dfs.append(matched_refs_df)
                    non_matched_dfs = non_matched_refs_df

                    if non_matched_refs_df.shape[0]:
                        # Removal of DMS_Code in the Reference:
                        non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(dms_code_removal, args=(dms_codes_list,)).apply(zero_at_beginning_removal)
                        master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(zero_at_beginning_removal)

                        matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'DMS Code Retrieval', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                        matched_dfs.append(matched_refs_df_step_2)
                        non_matched_dfs = non_matched_refs_df_step_2

                    matched_df_total = pd.concat(matched_dfs)
                    print('{} - Matched: {}'.format(master_file_brand, matched_df_total.shape[0]))
                    print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs.shape[0]))
                    non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                if master_file_brand == 'peugeot':
                    print('### PEUGEOT - {} Refs ###'.format(current_stock_master_file_filtered.shape[0]))
                    matched_dfs = []
                    # master_file['Part_Ref'] = master_file['Part_Ref'].apply(initial_code_removing)

                    # Plain Match
                    matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file, 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                    matched_dfs.append(matched_refs_df)
                    non_matched_dfs = non_matched_refs_df

                    if non_matched_refs_df.shape[0]:
                        # Removal of DMS_Code in the Reference:
                        non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(dms_code_removal, args=(dms_codes_list,)).apply(zero_at_beginning_removal)
                        master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(zero_at_beginning_removal)

                        matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'DMS Code Retrieval', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                        matched_dfs.append(matched_refs_df_step_2)
                        non_matched_dfs = non_matched_refs_df_step_2

                    matched_df_total = pd.concat(matched_dfs)
                    print('{} - Matched: {}'.format(master_file_brand, matched_df_total.shape[0]))
                    print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs.shape[0]))
                    non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                if master_file_brand == 'citroen':
                    print('### CITROEN - {} Refs ###'.format(current_stock_master_file_filtered.shape[0]))
                    matched_dfs = []
                    # master_file['Part_Ref'] = master_file['Part_Ref'].apply(initial_code_removing)
                    current_stock_master_file_filtered['Part_Ref'] = current_stock_master_file_filtered['Part_Ref'].apply(particular_case_references, args=(master_file_brand,))

                    # Plain Match
                    matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file, 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                    matched_dfs.append(matched_refs_df)
                    non_matched_dfs = non_matched_refs_df

                    if non_matched_refs_df.shape[0]:
                        # Removal of DMS_Code in the Reference:
                        non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(dms_code_removal, args=(dms_codes_list,)).apply(zero_at_beginning_removal)
                        master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(zero_at_beginning_removal)

                        matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'DMS Code Retrieval', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                        matched_dfs.append(matched_refs_df_step_2)
                        non_matched_dfs = non_matched_refs_df_step_2

                    matched_df_total = pd.concat(matched_dfs)
                    print('{} - Matched: {}'.format(master_file_brand, matched_df_total.shape[0]))
                    print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs.shape[0]))
                    non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                if master_file_brand == 'opel':
                    print('### OPEL - {} Refs ###'.format(current_stock_master_file_filtered.shape[0]))
                    matched_dfs = []

                    # Plain Match
                    matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file, 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                    matched_dfs.append(matched_refs_df)
                    non_matched_dfs = non_matched_refs_df

                    if non_matched_refs_df.shape[0]:
                        # Removal of DMS_Code in the Reference:
                        non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(dms_code_removal, args=(dms_codes_list,)).apply(zero_at_beginning_removal)
                        master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(zero_at_beginning_removal)

                        matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'DMS Code Retrieval', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                        matched_dfs.append(matched_refs_df_step_2)
                        non_matched_dfs = non_matched_refs_df_step_2

                    matched_df_total = pd.concat(matched_dfs)
                    print('{} - Matched: {}'.format(master_file_brand, matched_df_total.shape[0]))
                    print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs.shape[0]))
                    non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                if master_file_brand == 'chevrolet':
                    print('### CHEVROLET - {} Refs ###'.format(current_stock_master_file_filtered.shape[0]))
                    matched_dfs = []

                    # Plain Match
                    matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file, 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                    matched_dfs.append(matched_refs_df)
                    non_matched_dfs = non_matched_refs_df

                    if non_matched_refs_df.shape[0]:
                        # Removal of DMS_Code in the Reference:
                        non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(dms_code_removal, args=(dms_codes_list + ['CV', 'GD', 'CHEC'],)).apply(zero_at_beginning_removal)
                        master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(zero_at_beginning_removal)

                        matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'DMS Code Retrieval', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                        matched_dfs.append(matched_refs_df_step_2)
                        non_matched_dfs = non_matched_refs_df_step_2

                    matched_df_total = pd.concat(matched_dfs)
                    print('{} - Matched: {}'.format(master_file_brand, matched_df_total.shape[0]))
                    print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs.shape[0]))
                    non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                if master_file_brand == 'volkswagen':
                    print('### VAG - {} Refs ###'.format(current_stock_master_file_filtered.shape[0]))
                    matched_dfs = []

                    # Plain Match
                    matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file, 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                    matched_dfs.append(matched_refs_df)
                    non_matched_dfs = non_matched_refs_df

                    if non_matched_refs_df.shape[0]:
                        # Removal of DMS_Code in the Reference:
                        non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(dms_code_removal, args=(dms_codes_list + ['V'],)).apply(zero_at_beginning_removal)
                        master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(zero_at_beginning_removal)

                        matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'DMS Code Retrieval', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                        matched_dfs.append(matched_refs_df_step_2)
                        non_matched_dfs = non_matched_refs_df_step_2

                    matched_df_total = pd.concat(matched_dfs)
                    print('{} - Matched: {}'.format(master_file_brand, matched_df_total.shape[0]))
                    print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs.shape[0]))
                    # non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))
            else:
                print('No data found for the DMS Code(s): {}'.format(dms_codes_list))


def particular_case_references(string_to_process, brand):
    if brand == 'citroen':
        if re.match('LPTPBP159N', string_to_process):
            return 'LPTPBP159'
        elif re.match('LPTPTP0022-208L-1L', string_to_process):
            return 'LPTPTP0022'
        elif re.match('LPTPTP0022-208', string_to_process):
            return 'LPTPTP0022'
        else:
            return string_to_process


def references_merge(df, master_file, description_string, left_key, right_key):
    step_time = time.time()

    df = pd.merge(df, master_file[[right_key, 'Part_Desc_PT']], left_on=left_key, right_on=right_key, how='left').rename(columns={'Part_Desc_PT': 'Part_Desc_MF'})
    if left_key != right_key:
        df.drop([right_key], axis=1, inplace=True)

    if left_key != 'Part_Ref':
        df.drop([left_key], axis=1, inplace=True)

    non_matched = df['Part_Desc_MF'].isnull().sum()
    matched = df.shape[0] - non_matched
    print('{} - Matched/Non-Matched: {}/{} ({:.3f}s)'.format(description_string, matched, non_matched, time.time() - step_time))

    matched_df = df.loc[~df['Part_Desc_MF'].isnull(), :]
    non_matched_df = df.loc[df['Part_Desc_MF'].isnull(), :].drop(['Part_Desc_MF'], axis=1)

    return matched_df, non_matched_df


def zero_at_beginning_removal(string_to_process):
    # print('zero removal A', string_to_process)
    regex = re.compile(regex_dict['zero_at_beginning'])

    processed_string = regex.sub('', string_to_process)
    # print('zero removal B', processed_string)

    return processed_string


def initial_code_removing(string_to_process):
    regex = re.compile(regex_dict['001_beginning_code_removal'])

    processed_string = regex.sub('', string_to_process)

    return processed_string


def hifen_removal(string_to_process):
    regex = re.compile(regex_dict['remove_hifen'])

    processed_string = regex.sub('', string_to_process)
    return processed_string


def last_dot_removal(string_to_process):
    regex = re.compile(regex_dict['remove_last_dot'])

    processed_string = regex.sub('', string_to_process)
    return processed_string


def dms_code_addition(string_to_process, dms_code):
    processed_string = dms_code + string_to_process

    return processed_string


def fiat_error_handling(string_to_process):
    # string_to_process_len = len(string_to_process)
    regex = re.compile(regex_dict['2_letters_at_end'])

    processed_string = regex.sub('', string_to_process)
    # processed_string_len = len(processed_string)

    # if string_to_process_len > processed_string_len:
    #     print('Here A:', string_to_process)
    #     print('Here B:', processed_string)

    return processed_string


def dms_codes_retrieval(platforms, brand):
    query = ' UNION ALL '.join([options_file.dms_codes_per_franchise.format(x, y) for x, y in zip(platforms, [brand] * len(platforms))])
    df = sql_retrieve_df_specified_query(options_file.DSN, 'BI_AFR', options_file, query)
    dms_codes = list(np.unique(df['Original_Value'].values))

    return dms_codes



def dms_code_removal(string_to_process, dms_codes):
    # print('Here 0:', string_to_process)
    dms_codes.sort(key=len, reverse=True)  # I need to sort by length, from larger to smaller to avoid substrings of other dms codes. Ex: FI and FIA. FIA has to be searched first, otherwise FI will remove only FI and leave an erroneous A;

    # print(dms_codes)
    regex_code = r'^' + '|^'.join(dms_codes)
    # print(regex_code)
    regex = re.compile(regex_code)
    # print(string_to_process)
    processed_string = regex.sub('', string_to_process)
    # print('Here 1:', processed_string)

    return processed_string


def data_acquisition(platforms, dim_product_group_file, dim_clients_file):
    platforms_stock = []

    try:
        for platform in platforms:
            df_current_stock = read_csv('dbs/df_{}_current_stock_{}_section_A.csv'.format(platform, sel_month), index_col=0)
            print('File found for platform {}...'.format(platform))
            platforms_stock.append(df_current_stock)

        dim_product_group = read_csv(dim_product_group_file, index_col=0)
        dim_clients = read_csv(dim_clients_file, index_col=0)
    except FileNotFoundError:
        for platform in platforms:
            print('Retrieving from DW for platform {}...'.format(platform))
            df_current_stock = sql_retrieve_df_specified_query(options_file.DSN, options_file.sql_info['database_{}'.format(platform)],  options_file, options_file.current_stock_query.format(platform, sel_month))
            platforms_stock.append(df_current_stock)

        dim_product_group = sql_retrieve_df_specified_query(options_file.DSN, options_file.sql_info['database_BI_AFR'], options_file, options_file.dim_product_group_query)
        dim_clients = sql_retrieve_df_specified_query(options_file.DSN, options_file.sql_info['database_BI_GSC'], options_file, options_file.dim_clients_query)

        save_csv(platforms_stock + [dim_product_group, dim_clients], ['dbs/df_{}_current_stock_{}_section_A'.format(platform, sel_month) for platform in platforms] + ['dbs/dim_product_group_section_A', 'dbs/dim_clients_section_A'])
    return platforms_stock, dim_product_group, dim_clients


def data_processing(df_platforms_stock, dim_product_group, dim_clients):
    platforms_stock_unique = []

    try:
        for platform, df_platform_stock in zip(current_platforms, df_platforms_stock):
            current_stock_unique = read_csv('dbs/df_{}_current_stock_unique_{}_section_B_step_2.csv'.format(platform, sel_month), index_col=0)
            platforms_stock_unique.append(current_stock_unique)
    except FileNotFoundError:
        for platform, df_platform_stock in zip(current_platforms, df_platforms_stock):
            print('Platform: {}'.format(platform))
            start_a = time.time()
            df_platform_current_stock = lowercase_column_conversion(df_platform_stock, ['Part_Desc'])  # Lowercases the strings of these columns
            df_platform_current_stock = literal_removal(df_platform_current_stock, 'Part_Desc')  # Removes literals from the selected column
            df_platform_current_stock['Part_Desc'] = df_platform_current_stock['Part_Desc'].apply(string_punctuation_removal)  # Removes accents and punctuation from the selected column;
            print('a - elapsed time: {:.3f}'.format(time.time() - start_a))

            # start_b = time.time()
            # df_platform_current_stock['Part_Desc'] = df_platform_current_stock['Part_Desc'].apply(abbreviations_correction, args=(options_file.abbreviations_dict, ))
            # print('b - elapsed time: {:.3f}'.format(time.time() - start_b))

            start_c = time.time()
            df_platform_current_stock['Part_Desc'] = df_platform_current_stock['Part_Desc'].apply(stop_words_removal, args=(options_file.stop_words['Common_Stop_Words'] + options_file.stop_words[platform],))
            print('b - elapsed time: {:.3f}'.format(time.time() - start_c))

            start_d = time.time()
            df_current_stock_unique = description_merge_per_reference(df_platform_current_stock)  # Merges descriptions for the same reference and removes repeated tokens;
            print('c - elapsed time: {:.3f}'.format(time.time() - start_d))

            freq = pd.Series(' '.join(df_current_stock_unique['Part_Desc']).split()).value_counts()[:20]
            print(freq)

            print('total time: {:.3f}'.format(time.time() - start_a))
            save_csv([df_current_stock_unique], ['dbs/df_{}_current_stock_unique_{}_section_B_step_2'.format(platform, sel_month)])


def description_merge_per_reference(original_df):
    duplicated_refs = list(original_df.groupby('Part_Ref')['Part_Ref'].agg('count').where(lambda x: x > 1).dropna().index)

    selected_references_df = original_df.loc[original_df['Part_Ref'].isin(duplicated_refs), :]
    remaining_references_df = original_df.loc[~original_df['Part_Ref'].isin(duplicated_refs), :]

    selected_references_df = selected_references_df.groupby('Part_Ref').apply(repeated_references_description_concatenation).drop_duplicates()

    final_df = remaining_references_df.append(selected_references_df)

    return final_df


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
