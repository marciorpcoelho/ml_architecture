import re
import sys
import time
import nltk
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from traceback import format_exc
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer
import level_2_pa_part_reference_options as options_file
from level_2_pa_part_reference_options import regex_dict
from modules.level_1_a_data_acquisition import read_csv, sql_retrieve_df_specified_query
from modules.level_1_b_data_processing import lowercase_column_conversion, literal_removal, string_punctuation_removal, master_file_processing, regex_string_replacement, brand_code_removal, string_volkswagen_preparation, value_substitution, lemmatisation, stemming, unidecode_function, string_digit_removal, word_frequency, words_dataframe_creation
from modules.level_1_e_deployment import save_csv, time_tags
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
# nltk.download('stopwords')
# nltk.download('rslp')  # RSLP Portuguese stemmer

local_flag = 1
master_file_processing_flag = 0
current_platforms = ['BI_AFR', 'BI_CRP', 'BI_IBE', 'BI_CA']
sel_month = '202002'
part_desc_col = 'Part_Desc_Merged'
# 'Part_Desc' - Part Description from DW
# 'Part_Desc_PT' - Part Description from Master Files when available, and from DW when not available
# 'Part_Desc_Merge' - Part Description from Master Files merged with Part Description from the DW
time_tag_date, _ = time_tags(format_date="%m_%Y")
keywords_per_parts_family_dict = {}


def main():
    if master_file_processing_flag:
        master_file_processing(options_file.master_files_to_convert)

    platforms_stock, dim_product_group, dim_clients = data_acquisition(current_platforms, 'dbs/dim_product_group_section_A.csv', 'dbs/dim_clients_section_A.csv')
    platforms_stock = master_file_reference_match(platforms_stock, time_tag_date, dim_clients)
    data_processing(platforms_stock, dim_product_group, dim_clients)
    data_modelling(keywords_per_parts_family_dict)


def data_modelling(keyword_dictionary):
    train_dataset = pd.concat([pd.read_csv('output/{}_train_dataset.csv'.format(platform), index_col=False, low_memory=False) for platform in current_platforms])
    test_dataset = pd.concat([pd.read_csv('output/{}_test_dataset.csv'.format(platform), index_col=False, low_memory=False) for platform in current_platforms])

    requests_dict_2 = {}
    non_matched_keywords = []

    try:
        test_dataset_words_eval = pd.read_csv('output/df_top_words_parts_reference_project.csv', index_col='Part_Ref')
        print('Test Datasets Keywords Found...')
    except FileNotFoundError:
        print('Evaluation Test Dataset Keywords...')

        _, top_words_ticket_frequency = word_frequency(train_dataset, unit_col='Part_Ref', description_col=part_desc_col)
        test_dataset_words_eval, _ = words_dataframe_creation(test_dataset, top_words_ticket_frequency, unit_col='Part_Ref', description_col=part_desc_col)
        test_dataset_words_eval['New_Product_Group_DW'] = 'Não Definido'

        # print('top_words_ticket_frequency \n', top_words_ticket_frequency)
        # print('top_words_ticket_frequency number of words \n', len(top_words_ticket_frequency.keys()))
        # print('df_top_words \n', df_top_words)
        # print('df_top_words shape \n', df_top_words.shape)

        test_dataset_words_eval.to_csv('output/df_top_words_parts_reference_project.csv')

    references = ['54501BA60A', '545011Y210']

    # test_dataset = test_dataset.loc[test_dataset['Part_Ref'].isin(references)]
    # print(test_dataset)
    # test_dataset_words_eval = test_dataset_words_eval.loc[test_dataset_words_eval.index.isin(references)]
    for label in keyword_dictionary.keys():
        for keywords in keyword_dictionary[label]:

            # tokenized_key_word = nltk.tokenize.word_tokenize(keywords)
            # try:
            #     selected_cols = test_dataset_words_eval[tokenized_key_word]
            # except KeyError:
            #     print('Palavras chave {} que foram reduzidas a {} não foram encontradas de forma consecutiva.'.format(keywords, tokenized_key_word))
            #     continue
            #
            # matched_index = selected_cols[selected_cols == 1].dropna(axis=0).index.values  # returns the references with the keyword present
            # if matched_index is not None:
            #     test_dataset_words_eval.loc[test_dataset_words_eval.index.isin(matched_index), 'New_Product_Group_DW'] = label
            # else:
            #     print('Palavras chave {} que foram reduzidas a {} não foram encontradas de forma consecutiva.'.format(keywords, tokenized_key_word))

            try:
                rank = 0
                test_dataset_words_eval.loc[test_dataset_words_eval[keywords] == 1, 'New_Product_Group_DW'] = label
                requests_dict_2 = request_matches(label, keywords, rank, test_dataset_words_eval[test_dataset_words_eval[keywords] == 1].index.values, requests_dict_2)
            except KeyError:
                non_matched_keywords.append(keywords)
                # print('Palavra chave não encontrada: {}'.format(keywords))
                continue

    test_dataset.sort_values(by='Part_Ref', inplace=True)
    test_dataset_words_eval.sort_index(inplace=True)

    # print('test_dataset\n', test_dataset['Product_Group_DW'].values)
    # print('test_dataset_words_eval\n', test_dataset_words_eval['New_Product_Group_DW'].values)

    # if [(x, y) for (x, y) in zip(test_dataset['Part_Ref'].values, test_dataset_words_eval.index.values) if x != y]:
    #     unique_requests_df = test_dataset['Part_Ref'].unique()
    #     unique_requests_df_top_words = test_dataset_words_eval.index.values
    #     print('Referências não classificados no dataset original: {}'.format([x for x in unique_requests_df if x not in unique_requests_df_top_words]))
    #     print('Referências não classificados no dataset Top_Words: {}'.format([x for x in unique_requests_df_top_words if x not in unique_requests_df]))
    #     raise ValueError('Existem Referências sem classificação!')

    results = [1 if x == y else 0 for (x, y) in zip(test_dataset['Product_Group_DW'].values, test_dataset_words_eval['New_Product_Group_DW'].values)]
    correctly_classified = np.sum(results)
    total_classifications = len(results)
    print('Non Matched Keywords: \n{}'.format(non_matched_keywords))
    print('requests_dict_2 \n', requests_dict_2)

    print('Correctly Labeled: {}'.format(correctly_classified))
    print('Incorrectly Labeled: {}'.format(total_classifications - correctly_classified))
    print('% Correctly Labeled: {:.2f}'.format(correctly_classified / total_classifications * 100))
    return


def request_matches(label, keywords, rank, requests_list, dictionary):

    for request in requests_list:
        try:
            dictionary[request].append((label, rank))
        except KeyError:
            dictionary[request] = [(label, rank)]

    return dictionary


def master_file_reference_match(platforms_stock, time_tag_date_in, dim_clients):

    try:
        current_stock_master_file = read_csv('dbs/current_stock_all_platforms_master_stock_matched_{}.csv'.format(time_tag_date_in))
        print('Descriptions already matched from brand master files...')
    except FileNotFoundError:
        # current_stock_master_file = pd.concat([pd.read_csv('dbs/df_{}_current_stock_unique_202002_section_B_step_2.csv'.format(platform), index_col=False, low_memory=False) for platform in current_platforms])
        current_stock_master_file = pd.concat(platforms_stock)
        current_stock_master_file['Part_Desc_PT'] = np.nan
        # current_stock_master_file = current_stock_master_file.loc[current_stock_master_file['Part_Ref'] == 'A2118800905']

        # print('BEFORE \n', current_stock_master_file.head(10))
        # print('BEFORE Shape: ', current_stock_master_file.shape)
        # print('BEFORE Unique Refs', current_stock_master_file['Part_Ref'].nunique())
        # print('BEFORE Unique Refs', current_stock_master_file.drop_duplicates(subset='Part_Ref')['Part_Ref'].nunique())
        # print('BEFORE Null Descriptions: ', current_stock_master_file['Part_Desc_PT'].isnull().sum())

        for master_file_loc in options_file.master_files_converted:
            master_file_brands = options_file.master_files_and_brand[master_file_loc]
            master_file = read_csv(master_file_loc, index_col=0, dtype={'Part_Ref': str}, low_memory=False)
            # master_file = master_file.loc[master_file['Part_Ref'].isin(['000012008', '000010006', '000000999'])]

            for master_file_brand in master_file_brands:
                brand_codes_list = brand_codes_retrieval(current_platforms, master_file_brand)
                print('master_file_brand: {}, brand_codes_list: {}'.format(master_file_brand, brand_codes_list))

                current_stock_master_file_filtered = current_stock_master_file.loc[current_stock_master_file['Franchise_Code'].isin(brand_codes_list)]
                # print(current_stock_master_file_filtered.head(20))

                # current_stock_master_file_filtered = current_stock_master_file_filtered.loc[current_stock_master_file_filtered['Part_Ref'] == '000010006']
                # current_stock_master_file_filtered = current_stock_master_file.loc[current_stock_master_file['Franchise_Code'].isin(['VAG'])]  # This is for VAG file testing, where instead of filtering by all the selected brand Codes, I only select VAG
                # print('current_stock_master_file_filtered: \n', current_stock_master_file_filtered.head())

                if current_stock_master_file_filtered.shape[0]:
                    if master_file_brand == 'fiat':  # Fiat
                        print('### FIAT - {} Refs ###'.format(current_stock_master_file_filtered['Part_Ref'].nunique()))
                        matched_dfs = []

                        # Plain Match
                        _, matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file[['Part_Ref', 'Part_Desc_PT']], 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                        matched_dfs.append(matched_refs_df)
                        non_matched_dfs = non_matched_refs_df

                        if non_matched_refs_df.shape[0]:
                            # Erroneous References:
                            non_matched_refs_df['Part_Ref'] = non_matched_refs_df['Part_Ref'].apply(regex_string_replacement, args=(regex_dict['2_letters_at_end'],))

                            non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(brand_code_removal, args=(brand_codes_list,)).apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))
                            master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))

                            _, matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'Brand Code Removal', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                            matched_dfs.append(matched_refs_df_step_2)
                            non_matched_dfs = non_matched_refs_df_step_2

                        matched_df_brand = pd.concat(matched_dfs)
                        current_stock_master_file, _, _ = references_merge(current_stock_master_file, matched_df_brand[['Part_Ref', 'Part_Desc_PT']], 'Master Stock Match', left_key='Part_Ref', right_key='Part_Ref')
                        # print(matched_df_brand.head(20))
                        # _, _ = references_merge(current_stock_master_file, matched_df_brand[['Part_Ref', 'Part_Desc_MF']], 'Master Stock File', left_key='Part_Ref', right_key='Part_Ref')
                        # matched_df_brand['Part_Desc_Comparison'] = np.where(matched_df_brand['Part_Desc'] != matched_df_brand['Part_Desc_MF'], 1, 0)
                        # print('Different Descs: {}, Total Matched Desc: {}, Different Desc (%): {:.2f}'.format(matched_df_brand['Part_Desc_Comparison'].sum(), matched_df_brand.shape, matched_df_brand['Part_Desc_Comparison'].sum() / matched_df_brand.shape[0] * 100))
                        print('{} - Matched: {}'.format(master_file_brand, matched_df_brand['Part_Ref'].nunique()))
                        print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs['Part_Ref'].nunique()))
                        non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                    if master_file_brand == 'nissan':
                        print('### NISSAN - {} Refs ###'.format(current_stock_master_file_filtered['Part_Ref'].nunique()))
                        matched_dfs = []
                        # Plain Match
                        _, matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file[['Part_Ref', 'Part_Desc_PT']], 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                        matched_dfs.append(matched_refs_df)
                        non_matched_dfs = non_matched_refs_df

                        if non_matched_refs_df.shape[0]:
                            # Removal of Brand_Code in the Reference:
                            non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(brand_code_removal, args=(brand_codes_list,)).apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))
                            master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))

                            _, matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'Brand Code Removal', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                            matched_dfs.append(matched_refs_df_step_2)
                            non_matched_dfs = non_matched_refs_df_step_2

                            if non_matched_refs_df_step_2.shape[0]:
                                # Previous References Match
                                _, matched_refs_df_step_3, non_matched_refs_df_step_3 = references_merge(non_matched_refs_df_step_2, master_file[['Supersession Leader', 'Part_Desc_PT']], 'Previous References Match', left_key='Part_Ref', right_key='Supersession Leader')
                                matched_dfs.append(matched_refs_df_step_3)
                                non_matched_dfs = non_matched_refs_df_step_3

                                if non_matched_refs_df_step_3.shape[0]:
                                    # Error Handling
                                    non_matched_refs_df_step_3['Part_Ref_Step_4'] = non_matched_refs_df_step_3['Part_Ref'].apply(regex_string_replacement, args=(regex_dict['remove_hifen'],)).apply(regex_string_replacement, args=(regex_dict['remove_last_dot'],))
                                    master_file['Part_Ref_Step_4'] = master_file['Part_Ref'].apply(regex_string_replacement, args=(regex_dict['remove_hifen'],)).apply(regex_string_replacement, args=(regex_dict['remove_last_dot'],))
                                    master_file.drop_duplicates(subset='Part_Ref_Step_4', inplace=True)

                                    _, matched_refs_df_step_4, non_matched_refs_df_step_4 = references_merge(non_matched_refs_df_step_3, master_file[['Part_Ref_Step_4', 'Part_Desc_PT']], 'Erroneous References', left_key='Part_Ref_Step_4', right_key='Part_Ref_Step_4')
                                    matched_dfs.append(matched_refs_df_step_4)
                                    non_matched_dfs = non_matched_refs_df_step_4

                        matched_df_brand = pd.concat(matched_dfs)
                        current_stock_master_file, _, _ = references_merge(current_stock_master_file, matched_df_brand[['Part_Ref', 'Part_Desc_PT']], 'Master Stock Match', left_key='Part_Ref', right_key='Part_Ref')
                        # _, _ = references_merge(current_stock_master_file, matched_df_brand[['Part_Ref', 'Part_Desc_MF']], 'Master Stock File', left_key='Part_Ref', right_key='Part_Ref')
                        # print('Nissan: \n', matched_df_brand.head())
                        # matched_df_brand['Part_Desc_Comparison'] = np.where(matched_df_brand['Part_Desc'] != matched_df_brand['Part_Desc_MF'], 1, 0)
                        # print('Different Descs: {}, Total Matched Desc: {}, Different Desc (%): {:.2f}'.format(matched_df_brand['Part_Desc_Comparison'].sum(), matched_df_brand.shape, matched_df_brand['Part_Desc_Comparison'].sum() / matched_df_brand.shape[0] * 100))
                        print('{} - Matched: {}'.format(master_file_brand, matched_df_brand['Part_Ref'].nunique()))
                        print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs['Part_Ref'].nunique()))
                        non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                    if master_file_brand == 'seat':
                        print('### SEAT - {} Refs ###'.format(current_stock_master_file_filtered['Part_Ref'].nunique()))
                        matched_dfs = []
                        # Plain Match
                        _, matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file[['Part_Ref', 'Part_Desc_PT']], 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                        matched_dfs.append(matched_refs_df)
                        non_matched_dfs = non_matched_refs_df

                        if non_matched_refs_df.shape[0]:
                            # Removal of Brand_Code in the Reference:
                            non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(brand_code_removal, args=(brand_codes_list,)).apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))
                            master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))

                            _, matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'Brand Code Removal', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                            matched_dfs.append(matched_refs_df_step_2)
                            non_matched_dfs = non_matched_refs_df_step_2

                        matched_df_brand = pd.concat(matched_dfs)
                        current_stock_master_file, _, _ = references_merge(current_stock_master_file, matched_df_brand[['Part_Ref', 'Part_Desc_PT']].drop_duplicates(subset='Part_Desc_PT'), 'Master Stock Match', left_key='Part_Ref', right_key='Part_Ref')
                        # matched_df_brand['Part_Desc_Comparison'] = np.where(matched_df_brand['Part_Desc'] != matched_df_brand['Part_Desc_MF'], 1, 0)
                        # print('Different Descs: {}, Total Matched Desc: {}, Different Desc (%): {:.2f}'.format(matched_df_brand['Part_Desc_Comparison'].sum(), matched_df_brand.shape, matched_df_brand['Part_Desc_Comparison'].sum() / matched_df_brand.shape[0] * 100))
                        print('{} - Matched: {}'.format(master_file_brand, matched_df_brand['Part_Ref'].nunique()))
                        print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs['Part_Ref'].nunique()))
                        non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                    if master_file_brand == 'peugeot':
                        print('### PEUGEOT - {} Refs ###'.format(current_stock_master_file_filtered['Part_Ref'].nunique()))
                        matched_dfs = []
                        # master_file['Part_Ref'] = master_file['Part_Ref'].apply(initial_code_removing)

                        # Plain Match
                        _, matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file[['Part_Ref', 'Part_Desc_PT']], 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                        matched_dfs.append(matched_refs_df)
                        non_matched_dfs = non_matched_refs_df

                        if non_matched_refs_df.shape[0]:
                            # Removal of Brand_Code in the Reference:
                            non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(brand_code_removal, args=(brand_codes_list,)).apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))
                            master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))

                            _, matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'Brand Code Removal', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                            matched_dfs.append(matched_refs_df_step_2)
                            non_matched_dfs = non_matched_refs_df_step_2

                        matched_df_brand = pd.concat(matched_dfs)
                        current_stock_master_file, _, _ = references_merge(current_stock_master_file, matched_df_brand[['Part_Ref', 'Part_Desc_PT']], 'Master Stock Match', left_key='Part_Ref', right_key='Part_Ref')
                        # matched_df_brand['Part_Desc_Comparison'] = np.where(matched_df_brand['Part_Desc'] != matched_df_brand['Part_Desc_MF'], 1, 0)
                        # print('Different Descs: {}, Total Matched Desc: {}, Different Desc (%): {:.2f}'.format(matched_df_brand['Part_Desc_Comparison'].sum(), matched_df_brand.shape, matched_df_brand['Part_Desc_Comparison'].sum() / matched_df_brand.shape[0] * 100))
                        print('{} - Matched: {}'.format(master_file_brand, matched_df_brand['Part_Ref'].nunique()))
                        print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs['Part_Ref'].nunique()))
                        non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                    if master_file_brand == 'citroen':
                        print('### CITROEN - {} Refs ###'.format(current_stock_master_file_filtered['Part_Ref'].nunique()))
                        matched_dfs = []

                        current_stock_master_file_filtered_copy = current_stock_master_file_filtered.copy()
                        current_stock_master_file_filtered_copy['Part_Ref_Step_1'] = current_stock_master_file_filtered_copy['Part_Ref'].apply(particular_case_references, args=(master_file_brand,))

                        # Plain Match
                        _, matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered_copy, master_file[['Part_Ref', 'Part_Desc_PT']], 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                        matched_dfs.append(matched_refs_df)
                        non_matched_dfs = non_matched_refs_df

                        if non_matched_refs_df.shape[0]:
                            # Removal of Brand_Code in the Reference:
                            non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref_Step_1'].apply(brand_code_removal, args=(brand_codes_list,)).apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))
                            master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))

                            _, matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'Brand Code Removal', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                            matched_dfs.append(matched_refs_df_step_2)
                            non_matched_dfs = non_matched_refs_df_step_2

                        matched_df_brand = pd.concat(matched_dfs)
                        current_stock_master_file, _, _ = references_merge(current_stock_master_file, matched_df_brand[['Part_Ref', 'Part_Desc_PT']], 'Master Stock Match', left_key='Part_Ref', right_key='Part_Ref')
                        # matched_df_brand['Part_Desc_Comparison'] = np.where(matched_df_brand['Part_Desc'] != matched_df_brand['Part_Desc_MF'], 1, 0)
                        # print('Different Descs: {}, Total Matched Desc: {}, Different Desc (%): {:.2f}'.format(matched_df_brand['Part_Desc_Comparison'].sum(), matched_df_brand.shape, matched_df_brand['Part_Desc_Comparison'].sum() / matched_df_brand.shape[0] * 100))
                        print('{} - Matched: {}'.format(master_file_brand, matched_df_brand['Part_Ref'].nunique()))
                        print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs['Part_Ref'].nunique()))
                        non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                    if master_file_brand == 'opel':
                        print('### OPEL - {} Refs ###'.format(current_stock_master_file_filtered['Part_Ref'].nunique()))
                        matched_dfs = []

                        # Plain Match
                        _, matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file, 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                        matched_dfs.append(matched_refs_df)
                        non_matched_dfs = non_matched_refs_df

                        if non_matched_refs_df.shape[0]:
                            # Removal of Brand_Code in the Reference:
                            non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(brand_code_removal, args=(brand_codes_list,)).apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))
                            master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))

                            _, matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'Brand Code Removal', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                            matched_dfs.append(matched_refs_df_step_2)
                            non_matched_dfs = non_matched_refs_df_step_2

                        matched_df_brand = pd.concat(matched_dfs)
                        current_stock_master_file, _, _ = references_merge(current_stock_master_file, matched_df_brand[['Part_Ref', 'Part_Desc_PT']], 'Master Stock Match', left_key='Part_Ref', right_key='Part_Ref')
                        # matched_df_brand['Part_Desc_Comparison'] = np.where(matched_df_brand['Part_Desc'] != matched_df_brand['Part_Desc_MF'], 1, 0)
                        # print(matched_df_brand[['Part_Ref', 'Part_Desc', 'Part_Desc_MF']].head(20))
                        # print(matched_df_brand[['Part_Ref', 'Part_Desc', 'Part_Desc_MF']].tail(20))
                        # print('Different Descs: {}, Total Matched Desc: {}, Different Desc (%): {:.2f}'.format(matched_df_brand['Part_Desc_Comparison'].sum(), matched_df_brand.shape, matched_df_brand['Part_Desc_Comparison'].sum() / matched_df_brand.shape[0] * 100))
                        print('{} - Matched: {}'.format(master_file_brand, matched_df_brand['Part_Ref'].nunique()))
                        print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs['Part_Ref'].nunique()))
                        non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                    if master_file_brand == 'chevrolet':
                        print('### CHEVROLET - {} Refs ###'.format(current_stock_master_file_filtered['Part_Ref'].nunique()))
                        matched_dfs = []

                        # Plain Match
                        _, matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file[['Part_Ref', 'Part_Desc_PT']], 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                        matched_dfs.append(matched_refs_df)
                        non_matched_dfs = non_matched_refs_df

                        if non_matched_refs_df.shape[0]:
                            # Removal of Brand_Code in the Reference:
                            non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(brand_code_removal, args=(brand_codes_list + ['CV', 'GD', 'CHEC'],)).apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))
                            master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))

                            _, matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'Brand Code Removal', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                            matched_dfs.append(matched_refs_df_step_2)
                            non_matched_dfs = non_matched_refs_df_step_2

                        matched_df_brand = pd.concat(matched_dfs)
                        current_stock_master_file, _, _ = references_merge(current_stock_master_file, matched_df_brand[['Part_Ref', 'Part_Desc_PT']], 'Master Stock Match', left_key='Part_Ref', right_key='Part_Ref')
                        # matched_df_brand['Part_Desc_Comparison'] = np.where(matched_df_brand['Part_Desc'] != matched_df_brand['Part_Desc_MF'], 1, 0)
                        # print('Different Descs: {}, Total Matched Desc: {}, Different Desc (%): {:.2f}'.format(matched_df_brand['Part_Desc_Comparison'].sum(), matched_df_brand.shape, matched_df_brand['Part_Desc_Comparison'].sum() / matched_df_brand.shape[0] * 100))
                        print('{} - Matched: {}'.format(master_file_brand, matched_df_brand['Part_Ref'].nunique()))
                        print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs['Part_Ref'].nunique()))
                        non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                    # if master_file_brand == 'audi':
                    #     print('### {} - {} Refs ###'.format(master_file_brand, current_stock_master_file_filtered['Part_Ref'].nunique()))
                    #     # print(brand_codes_list)
                    #     matched_dfs = []
                    #
                    #     # Plain Match
                    #     _, matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file[['Part_Ref', 'Part_Desc_PT']], 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                    #     matched_dfs.append(matched_refs_df)
                    #     non_matched_dfs = non_matched_refs_df
                    #
                    #     if non_matched_refs_df.shape[0]:
                    #         # Removal of Brand_Code in the Reference:
                    #         non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(brand_code_removal, args=(brand_codes_list,)).apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))
                    #         master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))
                    #
                    #         _, matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'Brand Code Removal', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                    #         matched_dfs.append(matched_refs_df_step_2)
                    #         non_matched_dfs = non_matched_refs_df_step_2
                    #
                    #     matched_df_brand = pd.concat(matched_dfs)
                    #     current_stock_master_file, _, _ = references_merge(current_stock_master_file, matched_df_brand[['Part_Ref', 'Part_Desc_PT']], 'Master Stock Match', left_key='Part_Ref', right_key='Part_Ref')
                    #     # print(matched_df_brand)
                    #     # matched_df_brand['Part_Desc_Comparison'] = np.where(matched_df_brand['Part_Desc'] != matched_df_brand['Part_Desc_MF'], 1, 0)
                    #     # print('Different Descs: {}, Total Matched Desc: {}, Different Desc (%): {:.2f}'.format(matched_df_brand['Part_Desc_Comparison'].sum(), matched_df_brand.shape, matched_df_brand['Part_Desc_Comparison'].sum() / matched_df_brand.shape[0] * 100))
                    #     print('{} - Matched: {}'.format(master_file_brand, matched_df_brand['Part_Ref'].nunique()))
                    #     print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs['Part_Ref'].nunique()))
                    #     non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                    if master_file_brand == 'volkswagen' or master_file_brand == 'audi':
                        if current_stock_master_file_filtered.shape[0]:
                            print('### {} - {} Refs ###'.format(master_file_brand, current_stock_master_file_filtered['Part_Ref'].nunique()))
                            # print(brand_codes_list)
                            matched_dfs = []

                            # Space Removal and Word Characters removal at the beginning
                            current_stock_master_file_filtered_copy = current_stock_master_file_filtered.copy()
                            current_stock_master_file_filtered_copy['Part_Ref_Step_1'] = current_stock_master_file_filtered_copy['Part_Ref'].apply(regex_string_replacement, args=(regex_dict['space_removal'],)).apply(regex_string_replacement, args=(regex_dict['single_V_in_the_beginning'],))
                            master_file['Part_Ref_Step_1'] = master_file['Part_Ref'].apply(regex_string_replacement, args=(regex_dict['space_removal'],)).apply(regex_string_replacement, args=(regex_dict['single_V_in_the_beginning'],))
                            master_file.drop_duplicates(subset='Part_Ref_Step_1', inplace=True)  # ToDo: This is necessary due to duplicates in the Master File. There might also be problems regarding the descriptions.

                            _, matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered_copy, master_file[['Part_Ref_Step_1', 'Part_Desc_PT']], 'Initial Match', left_key='Part_Ref_Step_1', right_key='Part_Ref_Step_1', validate_option='many_to_one')

                            matched_dfs.append(matched_refs_df)
                            non_matched_dfs = non_matched_refs_df

                            if non_matched_refs_df.shape[0]:
                                non_matched_refs_df['Part_Ref_Step_1_5'] = non_matched_refs_df['Part_Ref_Step_1'].apply(regex_string_replacement, args=(regex_dict['letters_in_the_beginning'],))
                                master_file['Part_Ref_Step_1_5'] = master_file['Part_Ref_Step_1'].apply(regex_string_replacement, args=(regex_dict['letters_in_the_beginning'],))
                                master_file.drop_duplicates(subset='Part_Ref_Step_1_5', inplace=True)  # ToDo: This is necessary due to duplicates in the Master File. There might also be problems regarding the descriptions. But this move is safe because by removing letters the reference still refer to the same part's family, and therefore have very similiar, if not equal, descriptions

                                _, matched_refs_df_step_1_5, non_matched_refs_df_1_5 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_1_5', 'Part_Desc_PT']], 'Second Match', left_key='Part_Ref_Step_1_5', right_key='Part_Ref_Step_1_5', validate_option='many_to_one')

                                matched_dfs.append(matched_refs_df_step_1_5)
                                non_matched_dfs = non_matched_refs_df_1_5

                                if non_matched_refs_df_1_5.shape[0]:
                                    non_matched_refs_df_1_5['Part_Ref_Step_2'] = non_matched_refs_df_1_5['Part_Ref_Step_1'].apply(string_volkswagen_preparation)
                                    master_file['Part_Ref_Step_2'] = master_file['Part_Ref_Step_1']  # Just for merging purposes

                                    # Match by Reference Reorder
                                    _, matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df_1_5, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'Reference Reorder', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2', validate_option='many_to_one')

                                    matched_dfs.append(matched_refs_df_step_2)
                                    non_matched_dfs = non_matched_refs_df_step_2

                                    if non_matched_refs_df_step_2.shape[0]:
                                        non_matched_refs_df_step_2['Part_Ref_Step_3'] = non_matched_refs_df_step_2['Part_Ref_Step_2'].apply(regex_string_replacement, args=(regex_dict['up_to_2_letters_at_end'],))
                                        master_file['Part_Ref_Step_3'] = master_file['Part_Ref_Step_2'].apply(regex_string_replacement, args=(regex_dict['up_to_2_letters_at_end'],))
                                        master_file.drop_duplicates(subset='Part_Ref_Step_3', inplace=True)  # ToDo: This is necessary due to duplicates in the Master File. There might also be problems regarding the descriptions. But this move is safe because by removing letters the reference still refer to the same part's family, and therefore have very similiar, if not equal, descriptions

                                        _, matched_refs_df_step_3, non_matched_refs_df_step_3 = references_merge(non_matched_refs_df_step_2, master_file[['Part_Ref_Step_3', 'Part_Desc_PT']], 'Removal of last word characters', left_key='Part_Ref_Step_3', right_key='Part_Ref_Step_3', validate_option='many_to_one')
                                        matched_dfs.append(matched_refs_df_step_3)
                                        non_matched_dfs = non_matched_refs_df_step_3

                            # ToDO Need to remove all the Part_Ref_Step_X created
                            matched_df_brand = pd.concat(matched_dfs)
                            current_stock_master_file, _, _ = references_merge(current_stock_master_file, matched_df_brand[['Part_Ref', 'Part_Desc_PT']], 'Master Stock Match', left_key='Part_Ref', right_key='Part_Ref')
                            # print(matched_df_brand.head())
                            # _, _ = references_merge(current_stock_master_file, matched_df_brand[['Part_Ref', 'Part_Desc_MF']], 'Master Stock File', left_key='Part_Ref', right_key='Part_Ref', validate_option='many_to_one')
                            # matched_df_brand['Part_Desc_Comparison'] = np.where(matched_df_brand['Part_Desc'] != matched_df_brand['Part_Desc_MF'], 1, 0)
                            # print('Different Descs: {}, Total Matched Desc: {}, Different Desc (%): {:.2f}'.format(matched_df_brand['Part_Desc_Comparison'].sum(), matched_df_brand.shape, matched_df_brand['Part_Desc_Comparison'].sum() / matched_df_brand.shape[0] * 100))
                            print('{} - Matched: {}'.format(master_file_brand, matched_df_brand['Part_Ref'].nunique()))
                            print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs['Part_Ref'].nunique()))
                            non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                    if master_file_brand == 'skoda':
                        print('### SKODA - {} Refs ###'.format(current_stock_master_file_filtered['Part_Ref'].nunique()))
                        matched_dfs = []

                        # Plain Match
                        _, matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file[['Part_Ref', 'Part_Desc_PT']], 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                        matched_dfs.append(matched_refs_df)
                        non_matched_dfs = non_matched_refs_df

                        if non_matched_refs_df.shape[0]:
                            # Removal of Brand_Code in the Reference:
                            non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(brand_code_removal, args=(brand_codes_list,)).apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))
                            master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))

                            _, matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'Brand Code Removal', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                            matched_dfs.append(matched_refs_df_step_2)
                            non_matched_dfs = non_matched_refs_df_step_2

                        matched_df_brand = pd.concat(matched_dfs)
                        current_stock_master_file, _, _ = references_merge(current_stock_master_file, matched_df_brand[['Part_Ref', 'Part_Desc_PT']], 'Master Stock Match', left_key='Part_Ref', right_key='Part_Ref')
                        # matched_df_brand['Part_Desc_Comparison'] = np.where(matched_df_brand['Part_Desc'] != matched_df_brand['Part_Desc_MF'], 1, 0)
                        # print('Different Descs: {}, Total Matched Desc: {}, Different Desc (%): {:.2f}'.format(matched_df_brand['Part_Desc_Comparison'].sum(), matched_df_brand.shape, matched_df_brand['Part_Desc_Comparison'].sum() / matched_df_brand.shape[0] * 100))
                        print('{} - Matched: {}'.format(master_file_brand, matched_df_brand['Part_Ref'].nunique()))
                        print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs['Part_Ref'].nunique()))
                        non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                    if master_file_brand == 'mercedes':
                        print('### {} - {} Refs ###'.format(master_file_brand, current_stock_master_file_filtered['Part_Ref'].nunique()))
                        print('brand_codes_list', brand_codes_list)
                        matched_dfs = []

                        # print('1\n', master_file.loc[master_file['Part_Ref'] == 'A2118800905'])
                        # Plain Match
                        _, matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file[['Part_Ref', 'Part_Desc_PT']], 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                        matched_dfs.append(matched_refs_df)
                        non_matched_dfs = non_matched_refs_df

                        if non_matched_refs_df.shape[0]:
                            # Removal of Brand_Code in the Reference:
                            non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(brand_code_removal, args=(brand_codes_list + ['A'],)).apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],)).apply(regex_string_replacement, args=(regex_dict['middle_strip'],)).apply(regex_string_replacement, args=(regex_dict['zero_at_end'],))
                            master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(brand_code_removal, args=(brand_codes_list + ['A'],)).apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],)).apply(regex_string_replacement, args=(regex_dict['middle_strip'],)).apply(regex_string_replacement, args=(regex_dict['zero_at_end'],))
                            # print('2\n', master_file.loc[master_file['Part_Ref'] == 'A2118800905'])

                            _, matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'Brand Code Removal', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                            matched_dfs.append(matched_refs_df_step_2)
                            non_matched_dfs = non_matched_refs_df_step_2

                            if non_matched_refs_df_step_2.shape[0]:
                                # Removal of Brand_Code in the Reference:
                                non_matched_refs_df_step_2['Part_Ref_Step_3'] = non_matched_refs_df_step_2['Part_Ref_Step_2'].apply(regex_string_replacement, args=(regex_dict['bmw_dot'],)).apply(regex_string_replacement, args=(regex_dict['right_bar'],)).apply(regex_string_replacement, args=(regex_dict['all_letters_at_beginning'],)).apply(regex_string_replacement, args=(regex_dict['bmw_AT_end'],))
                                master_file['Part_Ref_Step_3'] = master_file['Part_Ref_Step_2'].apply(regex_string_replacement, args=(regex_dict['bmw_dot'],)).apply(regex_string_replacement, args=(regex_dict['right_bar'],)).apply(regex_string_replacement, args=(regex_dict['all_letters_at_beginning'],))
                                # print('3\n', master_file.loc[master_file['Part_Ref'] == 'A2118800905'])

                                _, matched_refs_df_step_3, non_matched_refs_df_step_3 = references_merge(non_matched_refs_df_step_2, master_file[['Part_Ref_Step_3', 'Part_Desc_PT']], 'Reference Cleanup', left_key='Part_Ref_Step_3', right_key='Part_Ref_Step_3')
                                matched_dfs.append(matched_refs_df_step_3)
                                non_matched_dfs = non_matched_refs_df_step_3

                        matched_df_brand = pd.concat(matched_dfs)
                        current_stock_master_file, _, _ = references_merge(current_stock_master_file, matched_df_brand[['Part_Ref', 'Part_Desc_PT']], 'Master Stock Match', left_key='Part_Ref', right_key='Part_Ref')
                        # matched_df_brand['Part_Desc_Comparison'] = np.where(matched_df_brand['Part_Desc'] != matched_df_brand['Part_Desc_MF'], 1, 0)
                        # print('Different Descs: {}, Total Matched Desc: {}, Different Desc (%): {:.2f}'.format(matched_df_brand['Part_Desc_Comparison'].sum(), matched_df_brand.shape, matched_df_brand['Part_Desc_Comparison'].sum() / matched_df_brand.shape[0] * 100))
                        print('{} - Matched: {}'.format(master_file_brand, matched_df_brand['Part_Ref'].nunique()))
                        print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs['Part_Ref'].nunique()))
                        non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                    if master_file_brand == 'bmw':
                        print('### {} - {} Refs ###'.format(master_file_brand, current_stock_master_file_filtered['Part_Ref'].nunique()))
                        matched_dfs = []

                        # Plain Match
                        _, matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file[['Part_Ref', 'Part_Desc_PT']], 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                        matched_dfs.append(matched_refs_df)
                        non_matched_dfs = non_matched_refs_df

                        if non_matched_refs_df.shape[0]:
                            # Removal of Brand_Code in the Reference:
                            non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(brand_code_removal, args=(brand_codes_list + ['ZZ' + 'ZG' + 'MN'],)).apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],)).apply(regex_string_replacement, args=(regex_dict['bmw_AT_end'],)).apply(regex_string_replacement, args=(regex_dict['bmw_dot'],))
                            master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(brand_code_removal, args=(brand_codes_list + ['ZZ' + 'ZG' + 'MN'],)).apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))

                            _, matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'Brand Code Removal', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                            matched_dfs.append(matched_refs_df_step_2)
                            non_matched_dfs = non_matched_refs_df_step_2

                        matched_df_brand = pd.concat(matched_dfs)
                        current_stock_master_file, _, _ = references_merge(current_stock_master_file, matched_df_brand[['Part_Ref', 'Part_Desc_PT']], 'Master Stock Match', left_key='Part_Ref', right_key='Part_Ref')
                        # matched_df_brand['Part_Desc_Comparison'] = np.where(matched_df_brand['Part_Desc'] != matched_df_brand['Part_Desc_MF'], 1, 0)
                        # print('Different Descs: {}, Total Matched Desc: {}, Different Desc (%): {:.2f}'.format(matched_df_brand['Part_Desc_Comparison'].sum(), matched_df_brand.shape, matched_df_brand['Part_Desc_Comparison'].sum() / matched_df_brand.shape[0] * 100))
                        print('{} - Matched: {}'.format(master_file_brand, matched_df_brand['Part_Ref'].nunique()))
                        print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs['Part_Ref'].nunique()))
                        non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                    if master_file_brand == 'ford':
                        print('### {} - {} Refs ###'.format(master_file_brand, current_stock_master_file_filtered['Part_Ref'].nunique()))
                        matched_dfs = []

                        # Plain Match
                        _, matched_refs_df, non_matched_refs_df = references_merge(current_stock_master_file_filtered, master_file[['Part_Ref', 'Part_Desc_PT']], 'Plain Match', left_key='Part_Ref', right_key='Part_Ref')
                        matched_dfs.append(matched_refs_df)
                        non_matched_dfs = non_matched_refs_df

                        if non_matched_refs_df.shape[0]:
                            # Removal of Brand_Code in the Reference:
                            non_matched_refs_df['Part_Ref_Step_2'] = non_matched_refs_df['Part_Ref'].apply(brand_code_removal, args=(brand_codes_list,)).apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))
                            master_file['Part_Ref_Step_2'] = master_file['Part_Ref'].apply(brand_code_removal, args=(brand_codes_list,)).apply(regex_string_replacement, args=(regex_dict['zero_at_beginning'],))

                            _, matched_refs_df_step_2, non_matched_refs_df_step_2 = references_merge(non_matched_refs_df, master_file[['Part_Ref_Step_2', 'Part_Desc_PT']], 'Brand Code Removal', left_key='Part_Ref_Step_2', right_key='Part_Ref_Step_2')
                            matched_dfs.append(matched_refs_df_step_2)
                            non_matched_dfs = non_matched_refs_df_step_2

                        matched_df_brand = pd.concat(matched_dfs)
                        current_stock_master_file, _, _ = references_merge(current_stock_master_file, matched_df_brand[['Part_Ref', 'Part_Desc_PT']], 'Master Stock Match', left_key='Part_Ref', right_key='Part_Ref')
                        # matched_df_brand['Part_Desc_Comparison'] = np.where(matched_df_brand['Part_Desc'] != matched_df_brand['Part_Desc_MF'], 1, 0)
                        # print('Different Descs: {}, Total Matched Desc: {}, Different Desc (%): {:.2f}'.format(matched_df_brand['Part_Desc_Comparison'].sum(), matched_df_brand.shape, matched_df_brand['Part_Desc_Comparison'].sum() / matched_df_brand.shape[0] * 100))
                        print('{} - Matched: {}'.format(master_file_brand, matched_df_brand['Part_Ref'].nunique()))
                        print('{} - Non Matched: {}'.format(master_file_brand, non_matched_dfs['Part_Ref'].nunique()))
                        non_matched_dfs.to_csv('dbs/non_matched_{}.csv'.format(master_file_brand))

                else:
                    print('No data found for the Brand Code(s): {}'.format(brand_codes_list))

        # print('AFTER \n', current_stock_master_file.head(10))
        # print('AFTER Shape: ', current_stock_master_file.shape)
        # print('AFTER Unique Refs', current_stock_master_file['Part_Ref'].nunique())
        # print('AFTER Unique Refs', current_stock_master_file.drop_duplicates(subset='Part_Ref')['Part_Ref'].nunique())
        # print('AFTER Null Descriptions: ', current_stock_master_file['Part_Desc_PT'].isnull().sum())

        current_stock_master_file['Part_Desc_Merged'] = current_stock_master_file['Part_Desc'].fillna('') + ' ' + current_stock_master_file['Part_Desc_PT'].fillna('')  # I'll start by merging both descriptions
        current_stock_master_file = value_substitution(current_stock_master_file, non_null_column='Part_Desc', null_column='Part_Desc_PT')  # For references which didn't match in the Master Files, use the DW Description;
        current_stock_master_file.to_csv('dbs/current_stock_all_platforms_master_stock_matched_{}.csv'.format(time_tag_date), index=False)

    platform_ids = [dim_clients[dim_clients['BI_Database'] == x]['Client_Id'].values[0] for x in current_platforms]
    return [current_stock_master_file[current_stock_master_file['Client_Id'] == platform_id] for platform_id in platform_ids]


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


def references_merge(df, master_file, description_string, left_key, right_key, validate_option=None):
    step_time = time.time()
    final = df.merge(master_file, left_on=left_key, right_on=right_key, how='left')
    final['Part_Desc_PT'] = final['Part_Desc_PT_y'].fillna(final['Part_Desc_PT_x'])
    final.drop(['Part_Desc_PT_x', 'Part_Desc_PT_y'], axis=1, inplace=True)

    if left_key != right_key:
        final.drop([right_key], axis=1, inplace=True)

    non_matched = final.loc[final['Part_Desc_PT'].isnull(), :]
    matched = final.loc[final['Part_Desc_PT'].notnull(), :]

    non_matched_desc = final['Part_Desc_PT'].isnull().sum()
    matched_desc = final['Part_Desc_PT'].notnull().sum()
    non_matched_references = non_matched['Part_Ref'].nunique()
    matched_references = matched['Part_Ref'].nunique()

    print('{} - Matched Description/Non-Matched Description: {}/{}'.format(description_string, matched_desc, non_matched_desc))
    # print('{} - Matched References/Non-Matched References: {}/{} -  ({:.3f}s)'.format(description_string, matched_references, non_matched_references, time.time() - step_time))

    matched_df = final.loc[~final['Part_Desc_PT'].isnull(), :].copy()
    non_matched_df = final.loc[final['Part_Desc_PT'].isnull(), :].copy()

    return final, matched_df, non_matched_df


def brand_codes_retrieval(platforms, brand):
    query = ' UNION ALL '.join([options_file.brand_codes_per_franchise.format(x, y) for x, y in zip(platforms, [brand] * len(platforms))])
    df = sql_retrieve_df_specified_query(options_file.DSN, 'BI_AFR', options_file, query)
    brand_codes = list(np.unique(df['Original_Value'].values))

    return brand_codes


def data_acquisition(platforms, dim_product_group_file, dim_clients_file):
    platforms_stock = []

    try:
        for platform in platforms:
            df_current_stock = read_csv('dbs/df_{}_current_stock_{}_section_A.csv'.format(platform, sel_month), index_col=0, low_memory=False)
            print('File found for platform {}...'.format(platform))
            platforms_stock.append(df_current_stock)

        dim_product_group = read_csv(dim_product_group_file, index_col=0)
        dim_clients = read_csv(dim_clients_file, index_col=0)
    except FileNotFoundError:
        for platform in platforms:
            print('Retrieving from DW for platform {}...'.format(platform))
            df_current_stock = sql_retrieve_df_specified_query(options_file.DSN, options_file.sql_info['database_{}'.format(platform)], options_file, options_file.current_stock_query.format(platform, sel_month))
            platforms_stock.append(df_current_stock)

        dim_product_group = sql_retrieve_df_specified_query(options_file.DSN, options_file.sql_info['database_BI_AFR'], options_file, options_file.dim_product_group_query)
        dim_clients = sql_retrieve_df_specified_query(options_file.DSN, options_file.sql_info['database_BI_GSC'], options_file, options_file.dim_clients_query)

        save_csv(platforms_stock + [dim_product_group, dim_clients], ['dbs/df_{}_current_stock_{}_section_A'.format(platform, sel_month) for platform in platforms] + ['dbs/dim_product_group_section_A', 'dbs/dim_clients_section_A'])
    return platforms_stock, dim_product_group, dim_clients


def data_processing(df_platforms_stock, dim_product_group, dim_clients):
    platforms_stock_unique = []

    try:
        keyword_dictionary_df = read_csv('output/keywords_per_parts_family.csv', index_col=0)
        print('Keyword Dictionary Found...')
        for key, row in keyword_dictionary_df.iterrows():
            keywords_per_parts_family_dict[key] = [x for x in row.values if x is not np.nan]

        return

    except FileNotFoundError:

        for platform, df_platform_stock in zip(current_platforms, df_platforms_stock):
            print('Platform: {}'.format(platform))

            try:
                df_current_stock_unique = read_csv('dbs/df_{}_current_stock_unique_{}_section_B_step_2.csv'.format(platform, sel_month), dtype={part_desc_col: str})
                print('Current Stock already processed...')
                platforms_stock_unique.append(df_current_stock_unique)
            except FileNotFoundError:
                print('Processing stock...')

                # df_platform_stock = df_platform_stock.loc[df_platform_stock['Part_Ref'].isin(['5X0807217F', '1Z5839697D']), :]

                start_a = time.time()
                df_platform_current_stock = lowercase_column_conversion(df_platform_stock, [part_desc_col])  # Lowercases the strings of these columns
                df_platform_current_stock = literal_removal(df_platform_current_stock, part_desc_col)  # Removes literals from the selected column
                df_platform_current_stock[part_desc_col] = df_platform_current_stock[part_desc_col].apply(string_punctuation_removal)  # Removes punctuation from string
                df_platform_current_stock[part_desc_col] = df_platform_current_stock[part_desc_col].apply(unidecode_function)  # Removes accents marks from selected description column;
                df_platform_current_stock[part_desc_col] = df_platform_current_stock[part_desc_col].apply(string_digit_removal)  # Removes accents marks from selected description column;
                print('a - elapsed time: {:.3f}'.format(time.time() - start_a))

                start_b = time.time()
                df_platform_current_stock[part_desc_col] = df_platform_current_stock[part_desc_col].apply(abbreviations_correction, args=(options_file.abbreviations_dict, ))  # Replacement of abbreviations by the respective full word
                print('b - elapsed time: {:.3f}'.format(time.time() - start_b))

                start_c = time.time()
                df_platform_current_stock[part_desc_col] = df_platform_current_stock[part_desc_col].apply(stop_words_removal, args=(options_file.stop_words['Common_Stop_Words'] + options_file.stop_words[platform] + options_file.stop_words['Parts_Specific_Common_Stop_Words'] + nltk.corpus.stopwords.words('portuguese'),))  # Removal of Stop Words
                print('c - elapsed time: {:.3f}'.format(time.time() - start_c))

                step_c_1 = time.time()
                stemmer_pt_2 = RSLPStemmer()
                df_platform_current_stock[part_desc_col] = df_platform_current_stock[part_desc_col].apply(stemming, args=(stemmer_pt_2,))  # Stems each word using a Portuguese Stemmer
                print('c_1 - elapsed time: {:.3f}'.format(time.time() - step_c_1))

                start_d = time.time()
                df_current_stock_unique = description_merge_per_reference(df_platform_current_stock)  # Merges descriptions for the same reference and removes repeated tokens;
                print('d - elapsed time: {:.3f}'.format(time.time() - start_d))

                start_f = time.time()
                df_current_stock_unique = removed_zero_length_descriptions(df_current_stock_unique)  # Removed descriptions with no information (length = 0)
                print('f - elapsed time: {:.3f}'.format(time.time() - start_f))

                print('total time: {:.3f}'.format(time.time() - start_a))
                save_csv([df_current_stock_unique], ['dbs/df_{}_current_stock_unique_{}_section_B_step_2'.format(platform, sel_month)], index=False)
            finally:
                keyword_selection(platform, df_current_stock_unique)

        # print('Keyword Dictionary: \n', keyword_dict_per_parts_family)
        # print('Keyword Dictionary Number of Keys', len(list(keyword_dict_per_parts_family.keys())))
        pd.DataFrame.from_dict(keywords_per_parts_family_dict, orient='index').to_csv('output/keywords_per_parts_family.csv')

        for key in keywords_per_parts_family_dict.keys():
            print('Product Group DW: {}, Number of Words: {}, Words: {}'.format(key, len(keywords_per_parts_family_dict[key]), keywords_per_parts_family_dict[key]))

        return


def keyword_selection(platform, df):
    start_k_1 = time.time()
    train_dataset, test_dataset = pd.DataFrame(), pd.DataFrame()

    df_grouped_product_group_dw = df.loc[df['Product_Group_DW'] != 1, :].groupby('Product_Group_DW')
    # df_train_grouped = df_train.groupby('Product_Group_DW')
    for key, group in df_grouped_product_group_dw:
        unique_parts_count = group['Part_Ref'].nunique()
        unique_descs_count = group[part_desc_col].nunique()
        if unique_descs_count > 1:
            # print('Product_Group DW: {}, Number of unique parts: {}, Number of unique descriptions: {}'.format(key, unique_parts_count, unique_descs_count))

            # group_train = group.sample(frac=0.8, random_state=42)
            # group_train, group_test = train_test_split(group, train_size=0.8, random_state=42)
            group_train, group_test = train_test_split(group, train_size=0.8, stratify=group['Product_Group_DW'], random_state=42)
            train_dataset = train_dataset.append(group_train)
            test_dataset = test_dataset.append(group_test)

            descriptions = list(group[part_desc_col])

            top_words = most_common_words(descriptions)
            # most_common_bigrams(descriptions)
            # most_common_trigrams(descriptions)

            cv = CountVectorizer(min_df=0.05, max_df=0.8, max_features=10000, ngram_range=(1, 1))
            try:
                x = cv.fit_transform(descriptions)

                tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
                tfidf_transformer.fit(x)
                feature_names = cv.get_feature_names()

                tf_idf_vector = tfidf_transformer.transform(cv.transform(descriptions))

                sorted_items = sort_coo(tf_idf_vector.tocoo())
                # extract only the top n; n here is 10
                tf_idf_keywords = extract_topn_from_vector(feature_names, sorted_items, 5)
                # print('TF-IDF Words: {}'.format(list(tf_idf_keywords.keys())))

                # now print the results
                # print("\nAbstract:")
                # print(descriptions)
                # print("\nTF-IDF Keywords:")
                # for k in tf_idf_keywords:
                #     print(k, tf_idf_keywords[k])
                common_and_tf_idf_words = list(set(list(tf_idf_keywords.keys()) + top_words))
                # print('Top Common Words + TF-IDF Words: {}'.format(common_and_tf_idf_words))
                keyword_dict_creation(key, common_and_tf_idf_words)

            except ValueError:
                print('TF-IDF Error - Product_Group DW: {}, Number of unique parts: {}, Number of unique descriptions: {}'.format(key, unique_parts_count, unique_descs_count))
                pass

        else:
            print('Not enough information - Product_Group DW: {}, Number of unique parts: {}, Number of unique descriptions: {}'.format(key, unique_parts_count, unique_descs_count))
            pass

    train_dataset.to_csv('output/{}_train_dataset.csv'.format(platform))
    test_dataset.to_csv('output/{}_test_dataset.csv'.format(platform))
    print('k_1 - elapsed time: {:.3f}'.format(time.time() - start_k_1))

    return


def keyword_dict_creation(product_group, keywords):
    try:
        keywords_per_parts_family_dict[product_group] = keywords_per_parts_family_dict[product_group] + keywords
    except KeyError:
        keywords_per_parts_family_dict[product_group] = keywords

    return


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def most_common_words(descriptions):
    top_words = get_top_n_words(descriptions, n=20)
    top_words = [x[0] for x in top_words]

    # top_df = pd.DataFrame(top_words)
    # top_df.columns = ["Word", "Freq"]
    # print(top_words)

    # sns.set(rc={'figure.figsize': (13, 8)})
    # g = sns.barplot(x="Word", y="Freq", data=top_df)
    # g.set_xticklabels(g.get_xticklabels(), rotation=30)
    # plt.show()

    return top_words


def most_common_bigrams(descriptions):
    top2_words = get_top_n2_words(descriptions, n=20)
    top2_df = pd.DataFrame(top2_words)
    top2_df.columns = ["Bi-gram", "Freq"]
    print(top2_df)

    sns.set(rc={'figure.figsize': (13, 8)})
    h = sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
    h.set_xticklabels(h.get_xticklabels(), rotation=45)

    # plt.show()


def most_common_trigrams(descriptions):
    top3_words = get_top_n3_words(descriptions, n=20)
    top3_df = pd.DataFrame(top3_words)
    top3_df.columns = ["Tri-gram", "Freq"]
    print(top3_df)

    sns.set(rc={'figure.figsize': (13, 8)})
    j = sns.barplot(x="Tri-gram", y="Freq", data=top3_df)
    j.set_xticklabels(j.get_xticklabels(), rotation=45)

    # plt.show()


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2, 2), max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec1.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3, 3), max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec1.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def stop_words_removal(string_to_process, stop_words_list):
    # tokenized_string_to_process = nltk.tokenize.word_tokenize(string_to_process)

    new_string = ' '.join([x for x in nltk.tokenize.word_tokenize(string_to_process) if x not in stop_words_list])

    return new_string


def abbreviations_correction(string_to_correct, abbreviations_dict):
    # tokenized_string_to_correct = nltk.tokenize.word_tokenize(string_to_correct)

    string_corrected = ' '.join([abbreviations_dict[x] if x in abbreviations_dict.keys() else x for x in nltk.tokenize.word_tokenize(string_to_correct)])

    return string_corrected


def description_merge_per_reference(original_df):
    duplicated_refs = list(original_df.groupby('Part_Ref')['Part_Ref'].agg('count').where(lambda x: x > 1).dropna().index)

    selected_references_df = original_df.loc[original_df['Part_Ref'].isin(duplicated_refs), :]
    remaining_references_df = original_df.loc[~original_df['Part_Ref'].isin(duplicated_refs), :]

    selected_references_df = selected_references_df.groupby('Part_Ref').apply(duplicate_references_description_concatenation).drop_duplicates(subset='Part_Ref')
    remaining_references_df[part_desc_col] = remaining_references_df[part_desc_col].apply(string_repeated_words_removal)

    final_df = remaining_references_df.append(selected_references_df)

    return final_df


def removed_zero_length_descriptions(df):
    df = df[df[part_desc_col].map(len) > 0]  # ToDo this might work for now, but will need to upgrade it. What about '  ' or '    ' ?
    return df


def duplicate_references_description_concatenation(x):
    concatenated_description = ' '.join(list(x[part_desc_col]))
    concatenated_description_unique_words = ' '.join(unique_list_creation(concatenated_description.split()))

    x[part_desc_col] = concatenated_description_unique_words

    return x


def string_repeated_words_removal(string_to_process):
    processed_string = ' '.join(set(nltk.tokenize.word_tokenize(string_to_process)))

    return processed_string


def unique_list_creation(old_list):
    new_list = []
    [new_list.append(x) for x in old_list if x not in new_list]
    return new_list


if __name__ == '__main__':
    main()
