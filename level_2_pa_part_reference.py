import re
import sys
import time
import logging
import numpy as np
import pandas as pd
from traceback import format_exc
from datetime import datetime, timedelta
import dgo_parts_dataiku_dataset_preparation
import level_2_pa_part_reference_options as options_file
from level_2_pa_part_reference_options import regex_dict
from modules.level_1_a_data_acquisition import read_csv, sql_retrieve_df_specified_query, project_units_count_checkup, sql_retrieve_df
from modules.level_1_b_data_processing import master_file_processing, regex_string_replacement, brand_code_removal, string_volkswagen_preparation
from modules.level_1_e_deployment import save_csv, time_tags, sql_inject
from modules.level_0_performance_report import performance_info_append, performance_info, error_upload, log_record, project_dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S @ %d/%m/%y', filename=options_file.log_files['full_log'], filemode='a')
logging.Logger('errors')
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # Allows the stdout to be seen in the console
logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))  # Allows the stderr to be seen in the console

local_flag = 1
master_file_processing_flag = 0
current_platforms = ['BI_AFR', 'BI_CRP', 'BI_IBE', 'BI_CA']
part_desc_col = 'Part_Desc_PT'
# 'Part_Desc' - Part Description from DW
# 'Part_Desc_PT' - Part Description from Master Files when available, and from DW when not available
# 'Part_Desc_Merge' - Part Description from Master Files merged with Part Description from the DW


def main():
    log_record('Projeto: Classificação de Peças - DGO', options_file.project_id)

    if master_file_processing_flag:
        master_file_processing(options_file.master_files_to_convert)

    sel_month, _ = time_tags(format_date='%Y%m')
    previous_day = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')

    platforms_stock, dim_product_group, dim_clients = data_acquisition(current_platforms, 'dbs/dim_product_group_section_A.csv', 'dbs/dim_clients_section_A.csv', previous_day, sel_month)
    current_stock_master_file, truncate_flag = master_file_reference_match(platforms_stock, previous_day, dim_clients)

    deployment(current_stock_master_file, options_file.sql_info['database_final'], options_file.sql_info['final_table'], truncate_flag)

    # Part 2
    if current_stock_master_file.shape[0]:
        main_families_metrics_dict, other_families_metrics_dict = dgo_parts_dataiku_dataset_preparation.main()
        performance_message = 'Escolhido modelo LGBM. Performance: \nFamílias Principais: \n{} \nFamílias de Outros: \n{}'.format(''.join(['{}: {:.2f}\n'.format(key, value) for key, value in zip(main_families_metrics_dict.keys(), main_families_metrics_dict.values())]), ''.join(['{}: {:.2f}\n'.format(key, value) for key, value in zip(other_families_metrics_dict.keys(), other_families_metrics_dict.values())]))
    else:
        performance_message = 'Não foram encontradas novas linhas. Modelo não foi retreinado.'

    performance_info(options_file.project_id, options_file, performance_message, current_stock_master_file['Part_Ref'].nunique())

    log_record('Conclusão com sucesso - Projeto {}.\n'.format(project_dict[options_file.project_id]), options_file.project_id)


def deployment(df, db, view, truncate_flag):

    if df.shape[0] and df is not None:
        df.rename(columns={'Client_Id': 'Client_ID'}, inplace=True)

        df['PLR_Account'] = df['PLR_Account'].fillna("")
        df['Part_Desc_PT'] = df['Part_Desc_PT'].fillna("")
        df['Part_Desc'] = df['Part_Desc'].fillna("")
        if df is not None:
            sql_inject(df, options_file.DSN_MLG_PRD, db, view, options_file, options_file.sel_cols, truncate=truncate_flag, check_date=1)
    else:
        log_record('Sem novas linhas para inserir em {}.'.format(view), options_file.project_id, flag=1)

    return


def master_file_reference_match(platforms_stock, previous_day, dim_clients):

    try:
        current_stock_master_file = read_csv('dbs/current_stock_all_platforms_master_stock_matched_{}.csv'.format(previous_day))

        print('Descriptions already matched from brand master files...')
    except FileNotFoundError:
        current_stock_master_file = pd.concat(platforms_stock)
        current_stock_master_file['Part_Desc_PT'] = np.NaN

        for master_file_brands in options_file.master_file_brands:
            master_file = sql_retrieve_df(options_file.DSN_MLG_PRD, options_file.sql_info['database_final'], options_file.sql_info['master_file_table'], options_file, query_filters={'Brand': ', '.join(master_file_brands)})

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
                        non_matched_dfs.to_csv('dbs/non_matched_{}_{}.csv'.format(master_file_brand, previous_day))

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

                            # if non_matched_refs_df_step_2.shape[0]:
                                # Previous References Match
                                # _, matched_refs_df_step_3, non_matched_refs_df_step_3 = references_merge(non_matched_refs_df_step_2, master_file[['Supersession Leader', 'Part_Desc_PT']], 'Previous References Match', left_key='Part_Ref', right_key='Supersession Leader')
                                # matched_dfs.append(matched_refs_df_step_3)
                                # non_matched_dfs = non_matched_refs_df_step_3

                            if non_matched_refs_df_step_2.shape[0]:
                                # Error Handling
                                non_matched_refs_df_step_2['Part_Ref_Step_4'] = non_matched_refs_df_step_2['Part_Ref'].apply(regex_string_replacement, args=(regex_dict['remove_hifen'],)).apply(regex_string_replacement, args=(regex_dict['remove_last_dot'],))
                                master_file['Part_Ref_Step_4'] = master_file['Part_Ref'].apply(regex_string_replacement, args=(regex_dict['remove_hifen'],)).apply(regex_string_replacement, args=(regex_dict['remove_last_dot'],))
                                master_file.drop_duplicates(subset='Part_Ref_Step_4', inplace=True)

                                _, matched_refs_df_step_4, non_matched_refs_df_step_4 = references_merge(non_matched_refs_df_step_2, master_file[['Part_Ref_Step_4', 'Part_Desc_PT']], 'Erroneous References', left_key='Part_Ref_Step_4', right_key='Part_Ref_Step_4')
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
                        non_matched_dfs.to_csv('dbs/non_matched_{}_{}.csv'.format(master_file_brand, previous_day))

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
                        non_matched_dfs.to_csv('dbs/non_matched_{}_{}.csv'.format(master_file_brand, previous_day))

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
                        non_matched_dfs.to_csv('dbs/non_matched_{}_{}.csv'.format(master_file_brand, previous_day))

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
                        non_matched_dfs.to_csv('dbs/non_matched_{}_{}.csv'.format(master_file_brand, previous_day))

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
                        non_matched_dfs.to_csv('dbs/non_matched_{}_{}.csv'.format(master_file_brand, previous_day))

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
                        non_matched_dfs.to_csv('dbs/non_matched_{}_{}.csv'.format(master_file_brand, previous_day))

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
                    #     non_matched_dfs.to_csv('dbs/non_matched_{}_{}.csv'.format(master_file_brand, previous_day))

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
                            non_matched_dfs.to_csv('dbs/non_matched_{}_{}.csv'.format(master_file_brand, previous_day))

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
                        non_matched_dfs.to_csv('dbs/non_matched_{}_{}.csv'.format(master_file_brand, previous_day))

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
                        non_matched_dfs.to_csv('dbs/non_matched_{}_{}.csv'.format(master_file_brand, previous_day))

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
                        non_matched_dfs.to_csv('dbs/non_matched_{}_{}.csv'.format(master_file_brand, previous_day))

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
                        non_matched_dfs.to_csv('dbs/non_matched_{}_{}.csv'.format(master_file_brand, previous_day))

                else:
                    print('No data found for the Brand Code(s): {}'.format(brand_codes_list))

        # print('AFTER \n', current_stock_master_file.head(10))
        # print('AFTER Shape: ', current_stock_master_file.shape)
        # print('AFTER Unique Refs', current_stock_master_file['Part_Ref'].nunique())
        # print('AFTER Unique Refs', current_stock_master_file.drop_duplicates(subset='Part_Ref')['Part_Ref'].nunique())
        # print('AFTER Null Descriptions: ', current_stock_master_file['Part_Desc_PT'].isnull().sum())

        current_stock_master_file['Part_Desc_Merged'] = current_stock_master_file['Part_Desc'].fillna('') + ' ' + current_stock_master_file['Part_Desc_PT'].fillna('')  # I'll start by merging both descriptions
        # current_stock_master_file = value_substitution(current_stock_master_file, non_null_column='Part_Desc', null_column='Part_Desc_PT')  # For references which didn't match in the Master Files, use the DW Description;
        current_stock_master_file.to_csv('dbs/current_stock_all_platforms_master_stock_matched_{}.csv'.format(previous_day), index=False)

    previous_master_file = sql_retrieve_df(options_file.DSN_MLG_PRD, options_file.sql_info['database_final'], options_file.sql_info['final_table'], options_file, column_renaming=1)

    new_df = pd.concat([current_stock_master_file, previous_master_file]).drop('Last_Sell_Date', axis=1)
    new_df = new_df.drop_duplicates()
    new_df.to_csv('dbs/part_ref_master_file_matched.csv', index=False)

    try:
        project_units_count_checkup(new_df, 'Part_Ref', options_file, sql_check=1)
        truncate_flag = 1
    except ValueError:
        new_df = missing_part_refs_in_master_file_table(new_df, previous_master_file)
        truncate_flag = 0

    return new_df, truncate_flag


def missing_part_refs_in_master_file_table(new_df, previous_master_file):

    new_df.rename(columns={'Client_Id': 'Client_ID'}, inplace=True)
    new_df['PLR_Account'] = new_df['PLR_Account'].fillna("")
    new_df['Part_Desc_PT'] = new_df['Part_Desc_PT'].fillna("")
    new_df['Part_Desc'] = new_df['Part_Desc'].fillna("")
    new_df['Average_Cost'] = new_df['Average_Cost'].round(4)
    new_df['PVP_1'] = new_df['PVP_1'].round(4)

    previous_master_file.rename(columns={'Client_Id': 'Client_ID'}, inplace=True)

    df = pd.concat([new_df[options_file.sel_cols], previous_master_file[options_file.sel_cols]])

    # for col in ['Average_Cost', 'PVP_1']:
    #     df[col] = df[col].apply('{:0<7}'.format)

    for col in list(df):
        df[col] = df[col].astype(str)  # If i don't convert, col types will cause problems and be considered different when removing duplicates (p.ex. 10 and '10' will be considered differet values)

    # test_df = new_df.loc[(new_df['Part_Ref'] == '1102600Q0H') & (new_df['Part_Desc'] == 'WASHER-DRAIN PL') & (new_df['Average_Cost'] == 1.1147), options_file.sel_cols]
    # previous_df_test = previous_master_file.loc[(previous_master_file['Part_Ref'] == '1102600Q0H') & (previous_master_file['Part_Desc'] == 'WASHER-DRAIN PL') & (previous_master_file['Average_Cost'] == 1.1147), options_file.sel_cols]
    # set_df = df.loc[(df['Part_Ref'] == '1102600Q0H') & (df['Part_Desc'] == 'WASHER-DRAIN PL') & (df['Average_Cost'] == '1.1147'), options_file.sel_cols]

    # for col in list(test_df):
        # print(set(test_df[col]))
        # print(set(previous_df_test[col]))
        # print(set(set_df[col]))

    # print(set_df)
    # print(df.loc[(df['Part_Ref'] == '1102600Q0H')])
    # df['Part_Desc'] = df['Part_Desc'].apply(unidecode_function)
    # df['Part_Desc_PT'] = df['Part_Desc_PT'].apply(unidecode_function)
    # print(df.loc[(df['Part_Ref'] == '1102600Q0H')])
    # print(previous_master_file.loc[(previous_master_file['Part_Ref'] == '1102600Q0H') & (previous_master_file['Part_Desc'] == 'WASHER-DRAIN PL') & (previous_master_file['Average_Cost'] == '1.1147'), options_file.sel_cols])

    # print(df.loc[df['Part_Ref'] == '03L115562', options_file.sel_cols])
    # print(df.loc[df['Part_Ref'] == '04152YZZA4', options_file.sel_cols])

    df = df.drop_duplicates(keep=False, ignore_index=True)
    # print(df.loc[df['Part_Ref'] == '03L115562', options_file.sel_cols])
    # print(df.loc[df['Part_Ref'] == '04152YZZA4', options_file.sel_cols])

    return df


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
    df = sql_retrieve_df_specified_query(options_file.DSN_SRV3_PRD, 'BI_AFR', options_file, query)
    brand_codes = list(np.unique(df['Original_Value'].values))

    return brand_codes


def data_acquisition(platforms, dim_product_group_file, dim_clients_file, previous_day, sel_month):
    platforms_stock = []
    # The way that the table is built, is that it only contains last day data. So, for sel_month = 202011, run on 05/11/2020, the table will only have data for 04/11/2020. That's why in the query i can use the condition Stock_Month = '202011', but when saving the files, I use the day to distinguish between 04/11/2020 and 05/11/2020 and so on.

    try:
        for platform in platforms:
            df_current_stock = read_csv('dbs/df_{}_current_stock_{}_section_A.csv'.format(platform, previous_day), index_col=0, low_memory=False)
            print('File found for platform {}...'.format(platform))
            platforms_stock.append(df_current_stock)

        dim_product_group = read_csv(dim_product_group_file, index_col=0)
        dim_clients = read_csv(dim_clients_file, index_col=0)
    except FileNotFoundError:
        for platform in platforms:
            print('Retrieving from DW for platform {}...'.format(platform))
            df_current_stock = sql_retrieve_df_specified_query(options_file.DSN_SRV3_PRD, options_file.sql_info['database_{}'.format(platform)], options_file, options_file.current_stock_query.format(platform, platform, previous_day))
            platforms_stock.append(df_current_stock)

        dim_product_group = sql_retrieve_df_specified_query(options_file.DSN_SRV3_PRD, options_file.sql_info['database_BI_AFR'], options_file, options_file.dim_product_group_query)
        dim_clients = sql_retrieve_df_specified_query(options_file.DSN_SRV3_PRD, options_file.sql_info['database_BI_GSC'], options_file, options_file.dim_clients_query)

        save_csv(platforms_stock + [dim_product_group, dim_clients], ['dbs/df_{}_current_stock_{}_section_A'.format(platform, previous_day) for platform in platforms] + ['dbs/dim_product_group_section_A', 'dbs/dim_clients_section_A'])
    return platforms_stock, dim_product_group, dim_clients


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        project_identifier, exception_desc = options_file.project_id, str(sys.exc_info()[1])
        log_record(exception_desc, project_identifier, flag=2)
        error_upload(options_file, project_identifier, format_exc(), exception_desc, error_flag=1)
        log_record('Falhou - Projeto: {}.'.format(project_dict[project_identifier]), project_identifier)

