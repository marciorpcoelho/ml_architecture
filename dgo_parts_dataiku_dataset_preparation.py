import pandas as pd
import numpy as np
from joblib import dump
from sklearn import feature_extraction
import time
import nltk
import string
import pyodbc
from modules.level_1_b_data_processing import lowercase_column_conversion, trim_columns
from modules.level_1_e_deployment import sql_inject
from modules.level_1_a_data_acquisition import sql_retrieve_df, sql_retrieve_df_specified_query
from modules.level_1_d_model_evaluation import algorithms_performance_dataset_creation, model_performance_saving
from modules.level_0_performance_report import log_record
import level_2_pa_part_reference_options as options_file
from dgo_parts_models_training import model_training
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

exact_matches_dict = {
    'elemento filtro oleo': '10',
    'esteira de insonorização': '42',
    'airbag': '30',
    'kit airbag': '30',
    'grelha esquerda': '34',
    'grelha': '34',
    'grelha,dianteira': '34',
    'grelha dianteira': '34',
    'autorradio': '57',
    'cera cavidades': '43',
    'kit de conservación cuero con prot. uv': '43',
    'higienizador': '43',
    'limpa-vidros': '43',
    'limpa-vidros concentrado': '43',
    'produto de limpeza para travões 2.0': '43',
    'aditivo hc mate': '44',
    'tuberia de aditivo': '29',
    'bomba manual adblue': '162',
    'cabo rep. airbag protecção dos peões': '42',
    'cabo rep.airbag cabeça its/unid.comando': '42',
    'cabo rep. airbag condutor/unid.comando': '42',
    'cabo rep. p/ airbag da cabeça its': '42',
    'cabo reparação airbag': '42',
    'camara parking': '150',
    'camara retroceso': '150',
    'camara replay': '147',
    'cadeado': '42',
    'lampada': '12',
    'lampara': '12',
    'altavoz bluetooth azul': '62',
    'grasa de vaselina': '48',
    'chaine': '5',
}
partial_matches_dict = {
    "^injector.{0,}|^inyector.{0,}|^injecteur.{0,}":
        "24",
    "^disco.{0,}\\strav.{0,}":
        "99",
    "^filtr.{0,}\\sole.+|^filtr.{0,}\\sóle.+":
        "10",
    "^jg\\smaxila.{0,}":
        "100",
    "^j.maxila.{0,}":
        "100",
    "^junta\\sde\\sculat.{0,}":
        "29",
    "^junta\\sde\\sculass.{0,}":
        "29",
    "^catalog.{0,}":
        "51",
    "^manual.{0,}":
        "51",
    "^colete.+ref.{0,}":
        "125",
    ".{0,}ad\\s{0,}blue.{0,}":
        "96",
    "^aditivo.{0,}":
        "102",
    "^disco\\s{0,}embraiagem.{0,}":
        "6",
    "^esteira\\s{0,}sen.{0,}":
        "170",
    ".{0,}gonflable.{0,}":
        "30",
    "^airbag.{0,}":
        "30",
    "^cobertura.{0,}airbag.{0,}|caperuza.{0,}airbag.{0,}":
        "42",
    ".{0,}sensor\\w{0,2}\\s{0,}parking.{0,}":
        "68",
    ".{0,}parking\\s{0,}aid.{0,}":
        "68",
    ".{0,}buzzer.{0,}":
        "68",
    ".{0,}alarm[e|a]\\s.{0,}":
        "56",
    ".{0,}alarm\\s.{0,}":
        "56",
    "^alternador.{0,}":
        "17",
    "^alternateur.{0,}":
        "17",
    "^alternator.{0,}":
        "17",
    "^altifalant[e|es].{0,}":
        "57",
    "^autoradio\\s.{0,}":
        "70",
    "^agente\\s{1,}fri.{0,}":
        "179",
    "^agente\\s{1,}ref.{0,}":
        "179",
    "^ambientado.{0,}":
        "43",
    "^kit\\s{0,}caneta.{0,}":
        "43",
    "^limpa\\svidros.{0,}|^liquido\\s{0,}limpa\\svidros.{0,}|^líquido\\s{0,}limpa\\svidros.{0,}":
        "7",
    "^higienizante\\s{0,}.{0,}":
        "46",
    "^vaselina\\s{0,}.{0,}":
        "48",
    "^spray\\s{0,}limpeza.{0,}|^spray\\s{0,}de\\s{0,}limpeza.{0,}":
        "46",
    "^adhesivo.{0,}":
        "48",
    "^kit\\s{0,}dist.{0,}":
        "97",
    "^chaine\\s{0,}d.{0,}":
        "97",
    "^correa\\s{0,}dist.{0,}":
        "97",
    "^correia\\s{0,}dist.{0,}":
        "97",
    "^corrente\\s{0,}dist.{0,}":
        "97",
    "^cadena\\s{0,}dist.{0,}|^cadena\\s{0,}de\\s{0,}la\\s{0,}dist.{0,}":
        "97",
    "^key\\s{0,}chain.{0,}":
        "77",
    "^additif.{0,}": "102"
}
parts_families_replacement = {
    '75': '75/77',
    '77': '75/77',
    '175': '29',
}
others_families_dict_str = {
    '147': 'O. Acessórios',
    '148': 'O. Acessórios',
    '149': 'O. Acessórios',
    '150': 'O. Acessórios',
    '152': 'O. Acessórios',
    '42': 'O. Colisão',
    '37': 'O. Colisão',
    '48': 'O. Consumíveis',
    '119': 'O. Manutenção',
    '78': 'O. Merchandising',
    '16': 'O. Manutenção',
    '144': 'O. Mota',
    '143': 'O. Mota',
    '170': 'O. Reparação',
    '29': 'O. Reparação',
    '172': 'O. Reparação',
    '173': 'O. Reparação',
    '54': 'O. Diversos',
    '75/77': 'Lazer/Marroquinaria',
}


def main():
    dgo_family_10_loc = 'dbs/dgo_familia_10_prepared.csv'
    dgo_family_13_loc = 'dbs/dgo_familia_13_prepared.csv'
    dgo_manual_classified_parts_loc = 'dbs/dgo_classified_parts.csv'

    log_record('Step 1 started.', options_file.project_id)
    master_file = sql_retrieve_df(options_file.DSN_MLG_PRD, options_file.sql_info['database_final'], options_file.sql_info['final_table'], options_file, column_renaming=1)
    master_file['Product_Group_DW'] = master_file['Product_Group_DW'].astype(str)
    log_record('Step 1 ended.', options_file.project_id)
    master_file = flow_step_2(master_file)
    control_prints(master_file, '2')
    master_file, keywords_df = flow_step_3(master_file)
    control_prints(master_file, '3')
    master_file = flow_step_4(master_file)
    control_prints(master_file, '4')
    master_file = flow_step_4_5(master_file, keywords_df)
    control_prints(master_file, '4.5')
    master_file, manual_classifications = flow_step_5(master_file, [dgo_family_10_loc, dgo_family_13_loc, dgo_manual_classified_parts_loc])
    control_prints(master_file, '5')
    master_file_non_classified, master_file_other_families, master_file_classified_families = flow_step_6(master_file)
    control_prints(master_file_non_classified, '6a')
    control_prints(master_file_classified_families, '6b')
    control_prints(master_file_other_families, '6c')
    master_file_classified_families_filtered = flow_step_7(master_file_classified_families)
    control_prints(master_file_classified_families_filtered, '7 - master_file_classified_families_filtered')
    master_file_other_families_filtered = flow_step_7(master_file_other_families)
    control_prints(master_file_other_families_filtered, '7 - master_file_other_families_filtered')
    master_file_final, main_families_cm, main_families_metrics_dict, other_families_cm, other_families_metrics_dict = flow_step_8(master_file_classified_families_filtered, master_file_other_families_filtered, master_file_non_classified)
    control_prints(master_file_final, '8')
    master_file_final = flow_step_9(master_file_final)
    control_prints(master_file_final, '9')
    master_file_final = flow_step_10(master_file_final, manual_classifications)
    control_prints(master_file_final, '10')
    master_file_final.to_csv('dbs/master_file_final.csv', index=False)
    deployment(master_file_final, main_families_cm, other_families_cm)

    return main_families_metrics_dict, other_families_metrics_dict


def control_prints(df, tag):
    # print(tag)
    # print(df.shape)
    # try:
    #     print(df['Part_Ref'].value_counts())
    # except KeyError:
    #     pass
    return


# compute_current_stock_all_platforms_master_stock_matched_04_2020_prepared_distinct
def flow_step_2(df):
    log_record('Step 2 started.', options_file.project_id)

    df = df.drop_duplicates(subset=['Part_Ref', 'Part_Desc', 'Product_Group_DW', 'Client_Id', 'Average_Cost', 'PVP_1', 'PLR_Account', 'Part_Desc_PT'])

    log_record('Step 2 ended.', options_file.project_id)
    return df


# compute_Dataset_w_Count_prepared
def flow_step_3(df):
    log_record('Step 3 started.', options_file.project_id)

    # Step 0
    lower_case_cols = ['Part_Desc_PT', 'Part_Desc']
    df = lowercase_column_conversion(df.copy(), lower_case_cols)

    # Step 1
    df = trim_columns(df.copy(), lower_case_cols)

    # Step 1.5
    text_cols = ['Part_Desc_PT', 'Part_Desc']
    for col in text_cols:
        df[col] = df[col].apply(remove_punctuations)
        df[col] = df[col].apply(lambda x: ' '.join([word for word in x.split() if word not in options_file.stop_words_common]))

    # Step 2
    df = feature_engineering_counts(df)
    # df = feature_engineering_part_desc(df)

    # Step 3
    # Removes numbers
    df['Part_Desc_PT'] = df['Part_Desc_PT'].str.replace(r'\d+', ' ')
    df['Part_Desc'] = df['Part_Desc'].str.replace(r'\d+', ' ')

    # Step 3.5
    # Creates keywords data
    part_desc_keywords = keyword_generation(df, 'Part_Desc')
    part_desc_pt_keywords = keyword_generation(df, 'Part_Desc_PT')
    keywords = pd.concat([part_desc_keywords, part_desc_pt_keywords])
    keywords.to_csv('dbs/keywords_per_product_group_dw.csv', index=False)

    # Step 4
    df = product_group_dw_corrections_on_desc(df.copy())

    # Step 5
    rows_to_remove_regex_filter_1 = r'(?<=^B.{11})AT$|(?<=^B.{15})AT$'
    rows_to_remove_regex_filter_2 = r'(?<=^BM.{11})AT$|(?<=^BM.{15})AT$'
    filter_3 = df['Part_Ref'].str.contains(rows_to_remove_regex_filter_1, na=False)
    filter_4 = df['Part_Ref'].str.contains(rows_to_remove_regex_filter_2, na=False)
    df = df[~filter_3 & ~filter_4]

    # Step 6
    df.loc[df['Part_Desc'] == df['Part_Desc_Copy'], 'Part_Desc_Copy'] = np.nan
    # df.loc[df['Part_Desc'] != df['Part_Desc_Copy'], 'Part_Desc_Copy'] = df['Product_Group_DW']
    df.loc[df['Part_Desc'] != df['Part_Desc_Copy'], 'Part_Desc_Copy'] = df['Part_Desc_Copy']

    # Step 7
    df.loc[df['Part_Desc_Copy'].isnull(), 'New_Product_Group_DW'] = df.loc[df['Part_Desc_Copy'].isnull(), 'Product_Group_DW']
    df.loc[~df['Part_Desc_Copy'].isnull(), 'New_Product_Group_DW'] = df.loc[~df['Part_Desc_Copy'].isnull(), 'Part_Desc_Copy']

    # Step 8
    df.drop(['Product_Group_DW', 'Part_Desc_Copy'], axis=1, inplace=True)

    # Step 9
    df.rename(columns={'New_Product_Group_DW': 'Product_Group_DW'}, inplace=True)

    # Step 10
    df = product_group_dw_complete_replaces(df.copy())

    log_record('Step 3 ended.', options_file.project_id)
    return df, keywords


def keyword_generation(df, text_col):

    final_df = pd.DataFrame()
    df_grouped = df[df['Product_Group_DW'] != '1'].groupby('Product_Group_DW')

    for key, group in df_grouped:
        _, df_words = get_ngrams_from_df(group, text_col, 3, ngrams_degree=2)
        df_words['Column'] = text_col

        final_df = pd.concat([final_df, df_words])

    final_df.to_csv('dbs/product_group_dw_{}_keywords.csv'.format(text_col), index=False)

    return df


def detect_keywords(df, col, list_words):

    lst_grams = [len(word.split(" ")) for word in list_words]
    vectorizer = feature_extraction.text.CountVectorizer(
        vocabulary=list_words,
        ngram_range=(min(lst_grams), max(lst_grams)))
    dtf_X = pd.DataFrame(vectorizer.fit_transform(df[col]).todense(), columns=[col + '_' + word for word in list_words])
    ## add the new features as columns
    df = pd.concat([df, dtf_X.set_index(df.index)], axis=1)

    return df


def get_ngrams_from_df(df, desc_col, top_ngrams_count, ngrams_degree=1):
    corpus = df[desc_col]
    lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))
    dict_words = {}
    df_words = pd.DataFrame()

    # Unigrams
    dic_words_freq = nltk.FreqDist(lst_tokens)
    df_words = pd.concat([df_words, pd.DataFrame(dic_words_freq.most_common(top_ngrams_count), columns=["Word", "Freq"])])
    df_words['ngram_degree'] = 1
    dict_words[1] = [word[0] for word in dic_words_freq.most_common(top_ngrams_count)]
    # print('1-grams: {}'.format(dict_words[1]))

    if ngrams_degree > 1:
        for ngram_degree in range(2, ngrams_degree+1):
            dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, ngrams_degree))
            dtf = pd.DataFrame(dic_words_freq.most_common(top_ngrams_count), columns=["Word", "Freq"])
            dtf["Word"] = dtf["Word"].apply(lambda x: " ".join(string for string in x))
            dtf['ngram_degree'] = ngram_degree
            df_words = pd.concat([df_words, dtf])
            list_words_ngram = [" ".join(string for string in word[0]) for word in dic_words_freq.most_common(top_ngrams_count)]
            # print('{}-grams: {}'.format(ngrams_degree, list_words_ngram))
            dict_words[ngram_degree] = list_words_ngram

    df_words.sort_values(by=['Freq', 'ngram_degree'], ascending=False, inplace=True)
    return dict_words, df_words


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


def feature_engineering_counts(df):
    # Get information from the part reference

    # Letters/numbers/space/etc count:
    start = time.time()
    df = digit_counts(df, 'Part_Ref')
    df = digit_counts(df, 'Part_Desc')
    df = digit_counts(df, 'Part_Desc_PT')
    print('Digit Counts Elapsed time: {:.2f}s'.format(time.time() - start))

    start_2 = time.time()
    df = brand_codes_selection(df, 'Part_Ref')
    print('Brand Codes Selection Elapsed time: {:.2f}s'.format(time.time() - start_2))

    return df


def feature_engineering_part_desc(df):

    part_desc_similarity(df, 'Part_Desc')
    part_desc_similarity(df, 'Part_Desc_PT')

    return df


def part_desc_similarity(df, col):

    df = get_data_product_group_sql(options_file.others_families_dict, options_file)
    stop_words_pt = ['de', 'da', 'e', 'O.', '+', '/', '-']
    df['PT_Product_Group_Desc_clean'] = df['PT_Product_Group_Desc'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words_pt]))
    unique_family_names = list(df['PT_Product_Group_Desc_clean'].unique())


def get_data_product_group_sql(others_dict, options_file_in):
    df = sql_retrieve_df_specified_query(options_file_in.DSN_SRV3_PRD, options_file_in.sql_info['database_BI_AFR'], options_file_in, options_file_in.product_group_complete_app_query)
    df['Product_Group_Code'] = df['Product_Group_Code'].astype('str')

    df = df[df['Product_Group_Code'] != '77']
    df.loc[df['Product_Group_Code'] == '75', 'Product_Group_Code'] = '75/77'
    df.loc[df['PT_Product_Group_Desc'] == 'Lazer', 'PT_Product_Group_Desc'] = 'Lazer/Marroquinaria'

    for key in others_dict.keys():
        df.loc[df['Product_Group_Code'] == str(key), 'PT_Product_Group_Desc'] = others_dict[key]

    return df


def digit_counts(df, col):

    df[col + '_length'] = df.apply(lambda x: len(x[col]), axis=1)
    df[col + '_word_count'] = df.apply(lambda x: len(str(x[col]).split(" ")), axis=1)
    df[col + '_digits_count'] = df.apply(lambda x: sum(c.isdigit() for c in x[col]), axis=1)
    df[col + '_letters_count'] = df.apply(lambda x: sum(c.isalpha() for c in x[col]), axis=1)
    df[col + '_spaces_count'] = df.apply(lambda x: sum(c.isspace() for c in x[col]), axis=1)
    df[col + '_others_count'] = df.apply(lambda x: len(x[col]) - x[col + '_digits_count'] - x[col + '_letters_count'] - x[col + '_spaces_count'], axis=1)

    return df


def brand_codes_detection(x, brand_codes_regex_dict, col):
    x['brand'] = x[[col]].str.extract(brand_codes_regex_dict[str(x['Client_Id'])], expand=False)[0]
    return x


def brand_codes_selection(df, col):
    brand_codes_regex_dict = brand_codes_retrieval()
    df = df.apply(lambda x: brand_codes_detection(x, brand_codes_regex_dict, col), axis=1)
    df['brand'] = df['brand'].fillna('N/A')

    return df


def brand_codes_retrieval():
    platforms = ['BI_AFR', 'BI_CRP', 'BI_IBE', 'BI_CA']
    query = ' UNION ALL '.join([options_file.brand_codes_per_platform.format(x) for x in platforms])
    brand_codes_df = sql_retrieve_df_specified_query(options_file.DSN_SRV3_PRD, 'BI_AFR', options_file, query)
    brand_codes_df['code_len'] = brand_codes_df['Franchise_Code_DMS'].str.len()
    brand_codes_df.sort_values(by='code_len', ascending=False, inplace=True)

    # brand_codes_platform = brand_codes_df[brand_codes_df['Client_ID'] == int(x['Client_ID'])]
    brand_codes_regex_dict = {}
    for client_id in brand_codes_df['Client_ID'].unique():
        filtered_df = brand_codes_df[brand_codes_df['Client_ID'] == client_id]
        unique_brands = filtered_df['Franchise_Code_DMS'].unique()
        regex_rules = 'r^(' + '|'.join(unique_brands) + ')'
        brand_codes_regex_dict[str(client_id)] = regex_rules

    return brand_codes_regex_dict


# compute_Dataset_w_Count_prepared_by_Part_Ref_2_1_1
def flow_step_4(df):
    log_record('Step 4 started.', options_file.project_id)

    # Step 0 (not in dataiku)
    previous_size = df.shape[0]
    df.dropna(subset=['Product_Group_DW'], inplace=True)
    if previous_size - df.shape[0] > 1:
        print('Problem with NaN in product_group_dw...')

    df['Part_Desc'].fillna('', inplace=True)
    df['Part_Desc_PT'].fillna('', inplace=True)
    df['Client_Id'].fillna('', inplace=True)
    df['PLR_Account'].fillna('', inplace=True)

    df = df.groupby('Part_Ref').apply(group_by_rules)
    df.reset_index(inplace=True)
    df.to_csv('dbs/df_after_flow_step_4.csv', index=False)

    log_record('Step 4 ended.', options_file.project_id)
    return df


def flow_step_4_5(df, keywords_df):
    log_record('Step 4.5 started', options_file.project_id)
    keywords_col_1 = list(set(keywords_df.loc[keywords_df['Column'] == 'Part_Desc', 'Word']))
    keywords_col_2 = list(set(keywords_df.loc[keywords_df['Column'] == 'Part_Desc_PT', 'Word']))

    # Detects keywords data
    df = detect_keywords(df, 'Part_Desc_concat', keywords_col_1)
    df = detect_keywords(df, 'Part_Desc_PT_concat', keywords_col_2)

    log_record('Step 4.5 ended.', options_file.project_id)
    return df


# compute_dataset_grouped_corrected
def flow_step_5(df, manual_classified_files_loc):
    log_record('Step 5 started.', options_file.project_id)

    # Step 0 - Merge all manual/application classifications
    manual_classified_families = get_dgo_manual_classifications(manual_classified_files_loc)
    pse_fact_pa_parts_classification_refs = get_fact_pa_classifications()
    all_classifications = pd.concat([manual_classified_families, pse_fact_pa_parts_classification_refs])

    # Step 1
    df = classification_corrections_start(df, all_classifications)

    df = product_group_dw_complete_replaces(df.copy())

    log_record('Step 5 ended.', options_file.project_id)
    return df, all_classifications


# split_Dataset_w_Count_prepared_by_Part_Ref_1
def flow_step_6(df):
    log_record('Step 6 started.', options_file.project_id)

    other_families = ['O. Acessórios', 'O. Colisão', 'O. Consumíveis', 'O. Manutenção', 'O. Merchandising', 'O. Mota', 'O. Reparação', 'O. Diversos']

    df_non_classified = df[df['Product_Group_DW'] == '1']
    df_other_families = df[(df['Product_Group_DW'] == 'O. Acessórios') |
                            (df['Product_Group_DW'] == 'O. Colisão') |
                            (df['Product_Group_DW'] == 'O. Consumíveis') |
                            (df['Product_Group_DW'] == 'O. Manutenção') |
                            (df['Product_Group_DW'] == 'O. Merchandising') |
                            (df['Product_Group_DW'] == 'O. Mota') |
                            (df['Product_Group_DW'] == 'O. Reparação') |
                            (df['Product_Group_DW'] == 'O. Diversos')]
    df_classified_families = df[~df['Product_Group_DW'].isin(other_families + ['1'])]

    log_record('Step 6 ended.', options_file.project_id)
    return df_non_classified, df_other_families, df_classified_families


# compute_full_dataset_min_count_part_ref_per_product_group_dw
# compute_full_dataset_min_count_part_ref_per_product_group_dw_2
def flow_step_7(df):
    log_record('Step 7 started.', options_file.project_id)

    full_dataset_df = df
    value_counts_series = full_dataset_df['Product_Group_DW'].value_counts()
    value_counts_series_filter = value_counts_series[value_counts_series >= 100].index.values

    full_dataset_min_count_part_ref_per_product_group_dw_df = full_dataset_df[full_dataset_df['Product_Group_DW'].isin(value_counts_series_filter)]

    log_record('Step 7 ended.', options_file.project_id)
    return full_dataset_min_count_part_ref_per_product_group_dw_df


def flow_step_8(master_file_classified_families_filtered, master_file_other_families_filtered, master_file_non_classified):
    log_record('Step 8 started.', options_file.project_id)

    starting_cols = [x for x in list(master_file_classified_families_filtered) if x not in ['Part_Ref']]

    _, main_families_clf, main_families_cm_train, main_families_cm_test, main_families_metrics_dict = model_training(master_file_classified_families_filtered, starting_cols, 'main_families_clf')  # Modelo conhece 50 familias
    dump(main_families_clf, options_file.main_families_clf_path)
    _, other_families_clf, other_families_cm_train, other_families_cm_test, other_families_metrics_dict = model_training(master_file_other_families_filtered, starting_cols, 'other_families_clf')  # Modelo conhece 8 familias
    dump(other_families_clf, options_file.other_families_clf_path)

    # print('Main Families CM (Test): \n{}'.format(main_families_cm_test))
    main_families_cm_test.to_csv('dbs/main_families_cm_temp.csv')
    # print('Other Families CM (Test): \n{}'.format(other_families_cm_test))
    other_families_cm_test.to_csv('dbs/other_families_cm_tmp.csv')

    # First Classification
    master_file_scored, _, _, _, _ = model_training(pd.concat([master_file_classified_families_filtered, master_file_other_families_filtered, master_file_non_classified]), starting_cols, 'second_main_families_clf', clf=main_families_clf)
    master_file_scored = prob_thres_col_creation(master_file_scored)

    # First 0.5 CutOff
    master_file_scored_over_50 = master_file_scored[master_file_scored['Max_Prob'] > 0.5]
    print('first classification, over 50 shape:', master_file_scored_over_50.shape)
    master_file_scored_sub_50 = master_file_scored[master_file_scored['Max_Prob'] <= 0.5]
    print('first classification, sub 50 shape:', master_file_scored_sub_50.shape)

    # Second Classification
    master_file_sub_50_scored, _, _, _, _ = model_training(master_file_scored_sub_50[starting_cols], starting_cols, 'second_others', clf=other_families_clf)
    master_file_sub_50_scored = prob_thres_col_creation(master_file_sub_50_scored)

    print('master_file_scored_over_50')
    print(list(master_file_scored_over_50))
    print(len(list(master_file_scored_over_50)))
    print('duplicated master_file_scored_over_50: {}'.format(master_file_scored_over_50.columns.duplicated()))

    print('master_file_sub_50_scored')
    print(list(master_file_sub_50_scored))
    print(len(list(master_file_sub_50_scored)))
    print('duplicated master_file_sub_50_scored: {}'.format(master_file_sub_50_scored.columns.duplicated()))

    master_file_final = pd.concat([master_file_scored_over_50, master_file_sub_50_scored])
    print(master_file_final.shape)

    models_metrics_dict = {
        'lgb_main_families': main_families_metrics_dict,
        'lgb_other_families': other_families_metrics_dict
    }

    performance_saving(models_metrics_dict)

    log_record('Step 8 ended.', options_file.project_id)
    return master_file_final, main_families_cm_test, main_families_metrics_dict, other_families_cm_test, other_families_metrics_dict


def performance_saving(models_metrics_dict):

    results_test = []
    models = models_metrics_dict.keys()

    for model in models:
        results_test.append(models_metrics_dict[model])
        df_results = algorithms_performance_dataset_creation(results_test[0], 'Test', 'lgb', options_file.project_id)
        model_performance_saving(df_results, options_file)

    return


def flow_step_9(df):
    log_record('Step 9 started.', options_file.project_id)

    # Step 1
    df = product_group_dw_corrections_on_desc_end(df.copy())

    # Step 2
    df.loc[:, 'Product_Group_DW'] = df['Product_Group_DW'].replace(others_families_dict_str)

    # Step 3
    df = probabilities_correction(df)

    # Step 4
    df.loc[df['Part_Desc_concat'] == df['Part_Desc_Copy'], 'Part_Desc_Copy'] = np.nan
    df.loc[df['Part_Desc_concat'] != df['Part_Desc_Copy'], 'Part_Desc_Copy'] = df['Part_Desc_Copy']

    # Step 5
    df.loc[df['Part_Desc_Copy'].isnull(), 'New_Product_Group_DW'] = df.loc[df['Part_Desc_Copy'].isnull(), 'Product_Group_DW']
    df.loc[~df['Part_Desc_Copy'].isnull(), 'New_Product_Group_DW'] = df.loc[~df['Part_Desc_Copy'].isnull(), 'Part_Desc_Copy']

    # Step 8
    df.drop(['Product_Group_DW', 'Part_Desc_Copy'], axis=1, inplace=True)

    # Step 9
    df.rename(columns={'New_Product_Group_DW': 'Product_Group_DW'}, inplace=True)

    # Step 10
    cols = list(df)
    df = df[[x for x in cols if not x.startswith('probability_')]]
    log_record('Step 9 ended.', options_file.project_id)
    return df


def flow_step_10(df, manual_classifications):
    log_record('Step 9 started.', options_file.project_id)

    df = classification_corrections_end(df, manual_classifications)

    log_record('Step 10 ended.', options_file.project_id)
    return df


def prob_thres_col_creation(df):
    proba_cols = [x for x in list(df) if x.startswith('proba')]

    df['Max_Prob'] = df[proba_cols].max(axis=1)
    df['New_Prediction_50'] = np.where(df['Max_Prob'] <= 0.5, 1, df['prediction'])
    df['New_Prediction_80'] = np.where(df['Max_Prob'] < 0.8, 1, df['prediction'])
    df['New_Prediction_85'] = np.where(df['Max_Prob'] < 0.85, 1, df['prediction'])
    df['New_Prediction_90'] = np.where(df['Max_Prob'] < 0.9, 1, df['prediction'])
    df['New_Prediction_95'] = np.where(df['Max_Prob'] < 0.95, 1, df['prediction'])
    df = df.drop(proba_cols, axis=1)

    return df


def classification_corrections_start(df, corrections):
    dataset_grouped_filtered_df = df
    pse_fact_pa_parts_classification_refs_df = corrections

    col_lists = list(dataset_grouped_filtered_df)
    dataset_grouped_corrected_df = dataset_grouped_filtered_df.merge(pse_fact_pa_parts_classification_refs_df, left_on=['Part_Ref'], right_on=['Part_Ref'], how='left')
    dataset_grouped_corrected_df.loc[dataset_grouped_corrected_df['New_Product_Group_DW'].isnull(), 'New_Product_Group_DW'] = dataset_grouped_corrected_df.loc[dataset_grouped_corrected_df['New_Product_Group_DW'].isnull(), 'Product_Group_DW']
    dataset_grouped_corrected_df.drop('Product_Group_DW', axis=1, inplace=True)
    dataset_grouped_corrected_df.rename(columns={'New_Product_Group_DW': 'Product_Group_DW'}, inplace=True)
    dataset_grouped_corrected_df = dataset_grouped_corrected_df[col_lists]

    return dataset_grouped_corrected_df


def classification_corrections_end(df, corrections):
    dataset_grouped_filtered_df = df
    pse_fact_pa_parts_classification_refs_df = corrections

    col_lists = list(dataset_grouped_filtered_df)
    dataset_grouped_corrected_df = dataset_grouped_filtered_df.merge(pse_fact_pa_parts_classification_refs_df, left_on=['Part_Ref'], right_on=['Part_Ref'], how='left')
    dataset_grouped_corrected_df.loc[dataset_grouped_corrected_df['New_Product_Group_DW'].notnull(), 'Max_Prob'] = 1
    dataset_grouped_corrected_df.loc[dataset_grouped_corrected_df['New_Product_Group_DW'].isnull(), 'New_Product_Group_DW'] = dataset_grouped_corrected_df.loc[dataset_grouped_corrected_df['New_Product_Group_DW'].isnull(), 'prediction']
    dataset_grouped_corrected_df.drop('prediction', axis=1, inplace=True)
    dataset_grouped_corrected_df.rename(columns={'New_Product_Group_DW': 'prediction'}, inplace=True)
    dataset_grouped_corrected_df = dataset_grouped_corrected_df[col_lists]

    return dataset_grouped_corrected_df


def get_dgo_manual_classifications(files_loc):
    manual_files = []

    for file_loc in files_loc:
        manual_family_file = pd.read_csv(file_loc)
        manual_files.append(manual_family_file)

    manual_classifications_df = pd.concat(manual_files)

    return manual_classifications_df


def get_fact_pa_classifications():
    pse_fact_pa_parts_classification_refs = sql_retrieve_df(options_file.DSN_MLG_PRD, options_file.sql_info['database_final'], options_file.sql_info['parts_classification_refs'], options_file, columns=['Part_Ref', 'Old_Product_Group_DW', 'New_Product_Group_DW'])

    return pse_fact_pa_parts_classification_refs


def group_by_rules(x):
    d = {}
    d['Part_Desc_concat'] = ' '.join(list(set(x['Part_Desc'])))
    # d['Product_Group_DW'] = ' '.join(list(set(x['Product_Group_DW'])))
    d['Product_Group_DW'] = x['Product_Group_DW'].value_counts().index[0]  # Selects only the most common Product_Group_DW. In draws, selects the highest value;
    d['Client_Id'] = x['Client_Id'].head(1).values[0]
    d['Average_Cost_avg'] = x['Average_Cost'].mean()
    d['PVP_1_avg'] = x['PVP_1'].mean()
    d['PLR_Account_first'] = x['PLR_Account'].head(1).values[0]
    d['Part_Desc_PT_concat'] = ' '.join(list(set(x['Part_Desc_PT'])))

    d['Part_Ref_length'] = x['Part_Ref_length'].mean()
    d['Part_Ref_digits_count'] = x['Part_Ref_digits_count'].mean()
    d['Part_Ref_letters_count'] = x['Part_Ref_letters_count'].mean()
    d['Part_Ref_spaces_count'] = x['Part_Ref_spaces_count'].mean()
    d['Part_Ref_others_count'] = x['Part_Ref_others_count'].mean()

    d['Part_Desc_length'] = x['Part_Desc_length'].mean()
    d['Part_Desc_digits_count'] = x['Part_Desc_digits_count'].mean()
    d['Part_Desc_letters_count'] = x['Part_Desc_letters_count'].mean()
    d['Part_Desc_spaces_count'] = x['Part_Desc_spaces_count'].mean()
    d['Part_Desc_others_count'] = x['Part_Desc_others_count'].mean()

    d['Part_Desc_PT_length'] = x['Part_Desc_PT_length'].mean()
    d['Part_Desc_PT_digits_count'] = x['Part_Desc_PT_digits_count'].mean()
    d['Part_Desc_PT_letters_count'] = x['Part_Desc_PT_letters_count'].mean()
    d['Part_Desc_PT_spaces_count'] = x['Part_Desc_PT_spaces_count'].mean()
    d['Part_Desc_PT_others_count'] = x['Part_Desc_PT_others_count'].mean()

    d['brand'] = x['brand'].head(1).values[0]

    return pd.Series(d, index=['Part_Desc_concat', 'Product_Group_DW', 'Client_Id', 'Average_Cost_avg', 'PVP_1_avg', 'PLR_Account_first', 'Part_Desc_PT_concat', 'part_ref_length', 'part_ref_digits_count', 'part_ref_letters_count', 'part_ref_spaces_count', 'part_ref_others_count', 'brand'])


def product_group_dw_complete_replaces(df):
    # Step 1
    df = df.loc[(df['Product_Group_DW'] != '44') & (df['Product_Group_DW'] != '45'), :]

    # Step 2 + Step 3
    df.loc[:, 'Product_Group_DW'] = df['Product_Group_DW'].replace(parts_families_replacement)

    # Step 4
    df.loc[:, 'Product_Group_DW'] = df['Product_Group_DW'].replace(others_families_dict_str)
    return df


def product_group_dw_corrections_on_desc(df):

    df['Part_Desc_Copy'] = df['Part_Desc']
    df['Part_Desc_Copy'] = df['Part_Desc_Copy'].replace(exact_matches_dict)
    df['Part_Desc_Copy'] = df['Part_Desc_Copy'].replace(partial_matches_dict, regex=True)

    filter_1 = df['Part_Desc'].str.match('^cabo\\s{0,}rep.{0,}')
    filter_2 = df['Product_Group_DW'] == '30'
    df.loc[filter_1 & filter_2, 'Part_Desc_Copy'] = '42'

    filter_3 = df['Part_Desc'].str.match('^injector.{0,}|^inyector.{0,}|^injecteur.{0,}')
    filter_4 = df['Average_Cost'] >= 20
    df.loc[filter_3 & filter_4, 'Part_Desc_Copy'] = '24'

    return df


def product_group_dw_corrections_on_desc_end(df):

    df['Part_Desc_Copy'] = df['Part_Desc_concat']
    df['Part_Desc_Copy'] = df['Part_Desc_Copy'].replace(exact_matches_dict)
    df['Part_Desc_Copy'] = df['Part_Desc_Copy'].replace(partial_matches_dict, regex=True)

    filter_1 = df['Part_Desc_concat'].str.match('^cabo\\s{0,}rep.{0,}')
    filter_2 = df['Product_Group_DW'] == '30'
    df.loc[filter_1 & filter_2, 'Part_Desc_Copy'] = '42'

    filter_3 = df['Part_Desc_concat'].str.match('^injector.{0,}|^inyector.{0,}|^injecteur.{0,}')
    filter_4 = df['Average_Cost_avg'] >= 20
    df.loc[filter_3 & filter_4, 'Part_Desc_Copy'] = '24'

    return df


def probabilities_correction(df):

    df['Max_Prob_forced'] = np.where(df['Part_Desc_concat'] == df['Part_Desc_Copy'], df['Max_Prob'], 1)
    df['prediction_forced'] = np.where(df['Part_Desc_concat'] == df['Part_Desc_Copy'], df['prediction'], df['Part_Desc_Copy'])
    df['New_Prediction_50_forced'] = np.where(df['Part_Desc_concat'] == df['Part_Desc_Copy'], df['New_Prediction_50'], df['Part_Desc_Copy'])
    df['New_Prediction_80_forced'] = np.where(df['Part_Desc_concat'] == df['Part_Desc_Copy'], df['New_Prediction_80'], df['Part_Desc_Copy'])
    df['New_Prediction_85_forced'] = np.where(df['Part_Desc_concat'] == df['Part_Desc_Copy'], df['New_Prediction_85'], df['Part_Desc_Copy'])
    df['New_Prediction_90_forced'] = np.where(df['Part_Desc_concat'] == df['Part_Desc_Copy'], df['New_Prediction_90'], df['Part_Desc_Copy'])
    df['New_Prediction_95_forced'] = np.where(df['Part_Desc_concat'] == df['Part_Desc_Copy'], df['New_Prediction_95'], df['Part_Desc_Copy'])

    df.drop(['Max_Prob', 'prediction', 'New_Prediction_50', 'New_Prediction_80', 'New_Prediction_85', 'New_Prediction_90', 'New_Prediction_95'], axis=1, inplace=True)
    df.rename(columns={'Max_Prob_forced': 'Max_Prob', 'prediction_forced': 'prediction', 'New_Prediction_50_forced': 'New_Prediction_50', 'New_Prediction_80_forced': 'New_Prediction_80', 'New_Prediction_85_forced': 'New_Prediction_85', 'New_Prediction_90_forced': 'New_Prediction_90', 'New_Prediction_95_forced': 'New_Prediction_95'}, inplace=True)

    return df


def deployment(df, main_families_cm, other_families_cm):
    sql_upload(main_families_cm, options_file.sql_info['database_final'], options_file.sql_info['matrix_lvl_1'])
    sql_upload(other_families_cm, options_file.sql_info['database_final'], options_file.sql_info['matrix_lvl_2'])

    df.rename(columns={'Client_Id': 'Client_ID', 'Part_Desc_concat': 'Part_Description', 'Average_Cost_avg': 'Part_Cost', 'PVP_1_avg': 'Part_PVP', 'prediction': 'Classification', 'Max_Prob': 'Classification_Prob'}, inplace=True)
    df['Classification_Flag'] = 0
    df['Classification_Prob'] = df['Classification_Prob'].round(2)
    df['Part_Cost'] = df['Part_Cost'].round(2)
    df['Part_PVP'] = df['Part_PVP'].round(2)
    df = df.astype({'Part_Ref': 'str', 'Client_ID': 'str', 'Part_Cost': 'str', 'Part_PVP': 'str', 'Classification_Prob': 'str'})
    df['Part_Description'] = df['Part_Description'].fillna("")
    df.dropna(subset=['Classification'], axis=0, inplace=True)
    sql_inject(df, options_file.DSN_MLG_PRD, options_file.sql_info['database_final'], options_file.sql_info['parts_classification_table'], options_file, columns=['Part_Ref', 'Part_Description', 'Part_Cost', 'Part_PVP', 'Client_ID', 'Product_Group_DW', 'Classification', 'Classification_Prob', 'Classification_Flag'], truncate=1, check_date=1)
    sql_inject(df, options_file.DSN_SRV3_PRD, options_file.sql_info['database_BI_GSC'], options_file.sql_info['parts_classification_table'], options_file, columns=['Part_Ref', 'Part_Description', 'Part_Cost', 'Part_PVP', 'Client_ID', 'Product_Group_DW', 'Classification', 'Classification_Prob', 'Classification_Flag'], truncate=1, check_date=1)
    sql_sp_run(options_file.DSN_SRV3_PRD, options_file.sql_info['database_BI_GSC'], options_file)
    return


def sql_sp_run(dsn, db, options_file_in):
    cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'.format(dsn, options_file_in.UID, options_file_in.PWD, db), searchescape='\\')
    cursor = cnxn.cursor()

    cursor.execute('EXEC {}.[dbo].{}'.format(db, options_file_in.sql_info['update_codes_sp']))

    cnxn.commit()
    cursor.close()
    cnxn.close()

    return


def sql_upload(df, db, view):
    df['Totals'] = df.sum(axis=1)
    df.index.rename('Actual', inplace=True)
    df.reset_index(inplace=True)

    sql_inject(df, options_file.DSN_MLG_PRD, db, view, options_file, list(df), truncate=1, check_date=1)

    return


if __name__ == '__main__':
    main()
