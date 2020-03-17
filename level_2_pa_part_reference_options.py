import os
from py_dotenv import read_dotenv
from multiprocessing import cpu_count
dotenv_path = '/info.env'
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
read_dotenv(base_path + dotenv_path)

if 'nt' in os.name:
    DSN = os.getenv('DSN_Prd')
    DSN_MLG = os.getenv('DSN_MLG')
elif 'posix' in os.name:
    DSN = os.getenv('DSN_Prd_Linux')
    DSN_MLG = os.getenv('DSN_MLG_Linux')
UID = os.getenv('UID')
PWD = os.getenv('PWD')


sql_info = {
    'database_BI_AFR': 'BI_RCG',
    'database_BI_CRP': 'BI_CRP',
    'database_BI_IBE': 'BI_IBE',
    'database_BI_CA': 'BI_CA',
    'database_BI_GSC': 'BI_GSC',
    'database_final': 'BI_MLG',
    }


current_stock_query = '''SELECT DISTINCT Part_Ref, Part_Desc, Product_Group_DW, Client_Id, Franchise_Code, Franchise_Code_DW
FROM {}.dbo.PSE_Fact_BI_Parts_Stock_Month WITH (NOLOCK)
WHERE Part_Ref <> '' and Part_Desc is not Null and Part_Desc <> '' and Stock_Month = '{}'
GROUP BY Part_Ref, Part_Desc, Product_Group_DW, Client_Id, Franchise_Code, Franchise_Code_DW'''

dms_franchises = '''SELECT *
FROM {}.dbo.PSE_MapDMS_Franchises'''


dim_product_group_query = '''SELECT [Product_Group_Code]
      ,[Product_Group_Level_1_Code]
      ,[Product_Group_Level_2_Code]
      ,[PT_Product_Group_Level_1_Desc]
      ,[PT_Product_Group_Level_2_Desc]
      ,[PT_Product_Group_Desc]
      ,[ES_Product_Group_Level_1_Desc]
      ,[ES_Product_Group_Level_2_Desc]
      ,[ES_Product_Group_Desc]
      ,[EN_Product_Group_Level_1_Desc]
      ,[EN_Product_Group_Level_2_Desc]
      ,[EN_Product_Group_Desc]
      ,[FR_Product_Group_Level_1_Desc]
      ,[FR_Product_Group_Level_2_Desc]
      ,[FR_Product_Group_Desc]
      ,[Product_Group_Display_Order]
  FROM [BI_AFR].[dbo].[PSE_Dim_Product_Groups_GSC]'''

dim_clients_query = '''SELECT Client_Id, Client_Name, Description, Description_Mobile, BI_Database
FROM [BI_GSC].[dbo].[GBL_Setup_Clients]'''


stop_words = {
    'Common_Stop_Words': ['de', 'do', 'da', 'a', 'p', 'para', 'com'],
    'BI_AFR': ['2012'],
    'BI_CRP': [],
    'BI_IBE': ['del', 'con'],
    'BI_CA': [],
}

abbreviations_dict = {
    'frente': ['fr', 'frt'],
    'traseiro': ['tr'],
    'esquerdo': ['esq'],
    'direito': ['drt'],
    'lh': ['left handed'],  # BI_AFR
    'rh': ['right handed'],  # BI_AFR
    'travao': ['trav'],  # BI_AFR
    'jg': ['jogo'],  # BI_AFR

}

dms_codes_per_franchise = '''
    select DISTINCT Client_Id, Franchise_Code_DMS as Original_Value
    from {}.dbo.PSE_MapDMS_Franchises WITH (NOLOCK)
    where Franchise like '%{}%'
'''

master_files_to_convert = {
    'dbs/Master_Files/Fiat_DiffusioneTariffaCJD': [[0, 13, 37, 71, 91, 109, 120, 123, 129], ['Part_Ref', 'Part_Ref#2', 'Part_Ref#3', 'Part_Desc_PT', 'Part_Desc_EN', 'Cod#2', 'Tag#1', 'Cod#3'], 0, 0],
    'dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213': [[0, 26, 34, 52, 67, 82, 86, 88, 136, 161, 178, 215, 227, 245, 271, 295, 319, 343, 367, 445, 447, 515, 605, 637, 651], ['Cod#1', 'Cod#2', 'Part_Ref', 'Part_Desc_PT', 'Part_Desc_FR', 'Tag#1', 'Tag#2', 'Cod#4', 'Cod#5', 'Cod#6', 'Cod#7', 'Cod#8', 'Cod#9', 'Cod#10', 'Cod#11', 'Cod#12', 'Cod#13', 'Cod#14', 'Cod#15', 'Cod#16', 'Cod#17', 'Cod#18', 'Part_Desc_PT#2', 'Cod#19'], 0, 0],
    'dbs/Master_Files/SEAT_I6459_Completa_200301_PO': [[3, 23, 46, 104, 108, 121, 125, 134], ['Part_Ref', 'Part_Desc_PT', 'Cod#2', 'Tag#1', 'Tag#2', 'Currency', 'Cod#3'], 1, 1],
    'dbs/Master_Files/VAG_TPCNCAVW': [[0, 15, 25, 38, 56, 63, 66], ['Part_Ref', 'Part_Desc_PT', 'Cod#2', 'Tag#2', 'Cod#3', 'Tag#3'], 0, 0],
    'dbs/Master_Files/VAG_TPCNCSK': [[0, 15, 25, 38, 56, 63, 66], ['Part_Ref', 'Part_Desc_PT', 'Cod#2', 'Tag#2', 'Cod#3', 'Tag#3'], 0, 0],
}

master_files_converted = [
    'dbs/Master_Files/Fiat_DiffusioneTariffaCJD.csv',
    'dbs/Master_Files/Nissan_tarifa.csv',  # The second Nissan file is the same as this (same references) but with one less column: Discount Code
    'dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv',
    'dbs/Master_Files/SEAT_I6459_Completa_200301_PO.csv',
    'dbs/Master_Files/VAG.csv',  # ToDo: need to handle this one later
]
master_files_and_brand = {
    'dbs/Master_Files/Fiat_DiffusioneTariffaCJD.csv': ['fiat'],
    'dbs/Master_Files/Nissan_tarifa.csv': ['nissan'],
    'dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv': ['peugeot', 'citroen', 'opel', 'chevrolet'],
    'dbs/Master_Files/SEAT_I6459_Completa_200301_PO.csv': ['seat'],
    'dbs/Master_Files/VAG.csv': ['volkswagen'],  # ToDo: need to handle this one later
}

master_files_afr_dict = {
    'FIA': ['dbs/Master_Files/Fiat_DiffusioneTariffaCJD.csv'],
    'NIS': ['dbs/Master_Files/Nissan_tarifa.csv'],
    'CHV': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'SEA': ['dbs/Master_Files/SEAT_I6459_Completa_200301_PO.csv'],
    'VAG': ['dbs/Master_Files/SEAT_I6459_Completa_200301_PO.csv', 'dbs/Master_Files/VAG_TPCNCAVW.csv', 'dbs/Master_Files/VAG_TPCNCSK.csv']
}

master_files_crp_dict = {
    'CH': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'CHV': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'CI': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'CTR': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'FI': ['dbs/Master_Files/Fiat_DiffusioneTariffaCJD.csv'],
    'FIA': ['dbs/Master_Files/Fiat_DiffusioneTariffaCJD.csv'],
    'NI': ['dbs/Master_Files/Nissan_tarifa.csv'],
    'NIS': ['dbs/Master_Files/Nissan_tarifa.csv'],
    'OP': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'OPL': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'PEU': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'PG': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'SE': ['dbs/Master_Files/SEAT_I6459_Completa_200301_PO.csv'],
    'SEA': ['dbs/Master_Files/SEAT_I6459_Completa_200301_PO.csv']
}

master_files_ca_dict = {
    'CHV': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'FIA': ['dbs/Master_Files/Fiat_DiffusioneTariffaCJD.csv'],
    'NIS': ['dbs/Master_Files/Nissan_tarifa.csv'],
    'OPL': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'PEU': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'SEA': ['dbs/Master_Files/SEAT_I6459_Completa_200301_PO.csv'],
}

master_files_ibe_dict = {
    'CHV': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'FIA': ['dbs/Master_Files/Fiat_DiffusioneTariffaCJD.csv'],
    'G': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'N': ['dbs/Master_Files/Nissan_tarifa.csv'],
    'NIS': ['dbs/Master_Files/Nissan_tarifa.csv'],
    'O': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'OPL': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'P': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'PEU': ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
    'S': ['dbs/Master_Files/SEAT_I6459_Completa_200301_PO.csv'],
    'SEA': ['dbs/Master_Files/SEAT_I6459_Completa_200301_PO.csv'],
    'U': ['dbs/Master_Files/Fiat_DiffusioneTariffaCJD.csv'],
}

# master_files_dict = {
#     8: ['dbs/Master_Files/Fiat_DiffusioneTariffaCJD.csv'],
#     33: ['dbs/Master_Files/Fiat_DiffusioneTariffaCJD.csv'],
#     22: ['dbs/Master_Files/Nissan_tarifa.csv'],
#     4: ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
#     6: ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
#     23: ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
#     24: ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
#     50: ['dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.csv'],
#     27: ['dbs/Master_Files/SEAT_I6459_Completa_200301_PO.csv'],
#     31: ['dbs/Master_Files/SEAT_I6459_Completa_200301_PO.csv'],
# }

# 'dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213.txt
# fields_1, fields_2, fields_3, fields_4, fields_5, fields_6 = [], [], [], [], [], []
# fields_7, fields_8, fields_9, fields_10, fields_11, fields_12 = [], [], [], [], [], []
# fields_13, fields_14, fields_15, fields_16, fields_17, fields_18 = [], [], [], [], [], []
# fields_19, fields_20, fields_21, fields_22, fields_23, fields_24 = [], [], [], [], [], []
# result = pd.DataFrame(columns={'Cod#1', 'Cod#2', 'Part_Ref', 'Part_Desc_PT', 'Part_Desc_FR', 'Tag#1', 'Tag#2', 'Cod#4', 'Cod#5', 'Cod#6', 'Cod#7', 'Cod#8', 'Cod#9', 'Cod#10', 'Cod#11', 'Cod#12', 'Cod#13', 'Cod#14', 'Cod#15', 'Cod#16', 'Cod#17', 'Cod#18', 'Part_Desc_PT#2', 'Cod#19'})
# field_1 = x[0:26].strip()
# field_2 = x[26:34].strip()
# field_3 = x[34:52].strip()
# field_4 = x[52:67].strip()
# field_5 = x[67:82].strip()
# field_6 = x[82:86].strip()
# field_7 = x[86:88].strip()
# field_8 = x[88:136].strip()
# field_9 = x[136:161].strip()
# field_10 = x[161:178].strip()
# field_11 = x[178:215].strip()
# field_12 = x[215:227].strip()
# field_13 = x[227:245].strip()
# field_14 = x[245:271].strip()
# field_15 = x[271:295].strip()
# field_16 = x[295:319].strip()
# field_17 = x[319:343].strip()
# field_18 = x[343:367].strip()
# field_19 = x[367:445].strip()
# field_20 = x[445:447].strip()
# field_21 = x[447:515].strip()
# field_22 = x[515:605].strip()
# field_23 = x[605:637].strip()
# field_24 = x[637:651].strip()
# result['Cod#1'] = fields_1
# result['Cod#2'] = fields_2
# result['Cod#3'] = fields_3
# result['Part_Desc_PT'] = fields_4
# result['Part_Desc_FR'] = fields_5
# result['Tag#1'] = fields_6
# result['Tag#2'] = fields_7
# result['Cod#4'] = fields_8
# result['Cod#5'] = fields_9
# result['Cod#6'] = fields_10
# result['Cod#7'] = fields_11
# result['Cod#8'] = fields_12
# result['Cod#9'] = fields_13
# result['Cod#10'] = fields_14
# result['Cod#11'] = fields_15
# result['Cod#12'] = fields_16
# result['Cod#13'] = fields_17
# result['Cod#14'] = fields_18
# result['Cod#15'] = fields_19
# result['Cod#16'] = fields_20
# result['Cod#17'] = fields_21
# result['Cod#18'] = fields_22
# result['Part_Desc_PT#2'] = fields_23
# result['Cod#19'] = fields_24

# 'dbs/Master_Files/SEAT_I6459_Completa_200301_PO.txt'
# fields_1, fields_2, fields_3, fields_4, fields_5, fields_6, fields_7 = [], [], [], [], [], [], []
# result = pd.DataFrame(columns={'Cod#1', 'Part_Desc_PT', 'Cod#2', 'Tag#1', 'Tag#2', 'Currency', 'Cod#3'})
# field_1 = x[0:23].strip()
# field_2 = x[23:46].strip()
# field_3 = x[46:104].strip()
# field_4 = x[104:108].strip()
# field_5 = x[108:121].strip()
# field_6 = x[121:125].strip()
# field_7 = x[125:134].strip()
# i += 1
# if i > 1:
#     fields_1.append(field_1)
#     fields_2.append(field_2)
#     fields_3.append(field_3)
#     fields_4.append(field_4)
#     fields_5.append(field_5)
#     fields_6.append(field_6)
#     fields_7.append(field_7)
# result['Cod#1'] = fields_1
# result['Part_Desc_PT'] = fields_2
# result['Cod#2'] = fields_3
# result['Tag#1'] = fields_4
# result['Tag#2'] = fields_5
# result['Currency'] = fields_6
# result['Cod#3'] = fields_7

# 'dbs/Master_Files/VAG_Familias 2020 03.txt'
# fields_1, fields_2 = [], []
# result = pd.DataFrame(columns={'Familia_Ref', 'Familia_Desc'})
# field_1 = x[0:18]
# field_2 = x[18:58]
# fields_1.append(field_1)
# fields_2.append(field_2)
# result['Familia_Ref'] = fields_1
# result['Familia_Desc'] = fields_2

# 'dbs/Master_Files/VAG_TPCNCAVW.txt'
# fields_1, fields_2, fields_3, fields_4, fields_5, fields_6 = [], [], [], [], [], []
# result = pd.DataFrame(columns={'Part_Ref', 'Part_Desc_PT', 'Cod#2', 'Tag#2', 'Cod#3', 'Tag#3'})
# field_1 = x[0:15].strip()
# field_2 = x[15:25].strip()
# field_3 = x[25:38].strip()
# field_4 = x[38:56].strip()
# field_5 = x[56:63].strip()
# field_6 = x[63:66].strip()
# fields_1.append(field_1)
# fields_2.append(field_2)
# fields_3.append(field_3)
# fields_4.append(field_4)
# fields_5.append(field_5)
# fields_6.append(field_6)
# result['Part_Ref'] = fields_1
# result['Part_Desc_PT'] = fields_2
# result['Cod#2'] = fields_3
# result['Tag#2'] = fields_4
# result['Cod#3'] = fields_5
# result['Tag#3'] = fields_6

# 'dbs/Master_Files/VAG_TPCNCSK.txt'
# fields_1, fields_2, fields_3, fields_4, fields_5, fields_6, fields_7 = [], [], [], [], [], [], []
# result = pd.DataFrame(columns={'Cod#1', 'Tag#1', 'Part_Desc_PT', 'Cod#2', 'Tag#2', 'Cod#3', 'Tag#3'})
# field_1 = x[0:12]
# field_2 = x[12:15]
# field_3 = x[15:25]
# field_4 = x[25:38]
# field_5 = x[38:56]
# field_6 = x[56:63]
# field_7 = x[63:66]
# fields_1.append(field_1)
# fields_2.append(field_2)
# fields_3.append(field_3)
# fields_4.append(field_4)
# fields_5.append(field_5)
# fields_6.append(field_6)
# fields_7.append(field_7)
# result['Cod#1'] = fields_1
# result['Tag#1'] = fields_2
# result['Part_Desc_PT'] = fields_3
# result['Cod#2'] = fields_4
# result['Tag#2'] = fields_5
# result['Cod#3'] = fields_6
# result['Tag#3'] = fields_7
# Also this file should be appended to the previous one

regex_dict = {
    'zero_at_beginning': r'^0*',
    'dms_code_at_beginning': r'^{}',
    '2_letters_at_end': r'[a-zA-Z]{2}?$',
    'remove_hifen': r'-',
    'remove_last_dot': r'\.{1}$',
    '001_beginning_code_removal': r'^001',
}

# if current_stock_master_file_filtered.shape[0]:
#     current_stock_master_file_filtered_refs = current_stock_master_file_filtered['Part_Ref'].unique()  # ToDo HERE
#     print('Current Stock References Count: {}'.format(len(current_stock_master_file_filtered_refs)))
#
#     if master_file_brand == 'fiat':
#         # Plain Match
#         step_1_time = time.time()
#         matched_refs_step_1 = [x for x in current_stock_master_file_filtered_refs if x in master_file_refs]
#         print('Step 1 - Elapsed time: {:.3f}'.format(time.time() - step_1_time))
#         print('Matched Step 1: {}'.format(len(matched_refs_step_1)))
#
#         if len(matched_refs_step_1) < len(current_stock_master_file_filtered_refs):
#             regex = re.compile(regex_dict['zero_at_beginning'])
#             step_2_time = time.time()
#             master_file_refs_stem = [regex.sub('', x) for x in master_file_refs]
#             matched_refs_step_2 = [x for x in current_stock_master_file_filtered_refs if regex.sub('', x) in master_file_refs_stem]
#
#             print('Step 2 - Elapsed time: {:.3f}'.format(time.time() - step_2_time))
#             print('Matched Step 2: {}'.format(len(matched_refs_step_2)))
#
#             if len(matched_refs_step_2) + len(matched_refs_step_1) < len(current_stock_master_file_filtered_refs):
#                 step_3_time = time.time()
#                 matched_refs_step_3 = [x for x in current_stock_master_file_filtered_refs if '000' + x in master_file_refs]
#                 print('Step 3 - Elapsed time: {:.3f}'.format(time.time() - step_3_time))
#                 print('Matched Step 3: {}'.format(len(matched_refs_step_3)))