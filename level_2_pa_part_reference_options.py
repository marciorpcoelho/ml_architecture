import os
from py_dotenv import read_dotenv
from multiprocessing import cpu_count
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
dotenv_path = '/info.env'
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
read_dotenv(base_path + dotenv_path)


if 'nt' in os.name:
    DSN_SRV3_PRD = os.getenv('DSN_SRV3_Prd')
    DSN_MLG_PRD = os.getenv('DSN_MLG_Prd')
    DSN_MLG_DEV = os.getenv('DSN_MLG_Dev')
elif 'posix' in os.name:
    DSN_SRV3_PRD = os.getenv('DSN_SRV3_Prd_Linux')
    DSN_MLG_PRD = os.getenv('DSN_MLG_Prd_Linux')
UID = os.getenv('UID')
PWD = os.getenv('PWD')

project_id = 2610
update_frequency_days = 0
documentation_url_app = 'https://gruposalvadorcaetano.sharepoint.com/:b:/s/rigor/6825_DGAA/EZBq5HGuxpxPkt6R2VrX0loBOLtKr_UKYPUmH0UTPlbibw?e=VeWKhZ'
sel_cols = ['Part_Ref', 'Part_Desc', 'Product_Group_DW', 'Client_ID', 'Average_Cost', 'PVP_1', 'PLR_Account', 'Part_Desc_PT']

sql_info = {
    'database_BI_AFR': 'BI_RCG',
    'database_BI_CRP': 'BI_CRP',
    'database_BI_IBE': 'BI_IBE',
    'database_BI_CA': 'BI_CA',
    'database_BI_GSC': 'BI_GSC',
    'database_final': 'BI_MLG',
    'final_table': 'PSE_Fact_PA_Parts_Stock_Master_File_Matched',
    'parts_classification_table': 'PSE_Fact_PA_Parts_Classification',
    'parts_classification_rules': 'PSE_Fact_PA_Parts_Classification_Rules',
    'parts_classification_refs': 'PSE_Fact_PA_Parts_Classification_Refs',
    'matrix_lvl_1': 'PSE_Fact_PA_Parts_Conf_Matrix_Lvl_1',
    'matrix_lvl_2': 'PSE_Fact_PA_Parts_Conf_Matrix_Lvl_2',
    'unit_count_number_history': 'LOG_Project_Units_Count_History',
    'master_file_table': 'PSE_Fact_PA_Parts_Master_Files',
    'update_codes_sp': 'usp_PSE_Parts_Classification_Upd'
}

sql_to_code_renaming = {
    'Client_ID': 'Client_Id'
}

column_translate_dict = {
    'Part_Ref': 'Referência',
    'Part_Description': 'Descrição',
    'Part_Cost': 'Preço de Custo',
    'Part_PVP': 'Preço de Venda',
    'Product_Group_DW': 'Família Original',
    'Product_Group_DW_desc': 'Família Original',
    'Classification': 'Família Classificada',
    'Classification_desc': 'Família Classificada',
    'Classification_Prob': 'Grau Confiança',
    'Percentage_Predicted': '% Conf.',
}

gridsearch_parameters = {
    # 'lr': [LogisticRegression, [{'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'newton-cg'], 'max_iter': [2000], 'multi_class': ['ovr', 'multinomial']}]],
    'lgb': [lgb.LGBMClassifier, [{'num_leaves': [15, 31, 50, 100], 'n_estimators': [50, 100, 200], 'max_depth': ['50', '100'], 'objective': ['multiclass']}]],

}

others_families_dict = {
    147: 'O. Acessórios',
    148: 'O. Acessórios',
    149: 'O. Acessórios',
    150: 'O. Acessórios',
    152: 'O. Acessórios',
    42: 'O. Colisão',
    37: 'O. Colisão',
    48: 'O. Consumíveis',
    119: 'O. Manutenção',
    16: 'O. Manutenção',
    78: 'O. Merchandising',
    144: 'O. Mota',
    143: 'O. Mota',
    170: 'O. Reparação',
    29: 'O. Reparação',
    172: 'O. Reparação',
    173: 'O. Reparação',
    54: 'O. Diversos',
    '75/77': 'Lazer/Marroquinaria',
}

warning_message_app_dict = {
    'starts': '{} - Não existem peças cuja descrição começa com: {}, nas condições de custo/preço definidas',
    'contains': '{} - Não existem peças cuja descrição começa com: {}, nas condições de custo/preço definidas',
}

classified_app_query = '''
    SELECT *
      FROM [BI_MLG].[dbo].[PSE_Fact_PA_Parts_Classification]
      where Product_Group_DW <> '1'
  '''

non_classified_app_query = '''
    SELECT *
      FROM [BI_MLG].[dbo].[PSE_Fact_PA_Parts_Classification]
  '''

product_group_app_query = '''
    SELECT [Product_Group_Code]
          ,[Product_Group_Level_1_Code]
          ,[Product_Group_Level_2_Code]
          ,[PT_Product_Group_Level_1_Desc]
          ,[PT_Product_Group_Level_2_Desc]
          ,[PT_Product_Group_Desc]
      FROM [BI_AFR].[dbo].[PSE_Dim_Product_Groups_GSC]
      where Product_Group_Code not in (1, 2, 44, 45)
    '''

# current_stock_query = '''SELECT DISTINCT Part_Ref, Part_Desc, Product_Group_DW, Client_Id, Franchise_Code, Franchise_Code_DW, Average_Cost, PVP_1, PLR_Account
# FROM {}.dbo.PSE_Fact_BI_Parts_Stock_Month WITH (NOLOCK)
# WHERE Part_Ref <> '' and Part_Desc is not Null and Part_Desc <> '' and Stock_Month = '{}'
# GROUP BY Part_Ref, Part_Desc, Product_Group_DW, Client_Id, Franchise_Code, Franchise_Code_DW, Average_Cost, PVP_1, PLR_Account'''

current_stock_query = '''
    WITH max_date as (
        SELECT 
            Part_Ref, MAX(ISNULL(Last_Sell_Date,'1')) as max_date_value
        FROM {}.dbo.PSE_Fact_BI_Parts_Stock_Month WITH (NOLOCK)
        WHERE 1=1
        GROUP BY Part_Ref
    )
    SELECT
        distinct a.Part_Ref, Part_Desc, Product_Group_DW, a.Client_Id, Franchise_Code, Franchise_Code_DW, AVG(a.Average_Cost / a.Currency_Rate) as Average_Cost, AVG(a.PVP_1 / a.Currency_Rate) as PVP_1, PLR_Account, Last_Sell_Date, Parts_DIM.PT_Product_Group_Level_2_Desc, Parts_DIM.Product_Group_Level_2_Code, Parts_DIM.Product_Group_Level_1_Code, Parts_DIM.PT_Product_Group_Level_1_Desc
    FROM {}.dbo.PSE_Fact_BI_Parts_Stock_Month as a WITH (NOLOCK)
    inner join max_date as b on a.Part_Ref = b.Part_Ref and ISNULL(a.Last_Sell_Date, '1') = b.max_date_value
    LEFT JOIN [BI_AFR].[dbo].[PSE_Dim_Product_Groups_GSC] as Parts_DIM on Parts_DIM.Product_Group_Code = a.Product_Group_DW
    WHERE 1=1 
    and Last_Sell_Date = '{}'
    and Warehouse_Code = -1
    GROUP BY a.Part_Ref, Part_Desc, Product_Group_DW, a.Client_Id, Franchise_Code, Franchise_Code_DW, PLR_Account, Last_Sell_Date, Parts_DIM.PT_Product_Group_Level_2_Desc, Parts_DIM.Product_Group_Level_2_Code, Parts_DIM.Product_Group_Level_1_Code, Parts_DIM.PT_Product_Group_Level_1_Desc
'''

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
    'Parts_Specific_Common_Stop_Words': ['esquerdo', 'direito', 'frente', 'tras', 'kit', 'jogo']  # Added 'esquerdo', 'direito', 'frente', 'tras' because they appear in too many Product Groups and provide nearly no information;
}

abbreviations_dict = {
    'str': 'string',
    'abbr': 'abbreviation',
    'spd': 'speed',
    'int': 'interior',
    'inf': 'inferior',
    'sup': 'superior',
    'fr': 'frente',
    'frt': 'frente',
    'fte': 'frente',
    'tr': 'tras',
    'trs': 'tras',
    'esq': 'esquerdo',
    'esqu': 'esquerdo',
    'esquerd': 'esquerdo',
    'drt': 'direito',
    'dta': 'direito',
    'dto': 'direito',
    'lh': 'esquerdo',  # BI_AFR
    'rh': 'direito',  # BI_AFR
    'trav': 'travao',  # BI_AFR
    'trava': 'travao',
    'jg': 'jogo',  # BI_AFR
    'j': 'jogo',
    'ign': 'ignicao',
    'c': 'com',
    'co': 'com',
    'cx': 'caixa',
    'filtr': 'filtro',
    'alar': 'alarme',
    # 'r': 'pneu'  # Cases when the description is like 155/65R14 or 215/55R17
    'paraf': 'parafuso',
    'casq': 'casquilho',
    'casqu': 'casquilho',
    'conj': 'conjunto',
    'anil': 'anilha',
    'eleo': 'oleo',  # I believe this is an enconding problem
    'direc': 'direcao',
    'bris': 'brisas',
    'abacadeira': 'abracadeira',
    'unid': 'unidade',
    'bluetoo': 'bluetooth',
    'blueto': 'bluetooth',
    'ext': 'exterior',
    'ole': 'oleo',
    'cil': 'cilindro',
    'transmiss': 'transmissão',
    'pain': 'painel',
    'crom': 'cromado',
    'choq': 'choque',
    'susp': 'suspensao',
    'veloc': 'velocidade',
    'amort': 'amortecedor',
    'indiv': 'individual',
    'embr': 'embraiagem',
    'amortecedo': 'amortecedor',
}

brand_codes_per_franchise = '''
    select DISTINCT Client_Id, Franchise_Code_DMS as Original_Value
    from {}.dbo.PSE_MapDMS_Franchises WITH (NOLOCK)
    where Franchise like '%{}%'
'''

master_files_to_convert = {
    'dbs/Master_Files/Fiat_DiffusioneTariffaCJD': [0, [0, 13, 37, 71, 91, 109, 120, 123, 129], ['Part_Ref', 'Part_Ref#2', 'Part_Ref#3', 'Part_Desc_PT', 'Part_Desc_EN', 'Cod#2', 'Tag#1', 'Cod#3'], 0, 0],
    'dbs/Master_Files/PCOC_Tabela_Precos_PSA_20200213': [0, [0, 26, 34, 52, 67, 82, 86, 88, 136, 161, 178, 215, 227, 245, 271, 295, 319, 343, 367, 445, 447, 515, 605, 637, 651], ['Cod#1', 'Cod#2', 'Part_Ref', 'Part_Desc_PT', 'Part_Desc_FR', 'Tag#1', 'Tag#2', 'Cod#4', 'Cod#5', 'Cod#6', 'Cod#7', 'Cod#8', 'Cod#9', 'Cod#10', 'Cod#11', 'Cod#12', 'Cod#13', 'Cod#14', 'Cod#15', 'Cod#16', 'Cod#17', 'Cod#18', 'Part_Desc_PT#2', 'Cod#19'], 0, 0],
    'dbs/Master_Files/SEAT_I6459_Completa_200301_PO': [0, [3, 23, 46, 104, 108, 121, 125, 134], ['Part_Ref', 'Part_Desc_PT', 'Cod#2', 'Tag#1', 'Tag#2', 'Currency', 'Cod#3'], 1, 1],
    'dbs/Master_Files/VAG_TPCNCAVW': [0, [0, 15, 25, 38, 56, 63, 66], ['Part_Ref', 'Part_Desc_PT', 'Cod#2', 'Tag#2', 'Cod#3', 'Tag#3'], 0, 0],
    'dbs/Master_Files/VAG_TPCNCSK': [0, [0, 15, 25, 38, 56, 63, 66], ['Part_Ref', 'Part_Desc_PT', 'Cod#2', 'Tag#2', 'Cod#3', 'Tag#3'], 0, 0],
    'dbs/Master_Files/MBF820P': [0, [0, 19, 46, 54, 129, 135, 150], ['Part_Ref', 'Cod#1', 'Tag#2', 'Part_Desc_PT', 'Cod#3', 'Cod#2'], 0, 0]
    # 'dbs/Master_Files/Skoda_RE_TARIF_SKO_20200318111652': [1, [], ['Part_Ref', 'Part_Desc_PT'], 0, 0],
    # 'dbs/Master_Files/Audi_VW_RE_TARIF_VAG_20200318112912': [1, [], ['Part_Ref', 'Part_Desc_PT'], 0, 0],
    # 'dbs/Master_Files/BMW_Motos_RE_TARIF_BMM_20200318111608': [1, [], ['Part_Ref', 'Part_Desc_PT'], 0, 0],
    # 'dbs/Master_Files/Hyundai_RE_TARIF_HYN_20200318142512': [1, [], ['Part_Ref', 'Part_Desc_PT'], 0, 0],
    # 'dbs/Master_Files/Honda_RE_TARIF_HON_20200318122510': [1, [], ['Part_Ref', 'Part_Desc_PT'], 0, 0],
    # 'dbs/Master_Files/Opel_RE_TARIF_OPL_20200318145935': [1, [], ['Part_Ref', 'Part_Desc_PT'], 0, 0],
}

master_file_brands = [
    ['fiat'],
    ['nissan'],
    ['peugeot', 'citroen', 'opel', 'chevrolet'],
    ['seat'],
    ['volkswagen'],
    ['skoda'],
    ['mercedes'],
    ['bmw'],
    ['ford']
]

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
    'zero_at_end': r'0*$',
    'dms_code_at_beginning': r'^{}',
    'up_to_2_letters_at_end': r'[a-zA-Z]{1,2}?$',
    '2_letters_at_end': r'[a-zA-Z]{2}?$',
    '1_letter_at_beginning': r'^[a-zA-Z]{1}?',
    'all_letters_at_beginning': r'^[a-zA-Z]+',
    'remove_hifen': r'-',
    'remove_last_dot': r'\.{1}$',
    # '001_beginning_code_removal': r'^001',
    'dms_code_or_condition': r'^' + '|^',
    'space_removal': r'\s+',
    'letters_in_the_beginning': r'^[a-zA-Z]{2}',
    'single_V_in_the_beginning': r'^[V]',
    'bmw_AT_end': r'AT$',
    'bmw_dot': r'\.',
    'right_bar': r'/',
    'middle_strip': r'\s+',
}

stop_words_list = [
    'de', 'a', 'o', 'que', 'e', 'é', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'à', 'seu', 'sua', 'ou',
    'quando', 'muito', 'nos', 'já', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'depois', 'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'você', 'essa', 'num',
    'nem', 'suas', 'meu', 'às', 'minha', 'numa', 'pelos', 'elas', 'qual', 'nós', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua',
    'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 'estou', 'está', 'estamos', 'estão', 'estive',
    'esteve', 'estivemos', 'estiveram', 'estava', 'estávamos', 'estavam', 'estivera', 'estivéramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivéssemos', 'estivessem', 'estiver', 'estivermos',
    'estiverem', 'hei', 'há', 'havemos', 'hão', 'houve', 'houvemos', 'houveram', 'houvera', 'houvéramos', 'haja', 'hajamos', 'hajam', 'houvesse', 'houvéssemos', 'houvessem', 'houver', 'houvermos',
    'houverem', 'houverei', 'houverá', 'houveremos', 'houverão', 'houveria', 'houveríamos', 'houveriam', 'sou', 'somos', 'são', 'era', 'éramos', 'eram', 'fui', 'foi', 'fomos', 'foram', 'fora', 'fôramos',
    'seja', 'sejamos', 'sejam', 'fosse', 'fôssemos', 'fossem', 'for', 'formos', 'forem', 'serei', 'será', 'seremos', 'serão', 'seria', 'seríamos', 'seriam', 'tenho', 'tem', 'temos', 'tém', 'tinha',
    'tínhamos', 'tinham', 'tive', 'teve', 'tivemos', 'tiveram', 'tivera', 'tivéramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 'tivéssemos', 'tivessem', 'tiver', 'tivermos', 'tiverem', 'terei', 'terá',
    'teremos', 'terão', 'teria', 'teríamos', 'teriam', '2012', 'del', 'con', 'esquerdo', 'direito', 'frente', 'tras', 'kit', 'jogo'
]

log_files = {
    'full_log': 'logs/dgo_parts.txt'
}
