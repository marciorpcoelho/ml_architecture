import os
from py_dotenv import read_dotenv
dotenv_path = 'info.env'
read_dotenv(dotenv_path)

DSN_PRD = os.getenv('DSN_Prd')
DSN_MLG = os.getenv('DSN_MLG')
UID = os.getenv('UID')
PWD = os.getenv('PWD')

project_id = 2406
update_frequency_days = 0
stock_days_threshold = 150

sql_info = {
    'database_source': 'BI_DTR',
    'database_final': 'BI_MLG',
    'product_db': 'VHE_Dim_VehicleData_DTR',
    'sales': 'VHE_Fact_BI_Sales_DTR',
    'stock': 'VHE_Fact_BI_Stock_DTR',
    'final_table': 'VHE_Fact_BI_Sales_DTR_Temp',
}

sales_query_filtered = '''
        SELECT *
        FROM [BI_DTR].dbo.[VHE_Fact_BI_Sales_DTR] WITH (NOLOCK)
        where Registration_Flag = '1' and Chassis_Flag = '1' and VehicleData_Code <> '1'
            and Sales_Type_Code_DMS in ('RAC', 'STOCK', 'UVENDA', 'VENDA') and Sales_Type_Dealer_Code <> 'Demo'
        UNION ALL
        SELECT *
        FROM [BI_DW_History].dbo.[VHE_Fact_BI_Sales_DTR] WITH (NOLOCK) 
        where Registration_Flag = '1' and Chassis_Flag = '1' and VehicleData_Code <> '1'
            and Sales_Type_Code_DMS in ('RAC', 'STOCK', 'UVENDA', 'VENDA') and Sales_Type_Dealer_Code <> 'Demo' '''


sales_query = '''
        SELECT *
        FROM [BI_DTR].dbo.[VHE_Fact_BI_Sales_DTR] WITH (NOLOCK) 
        UNION ALL 
        SELECT *
        FROM [BI_DW_History].dbo.[VHE_Fact_BI_Sales_DTR] WITH (NOLOCK)'''

stock_query = '''
        select *
        from [BI_DTR].dbo.[VHE_Fact_BI_Stock_DTR] WITH (NOLOCK)'''

product_db_query = '''
        SELECT *
        FROM [BI_DTR].dbo.[VHE_Dim_VehicleData_DTR] WITH (NOLOCK)
        UNION ALL
        SELECT *
        FROM [BI_DW_History].dbo.[VHE_Dim_VehicleData_DTR] WITH (NOLOCK)'''

# Motorização
motor_translation = {
    '1.0': ['1.0 lpgi'],
    '1.0i': ['1.0 t-gdi', '1.0i', '1.0l'],
    '1.1d': ['1.1 crdi'],
    '1.2i': ['1.2i'],
    '1.3i': ['1.3l'],
    '1.4d': ['1.4 crdi'],
    '1.4i': ['1.4 t-gdi'],
    '1.5i': ['1.5l'],
    # '1.6': [],
    '1.6d': ['1.6l', '1.6 crdi'],
    '1.6i': ['1.6 t-gdi', '1.6 gdi'],
    '1.7d': ['1.7 crdi'],
    '2.0d': ['2.0 crdi'],
    '2.0i': ['2.0l', '2.0 t-gdi'],
    '2.2d': ['2.2 crdi'],
    '2.5d': ['2.5 crdi'],
    'eletrico': ['motor elétrico'],
    'NÃO_PARAMETRIZADOS': ['1.0 mpi', '1.2 mpi'],
}

# Transmissão
transmission_translation = {
    'Manual': ['manual 6 velocidades', 'manual 5 velocidades', 'mt'],
    'Auto': ['at', 's/info', 'caixa automática 4 velocidades', 'caixa automática 6 velocidades', 'caixa automática 8 velocidades'],
    'CVT': ['cvt'],
    'DCT': ['dct', 'automática de dupla embraiagem de 6 velocidades (6 dct)', 'automática de dupla embraiagem de 7 velocidades (7 dct)'],
}

# Versão
version_translation = {
    'Access': ['access', 'access plus', 'access my17'],
    'Comfort': ['comfort ', 'comfort', 'comfort + connect navi ', 'comfort', 'van 3 lugares', 'comfort my19', 'comfort navi', 'blue comfort my17', 'blue comfort hp my17', 'comfort + navi', 'comfort + connect navi', 'blue comfort'],
    'Creative': ['creative plus'],
    'Dynamic': ['dynamic', 'dynamic + connect navi'],
    'Elegance': ['elegance navi', '1.5 i-vtec turbo cvt elegance navi', '1.6 i-dtec turbo elegance navi', 'elegance ', 'elegance + connect navi ', 'elegance plus + connect n', 'elegance', 'elegance + connect navi', '1.5 i-vtec turbo elegance'],
    'EV': ['ev'],
    'Executive': ['executive ', 'executive', 'executive premium', '1.5 i-vtec turbo executive', '1.5 i-vtec turbo cvt executive', '1.6 i-dtec turbo executive', 'executive', 'executive my19'],
    'GO': ['go', 'go+', 'go!', 'go!+'],
    'HEV': ['hev'],
    'Launch': ['launch edition'],
    'Lifestyle': ['lifestyle', 'lifestyle + navi', 'lifestyle + connect navi'],
    'Performance': ['performance pack'],
    'PHEV': ['phev'],
    'Premium': ['premium', 'premium my19', 'premium my19 + pack pele'],
    'Prestige': ['prestige'],
    'Pro': ['pro edition'],
    'Sport': ['sport plus', 'sport'],
    'Style': ['style', 'comfort my18', 'style my18', 'style plus my18', 'style+', 'blue style hp my17', 'blue style', 'style my19'],
    'Type R': ['gt pack', 'gt'],
    'NÃO_PARAMETRIZADOS': ['trend', 'trend ', 'x-road navi', 'dynamic + connect navi ', 'auto ribeiro', 'teclife', 'van 6 lugares', 'turbo', 'led', 'panorama', 'style + navi', '250cv', 'my18']
}

# Cor Exterior
ext_color_translation = {
    'Amarelo': ['acid yellow', 'acid yellow (tt)'],
    'Azul': ['intense blue', 'brilliant sporty blue m.', 'morpho blue p.', 'stargazing blue', 'champion blue', 'ceramic blue', 'stellar blue', 'blue lagoon', 'performance blue', 'morning blue', 'ara blue', 'marina blue', 'ceramic blue (tt)', 'blue lagoon (tt)', 'skyride blue m.', 'twilight blue m.'],
    'Branco': ['taffeta white', 'platinum white p.', 'white orchid p.', 'polar white', 'white sand', 'creamy white', 'chalk white', 'pure white', 'white crystal', 'white cream', 'chalk white (tt)', 'championship white'],
    'Castanho': ['golden brown m.', 'cashmere brown', 'tan brown', 'demitasse brown', 'premium agate brown p.'],
    'Cinzento': ['shining grey m.', 'modern steel m.', 'micron grey', 'galactic grey', 'iron gray', 'galactic grey (tt)', 'sonic grey p.', 'shadow grey', 'stone gray'],
    'Laranja': ['sunset orange ii'],
    'Prateado': ['lunar silver m.', 'platinum silver', 'sleek silver', 'lake silver', 'aurora silver', 'titanium silver', 'platinum silve', 'typhoon silver', 'lake silver (tt)', 'alabaster silver m.', 'tinted silver m.'],
    'Preto': ['crystal black p.', 'ruse black m.', 'phantom black'],
    'Vermelho': ['rallye red', 'milano red', 'fiery red', 'passion red', 'tomato red', 'pulse red', 'engine red', 'magma red', 'pulse red (tt)', 'passion red p.'],
    'NÃO_PARAMETRIZADOS': ['ral3000', 'ral1016', 'dark knight', 'tangerine comet', 'velvet dune', 'velvet dune (tt)', 'tangerine comet (tt)', 'dark knight (tt)', 'ocean view', 'magnetic force', 'stormy sea', 'rain forest',
                           'wild explorer', 'polished metal m.', 'midnight burgundy p.', 'star dust', 'aqua sparkling', 'iced coffee', 'urban titanium m.', 'moon rock', 'clean slate', 'olivine grey']
}

int_color_translation = {
    'Azul': ['blue'],
    'Bege': ['beige', 'sahara beige', 'dark beige', 'elegant beige'],
    'Bege/Preto': ['beige + black'],
    'Branco': ['ivory'],
    'Cinzento': ['grey', 'pele sintética cinza'],
    'Cinzento Escuro': ['dark grey'],
    'Laranja': ['orange'],
    'Preto': ['black', 'black 2', 'black 3', 'neutral black'],
    'Preto/Castanho': ['black / brown'],
    'Preto/Cinzento': ['black/charcoal'],
    'Preto/Laranja': ['black/orange'],
    'Vermelho': ['red'],
    'NÃO_PARAMETRIZADOS': ['lava stone', 'red point', 'blue point', 'brilliant brown', 'blue grey', 'black + red point'],
}



