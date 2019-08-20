import os
import sys
import numpy as np
import time
import pandas as pd
from level_1_a_data_acquisition import sql_retrieve_df_specified_query, read_csv
from level_1_b_data_processing import df_join_function, null_analysis, lowercase_column_convertion, col_group, value_count_histogram, value_substitution, na_fill_hyundai, remove_columns, measures_calculation_hyundai
from level_1_e_deployment import sql_inject_v2, time_tags
import level_0_performance_report
import level_2_order_optimization_hyundai_options as options_file


def main():
    models = ['dt', 'rf', 'lr', 'ab', 'xgb', 'lgb']
    tentative_daysinstock_limit = 30  # DaysInStock_Global
    tentative_margin_limit = 3  # Fixed_Margin_I_%

    df_sales, df_stock, df_pdb_dim = data_acquisition()
    df_sales = data_processing(df_sales, df_stock, df_pdb_dim)
    best_models, running_times = data_modelling(df_sales, models)
    # deployment(df_sales, options_file.sql_info['database_final'], options_file.sql_info['final_table'])


def data_acquisition():
    print('Starting section A...')
    start = time.time()

    sales_info = ['dbs/df_sales', options_file.sales_query]
    stock_info = ['dbs/df_stock', options_file.stock_query]
    product_db = ['dbs/df_pdb', options_file.product_db]

    current_date, _ = time_tags()

    dfs = []

    for dimension in [sales_info, stock_info, product_db]:
        file_name = dimension[0] + '_' + str(current_date)
        try:
            df = pd.read_csv(file_name + '.csv', index_col=0, low_memory=False, dtype={'SLR_Document_Period_CHS': 'Int64', 'SLR_Document_Year_CHS': 'Int64', 'SLR_Document_Type_CHS': 'Int64', 'Analysis_Period_RGN': 'Int64', 'Analysis_Year_RGN': 'Int64',
                                                                                          'PDB_Vehicle_Type_Code_DMS': 'Int64', 'PDB_Fuel_Type_Code_DMS': 'Int64', 'PDB_Transmission_Type_Code_DMS': 'Int64'})
            print('{} file found.'.format(file_name))
        except FileNotFoundError:
            print('{} file not found. Retrieving data from SQL...'.format(file_name))
            df = sql_retrieve_df_specified_query(options_file.DSN_PRD, options_file.sql_info['database_source'], options_file, dimension[1])
            df.to_csv(file_name + '.csv')

        dfs.append(df)

    df_sales = dfs[0]
    df_stock = dfs[1]
    df_pdb = dfs[2]

    # df_sales['SLR_Document_Date'] = pd.to_datetime(df_sales['SLR_Document_Date'], format='%Y%m%d')
    # df_sales['WIP_Date_Created'] = pd.to_datetime(df_sales['WIP_Date_Created'], format='%Y%m%d')
    # df_sales['Movement_Date'] = pd.to_datetime(df_sales['Movement_Date'], format='%Y%m%d')

    print('Ended section A - Elapsed time: {:.2f}'.format(time.time() - start))

    return df_sales, df_stock, df_pdb


def data_processing(df_sales, df_stock, df_pdb_dim):
    current_date, _ = time_tags()

    try:
        df_sales = read_csv('dbs/df_sales_importador_processed_{}.csv'.format(current_date), index_col=0, parse_dates=['NLR_Posting_Date', 'SLR_Document_Date_CHS', 'Analysis_Date_RGN', 'SLR_Document_Date_RGN', 'Record_Date'])
        print('Current day file found. Skipping to step 2...')
    except FileNotFoundError:
        print('Current day file not found. Processing...')

        # Step 1 - Dataset cleaning and transforming to 1 line per sale
        # df_sales = df_sales[df_sales['VehicleData_Code'] == 38]

        columns_to_convert_to_datetime = ['Ship_Arrival_Date', 'SLR_Document_Date_CHS', 'Registration_Request_Date', 'SLR_Document_Date_RGN']
        for column in columns_to_convert_to_datetime:
            df_sales[column] = pd.to_datetime(df_sales[column])

        # Filtering
        print('\nInitial Unique Chassis Count: {}'.format(df_sales['Chassis_Number'].nunique()))
        print('Initial Unique Registration Count: {}'.format(df_sales['Registration_Number'].nunique()))

        print('Removal of 49-VG-94 Registration Plate, which presents two Chassis Number')
        df_sales = df_sales[~(df_sales['Registration_Number'] == '49-VG-94')].copy()

        print('\nInitial Unique Chassis Count: {}'.format(df_sales['Chassis_Number'].nunique()))
        print('Initial Unique Registration Count: {}'.format(df_sales['Registration_Number'].nunique()))

        # Sorting
        df_sales.sort_values(['Ship_Arrival_Date', 'SLR_Document_Date_CHS', 'Registration_Request_Date', 'SLR_Document_Date_RGN'])

        df_sales['No_Registration_Number_Flag'] = 0
        df_sales['Registration_Number_No_SLR_Document_RGN_Flag'] = 0
        df_sales['SLR_Document_RGN_Flag'] = 0
        df_sales['Undefined_VHE_Status'] = 0

        df_sales_grouped_3 = df_sales.groupby(['Chassis_Number', 'Registration_Number'])
        df_sales = na_fill_hyundai(df_sales_grouped_3)
        # print(null_analysis(df_sales))

        # New Column Creation
        df_sales_grouped = df_sales.groupby(['VehicleData_Code'])
        df_sales['Quantity_Sold'] = df_sales_grouped['Quantity_CHS'].transform('sum')
        df_sales['Quantity_Sold'] = df_sales['Quantity_Sold'].astype(np.int64, errors='ignore')

        df_sales_unique_chassis = df_sales.drop_duplicates(subset=['VehicleData_Code', 'Chassis_Number']).copy()
        df_sales_grouped_2 = df_sales_unique_chassis.groupby(['VehicleData_Code'])
        df_sales['Average_DaysInStock_Global'] = df_sales_grouped_2['DaysInStock_Global'].transform('mean').round(3)

        df_sales.to_csv('dbs/df_sales_importador_processed_{}.csv'.format(current_date))

    # description_cols = [x for x in list(df_pdb_dim) if x.endswith('_Desc')]
    description_cols = ['PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc', 'PT_PDB_Exterior_Color_Desc', 'PT_PDB_Interior_Color_Desc']

    # Step 2: ML Processing
    df_sales = df_join_function(df_sales, df_pdb_dim[['VehicleData_Code'] + description_cols].set_index('VehicleData_Code'), on='VehicleData_Code')

    df_sales = lowercase_column_convertion(df_sales, description_cols)

    # print(df_sales.head())

    # Filtering rows with no information
    df_sales = df_sales[df_sales['VehicleData_Code'] != 1]
    df_sales = df_sales[df_sales['Sales_Type_Dealer_Code'] != 'Demo']
    df_sales = df_sales[df_sales['Sales_Type_Code_DMS'].isin(['RAC', 'STOCK', 'VENDA'])]

    # Remove unnecessary columns:
    df_sales = remove_columns(df_sales, ['VehicleData_Code', 'Client_Id', 'Record_Type', 'Vehicle_ID', 'SLR_Document', 'SLR_Document_Account', 'VHE_Type_Orig', 'VHE_Type', 'Registration_Number',
                                         'NLR_Posting_Date', 'SLR_Document_Category', 'Chassis_Flag', 'SLR_Document_Date_CHS', 'SLR_Document_Period_CHS', 'SLR_Document_Year_CHS', 'SLR_Document_CHS', 'SLR_Document_Type_CHS',
                                         'SLR_Account_CHS', 'SLR_Account_CHS_Key', 'Quantity_CHS', 'Registration_Flag', 'Analysis_Date_RGN', 'Analysis_Period_RGN', 'Analysis_Year_RGN', 'SLR_Document_Date_RGN',
                                         'SLR_Document_RGN', 'SLR_Document_Type_RGN', 'SLR_Account_RGN', 'SLR_Account_RGN_Key', 'Quantity_RGN', 'Sales_Type_Code_DMS', 'Location_Code', 'VehicleData_Key',
                                         'Record_Date', 'Currency_Rate', 'Currency_Rate2', 'Currency_Rate3', 'Currency_Rate4', 'Currency_Rate5', 'Currency_Rate6', 'Currency_Rate7', 'Currency_Rate8',
                                         'Dispatch_Type_Code', 'Currency_Rate9', 'Currency_Rate10', 'Currency_Rate11', 'Currency_Rate12', 'Currency_Rate13', 'Currency_Rate14', 'Currency_Rate15', 'Stock_Age_Distributor_Code',
                                         'Stock_Age_Dealer_Code', 'Stock_Age_Global_Code', 'Immobilized_Number', 'SLR_Account_Dealer_Code', 'Salesman_Dealer_Code', 'Ship_Arrival_Date', 'Registration_Request_Date', 'Registration_Date', 'Vehicle_Code'], options_file.project_id)

    # Specific Measures Calculation
    df_sales = measures_calculation_hyundai(df_sales)

    # Fill values
    df_sales['Total_Discount_%'] = df_sales['Total_Discount_%'].replace([np.inf, np.nan, -np.inf], 0)  # Is this correct? This is caused by Total Sales = 0
    df_sales['Fixed_Margin_I_%'] = df_sales['Fixed_Margin_I_%'].replace([np.inf, np.nan, -np.inf], 0)  # Is this correct? This is caused by Total Net Sales = 0

    print(df_sales['PT_PDB_Interior_Color_Desc'].unique())
    print(df_sales['PT_PDB_Interior_Color_Desc'].nunique())

    df_sales = parameter_processing(df_sales)

    print(df_sales['PT_PDB_Interior_Color_Desc'].unique())
    print(df_sales['PT_PDB_Interior_Color_Desc'].nunique())

    # print(df_sales.head())
    # print(df_sales.describe())
    # null_analysis(df_sales)

    # print(df_sales.shape)
    # print(df_sales['Chassis_Number'].nunique())
    # print(df_sales['Chassis_Number'].value_counts())
    # print(df_sales[df_sales['Chassis_Number'] == 'KMHC851CGJU091903'])
    # print('Number of rows with infinity: {}'.format(df_sales[df_sales['Fixed_Margin_I_%'].isin([np.inf, -np.inf])].shape[0]))

    return df_sales


def data_modelling(df_sales, models):
    print('Started Step C...')



    print('Finished Step C.')
    a, b = 1, 2
    return a, b


def parameter_processing(df_sales):

    print(df_sales.head())
    description_cols = ['PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc', 'PT_PDB_Exterior_Color_Desc', 'PT_PDB_Interior_Color_Desc']

    # Modelo
    df_sales = lowercase_column_convertion(df_sales, description_cols)  # Lowercases the strings of these columns
    df_sales.loc[:, 'Modelo'] = df_sales['PT_PDB_Model_Desc'].str.split().str[0]

    translation_dictionaries = [options_file.motor_translation, options_file.transmission_translation, options_file.version_translation, options_file.ext_color_translation, options_file.int_color_translation]

    df_sales = col_group(df_sales, description_cols, translation_dictionaries, options_file.project_id)

    return df_sales


def deployment(df, db, view):

    columns_to_convert_to_datetime = ['NLR_Posting_Date', 'SLR_Document_Date_CHS', 'Analysis_Date_RGN', 'SLR_Document_Date_RGN', 'Ship_Arrival_Date', 'Registration_Request_Date', 'Registration_Date', 'Record_Date']
    columns_to_convert_to_int = ['SLR_Document_Period_CHS', 'Quantity_Sold', 'SLR_Document_Year_CHS']
    # columns_to_convert_to_string = ['SLR_Document_Year_CHS']

    # for column in columns_to_convert_to_datetime:
    #     df[column] = pd.to_datetime(df[column])

    # for column in columns_to_convert_to_int:
    #     df[column] = df[column].astype(np.int64, errors='ignore')

    # for column in columns_to_convert_to_int:
    #     df[column] = pd.to_numeric(df[column])

    df = df.astype(object).where(pd.notnull(df), 'NULL')

    # columns = ['Client_Id', 'NLR_Code', 'Environment', 'DMS_Type', 'Value_Type_Code', 'Record_Type', 'Vehicle_ID', 'SLR_Document', 'SLR_Document_Account', 'VHE_Type_Orig', 'VHE_Type', 'Chassis_Number', 'Registration_Number', 'NLR_Posting_Date', 'SLR_Document_Category', 'Chassis_Flag', 'SLR_Document_Date_CHS', 'SLR_Document_Period_CHS', 'SLR_Document_Year_CHS', 'SLR_Document_CHS', 'SLR_Document_Type_CHS', 'SLR_Account_CHS', 'SLR_Account_CHS_Key', 'Quantity_CHS', 'Registration_Flag', 'Analysis_Date_RGN', 'Analysis_Period_RGN', 'Analysis_Year_RGN', 'SLR_Document_Date_RGN', 'SLR_Document_RGN', 'SLR_Document_Type_RGN', 'SLR_Account_RGN', 'SLR_Account_RGN_Key', 'Quantity_RGN', 'Product_Code', 'Sales_Type_Code_DMS', 'Sales_Type_Code', 'Location_Code', 'VehicleData_Key', 'VehicleData_Code', 'Vehicle_Code', 'PDB_Vehicle_Type_Code_DMS', 'Vehicle_Type_Code', 'PDB_Fuel_Type_Code_DMS', 'Fuel_Type_Code', 'PDB_Transmission_Type_Code_DMS', 'Transmission_Type_Code', 'Vehicle_Area_Code', 'Dispatch_Type_Code', 'Sales_Status_Code', 'Ship_Arrival_Date', 'Registration_Request_Date', 'Registration_Date', 'DaysInStock_Distributor', 'Stock_Age_Distributor_Code', 'DaysInStock_Dealer', 'Stock_Age_Dealer_Code', 'DaysInStock_Global', 'Stock_Age_Global_Code', 'Immobilized_Number', 'SLR_Account_Dealer_Code', 'Salesman_Dealer_Code', 'Sales_Type_Dealer_Code', 'Measure_1', 'Measure_2', 'Measure_3', 'Measure_4', 'Measure_5', 'Measure_6', 'Measure_7', 'Measure_8', 'Measure_9', 'Measure_10', 'Measure_11', 'Measure_12', 'Measure_13', 'Measure_14', 'Measure_15', 'Measure_16', 'Measure_17', 'Measure_18', 'Measure_19', 'Measure_20', 'Measure_21', 'Measure_22', 'Measure_23', 'Measure_24', 'Measure_25', 'Measure_26', 'Measure_27', 'Measure_28', 'Measure_29', 'Measure_30', 'Measure_31', 'Measure_32', 'Measure_33', 'Measure_34', 'Measure_35', 'Measure_36', 'Measure_37', 'Measure_38', 'Measure_39', 'Measure_40', 'Measure_41', 'Measure_42', 'Measure_43', 'Measure_44', 'Measure_45', 'Measure_46', 'Currency_Rate', 'Currency_Rate2', 'Currency_Rate3', 'Currency_Rate4', 'Currency_Rate5', 'Currency_Rate6', 'Currency_Rate7', 'Currency_Rate8', 'Currency_Rate9', 'Currency_Rate10', 'Currency_Rate11', 'Currency_Rate12', 'Currency_Rate13', 'Currency_Rate14', 'Currency_Rate15', 'Record_Date', 'Quantity_Sold', 'Average_DaysInStock_Global']

    # for column in list(df):
    #     df[column] = df[column].astype(str)

    if df is not None:
        sql_inject_v2(df, options_file.DSN_MLG, db, view, options_file, list(df), truncate=1, check_date=1)
        # sql_inject_v1(df, options_file.DSN_MLG, db, view, options_file, list(df), truncate=1, check_date=1)


if __name__ == '__main__':
    main()
