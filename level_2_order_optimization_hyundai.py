import sys
import time
import logging
import numpy as np
import pandas as pd
from modules.level_1_a_data_acquisition import sql_retrieve_df_specified_query, read_csv, missing_customer_info_treatment
from modules.level_1_b_data_processing import robust_scaler_function, skewness_reduction, pandas_object_columns_categorical_conversion_auto, pandas_object_columns_categorical_conversion, ohe, constant_columns_removal, dataset_split, new_features, df_join_function, parameter_processing_hyundai, col_group, score_calculation, null_analysis, inf_analysis, lowercase_column_conversion, na_fill_hyundai, remove_columns, measures_calculation_hyundai
from modules.level_1_c_data_modelling import regression_model_training, save_model
from modules.level_1_d_model_evaluation import performance_evaluation_regression, plot_roc_curve, multiprocess_model_evaluation, model_choice, feature_contribution, heatmap_correlation_function
from modules.level_1_e_deployment import sql_inject_v2, time_tags
from modules.level_0_performance_report import log_record
import level_2_order_optimization_hyundai_options as options_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S @ %d/%m/%y', filename=options_file.log_files['full_log'], filemode='a')
logging.Logger('errors')
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # Allows the stdout to be seen in the console
logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))  # Allows the stderr to be seen in the console

oversample_flag = 0


def main():
    models = ['rf', 'lgb', 'xgb', 'ridge', 'll_cv', 'elastic_cv', 'svr']
    # models = ['rf', 'lgb', 'xgb']  # All Available atm
    target = 'DaysInStock_Global'
    configuration_parameters = ['PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc', 'PT_PDB_Exterior_Color_Desc', 'PT_PDB_Interior_Color_Desc']

    df_sales, df_stock, df_pdb_dim, df_customers, df_dealers = data_acquisition()
    df_sales, datasets, datasets_non_ohe = data_processing(df_sales, df_stock, df_pdb_dim, configuration_parameters, target)
    # best_models, running_times = data_modelling(df_sales, datasets, datasets_non_ohe, models)
    # model_choice_message, best_model_name = model_evaluation(df_sales, models, best_models, running_times, datasets, datasets_non_ohe, options_file, configuration_parameters, options_file.project_id)
    # print(model_choice_message, best_model_name)
    deployment(df_sales, options_file.sql_info['database_final'], options_file.sql_info['final_table'])


def data_acquisition():
    log_record('Início Secção A...', options_file.project_id)
    start = time.time()

    sales_info = ['dbs/df_sales', options_file.sales_query]
    stock_info = ['dbs/df_stock', options_file.stock_query]
    product_db = ['dbs/df_pdb', options_file.product_db_query]
    customer_group_info = ['dbs/df_customer_group', options_file.customer_group_query]
    dealers_info = ['dbs/df_dealers', options_file.dealers_query]

    current_date, _ = time_tags()
    # current_date = '2019-09-02'

    dfs = []

    for dimension in [sales_info, stock_info, product_db, customer_group_info, dealers_info]:
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
    df_customers = dfs[3]
    df_dealers = dfs[4]

    df_pdb.drop_duplicates(subset='VehicleData_Code', inplace=True)  # There are repeated VehicleData_Code inside this union between BI_DTR and BI_DW_History

    # df_sales['SLR_Document_Date'] = pd.to_datetime(df_sales['SLR_Document_Date'], format='%Y%m%d')
    # df_sales['WIP_Date_Created'] = pd.to_datetime(df_sales['WIP_Date_Created'], format='%Y%m%d')
    # df_sales['Movement_Date'] = pd.to_datetime(df_sales['Movement_Date'], format='%Y%m%d')
    df_sales['NLR_Code'] = pd.to_numeric(df_sales['NLR_Code'], errors='ignore')
    # df_sales['Product_Code'] = pd.to_numeric(df_sales['Product_Code'], errors='ignore')

    # df_customers = pd.read_csv('dbs/customers_group.csv', encoding='latin-1', sep=';')
    # df_dealers = pd.read_csv('dbs/dealers_dtr.csv', encoding='latin-1', sep=';')

    # Adding missing information regarding customers
    missing_customer_info_treatment(df_sales)

    # Addition of customer information
    df_customers_and_dealers = df_join_function(df_dealers, df_customers[['Customer_Group_Code', 'Customer_Group_Desc']].set_index('Customer_Group_Code'), on='Customer_Group_Code', how='left')
    df_sales = df_join_function(df_sales, df_customers_and_dealers[['SLR_Account_CHS_Key', 'NDB_VATGroup_Desc', 'VAT_Number_Display', 'NDB_Contract_Dealer_Desc', 'NDB_VHE_PerformGroup_Desc', 'NDB_VHE_Team_Desc', 'Customer_Display', 'Customer_Group_Code', 'Customer_Group_Desc']].set_index('SLR_Account_CHS_Key'), on='SLR_Account_CHS_Key', how='left')

    print('Ended section A - Elapsed time: {:.2f}'.format(time.time() - start))
    log_record('Fim Secção A.', options_file.project_id)
    return df_sales, df_stock, df_pdb, df_customers, df_dealers


def data_processing(df_sales, df_stock, df_pdb_dim, configuration_parameters_cols, target):
    # print('Starting section B...')
    log_record('Início Secção B...', options_file.project_id)
    start = time.time()
    current_date, _ = time_tags()
    # current_date = '2019-10-10'

    try:
        df_ohe = read_csv('dbs/df_hyundai_dataset_ml_version_ohe_{}.csv'.format(current_date), index_col=0, dtype={'NDB_VATGroup_Desc': 'category', 'VAT_Number_Display': 'category', 'NDB_Contract_Dealer_Desc': 'category',
                                                                                                                   'NDB_VHE_PerformGroup_Desc': 'category', 'NDB_VHE_Team_Desc': 'category', 'Customer_Display': 'category',
                                                                                                                   'Customer_Group_Desc': 'category', 'SLR_Account_Dealer_Code': 'category', 'Product_Code': 'category',
                                                                                                                   'Sales_Type_Dealer_Code': 'category', 'Sales_Type_Code': 'category', 'Vehicle_Type_Code': 'category', 'Fuel_Type_Code': 'category',
                                                                                                                   'PT_PDB_Model_Desc': 'category', 'PT_PDB_Engine_Desc': 'category', 'PT_PDB_Transmission_Type_Desc': 'category', 'PT_PDB_Version_Desc': 'category',
                                                                                                                   'PT_PDB_Exterior_Color_Desc': 'category', 'PT_PDB_Interior_Color_Desc': 'category'})
        df_non_ohe = read_csv('dbs/df_hyundai_dataset_ml_version_{}.csv'.format(current_date), index_col=0, dtype={'NDB_VATGroup_Desc': 'category', 'VAT_Number_Display': 'category', 'NDB_Contract_Dealer_Desc': 'category',
                                                                                                                   'NDB_VHE_PerformGroup_Desc': 'category', 'NDB_VHE_Team_Desc': 'category', 'Customer_Display': 'category',
                                                                                                                   'Customer_Group_Desc': 'category', 'SLR_Account_Dealer_Code': 'category', 'Product_Code': 'category',
                                                                                                                   'Sales_Type_Dealer_Code': 'category', 'Sales_Type_Code': 'category', 'Vehicle_Type_Code': 'category', 'Fuel_Type_Code': 'category',
                                                                                                                   'PT_PDB_Model_Desc': 'category', 'PT_PDB_Engine_Desc': 'category', 'PT_PDB_Transmission_Type_Desc': 'category', 'PT_PDB_Version_Desc': 'category',
                                                                                                                   'PT_PDB_Exterior_Color_Desc': 'category', 'PT_PDB_Interior_Color_Desc': 'category'})
        df_sales = read_csv('dbs/df_hyundai_dataset_all_info_{}.csv'.format(current_date), index_col=0, parse_dates=['NLR_Posting_Date', 'SLR_Document_Date_CHS', 'Analysis_Date_RGN', 'SLR_Document_Date_RGN', 'Record_Date', 'Registration_Request_Date'])

        # df_non_ohe = pandas_object_columns_categorical_conversion(df_non_ohe, df_non_ohe.columns + ['SLR_Account_Dealer_Code', 'Product_Code', 'Sales_Type_Dealer_Code', 'Sales_Type_Code', 'Vehicle_Type_Code', 'Fuel_Type_Code'])
        print('Current day file found and processed. Skipping to Section C...')
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

        # df_sales.to_csv('dbs/df_sales_importador_processed_{}.csv'.format(current_date))

        # Step 2: ML Processing
        # print('Number of unique Chassis: {} and number of rows: {}'.format(df_sales['Chassis_Number'].nunique(), df_sales.shape[0]))
        print('before join - \n{} \n{}'.format(df_sales.head(), df_sales.shape))
        df_sales = df_join_function(df_sales, df_pdb_dim[['VehicleData_Code'] + configuration_parameters_cols].set_index('VehicleData_Code'), on='VehicleData_Code', how='left')
        print('after join - \n{} \n{}'.format(df_sales.head(), df_sales.shape))

        df_sales = lowercase_column_conversion(df_sales, configuration_parameters_cols)

        # Filtering rows with no relevant information
        # print('1 - Number of unique Chassis: {} and number of rows: {}'.format(df_sales['Chassis_Number'].nunique(), df_sales.shape[0]))
        # df_sales = df_sales[df_sales['NLR_Code'] == 702]  # Escolha de viaturas apenas Hyundai
        print('2 - Number of unique Chassis: {} and number of rows: {}'.format(df_sales['Chassis_Number'].nunique(), df_sales.shape[0]))
        df_sales = df_sales[df_sales['VehicleData_Code'] != 1]
        print('3 - Number of unique Chassis: {} and number of rows: {}'.format(df_sales['Chassis_Number'].nunique(), df_sales.shape[0]))
        df_sales = df_sales[df_sales['Sales_Type_Dealer_Code'] != 'Demo']
        print('4 - Number of unique Chassis: {} and number of rows: {}'.format(df_sales['Chassis_Number'].nunique(), df_sales.shape[0]))
        df_sales = df_sales[df_sales['Sales_Type_Code_DMS'].isin(['RAC', 'STOCK', 'VENDA'])]
        print('5 - Number of unique Chassis: {} and number of rows: {}'.format(df_sales['Chassis_Number'].nunique(), df_sales.shape[0]))
        df_sales = df_sales[~df_sales['Dispatch_Type_Code'].isin(['AMBULÂNCIA', 'TAXI', 'PSP'])]
        print('6 - Number of unique Chassis: {} and number of rows: {}'.format(df_sales['Chassis_Number'].nunique(), df_sales.shape[0]))
        df_sales = df_sales[df_sales['DaysInStock_Global'] >= 0]  # Filters rows where, for some odd reason, the days in stock are negative
        print('7 - Number of unique Chassis: {} and number of rows: {}'.format(df_sales['Chassis_Number'].nunique(), df_sales.shape[0]))
        df_sales = df_sales[df_sales['Registration_Number'] != 'G.FORCE']  # Filters rows where, for some odd reason, the days in stock are negative
        print('8 - Number of unique Chassis: {} and number of rows: {}'.format(df_sales['Chassis_Number'].nunique(), df_sales.shape[0]))
        df_sales = df_sales[df_sales['Customer_Group_Code'].notnull()]  # Filters rows where there is no client information;
        print('9 - Number of unique Chassis: {} and number of rows: {}'.format(df_sales['Chassis_Number'].nunique(), df_sales.shape[0]))

        df_sales = new_features(df_sales, configuration_parameters_cols, options_file.project_id)

        # Remove unnecessary columns:
        # df_sales = remove_columns(df_sales, ['Client_Id', 'Record_Type', 'Vehicle_ID', 'SLR_Document', 'SLR_Document_Account', 'VHE_Type_Orig', 'VHE_Type', 'Registration_Number',
        #                                      'NLR_Posting_Date', 'SLR_Document_Category', 'Chassis_Flag', 'SLR_Document_Date_CHS', 'SLR_Document_Period_CHS', 'SLR_Document_Year_CHS', 'SLR_Document_CHS', 'SLR_Document_Type_CHS',
        #                                      'SLR_Account_CHS', 'SLR_Account_CHS_Key', 'Quantity_CHS', 'Registration_Flag', 'Analysis_Date_RGN', 'Analysis_Period_RGN', 'Analysis_Year_RGN', 'SLR_Document_Date_RGN',
        #                                      'SLR_Document_RGN', 'SLR_Document_Type_RGN', 'SLR_Account_RGN', 'SLR_Account_RGN_Key', 'Quantity_RGN', 'Sales_Type_Code_DMS', 'Location_Code', 'VehicleData_Key',
        #                                      'Record_Date', 'Currency_Rate', 'Currency_Rate2', 'Currency_Rate3', 'Currency_Rate4', 'Currency_Rate5', 'Currency_Rate6', 'Currency_Rate7', 'Currency_Rate8',
        #                                      'Dispatch_Type_Code', 'Currency_Rate9', 'Currency_Rate10', 'Currency_Rate11', 'Currency_Rate12', 'Currency_Rate13', 'Currency_Rate14', 'Currency_Rate15', 'Stock_Age_Distributor_Code',
        #                                      'Stock_Age_Dealer_Code', 'Stock_Age_Global_Code', 'Immobilized_Number', 'SLR_Account_Dealer_Code', 'Salesman_Dealer_Code', 'Ship_Arrival_Date', 'Registration_Request_Date', 'Registration_Date', 'Vehicle_Code',
        #                                      'PDB_Vehicle_Type_Code_DMS', 'PDB_Fuel_Type_Code_DMS', 'PDB_Transmission_Type_Code_DMS'], options_file.project_id)

        # df_sales = remove_columns(df_sales, ['Client_Id', 'Record_Type', 'Vehicle_ID', 'SLR_Document', 'SLR_Document_Account', 'VHE_Type_Orig', 'VHE_Type',
        #                                      'SLR_Document_Category', 'Chassis_Flag', 'SLR_Document_Period_CHS', 'SLR_Document_Year_CHS', 'SLR_Document_CHS', 'SLR_Document_Type_CHS',
        #                                      'SLR_Account_CHS', 'Quantity_CHS', 'Registration_Flag', 'Analysis_Date_RGN', 'Analysis_Period_RGN', 'Analysis_Year_RGN',
        #                                      'SLR_Document_RGN', 'SLR_Document_Type_RGN', 'SLR_Account_RGN', 'SLR_Account_RGN_Key', 'Quantity_RGN', 'Sales_Type_Code_DMS', 'VehicleData_Key',
        #                                      'Record_Date', 'Currency_Rate', 'Currency_Rate2', 'Currency_Rate3', 'Currency_Rate4', 'Currency_Rate5', 'Currency_Rate6', 'Currency_Rate7', 'Currency_Rate8',
        #                                      'Currency_Rate9', 'Currency_Rate10', 'Currency_Rate11', 'Currency_Rate12', 'Currency_Rate13', 'Currency_Rate14', 'Currency_Rate15', 'Stock_Age_Distributor_Code',
        #                                      'Stock_Age_Dealer_Code', 'Stock_Age_Global_Code', 'Immobilized_Number', 'Salesman_Dealer_Code', 'Vehicle_Code',
        #                                      'PDB_Vehicle_Type_Code_DMS', 'PDB_Fuel_Type_Code_DMS', 'PDB_Transmission_Type_Code_DMS', 'Transmission_Type_Code', 'Customer_Group_Code'], options_file.project_id)

        # Specific Measures Calculation
        df_sales = measures_calculation_hyundai(df_sales)

        # Fill values
        df_sales['Total_Discount_%'] = df_sales['Total_Discount_%'].replace([np.inf, np.nan, -np.inf], 0)  # Is this correct? This is caused by Total Sales = 0
        df_sales['Fixed_Margin_I_%'] = df_sales['Fixed_Margin_I_%'].replace([np.inf, np.nan, -np.inf], 0)  # Is this correct? This is caused by Total Net Sales = 0

        df_sales = lowercase_column_conversion(df_sales, configuration_parameters_cols)  # Lowercases the strings of these columns

        df_sales = parameter_processing_hyundai(df_sales, options_file, configuration_parameters_cols)

        translation_dictionaries = [options_file.motor_translation, options_file.transmission_translation, options_file.version_translation, options_file.ext_color_translation, options_file.int_color_translation]
        grouping_dictionaries = [options_file.motor_grouping, options_file.transmission_grouping, options_file.version_grouping, options_file.ext_color_grouping, options_file.int_color_grouping]

        # Parameter Translation
        df_sales = col_group(df_sales, [x for x in configuration_parameters_cols if 'Model' not in x], translation_dictionaries, options_file.project_id)
        df_sales = df_sales[df_sales['PT_PDB_Version_Desc'] != 'NÃO_PARAMETRIZADOS']
        print('10 - Number of unique Chassis: {} and number of rows: {}'.format(df_sales['Chassis_Number'].nunique(), df_sales.shape[0]))

        # Target Variable Calculation
        # df_sales = score_calculation(df_sales, options_file.stock_days_threshold, options_file.margin_threshold, options_file.project_id)

        # value_count_histogram(df_sales, configuration_parameters_cols + ['target_class'] + ['DaysInStock_Global'], 'hyundai_2406_translation')

        # df_sales.to_csv('output/processing_test.csv')
        # Parameter Grouping
        # df_sales.to_csv('output/df_test_unit.csv')
        print('### NO GROUPING ###')
        # df_sales = col_group(df_sales, [x for x in configuration_parameters_cols if 'Model' not in x], grouping_dictionaries, options_file.project_id)

        print('Number of Different VehicleData_Code: {}'.format(df_sales['VehicleData_Code'].nunique()))
        df_sales_grouped_conf_cols = df_sales.groupby(configuration_parameters_cols)

        print('Number of Different Configurations: {}'.format(len(df_sales_grouped_conf_cols)))

        # value_count_histogram(df_sales, configuration_parameters_cols + ['target_class'] + ['DaysInStock_Global'], 'hyundai_2406_grouping')

        # New VehicleData_Code Creation
        df_sales['ML_VehicleData_Code'] = df_sales.groupby(configuration_parameters_cols).ngroup()
        df_sales.to_csv('dbs/df_hyundai_dataset_all_info_{}.csv'.format(current_date))

        df_sales = df_sales[df_sales['NLR_Code'] == 702]  # Escolha de viaturas apenas Hyundai

        columns_with_too_much_info_2 = ['Client_Id', 'Record_Type', 'Vehicle_ID', 'SLR_Document', 'SLR_Document_Account', 'VHE_Type_Orig', 'VHE_Type',
                                        'SLR_Document_Category', 'Chassis_Flag', 'SLR_Document_Period_CHS', 'SLR_Document_Year_CHS', 'SLR_Document_CHS', 'SLR_Document_Type_CHS',
                                        'SLR_Account_CHS', 'Quantity_CHS', 'Registration_Flag', 'Analysis_Date_RGN', 'Analysis_Period_RGN', 'Analysis_Year_RGN',
                                        'SLR_Document_RGN', 'SLR_Document_Type_RGN', 'SLR_Account_RGN', 'SLR_Account_RGN_Key', 'Quantity_RGN', 'Sales_Type_Code_DMS', 'VehicleData_Key',
                                        'Record_Date', 'Currency_Rate', 'Currency_Rate2', 'Currency_Rate3', 'Currency_Rate4', 'Currency_Rate5', 'Currency_Rate6', 'Currency_Rate7', 'Currency_Rate8',
                                        'Currency_Rate9', 'Currency_Rate10', 'Currency_Rate11', 'Currency_Rate12', 'Currency_Rate13', 'Currency_Rate14', 'Currency_Rate15', 'Stock_Age_Distributor_Code',
                                        'Stock_Age_Dealer_Code', 'Stock_Age_Global_Code', 'Immobilized_Number', 'Salesman_Dealer_Code', 'Vehicle_Code',
                                        'PDB_Vehicle_Type_Code_DMS', 'PDB_Fuel_Type_Code_DMS', 'PDB_Transmission_Type_Code_DMS', 'Transmission_Type_Code', 'Customer_Group_Code']

        df_sales.to_csv('output/df_hyundai_dataset_all_info_{}.csv'.format(current_date))

        columns_with_too_much_info = ['Measure_' + str(x) for x in [2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 40, 41, 42, 43]] + \
                                     ['Chassis_Number', 'Total_Sales', 'Total_Discount', 'Total_Discount_%', 'Total_Net_Sales', 'Fixed_Margin_I', 'Fixed_Margin_II', 'Fixed_Margin_I_%', 'stock_days_class', 'DaysInStock_Dealer',
                                      'DaysInStock_Distributor', 'Average_DaysInStock_Global', 'HME_Support', 'Quality_Margin', 'Total_Network_Support', 'Registration_Number', 'NLR_Posting_Date', 'SLR_Document_Date_CHS', 'SLR_Account_CHS_Key', 'SLR_Document_Date_RGN', 'Location_Code',
                                      'Dispatch_Type_Code', 'SLR_Account_Dealer_Code', 'Ship_Arrival_Date', 'Registration_Request_Date', 'Registration_Date', 'Fixed_Margin_II_%', 'DaysInStock_Global']

        df_non_ohe = remove_columns(df_sales, [x for x in columns_with_too_much_info + columns_with_too_much_info_2 if x not in target], options_file.project_id)

        df_non_ohe = constant_columns_removal(df_non_ohe, options_file.project_id)

        # heatmap_correlation_function(df_ohe, 'target_class', 'heatmap_hyundai_v1')

        df_non_ohe = pandas_object_columns_categorical_conversion_auto(df_non_ohe)
        df_non_ohe = pandas_object_columns_categorical_conversion(df_non_ohe, ['Product_Code', 'Sales_Type_Dealer_Code', 'Sales_Type_Code', 'Vehicle_Type_Code', 'Fuel_Type_Code'])

        df_non_ohe.to_csv('dbs/df_hyundai_dataset_ml_version_{}_non_scaled.csv'.format(current_date))

        df_non_ohe = skewness_reduction(df_non_ohe, target)
        df_non_ohe = robust_scaler_function(df_non_ohe, target)

        df_non_ohe.to_csv('dbs/df_hyundai_dataset_ml_version_{}.csv'.format(current_date))

        df_ohe = ohe(df_non_ohe, configuration_parameters_cols + ['Product_Code', 'Fuel_Type_Code', 'Sales_Type_Code', 'Vehicle_Type_Code', 'Sales_Type_Dealer_Code', 'NDB_VATGroup_Desc', 'VAT_Number_Display', 'NDB_Contract_Dealer_Desc', 'NDB_VHE_PerformGroup_Desc', 'NDB_VHE_Team_Desc', 'Customer_Display', 'Customer_Group_Desc'])
        df_ohe.to_csv('dbs/df_hyundai_dataset_ml_version_ohe_{}.csv'.format(current_date))

        # heatmap_correlation_function(df_ohe, 'target_class', 'heatmap_hyundai_ohe_v1')

    datasets_non_ohe = dataset_split(df_non_ohe, [target], oversample_flag, objective='regression')
    datasets = dataset_split(df_ohe, [target], oversample_flag, objective='regression')

    print('Ended section B - Elapsed time: {:.2f}'.format(time.time() - start))
    log_record('Fim Secção B.', options_file.project_id)
    return df_sales, datasets, datasets_non_ohe


def data_modelling(df_sales, datasets, datasets_non_ohe, models):
    log_record('Início Secção C...', options_file.project_id)
    start = time.time()

    best_models, running_times = regression_model_training(models, datasets['train_x'], datasets_non_ohe['train_x'], datasets['train_y'], datasets_non_ohe['train_y'], options_file.regression_models_standard, options_file.k, options_file.gridsearch_score, options_file.project_id)
    save_model(best_models, models, options_file.project_id)

    print('Ended section C - Elapsed time: {:.2f}'.format(time.time() - start))
    log_record('Fim Secção C.', options_file.project_id)
    return best_models, running_times


def model_evaluation(df_sales, models, best_models, running_times, datasets, datasets_non_ohe, in_options_file, configuration_parameters, project_id):
    # print('Starting Step D...')
    log_record('Início Secção D...', options_file.project_id)
    start = time.time()

    results_training, results_test, predictions = performance_evaluation_regression(models, best_models, running_times, datasets, datasets_non_ohe, in_options_file, project_id)  # Creates a df with the performance of each model evaluated in various metrics

    # df_model_dict = multiprocess_model_evaluation(df_sales, models, datasets, best_models, predictions, configuration_parameters, oversample_flag, project_id)

    model_choice_message, best_model_name, _, section_e_upload_flag = model_choice(options_file.DSN_MLG, options_file, results_test)
    # best_model = df_model_dict[best_model_name]

    print('Best Model Name: {}'.format(best_model_name))

    # feature_contribution(best_model, configuration_parameters, 'PT_PDB_Model_Desc', options_file, project_id)

    print('Ended section D - Elapsed time: {:.2f}'.format(time.time() - start))
    log_record('Fim Secção D.', options_file.project_id)
    return model_choice_message, best_model_name


def deployment(df, db, view):
    log_record('Início Secção E...', options_file.project_id)
    start = time.time()

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
        sql_inject_v2(df[options_file.sql_columns_vhe_fact_bi], options_file.DSN_MLG, db, view, options_file, list(df[options_file.sql_columns_vhe_fact_bi]), truncate=1, check_date=1)
        # sql_inject_v1(df, options_file.DSN_MLG, db, view, options_file, list(df), truncate=1, check_date=1)

    print('Secção E terminada - Duração: {:.2f}'.format(time.time() - start))
    log_record('Fim Secção E.', options_file.project_id)


if __name__ == '__main__':
    main()
