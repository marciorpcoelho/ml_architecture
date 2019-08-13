import time
import pandas as pd
import sys
from level_1_a_data_acquisition import dw_data_retrieval, autoline_data_retrieval
from level_1_b_data_processing import apv_dataset_treatment
from level_1_c_data_modelling import apv_stock_evolution_calculation, part_ref_selection
from level_1_e_deployment import time_tags
import level_2_order_optimization_apv_baviera_options as options_file

update = 1  # Decides whether to fetch new datasets from the DW or not


def main():
    selected_parts = []
    min_date = '20180131'  # This is a placeholder for the minimum possible date. It should search for last date with processed data
    max_date = '20190731'  # This will be replaced by current date
    print('Full Available Data: {} to {}'.format(min_date, max_date))

    df_sales, df_purchases, df_stock, df_reg, df_reg_al_clients, df_al = data_acquistion(options_file)
    df_sales_cleaned, df_purchases_cleaned, df_stock = data_processing(df_sales, df_purchases, df_stock, options_file)
    results = data_modelling(options_file.pse_code, selected_parts, df_sales_cleaned, df_al, df_stock, df_reg_al_clients, df_purchases_cleaned, min_date, max_date)


def data_acquistion(options_info):
    print('Starting section A...')

    pse_code = options_info.pse_code
    start = time.time()
    current_date, _ = time_tags(format_date='%Y%m%d')

    df_sales, df_purchases, df_stock, df_reg, df_reg_al_clients = dw_data_retrieval(pse_code, current_date, options_info, update)
    df_al = autoline_data_retrieval(pse_code)
    print('Ended section A - Elapsed time: {:.2f}'.format(time.time() - start))

    return df_sales, df_purchases, df_stock, df_reg, df_reg_al_clients, df_al


def data_processing(df_sales, df_purchases, df_stock, options_info):
    print('Dataset processing started.')
    start_treatment = time.time()

    df_sales, df_purchases, df_stock = apv_dataset_treatment(df_sales, df_purchases, df_stock, options_info.pse_code, options_info.urgent_purchases_flags)

    print('Dataset processing finished. Elapsed time: {:.2f}'.format(time.time() - start_treatment))
    return df_sales, df_purchases, df_stock


def data_modelling(pse_code, selected_parts, df_sales, df_al, df_stock, df_reg_al_clients, df_purchases, min_date, max_date):
    print('Data modelling started...')

    if pse_code == '0I':
        selected_parts = ['BM83.21.2.405.675', 'BM07.12.9.952.104', 'BM07.14.9.213.164', 'BM83.19.2.158.851', 'BM64.11.9.237.555']  # PSE_Code = 0I, Lisboa - Expo
    if pse_code == '0B':
        selected_parts = ['BM83.21.0.406.573', 'BM83.13.9.415.965', 'BM51.18.1.813.017', 'BM11.42.8.507.683', 'BM64.11.9.237.555']  # PSE_Code = 0B, Gaia

    selected_parts = part_ref_selection(df_al, max_date)
    results = apv_stock_evolution_calculation(pse_code, selected_parts, df_sales, df_al, df_stock, df_reg_al_clients, df_purchases, min_date, max_date)

    return results


if __name__ == '__main__':
    main()
