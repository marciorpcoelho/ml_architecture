import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')) + '\\'

"""
# Streamlit second test
Here's my first attempt at deploying a solver
"""

configuration_parameters_full = ['Motor_Desc', 'Alarm', 'AC_Auto', 'Open_Roof', 'Auto_Trans', 'Colour_Ext', 'Colour_Int', 'LED_Lights', 'Xenon_Lights', 'Rims_Size', 'Model_Code', 'Navigation', 'Park_Front_Sens', 'Roof_Bars', 'Interior_Type', 'Version']
extra_parameters = ['Average_Score_Euros_Local_Fase2_Level_1', 'Number_Cars_Sold_Local_Fase2_Level_1', 'Sales_Place_Fase2_Level_1']


def main():
    dataset_name = base_path + 'output/bmw_dataset.csv'
    data = get_data(dataset_name)

    print(data[['Model_Code', 'Average_Score_Euros_Local_Fase2_Level_1']].head())
    st.write('Dataset:', data)
    st.write('Dataset Size:', data.shape)
    max_number_of_cars_sold = max(data['Number_Cars_Sold_Local_Fase2_Level_1'])

    sel_local = st.sidebar.selectbox('Please select a Location', ['None'] + list(data['Sales_Place_Fase2_Level_1'].unique()), index=0)
    sel_model = st.sidebar.selectbox('Please select a model to optimize the order for:', ['None'] + list(data['Model_Code'].unique()), index=0)

    if 'None' not in [sel_local, sel_model]:
        max_number_of_cars_sold = max(data[(data['Sales_Place_Fase2_Level_1'] == sel_local) & (data['Model_Code'] == sel_model)]['Number_Cars_Sold_Local_Fase2_Level_1'])

    sel_min_sold_cars = st.sidebar.number_input('Please select a minimum value for sold cars per configuration per local (max value is {}):'.format(max_number_of_cars_sold), 1, max_number_of_cars_sold, value=1)

    sel_values_filters = [sel_local, sel_min_sold_cars, sel_model]
    sel_values_col_filters = ['Sales_Place_Fase2_Level_1', 'Number_Cars_Sold_Local_Fase2_Level_1', 'Model_Code']

    if 'None' not in sel_values_filters:
        data_filtered = filter_data(data, sel_values_filters, sel_values_col_filters)

        if data_filtered.shape[0]:
            max_number_ext_colors = data_filtered['Colour_Ext'].nunique()

            st.write('Dataset after Filter:', data_filtered)
            st.write('Dataset Size:', data_filtered.shape)

            sel_ext_color_count = st.number_input('Please select number of different exterior colors to select (minimum value is {} and maximum value is {})'.format(1, max_number_ext_colors), 1, max_number_ext_colors, value=max_number_ext_colors - 1)

            sel_order_size = st.sidebar.number_input('Please select the number of vehicles for the order:', 1, 1000, value=50)

            if sel_order_size < sel_ext_color_count:
                st.write('Please increase the order/decrease the number of different exterior colors.')
            else:
                with st.spinner('Processing...'):
                    status, total_value_optimized, selection = solver(data_filtered, sel_ext_color_count, sel_order_size)

                st.write('Optimization Status: {}'.format(status))
                st.write('Max solution achieved: {:.2f}'.format(total_value_optimized))
                st.write('Solution: {}'.format(selection))

                data_filtered['Order'] = selection
                st.write('Order:', data_filtered.loc[data_filtered['Order'] > 0, ['Order'] + configuration_parameters_full])

        else:
            st.write('Please lower the number of sold cars per local as no configuration meets your criteria.')


def solver(dataset, sel_ext_color_count, sel_order_size):

    ext_color_dummies = np.array(pd.get_dummies(dataset['Colour_Ext']))
    unique_ext_color_count = ext_color_dummies.shape[1]

    unique_ids = dataset['Configuration_ID'].unique()
    unique_ids_count = dataset['Configuration_ID'].nunique()
    scores = dataset['Average_Score_Euros_Local_Fase2_Level_1'].values.tolist()

    selection = cp.Variable(unique_ids_count, integer=True)

    ext_color_multiplication = selection * ext_color_dummies
    st.write(ext_color_multiplication)

    # ext_color_restriction = cp.sum(ext_color_multiplication / ext_color_multiplication, axis=0, keepdims=True)

    ext_color_restriction_2 = cp.length(ext_color_multiplication)

    order_size_restriction = cp.sum(selection) <= sel_order_size
    total_value = selection * scores

    problem = cp.Problem(cp.Maximize(total_value), [selection >= 0, selection <= 100, order_size_restriction, ext_color_restriction_2 >= sel_ext_color_count])

    result = problem.solve(solver=cp.GLPK_MI, verbose=False, qcp=True)

    return problem.status, result, selection.value


@st.cache
def get_data(dataset_name):
    df = pd.read_csv(dataset_name, encoding='latin-1', delimiter=';', usecols=configuration_parameters_full + extra_parameters)

    return df


def filter_data(dataset, filters_list, col_filters_list):
    data_filtered = dataset.copy()

    for col_filter, filter_value in zip(col_filters_list, filters_list):
        if col_filter != 'Number_Cars_Sold_Local_Fase2_Level_1':
            data_filtered = data_filtered[data_filtered[col_filter] == filter_value]
        else:
            data_filtered = data_filtered[data_filtered[col_filter].ge(filter_value)]

    data_filtered['Configuration_ID'] = data_filtered.groupby(configuration_parameters_full).ngroup()
    data_filtered.drop_duplicates(subset='Configuration_ID', inplace=True)
    data_filtered.sort_values(by='Average_Score_Euros_Local_Fase2_Level_1', ascending=False, inplace=True)

    data_filtered = data_filtered[data_filtered['Average_Score_Euros_Local_Fase2_Level_1'] > 0]

    return data_filtered


if __name__ == '__main__':
    main()
