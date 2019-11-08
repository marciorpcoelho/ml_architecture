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
extra_parameters = ['Average_Score_Euros', 'Number_Cars_Sold', 'Average_Score_Euros_Local', 'Number_Cars_Sold_Local', 'Sales_Place']


def main():
    dataset_name = base_path + 'dbs/vhe_crp.csv'
    data = get_data(dataset_name)

    st.write('Dataset:', data)
    st.write('Dataset Size:', data.shape)

    sel_local = st.selectbox('Please select a Location', ['None'] + list(data['Sales_Place'].unique()), index=0)
    sel_min_sold_cars = st.slider('Please select a minimum value for sold cars per configuration per local:', 1, max(data['Number_Cars_Sold_Local']), value=1)
    sel_model = st.selectbox('Please select a model to optimize the order for:', ['None'] + list(data['Model_Code'].unique()), index=0)

    sel_values_filters = [sel_local, sel_min_sold_cars, sel_model]
    sel_values_col_filters = ['Sales_Place', 'Number_Cars_Sold_Local', 'Model_Code']

    if 'None' not in sel_values_filters:
        data_filtered = filter_data(data, sel_values_filters, sel_values_col_filters)

        sel_ext_color_count = st.number_input('Please select number of different exterior colors to select (minimum value is {} and maximum value is {})'.format(1, data_filtered['Colour_Ext'].nunique()), 1, data_filtered['Colour_Ext'].nunique(), value=4)

        st.write('Dataset:', data_filtered)
        st.write('Dataset Size:', data_filtered.shape)

        status, total_value_optimized, selection = solver(data_filtered, sel_ext_color_count)

        st.write('Optimization Status: {}'.format(status))
        st.write('Max solution achieved: {:.2f}'.format(total_value_optimized))
        st.write('Solution: {}'.format(selection))


def solver(dataset, sel_ext_color_count):

    ext_color_dummies = np.array(pd.get_dummies(dataset['Colour_Ext'])).T

    unique_ids = dataset['Configuration_ID'].unique()
    unique_ids_count = dataset['Configuration_ID'].nunique()

    scores = dataset['Average_Score_Euros_Local'].values.tolist()
    selection = cp.Variable(unique_ids_count, integer=True)

    ext_color_restriction = cp.multiply(selection, ext_color_dummies)

    total_value = selection * scores

    problem = cp.Problem(cp.Maximize(total_value), [selection >= 0, selection <= 100])

    result = problem.solve(solver=cp.GLPK_MI, verbose=False)

    return problem.status, result, selection.value


@st.cache
def get_data(dataset_name):
    df = pd.read_csv(dataset_name, encoding='utf-8', delimiter=';', usecols=configuration_parameters_full + extra_parameters)

    return df


def filter_data(dataset, filters_list, col_filters_list):
    data_filtered = dataset.copy()

    for col_filter, filter_value in zip(col_filters_list, filters_list):
        if col_filter != 'Number_Cars_Sold_Local':
            data_filtered = data_filtered[data_filtered[col_filter] == filter_value]
        else:
            data_filtered = data_filtered[data_filtered[col_filter].ge(filter_value)]

    data_filtered['Configuration_ID'] = data_filtered.groupby(configuration_parameters_full).ngroup()
    data_filtered.drop_duplicates(subset='Configuration_ID', inplace=True)
    data_filtered.sort_values(by='Average_Score_Euros_Local', ascending=False, inplace=True)

    data_filtered = data_filtered[data_filtered['Average_Score_Euros_Local'] > 0]

    return data_filtered


if __name__ == '__main__':
    main()
