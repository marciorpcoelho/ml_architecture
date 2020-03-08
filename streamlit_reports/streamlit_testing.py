import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')) + '\\'
sys.path.insert(1, base_path)
import level_2_order_optimization_hyundai_options as options_file

"""
# Forecast Dias em Stock - Importador
Previsão de Dias em Stock - Machine Learning Forecast
"""

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

configuration_parameters = ['PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc', 'PT_PDB_Exterior_Color_Desc', 'PT_PDB_Interior_Color_Desc']
# selection_history = pd.DataFrame(columns=['PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc', 'PT_PDB_Exterior_Color_Desc', 'PT_PDB_Interior_Color_Desc', 'Measure_9', 'Measure_10', 'number_prev_sales', 'Prediction'])


def main():
    last_date = '2020-02-26'
    # st.title('ML Deploy Testing')

    data = get_data(base_path, last_date)
    last_predictions = get_last_predictions(base_path)
    predictive_models = get_models(base_path)
    # st.write("### Top Days in Stock", data.sort_values(by='DaysInStock_Global', ascending=False))

    # daysinstock_to_filter = st.slider('Days in Stock', 0, 1000, 100)
    # filtered_data = data[data['DaysInStock_Global'] <= daysinstock_to_filter]

    # default_selection = ['h-1', '1.0i/g', 'Manual', 'Premium', 'Vermelho', 'Cinzento', 13984.388671875, 41.91549301147461, 0]

    # Filters
    # Note: Can an option not be present in the dataset?
    st.sidebar.title('Parâmetros:')
    model_filter = st.sidebar.selectbox('Por favor escolha um modelo:', ['-'] + list(data['PT_PDB_Model_Desc'].unique()), index=0)
    engine_filter = st.sidebar.selectbox('Por favor escolha uma motorização:', ['-'] + list(data['PT_PDB_Engine_Desc'].unique()), index=0)
    transmission_filter = st.sidebar.selectbox('Por favor escolha uma transmissão:', ['-'] + list(data['PT_PDB_Transmission_Type_Desc'].unique()), index=0)
    version_filter = st.sidebar.selectbox('Por favor escolha uma versão:', ['-'] + list(data['PT_PDB_Version_Desc'].unique()), index=0)
    ext_color_filter = st.sidebar.selectbox('Por favor escolha uma cor exterior:', ['-'] + list(data['PT_PDB_Exterior_Color_Desc'].unique()), index=0)
    int_color_filter = st.sidebar.selectbox('Por favor escolha uma cor interior:', ['-'] + list(data['PT_PDB_Interior_Color_Desc'].unique()), index=0)
    measure_9_filter = st.sidebar.slider('Por favor escolha um custo base:', 0.0, 40000.0, value=data['Measure_9'].mean())
    measure_10_filter = st.sidebar.slider('Por favor escolha um valor para custo base - outros:', 0.0, 2000.0, value=data['Measure_10'].mean())

    if st.sidebar.button('Prever Dias em Stock'):
        selections = [model_filter, engine_filter, transmission_filter, version_filter, ext_color_filter, int_color_filter]
        if '-' in selections:
            st.error("## Por favor escolha todos os parâmetros para a previsão.")
        else:
            col_filters = ['PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc', 'PT_PDB_Exterior_Color_Desc', 'PT_PDB_Interior_Color_Desc']

            _, number_prev_sales = check_number_prev_sales(data, selections, col_filters)

            selection_to_predict = get_prediction(selections + [measure_9_filter, measure_10_filter] + [number_prev_sales])

            predictions, feature_importances = model_prediction(predictive_models, selection_to_predict)
            feature_importances_normalized = feature_importance_treatment(feature_importances)

            # chart_data_v1 = pd.DataFrame()
            # chart_data_v1['features'] = list(selection_to_predict)[0:9]
            # chart_data_v1['feature_importance'] = feature_importances_normalized[1]
            # chart_data_v1.sort_values(by='feature_importance', ascending=False, inplace=True)
            #
            # chart_v1 = alt.Chart(chart_data_v1).mark_bar().encode(
            #     alt.X('features:N', axis=alt.Axis(labelAngle=-30)),
            #     alt.Y('feature_importance:Q'),
            #     tooltip=['features']
            # ).interactive()

            st.write("### Número de vendas anteriores: {}".format(number_prev_sales))
            # st.write("### Selection to predict: {}".format(selection_to_predict.head(1).values))

            if predictions[0][0] < predictions[1][0]:
                st.write("### Previsão de Dias em Stock (Percentil Inferior): {:.2f} dias.".format(predictions[0][0]))

            st.write("### Previsão de Dias em Stock: {:.2f} dias.".format(predictions[1][0]))

            if predictions[2][0] > predictions[1][0]:
                st.write("### Previsão de Dias em Stock (Percentil Superior): {:.2f} dias.".format(predictions[2][0]))

            # st.write("Parâmetros mais importantes:", "", chart_v1)

            save_predictions(last_predictions, selection_to_predict, predictions, base_path)

            # shap_plot(data, predictive_models[1], selection_to_predict)

    if last_predictions is not None:
        last_predictions.sort_values(by='Date', inplace=True, ascending=False)
        st.write('### Histórico de Previsões:', last_predictions[[x for x in list(last_predictions) if x not in ['Date', 'Previsão - Percentil Inferior (dias)', 'Previsão - Percentil Superior (dias)']]].head(5)
                 .rename(columns=options_file.column_translate_dict)
                 .reset_index(drop=True)
                 .style.format({'Previsão (dias)': '{:.0f}', 'Custo Base': '{:.2f}', 'Custo Base - Outros': '{:.2f}'}))

    st.button("Reset")


def feature_importance_treatment(feature_importances):
    feature_importances_normalized = []
    for feature_importance in feature_importances:
        feature_importances_normalized.append(feature_importance / np.sum(feature_importance))  # Normalization

    return feature_importances_normalized


# def shap_plot(data, model, selection_to_predict):
#     selection_to_predict_no_date = selection_to_predict[[x for x in list(selection_to_predict) if x != 'Date']]
#
#     shap_explainer = shap.TreeExplainer(model)
#     shap_values = shap_explainer.shap_values(data.values)
#
#     print(shap_explainer.expected_value)
#     print(shap_values)


@st.cache(show_spinner=True)
def get_data(path, last_date):
    dataset_name = path + 'output/hyundai_ml_dataset_streamlined_{}.csv'.format(last_date)
    df = pd.read_csv(dataset_name, index_col=0, dtype={'NDB_VATGroup_Desc': 'category', 'VAT_Number_Display': 'category', 'NDB_Contract_Dealer_Desc': 'category',
                                                       'NDB_VHE_PerformGroup_Desc': 'category', 'NDB_VHE_Team_Desc': 'category', 'Customer_Display': 'category',
                                                       'Customer_Group_Desc': 'category', 'Product_Code': 'category',
                                                       'Sales_Type_Dealer_Code': 'category', 'Sales_Type_Code': 'category', 'Vehicle_Type_Code': 'category', 'Fuel_Type_Code': 'category',
                                                       'PT_PDB_Engine_Desc': 'category', 'PT_PDB_Transmission_Type_Desc': 'category', 'PT_PDB_Version_Desc': 'category',
                                                       'PT_PDB_Exterior_Color_Desc': 'category', 'PT_PDB_Interior_Color_Desc': 'category'})
    df['Measure_9'] = df['Measure_9'] * (-1)
    df['Measure_10'] = df['Measure_10'] * (-1)

    # Remove a single case where cost is positive
    df = df[df['Measure_9'] >= 0]

    for model in df['PT_PDB_Model_Desc'].unique():
        if model == 'cr-v':
            df.loc[df['PT_PDB_Model_Desc'] == model, 'PT_PDB_Model_Desc'] = 'CR-V'
        elif model == 'hr-v':
            df.loc[df['PT_PDB_Model_Desc'] == model, 'PT_PDB_Model_Desc'] = 'HR-V'
        elif model == 'ioniq':
            df.loc[df['PT_PDB_Model_Desc'] == model, 'PT_PDB_Model_Desc'] = 'Ioniq'
        elif model[0] != 'i':
            df.loc[df['PT_PDB_Model_Desc'] == model, 'PT_PDB_Model_Desc'] = model.capitalize()

    return df


def get_last_predictions(path):
    try:
        last_predictions = pd.read_csv(path + 'dbs/predictions_history.csv', index_col=0)
        return last_predictions
    except FileNotFoundError:
        return


def save_predictions(last_predictions, selection, predictions, path):
    timestamp = datetime.datetime.now()

    selection.loc[0, 'Previsão - Percentil Inferior (dias)'] = predictions[0]
    selection.loc[0, 'Previsão (dias)'] = predictions[1]
    selection.loc[0, 'Previsão - Percentil Superior (dias)'] = predictions[2]
    selection.loc[0, 'Date'] = timestamp

    if last_predictions is not None:
        last_predictions = last_predictions.append(selection)
        last_predictions.to_csv(path + 'dbs/predictions_history.csv')
    else:
        selection.to_csv(path + 'dbs/predictions_history.csv')


@st.cache
def get_models(path):
    model_name_lower_quantile = path + 'models/project_{}_{}_best_{}.sav'.format(2406, 'hyundai_regression_model_streamlined_lower', '7_11_2019')
    model_name_standard = path + 'models/project_{}_{}_best_{}.sav'.format(2406, 'hyundai_regression_model_streamlined', '7_11_2019')
    model_name_upper_quantile = path + 'models/project_{}_{}_best_{}.sav'.format(2406, 'hyundai_regression_model_streamlined_upper', '7_11_2019')

    with open(model_name_lower_quantile, 'rb') as f:
        ml_model_lower = pickle.load(f)

    with open(model_name_standard, 'rb') as f:
        ml_model_default = pickle.load(f)

    with open(model_name_upper_quantile, 'rb') as f:
        ml_model_upper = pickle.load(f)

    return [ml_model_lower, ml_model_default, ml_model_upper]


def check_number_prev_sales(dataset, filters_list, col_filters_list):

    data_filtered = dataset.copy()
    if '-' in filters_list:
        default_prediction = ['kauai', '1.0i', 'Manual', 'Premium', 'Vermelho', 'Preto', -14357.429999999995, 0.0, 161]
        default_data = get_prediction(default_prediction)
        return default_data, 0
    else:
        for col_filter, filter_value in zip(col_filters_list, filters_list):

                # try:
                data_filtered = data_filtered[data_filtered[col_filter] == filter_value]
                # except

        if data_filtered.shape[0]:
            number_previous_sales = data_filtered.shape[0]
            return data_filtered, number_previous_sales
        else:
            number_previous_sales = 0
            return data_filtered, number_previous_sales


def model_prediction(ml_models, selection):
    # dataset order: ['PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc',
    # 'PT_PDB_Exterior_Color_Desc', 'PT_PDB_Interior_Color_Desc', 'Measure_9', 'Measure_10', 'number_prev_sales']

    model_predictions = []
    model_feature_importance = []
    for model in ml_models:
        model_predictions.append(model.predict(selection))
        model_feature_importance.append(model.feature_importances_)

    return model_predictions, model_feature_importance


def get_prediction(selection):
    selection_df = pd.DataFrame(columns=['PT_PDB_Model_Desc', 'PT_PDB_Engine_Desc', 'PT_PDB_Transmission_Type_Desc', 'PT_PDB_Version_Desc', 'PT_PDB_Exterior_Color_Desc', 'PT_PDB_Interior_Color_Desc', 'Measure_9', 'Measure_10', 'number_prev_sales'])

    if '-' in selection:
        selection_df.loc[0, :] = ['kauai', '1.0i', 'Manual', 'Premium', 'Vermelho', 'Preto', -14357.429999999995, 0.0, 161]
    else:
        selection_df.loc[0, :] = selection

    for col in configuration_parameters:
        selection_df[col] = selection_df[col].astype('category')
    for col in ['Measure_9', 'Measure_10']:
        selection_df[col] = selection_df[col].astype('float64')
    selection_df['number_prev_sales'] = selection_df['number_prev_sales'].astype('int')

    return selection_df


def historic_predictions(selection_history_var, selection_to_predict_copy, prediction):
    selection_to_predict_copy.loc[0, 'Prediction'] = prediction

    selection_history_var.loc[selection_history_var.shape[0] + 1, :] = selection_to_predict_copy.head(1).values

    return selection_history_var


if __name__ == "__main__":
    main()


