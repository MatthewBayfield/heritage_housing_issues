import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import phik
from phik.phik import phik_matrix
from src.data_management import load_csv, load_pkl_file
from src.ml import transformers_and_functions as tf


def sale_price_predictor_body():
    data_cleaning_and_feature_engineering_pipeline = load_pkl_file('src/ml/data_cleaning_and_feature_engineering_pipeline.pkl')
    regressor_model_pipeline = load_pkl_file('src/ml/model_pipeline.pkl')
    train_set_df = load_csv('src/ml/train_set_df.csv', index_col=0)
    y_train = train_set_df['SalePrice']
    inherited_properties_df = load_csv('outputs/datasets/collection/inherited_houses.csv').rename(index={0:'House 1', 1:'House 2',2: 'House 3',3: 'House 4'})
    inherited_properties_df['GarageYrBlt'] = inherited_properties_df['GarageYrBlt'].round(0).astype('int')
    inherited_houses_predicted_prices_df = load_csv('src/ml/inherited_houses_predicted_prices_df.csv', index_col=0)
    model_variables = data_cleaning_and_feature_engineering_pipeline[-1].get_feature_names_out()
    model_variables.append('SalePrice')

    @st.cache()
    def extract_train_set_variable_ranges():
        """
        Extracts the train set variable value ranges or values, and returns a dataframe containing them.
        """
        train_set_variable_ranges = pd.DataFrame(columns=['values or value interval'])
        variables = train_set_df[model_variables].select_dtypes(exclude='object').columns.to_list()
        for variable in variables:
            train_set_variable_ranges = pd.concat([train_set_variable_ranges,
                                                pd.DataFrame(index=[variable], data=[str([train_set_df[variable].min(),
                                                                train_set_df[variable].max()])], columns=['values or value interval'])])
        variables = train_set_df[model_variables].select_dtypes(include='object').columns.to_list()
        for variable in variables:
            train_set_variable_ranges = pd.concat([train_set_variable_ranges,
                                                pd.DataFrame(index=[variable], data=str(train_set_df[variable].dropna().unique().tolist()), columns=['values or value interval'])])
        return train_set_variable_ranges


    st.write('## Sale Price Predictor')

    st.write('### Model performance')

    st.success(f'The current version of the Machine Learning (ML) model, or more completely the ML pipeline, successfully meets the agreed model\n'
               f'performance success criteria.\n')
    st.markdown('To be successful the regressor needed to satisfy the scoring metric of $R^2\ge0.75$; the current model performance has $R^2=0.894$.')
    
    sale_price_predictor = st.radio(label='Sale Price Predictions', options=['Inherited Properties', 'Other'])

    if sale_price_predictor == 'Inherited Properties':
        st.write("### Predicted sale prices of the client's four inherited properties")
        
        st.write("Below is a dataframe containing the house attributes for the client's four inherited properties:")
        st.dataframe(inherited_properties_df)

        st.write(f'Using the current best ML regressor model, both the median sale price prediction and prediction intervals are given for each inherited property\n'
                 f'in the table below. The prediction intervals guarantee that the predicted sale price has at least a 50% probability of being in the indicated interval, and\n'
                 f' more than likely approaches a 96% probability of being in the interval.\n')

        with st.spinner('Loading, please wait'):
            st.table(inherited_houses_predicted_prices_df)
    
    else:
        st.write("### Predict the sale price of any property in Ames, Iowa")

        st.write("#### Range of validity of the model")
        st.warning(f'Strictly speaking predictions made by the model for a property are only valid if its relevant house attribute values fall within the same ranges as the\n'
                f'attribute values of the properties that the model was trained on. In addition only properties whose actual sale price lies in the sale price range of the\n'
                f'properties the model was trained on, can be predicted. With this in mind the values/value ranges for the house attributes and sale price that featured in\n'
                f'the data used to train the model, are displayed in the dataframe below. (The units have been omitted, but can be found on the sale price correlation\n'
                f'study app page)')
        with st.spinner('Loading, please wait...'):
            st.write(extract_train_set_variable_ranges())

        st.write('#### Obtain a prediction')

        st.info(f"Enter valid attribute values for the following house attributes below, and obtain a prediction of the corresponding house sale price by\n"
                f"clicking the 'Get Prediction' button.")

        col1, col2, col3 = st.beta_columns(3)

        with col1:
            overall_quality = st.number_input('OverallQual', min_value=1, max_value=10)
            gr_liv_area = st.number_input('GrLivArea (SF)', min_value=334.0, max_value=3627.0, step=0.1)
            lot_area = st.number_input('LotArea (SF)', min_value=1300.0, max_value=215245.0, step=0.1)
            second_flr_sf = st.number_input('2ndFlrSF', min_value=0.0, max_value=1818.0, step=0.1)
        
        with col2:
            total_bsmt_sf = st.number_input('TotalBsmtSF', min_value=0.0, max_value=3206.0, step=0.1)
            garage_area =  st.number_input('GarageArea (SF)', min_value=0.0, max_value=1390.0, step=0.1)
            bsmt_fin_sf1 = st.number_input('BsmtFinSF1', min_value=0.0, max_value=2188.0, step=0.1)
            mas_vnr_area = st.number_input('MasVnrArea (SF)', min_value=0.0, max_value=1600.0, step=0.1)

        with col3:
            year_built = st.number_input('YearBuilt', min_value=1872, max_value=2010)
            year_remod_add = st.number_input('YearRemodAdd', min_value=1950, max_value=2010)
            kitchen_qual = st.selectbox('KitchenQual', ['TA','Gd', 'Fa', 'Ex'])
            open_porch_sf = st.number_input('OpenPorchSF', min_value=0.0, max_value=547.0, step=0.1)


        @st.cache()
        def make_predictions(data):
            """
            Calculates the predicted sale price for a property with attribute values contained in the data parameter.

            Args:
                data: a dataframe containing the house attribute data for a property.

            Returns a dataframe containing the predicted sale price for the property.
            """
            inverse_transform = tf.scale_target(y_train, y_train)[1]
            transformed_data = data_cleaning_and_feature_engineering_pipeline[2:].transform(data)
            prediction = inverse_transform(pd.DataFrame(data=regressor_model_pipeline.predict(transformed_data)))
            return pd.DataFrame(data=prediction, index=[''], columns=['Predicted Sale Price ($)']).round(0).astype('int')

        predict = st.button('Get Prediction')

        if predict:
            with st.spinner('Generating Prediction. This may take a few mins...'):
                data = data_cleaning_and_feature_engineering_pipeline[0:2].transform(train_set_df.drop('SalePrice', axis=1)).head(1)
                data['OverallQual'] = overall_quality
                data['GrLivArea'] = gr_liv_area
                data['LotArea'] = lot_area
                data['2ndFlrSF'] = second_flr_sf
                data['TotalBsmtSF'] = total_bsmt_sf
                data['GarageArea'] = garage_area
                data['BsmtFinSF1'] = bsmt_fin_sf1
                data['MasVnrArea'] = mas_vnr_area
                data['YearBuilt'] = year_built
                data['YearRemodAdd'] = year_remod_add
                data['KitchenQual'] = kitchen_qual
                data['OpenPorchSF'] = open_porch_sf

                prediction = make_predictions(data)
                st.table(prediction)
