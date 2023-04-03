import pandas as pd
import numpy as np
import pingouin as pg
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest
from phik.phik import phik_matrix
from phik.report import plot_correlation_matrix
from phik.significance import significance_matrix
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from feature_engine.selection import SmartCorrelatedSelection, DropFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV


class IndependentKNNImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values for numerical features using a KNNImputer.
    
    Allows independent fitting and transforming for train and test sets.
    """
    def fit(self, x, y):
        return self

    def transform(self, x):
        """
        Transforms using a KNNImputer.
        """
        self.numeric_df = x.select_dtypes(exclude='object')
        imputer = KNNImputer()
        imputer.set_output(transform='pandas')
        imputer.fit(self.numeric_df)
        x[self.numeric_df.columns] = imputer.transform(self.numeric_df)
        return x

    def set_output(self, transform):
        pass


class EqualFrequencyImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values for categorical features using an equal frequency replacement with possible feature values.
    """

    def set_output(self, transform):
        pass

    def fit(self, x, y):
        return self

    def equal_frequency_imputer_categorical_features(self, categorical_df, transform_df, missing_data_df):
            """
            Imputes missing values for categorical features using an equal frequency value replacement method. Produces before and after count plots.

            The missing values are individually replaced in sequence by repeatedly cycling through one of the possible values.

            Args:
                categorical_df: dataframe containing all of the categorical features only.
                transform_df: dataframe which is updated with imputed values, and from which the categorical features originate.
                missing_data_df: dataframe containing columns (all types) with missing data only.
                
            """
            counter = 0
            imputed_columns = []
            while counter < len(categorical_df.columns):
                if categorical_df.iloc[:, counter].name in missing_data_df.columns:
                    imputed_columns.append(categorical_df.iloc[:, counter].name)
                counter += 1

            for col in imputed_columns:
                number_of_nans = categorical_df[col].loc[categorical_df[col].isna() == True].size
                unique_values = categorical_df[col].unique()
                index_no = 0
                while number_of_nans > 0:
                    if index_no + 1 >= unique_values.size:
                        index_no = 0
                    categorical_df[col].fillna(value=unique_values[index_no], limit=1, inplace=True)
                    transform_df[col].fillna(value=unique_values[index_no], limit=1, inplace=True)
                    index_no += 1
                    number_of_nans = categorical_df[col].loc[categorical_df[col].isna() == True].size

    def transform(self, x):
        """
        Transform the features by applying the equal frequency imputer function.
        """
        self.categorical_df = x.select_dtypes(include='object')
        self.missing_data_df = x.loc[:, x.isna().any()]

        self.equal_frequency_imputer_categorical_features(self.categorical_df, x, self.missing_data_df)

        return x
        

class CompositeSelectKBest(BaseEstimator, TransformerMixin):
    """
    Custom transformer that is a composition of two SelectKBest transformers.

    Filters out a set union of best features, from a dataset with a SalePrice target.
    """
    def spearman_score_function(self, feature_array, target_array):
        """
        Calculates spearman correlation coefficients for x's features and y's target.

        To be used as the score function in the sklearn.feature_selection.SelectKBest function.

        Args:
            feature_array = array-like, containing x data.
            target_array = array-like, containing y data.

        Returns tuple of the spearman r values series, and the spearman p values series for the dataset.
        """
        df = pd.DataFrame(data=feature_array, columns=self.x_column_names)
        df['SalePrice'] = target_array
        spearman_df = df.pairwise_corr(columns=['SalePrice'], method='spearman')
        return (spearman_df['r'], spearman_df['p-unc'])


    def phik_score_function(self, feature_array, target_array):
        """
        Calculates phi_k correlation coefficients for x's features and y's target.

        To be used as the score function in the sklearn.feature_selection.SelectKBest function.

        Args:
            feature_array = array-like, containing x data.
            target_array = array-like, containing y data

        Returns tuple of the phi_k correlation values series, and the significance values series for the dataset.
        """
        df = pd.DataFrame(data=feature_array, columns=self.x_column_names)
        df['SalePrice'] = target_array
    
        phik_matrix_df = phik_matrix(df)[['SalePrice']].drop('SalePrice', axis=0)
        significance_matrix_df = significance_matrix(df)[['SalePrice']].drop('SalePrice', axis=0)
        return (phik_matrix_df['SalePrice'], significance_matrix_df['SalePrice'])


    def fit(self, x, y):
        """
        Composed of two SelectKBest fittings with spearman and phi_k score functions
        """
        self.x_column_names = x.columns.to_list()
        select_k_best_spear = SelectKBest(score_func=self.spearman_score_function)
        select_k_best_spear.set_output(transform='pandas')
        self.spear_trans = select_k_best_spear.fit(x, y)

        select_k_best_phi = SelectKBest(score_func=self.phik_score_function)
        select_k_best_phi.set_output(transform='pandas')
        self.phik_trans = select_k_best_phi.fit(x, y)

        return self

    def transform(self, x):
        """
        Filters out the union of each set of the 10 best features obtained using the SelectKBest method with a phi_k and spearman score function.
        """
        spearman_transformed_df = self.spear_trans.transform(x)
        phik_transformed_df = self.phik_trans.transform(x)

        selected_features = list(set(spearman_transformed_df.columns.to_list() + phik_transformed_df.columns.to_list()))
        x = x[selected_features]
        return x
    
    def set_output(self, transform):
        pass

def scale_target(y_fit, y_transform):
    """
    Scales target for a subset, having been trained on another subset.

    Args:
        y_fit: target values used for fitting.
        y_transform: target values transformed.

    Returns a tuple of the scaled target series, as well as the inverse transform.
    """
    y_fit = pd.DataFrame(data=y_fit)
    y_transform = pd.DataFrame(data=y_transform)
    min_max_scaler = MinMaxScaler()
    min_max_scaler.set_output(transform='pandas')
    min_max_scaler.fit(y_fit)
    y_transform = min_max_scaler.transform(y_transform)
    inverse_transform = min_max_scaler.inverse_transform

    return (y_transform.iloc[:, 0], inverse_transform)