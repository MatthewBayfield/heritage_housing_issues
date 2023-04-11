import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import phik
from phik.phik import phik_matrix
from src.data_management import load_csv, load_pkl_file
from src.ml import transformers_and_functions as tf


def ml_model_body():
    data_cleaning_and_feature_engineering_pipeline = load_pkl_file('src/ml/data_cleaning_and_feature_engineering_pipeline.pkl')
    data_cleaning_and_feature_engineering_pipeline_img = plt.imread('media/data_cleaning_engineering_pipeline.png')
    regressor_model_pipeline = load_pkl_file('src/ml/model_pipeline.pkl')
    train_set_df = load_csv('src/ml/train_set_df.csv', index_col=0)
    y_train = train_set_df['SalePrice']
    test_set_df = load_csv('src/ml/test_set_df.csv', index_col=0)
    x_test = test_set_df.drop('SalePrice', axis=1)
    y_test = test_set_df['SalePrice']

    st.write('## The ML model/pipeline employed to predict the sale price of a house in Ames, Iowa using its attributes')

    st.write('### Overview')
    st.info(f"A ML pipeline was constructed in order to satisfy the client's business requirement to be able to predict the sale price of their inherited properties\n"
            f'as well as for any other property in the area. A regressor model was the natural choice of ML algorithm. ')

    st.write('### Pipelines')
    st.write('#### Data cleaning and engineering pipeline')
    st.write('##### Optimising the dataset split')
    st.write(f'Before creating any steps of the pipeline, much attention was paid to obtaining an optimal dataset split into train and test subsets. \n'
             f'The train set would be used to choose, train, and optimise a candidate model, while the test set would act as a holdout set to independently assess the \n'
             f'model performance on unseen data. \n\n'
             f'An optimal split is one that maximises model performance on both the train and test set, and that is therefore generalisable.\n'
             f'For this to be achieved the test and train sets must be sufficiently representative of the whole dataset from which they were derived, under the\n'
             f'assumption that the whole dataset itself is representative of the parent population, which in turn is assumed to be static. The quality of the split\n'
             f'was determined by the value of the random_state parameter in the sklearn train_test_split function. The sizes of the train and test sets also indirectly\n'
             f'impact the likelihood that a random split is representative, namely the more data in each the more likely it is; thus the larger the parent set as a whole,\n'
             f'the less likely the random split will affect model performance significantly. The house prices dataset size is fairly small (~1500), and the random split is\n'
             f'likely to affect model performance.\n\n'
             f'To ensure a representative split with respect to the train/test target distribution, the dataset was stratified by the target (sale price) during splitting.\n'
             f'A sample of 100 different random splits were performed and their relative quality assessed using significance tests to compare the distributions\n'
             f'of the features in both the train and test sets with the whole dataset. The best ranked split with respect to the number of features for which the\n'
             f'distributions are statistically the same (alpha=0.5), and also the product p-value, was then chosen.\n\n'
             f'A final decision related to the dataset split, was the relative proportions of the train and test sets. The larger the train set, the more likely\n'
             f'the model will learn any patterns in the data, and thus be able to make better predictions. The larger the test set size the greater the confidence in\n'
             f'the accuracy and reliability of the model performance metric scores. Thus naturally there is a trade-off between having an optimally large test set\n'
             f'or train set, at least for small datasets. It was decided to choose a test set proportion of 0.25.\n\n')
    
    st.write('##### Outliers, Imputing and Encoding')
    st.write(f'It was determined from the significant features EDA notebook that for the most significant continuous numeric features in relation to sale price,\n'
             f'there were several instances whose vector components were outliers in at least 50% of the continuous features, thus making them more likely multivariate outliers.\n'
             f"What's more it was discovered that the components of these instances corresponded to the extremest outliers for multiple features, supporting the idea of a\n"
             f"correlation between the number of features for which an instance's component is an outlier, and the extremity of the outliers.\n"
             f'Outliers for each feature (using the whole dataset) were determined using the IQR method, and the indices of the outliers tracked and counted to determine\n'
             f'if the same instance gave rise to outliers for other features. These identified outliers were trimmed from the whole dataset.\n\n'
             
             f'With regard to handling missing data, imputation was the preferred method. For numeric features the sklearn KNNImputer (k-nearest-neighbours) was used, whilst\n'
             f'for non-numeric features a custom equal frequency imputer was used. The motivation for the use of the KNNImputer is that it more realistically recreates real\n'
             f'data, in that it fills in missing values by averaging over actual values taken from the k most similar instances in the train/test set. Since there are \n'
             f'correlations between the features, instances that have similar values for some features, likely have similar values for other features. The equal frequency imputer\n'
             f'sequentially inserts one of the possible feature values for each missing value, with the aim to not distort the overall feature value distribution\n'
             f'in the subset.\n\n'
             f'Categorical features were ordinally encoded, as this was the natural choice, since all categorical features possess a natural ordering.\n\n')

    st.write('##### Feature and target scaling')
    st.write(f'The most significant features with respect to their correlation to the target, as identified in the significant feature EDA notebook, were scaled using\n'
             f'either a variance stabilising transformation or a sklearn MinMaxScaler transformer; the goal being to normalise and or standardise their distributions,\n'
             f'which is important for optimising some model algorithms. In addition the target was also scaled using a MinMaxScaler transformer.\n\n')

    st.write('##### Feature reduction/selection')
    st.write(f'In the initial pipeline two feature selections were performed. First a custom SelectKBest transformer was used that selects the set union of the top 10\n'
             f'correlated features to sale price obtained using both the Spearman and phik correlation tests. A SmartCorrelatedSelection transformer was then applied to remove\n'
             f'redundant features: that is a feature of each strongly correlated feature-feature pair, chosen by its relative importance in a fitted decision tree\n'
             f'regressor estimator.\n'
             f'In other trialed pipelines a sklearn SequentialFeatureSelector transformer instead of the SelectKBest transformer,  was used to remove features one at a time\n'
             f'in a way that improved model performance. Finally pipelines that involved dropping or pruning features from the current best model were trialed, in order to\n'
             f'attempt to improve the model performance further and reduce model complexity.')
    
    st.write('##### Chosen pipeline')
    st.write(f'The chosen data cleaning and engineering pipeline is shown below. It turned out that the initial SelectKBest step containing pipeline optimised the overall\n'
             f'model performance; reducing the number of features further did not improve model performance. The pipeline containing the backward SequentialFeatureSelector\n'
             f'step produced a model with higher complexity, but similar performance.')
    st.image(data_cleaning_and_feature_engineering_pipeline_img)

    