import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import phik
from phik.phik import phik_matrix
from phik.report import plot_correlation_matrix
from src.data_management import load_csv

def sale_price_study_body():
    st.write("## The relationship between a house's attributes and its sale price")
    st.write('### Overview')
    st.write(f'In the dataset there were 22 individual attributes, often related, and groupable by being a measure of size, quality, age,\n'
             f'condition or form.\n'
             f'Despite the large number of attributes, the actual number of independent features is far less, due to there being many moderate-to-strong\n'
             f'correlations between attributes.\n\n'

             f'Some possible limitations of the dataset worth noting, are the lack of any attributes relating to its location,\n'
             f'or the time of sale, both of which likely impact the sale price.\n\n'

             f'Irrespective of any dataset limitations, statistically significant correlations/relationships were discovered between various house attributes\n'
             f'and the sale price, which shall now be explored.\n')

    st.write('### The house attributes')
    st.write(f'Here is a table of the house attributes in the dataset, their meanings, abbreviations, and possible values:\n\n')
    st.markdown(f'''|Variable|Meaning|Values|
                    |:----|:----|:----|
                    |1stFlrSF|First Floor square feet|334 - 4692|
                    |2ndFlrSF|Second-floor square feet|0 - 2065|
                    |BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
                    |BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
                    |BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
                    |BsmtFinSF1|Type 1 finished square feet|0 - 5644|
                    |BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
                    |TotalBsmtSF|Total square feet of basement area|0 - 6110|
                    |GarageArea|Size of garage in square feet|0 - 1418|
                    |GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
                    |GarageYrBlt|Year garage was built|1900 - 2010|
                    |GrLivArea|Above grade (ground) living area square feet|334 - 5642|
                    |KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
                    |LotArea| Lot size in square feet|1300 - 215245|
                    |LotFrontage| Linear feet of street connected to property|21 - 313|
                    |MasVnrArea|Masonry veneer area in square feet|0 - 1600|
                    |EnclosedPorchSF|Enclosed porch area in square feet|0 - 286|
                    |OpenPorchSF|Open porch area in square feet|0 - 547|
                    |OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
                    |OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
                    |WoodDeckSF|Wood deck area in square feet|0 - 736|
                    |YearBuilt|Original construction date|1872 - 2010|
                    |YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
                    |SalePrice|Sale Price|34900 - 755000|''')
                    
    st.write('### Correlation tests')
    st.info(f'Two types of correlation tests that calculate correlation coefficients that range from 0 to 1, and indicate the strength of the correlation,\n'
            f'were performed.\n\n'
            f'- The first test is Spearmans correlation test that measures monotonic relationships, meaning for the case of a positive monotonic relationship ---\n'
            f'indicated by a positive coefficient --- that if an attribute increases, the sale price never decreases.\n'

            f'- The second test is the phi k correlation test; this measures whether there is any type of relationship, perhaps non-linear or otherwise.\n\n'
            f'Collectively these two tests likely will reveal most strong relationships between any attribute and the sale price.\n')
    
    correlation_test = st.radio(label='Correlation test', options=['Spearman', 'phi k'])

    if correlation_test == 'Spearman':

        st.write('#### Spearman')
        st.write(f'Below is a heatmap displaying the calculated spearman coefficients for each sale price - attribute pairing:')
        with st.spinner('Loading, please wait'):
            spearman_df = load_csv('src/sale_price_study/spearman_df.csv')
            global fig
            fig, axis = plt.subplots()    
            spearman_heatmap = sns.heatmap(spearman_df.pivot(index='Y', columns=['X'], values=['r']).sort_values(by=('r', 'SalePrice'), ascending=False), annot=True,
                                            vmax=1, vmin=-1, xticklabels=['SalePrice'], linecolor='black', linewidth=0.05, ax=axis)
            spearman_heatmap.set(xlabel='', ylabel='House attribute', title='SalePrice-Attribute pair spearman correlations')
        st.pyplot(fig)
        st.write(f'All of the coefficients are statistically significant except for the attributes EnclosedPorchSF and OverallCond.\n'
                f'This means there exists positive monotonic relationships for all the remaining attributes and the sale price. In the case of \n'
                f'EnclosedPorchSF and OverallCond, there likely exists no correlation to sale price or a weak negative one.\n\n'

                f'With regard to the strength of the relationship as indicated by the magnitude of the coefficient (the larger the stronger),\n'
                f'each attribute can be described as being correlated to the sale price as follows:\n')

        def strength_label(x):
            strength = ''
            if abs(x) > 0.65:
                strength = 'strong'
            elif abs(x) > 0.35:
                strength = 'moderate'
            elif abs(x) > 0:
                strength = 'weak'
            
            return strength

        @st.cache
        def create_spearman_strength_df():
            spearman_coeff_df = spearman_df[['Y', 'r']].set_index('Y').sort_values(by='r', ascending=False)
            return spearman_coeff_df.applymap(strength_label).reset_index().rename(columns={'r': 'Correlation Strength', 'Y':'Attribute'})

        st.table(create_spearman_strength_df())

        st.write('**Takeaway:**')
        st.write(f'As can be seen from the heatmap, according to the Spearman test, the **top five most significantly correlated attributes**, in descending order, are\n'
                f'OverallQual, GrLivArea, KitchenQual, YearBuilt, and GarageArea. **Increasing** any of these attributes **never decreases** the **sale price**.\n\n')

    else:
        st.write('#### Phi k')
        st.write(f'Below is a heatmap displaying the calculated spearman coefficients for each sale price - feature pairing:')

        fig, ax = plt.subplots()
        phik_matrix_df = pd.read_csv('src/sale_price_study/phik_matrix_df.csv')
        matrix_plot = plot_correlation_matrix(phik_matrix_df.values, x_labels=phik_matrix_df.columns, y_labels=phik_matrix_df.index, figsize=(10,10), vmin=0, vmax=1,
                                        y_label='Feature', title='SalePrice-Feature pair $\phi_k$ correlations')
        
        st.pyplot(data=matrix_plot)