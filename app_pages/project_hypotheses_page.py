import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import phik
from phik.phik import phik_matrix
from src.data_management import load_csv



def project_hypotheses_body():
    st.write('## Project Hypotheses')

    st.write('### Overview')

    st.info(f'Before conducting the data analysis in to how the sale price of a house was related to its attributes aka features, several sets of hypotheses regarding\n'
            f'how the various attributes are related to sale price, were made.\n\n'

            f'In particular formal statistical hypotheses were made regarding the form of any correlations; and in addition informal hypotheses/predictions were made with\n'
            f'respect to the stength of the correlations and how attributes may be grouped based on their nature.\n\n')

    st.write('### Hypothesis statements')
    st.write('#### Hypotheses set 1')
    st.write('Feature Group 1 (Size group - More space, more rooms tends to significantly increase sale price):')
    st.write(f'* First Floor square feet (1stFlrSF)\n'
             f'* Second Floor square feet (2ndFlrSF)\n'
             f'* Bedrooms above grade (BedroomAbvGr)\n'
             f'* Total square feet of basement area (TotalBsmtSF)\n'
             f'* Above grade (ground) living area square feet (GrLivArea)\n'
             f'* Type 1 finished square feet (BsmtFinSF1)\n'
             f'* Unfinished square feet of basement area (BsmtUnfSF)\n'
             f'* LotArea: Lot size in square feet\n'
             f'* LotFrontage: Linear feet of street connected to property\n\n'
             f'**Expect a statistically significant strong positive monotonic correlation between these features and the sale price**.\n\n')

    st.write(f'#### Hypotheses set 2')
    st.write(f'Feature Group 2 (Quality group - Higher quality normally means higher sale prices):')
    st.write(f'* OverallQual: Rates the overall material and finish of the house\n'
             f'* KitchenQual: Kitchen quality\n\n'
             f'**Expect a statistically significant moderate positive monotonic correlation between these features and the sale price**.\n\n')

    st.write(f'#### Hypotheses set 3')
    st.write(f'Feature Group 3 (Age/condition group - newer or renovated houses, or houses in better condition tend to have higher sale prices):')
    st.write(f'* YearBuilt: original construction date\n'
             f'* YearRemodAdd: Remodel date\n'
             f'* OverallCond: Rates the overall condition of the house\n\n'
             f'**Expect a statistically significant moderate positive monotonic correlation between these features and the sale price**.\n\n')

    st.write(f'#### Hypotheses set 4')
    st.write(f'Feature Group 4 (These features are not normally the most significant in determining sale price):')
    st.write(f'* GarageFinish: Interior finish of the garage\n'
             f'* GarageYrBlt: Year garage was built\n'
             f'* GarageArea: Size of garage in square feet\n'
             f'* EnclosedPorchSF: Enclosed porch area in square feet\n'
             f'* OpenPorchSF: Open porch area in square feet\n'
             f'* MasVnrArea: Masonry veneer area in square feet\n'
             f'* WoodDeckSF: Wood deck area in square feet\n'
             f'* BsmtExposure: Refers to walkout or garden level walls\n\n'
             f'**Expect a statistically significant weak positive monotonic correlation between these features and the sale price**.\n\n')
    st.write("### Validation methods")
    st.write(f'The formal part of each hypothesis, namely that a statistically significant positive monotonic correlation exists, was treated as an alternative hypothesis to the\n'
             f'null hypothesis: that no or a negative monotonic correlation exists. Spearman correlation coefficients were calculated, as well as their\n'
             f'significance to determine whether the alternative hypotheses could be accepted.\n\n'
             
             f'The magnitude of the spearman coefficients, along with the magnitude and significance of phi k correlation coefficients were used to establish\n'
             f'the strength of the relationship with sale price, as well as whether any other relatonship exists not monotonic in nature. Finally\n'
             f'Predictive Power Scores(PPS) were calulated and plots used to establish the strength of any relationship, and to verify the results of the correlation tests.\n\n')

    st.write('### Outcomes')

    st.write('#### Formal hypotheses')
    st.write(f'The **common alternative hypothesis** (existence of a statistically significant positive monotonic correlation) for **all features can be accepted**,\n'
             f'**except for** the OverallCond and EnclosedPorch features. Positive spearman coefficients were found that were statistically significant except for\n'
             f'the OverallCond and EnclosedPorch features, where small negative not statistically significant coefficients were obtained; this implies these features\n'
             f'have no monotonic correlation to sale price or a weak negative monotonic relationship.\n\n')

    st.write('#### Informal hypotheses')
    st.write(f'**With regard to the strength of any relationship to the sale price, as implied by the correlation tests**:\n'  
             f'* The quality feature group features have a strong relationship to sale price.\n'
             f'* Of the size group features, all but the 2ndFlrSF, BedroomAbvGr, BsmtUnfSF, BsmtFinSF1 have at least moderate Spearman correlations; whilst all\n'
             f'but LotArea, LotFrontage, BedroomAbvGr have at least a moderate dependence on sale price.\n'
             f'* The garage related features have at least a moderate correlation/dependence with sale price.\n'
             f'* On the whole the enclosed porch related feature has a weak relationship to the sale price, whilst the open porch feature may have a moderate relationship.\n'
             f'* Age related features, and the MasVnrArea feature have at least a moderate relationship to the sale price.\n'
             f'* The OverallCond, BedroomAbvGr and WoodDeckSF features have a weak relationship to the sale price.\n\n'

             f'**Scatter plots**:\n'

             f'* The scatter plots largely agree with the relationships implied by the correlation tests.\n'

             f'* They also to some extent explain why certain features appear to have a weak monotonic relationship, but at least a moderate dependence on\n'
             f'the sale price: namely because of certain feature values having greater variations in sale price; likewise weak relationships are illustrated\n'
             f'in the plots as clustering with less variation.\n\n'

             f'**PPS**:\n'

             f"* The PPS's for all features are not that strong or always consistent with strength of the relationships implied by the correlation coefficients,\n"
             f'but this is more likely because multiple features are necessary to predict the sale price. However some of the strongest correlated features,\n'
             f"also have the largest PPS's.\n\n")

    st.info('**View the sale price correlation study page to see correlation coefficient heatmaps, correlation strength description tables, as well as scatter plots**.')
