import streamlit as st

def page_summary_body():
    st.write('## Project Summary')
    st.write('### Background')
    st.info(f'The client recently received an inheritance from a deceased great-grandfather located in Ames, Iowa. Part of this\n'
            f'inheritance includes four properties, all situated in Ames, Iowa. The client is considering selling some, if not all,\n'
            f'of these properties, and naturally wants to maximise their sale price. \n\n'
             
            f'The client has an excellent understanding of property prices in their own state and residential area, however this knowledge may not generalise,\n'
            f'and may lead to inaccurate appraisals; consequently they desired the help of a data practitioner to assist in accurately valuing their properties.\n\n'

            f'To this end, the client desired to know what makes a house desirable and valuable in Ames, Iowa, in order to be able to\n'
            f'achieve a price for each property that reflects its value.')
    
    st.write('### Business requirements')
    st.info(f'1 - The client is interested in discovering how the house attributes correlate with the sale price in Ames, Iowa. Therefore, the client expects\n'
            f'data visualisations of the correlated house attributes against the sale price to illustrate any relationships.\n\n'

            f'2 - The client is interested in predicting the house sale price for their four inherited houses, and more generally any other house in Ames, Iowa.\n\n')

    st.write('### Dataset Overview')
    st.info(f'- The dataset consists of thousand rows corresponding to housing records from Ames, Iowa, indicating house profile\n'
    f'(Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built), and its respective sale price for houses built between 1872 and 2010.\n\n'

    f'- It can be found on [kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data).\n\n'

    f'- Each housing record consists of 22 house attributes, and the house sale price.\n\n'
    )

    st.write('### More Information')
    st.info('For more information [see](https://github.com/MatthewBayfield/heritage_housing_issues).')

