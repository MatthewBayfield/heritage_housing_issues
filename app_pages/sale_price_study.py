import streamlit as st

def sale_price_study_body():
    st.write("## The relationship between a house's attributes and its sale price")
    st.write('### Overview')
    st.write(f'In the dataset there were 22 individual attributes, often related, and groupable by being a measure of size, quality, age,\n'
             f'condition or form.\n'
             f'Despite the large number of attributes, the actual number of independent features is far less, due to there being many moderate-to-strong\n'
             f'correlations between attributes.\n\n'

             f'Some possible limitations of the dataset worth noting, are the lack of any attributes relating to its location,\n'
             f'or the time of sale, both which likely impact the sale price.\n\n'

             f'Irrespective of any dataset limitations, statistically significant correlations/relationships were discovered between various house attributes\n'
             f'and the sale price, which shall now be explored.'
             )
