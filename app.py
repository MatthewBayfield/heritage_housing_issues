import streamlit as st
from app_pages.multipage import MultiPage
from app_pages.page_summary import page_summary_body
from app_pages.sale_price_study import sale_price_study_body

app = MultiPage(app_name= "Heritage Housing Issues")

app.add_page('Project Summary', page_summary_body)
app.add_page("How are a house's attributes related to its sale price", sale_price_study_body)

app.run()