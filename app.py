import streamlit as st
from app_pages.multipage import MultiPage
from app_pages.page_summary import page_summary_body
from app_pages.sale_price_study import sale_price_study_body
from app_pages.project_hypotheses_page import project_hypotheses_body
from app_pages.sale_price_predictor import sale_price_predictor_body

app = MultiPage(app_name= "Heritage Housing Issues")

app.add_page('Project Summary', page_summary_body)
app.add_page('Sale price correlation study', sale_price_study_body)
app.add_page('Sale price predictor', sale_price_predictor_body)
app.add_page('Project Hypotheses', project_hypotheses_body)

app.run()