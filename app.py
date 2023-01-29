import streamlit as st
from app_pages.multipage import MultiPage
from app_pages.page_summary import page_summary_body

app = MultiPage(app_name= "Heritage Housing Issues")

app.add_page('Project Summary', page_summary_body)

app.run()