import streamlit as st
from app_pages.multipage import MultiPage

app = MultiPage(app_name= "Heritage Housing Issues") 

app.run()