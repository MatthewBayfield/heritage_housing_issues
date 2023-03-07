import streamlit as st
import pandas as pd

@st.cache
def load_csv(filepath):
    df = pd.read_csv(filepath)
    return df