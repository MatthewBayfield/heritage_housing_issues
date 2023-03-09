import streamlit as st
import pandas as pd
import joblib

@st.cache
def load_csv(filepath):
    df = pd.read_csv(filepath)
    return df

@st.cache
def load_pkl_file(file_path):
    return joblib.load(filename=file_path)