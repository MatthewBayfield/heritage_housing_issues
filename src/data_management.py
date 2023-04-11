import streamlit as st
import pandas as pd
import joblib

@st.cache(allow_output_mutation=True)
def load_csv(filepath, **kwargs):
    df = pd.read_csv(filepath, **kwargs)
    return df

@st.cache(allow_output_mutation=True)
def load_pkl_file(file_path, **kwargs):
    return joblib.load(filename=file_path, **kwargs)