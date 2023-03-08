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
    st.write("### Validation methods")
    st.write('### Outcomes')
    