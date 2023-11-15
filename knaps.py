import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from streamlit_option_menu import option_menu  #pustaka yang memberikan fungsi tambahan untuk membuat menu pilihan dengan Streamlit
with st.sidebar: #Fungsi tersebut menghasilkan objek pilihan menu
    choose = option_menu("Linear Regression (Polynomial)", ["Home", "Dataset", "Prepocessing", "Predict", "Help"],
                             icons=['house', 'table', 'cloud-upload', 'boxes','check2-circle'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding":"5!important", "background-color": "10A19D"}, #Mengatur tampilan kontainer (wadah) dari menu pilihan
            "icon": {"color": "blue", "font-size": "25px"},  #Mengatur tampilan ikon dalam menu pilihan. Properti "color" mengatur warna ikon. Properti "font-size" mengatur ukuran font ikon.
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"}, #Mengatur tampilan tautan dalam menu pilihan
            "nav-link-selected": {"background-color": "#00FFFF"}, #Mengatur tampilan tautan yang dipilih dalam menu pilihan
