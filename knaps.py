import streamlit as st 
import pandas as pd 
import numpy as np 
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu("Main Menu", ["Home", "PreProcessing Data", "Klasifikasi ERNN", "Korelasi Data", "Uji Coba"], icons=['house', 'table', 'boxes', 'boxes','check2-circle'], menu_icon="cast", default_index=1, orientation='vertical')

if selected=='Home':
    st.markdown('<h1 style = "text-align: center;"> Website Klasifikasi Hipertensi </h1>', unsafe_allow_html = True)

elif selected=='PreProcessing Data':
    st.write("You are at home")
 
elif selected=='Klasifikasi ERNN':
    st.write("You are at home")
  
elif selected=='Korelasi Data':
    st.write("You are at home")
    
elif selected=='Uji Coba':
    st.write("You are at home")

with st.sidebar:
    upload_file = st.sidebar.file_uploader("Masukkan file excel atau csv disini", key=1)
