import streamlit as st 
import pandas as pd 
import numpy as np 
from streamlit_option_menu import option_menu

selected = om("Main Menu", ["Home", "PreProcessing Data", "Klasifikasi ERNN", "Korelasi Data", "Uji Coba"], icons=['house', 'table', 'boxes', 'boxes','check2-circle'], menu_icon="cast", default_index=1, orientation='vertical')
selected

if selected=='Home':
    st.write("You are at home")

elif selected=='PreProcessing Data':
    st.write("You are at home")
 
elif selected=='Klasifikasi ERNN':
    st.write("You are at home")
  
elif selected=='Korelasi Data':
    st.write("You are at home")
    
elif selected=='Uji Coba':
    st.write("You are at home")
