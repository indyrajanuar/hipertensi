import streamlit as st 
import pandas as pd 
import numpy as np 
from streamlit_option_menu import option_menu

with st.sidebar:
    choose = option_menu("Main Menu", ["Home", "PreProcessing Data", "Klasifikasi ERNN", "Korelasi Data", "Uji Coba"],
                             icons=['house', 'table', 'boxes', 'boxes','check2-circle'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding": "5!important", "background-color": "10A19D"},
            "icon": {"color": "blue", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#00FFFF"},
        }
        )
if choose=='Home':

elif choose=='PreProcessing Data':
 
elif choose=='Klasifikasi ERNN':
  
elif choose=='Korelasi Data':
    
elif choose == 'Uji Coba':

if __name__ == "__main__":
    main()
