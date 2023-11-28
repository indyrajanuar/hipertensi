import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Home", "PreProcessing Data", "Klasifikasi ERNN", "Korelasi Data", "Uji Coba"],
        icons=['house', 'table', 'boxes', 'boxes', 'check2-circle'],
        menu_icon="cast",
        default_index=1,
        orientation='vertical'
    )

if selected == 'Home':
    st.markdown('<h1 style="text-align: center;"> Website Klasifikasi Hipertensi </h1>', unsafe_allow_html=True)
    
    # Display the uploaded file content
    if 'upload_file' in st.session_state:
        uploaded_file = st.session_state.upload_file
        if uploaded_file is not None:
            try:
                # Try reading the CSV file with 'latin1' encoding
                df = pd.read_csv(uploaded_file, encoding='latin1')
                st.write("Uploaded File Content:")
                st.write(df)
            except UnicodeDecodeError:
                st.error("Unable to decode the file. Please check the file encoding and try again.")

elif selected == 'PreProcessing Data':
    st.write("You are at PreProcessing Data")

elif selected == 'Klasifikasi ERNN':
    st.write("You are at Klasifikasi ERNN")

elif selected == 'Korelasi Data':
    st.write("You are at Korelasi Data")

elif selected == 'Uji Coba':
    st.write("You are at Uji Coba")

with st.sidebar:
    upload_file = st.sidebar.file_uploader("Masukkan file excel atau csv disini", key=1)

    # Save the uploaded file to session state
    if upload_file is not None:
        st.session_state.upload_file = upload_file
