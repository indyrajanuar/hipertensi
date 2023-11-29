import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

def clean_data(data, categorical_features_to_remove):
    # Hapus objek kategorikal pada fitur tertentu
    data = data.drop(columns=categorical_features_to_remove)
    return data

with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Home", "PreProcessing Data", "Klasifikasi ERNN", "Korelasi Data", "Uji Coba"],
        icons=['house', 'table', 'boxes', 'boxes', 'check2-circle'],
        menu_icon="cast",
        default_index=1,
        orientation='vertical')

with st.sidebar:
    upload_file = st.sidebar.file_uploader("Masukkan file csv disini", key=1)

if selected == 'Home':
    st.markdown('<h1 style="text-align: center;"> Website Klasifikasi Hipertensi </h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: left;"> Hipertensi </h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: left;"> View Data </h1>', unsafe_allow_html=True)
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.write("Data yang digunakan yaitu data Penyakit Hipertensi dari UPT Puskesmas Modopuro Mojokerto.")
        st.dataframe(df)

elif selected == 'PreProcessing Data':
    st.markdown('<h3 style="text-align: left;"> Data Asli </h1>', unsafe_allow_html=True)
    st.write("Berikut merupakan data asli yang didapat dari UPT Puskesmas Modopuro Mojokerto.")
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.dataframe(df)
        st.markdown('<h3 style="text-align: left;"> Melakukan Cleaning Data </h1>', unsafe_allow_html=True)

        # Pilihan fitur kategorikal yang akan dihapus
        categorical_features_to_remove = st.multiselect(
            "Umur Tahun", data.columns
        )
        
        # Button to clean the data
        if st.button("Clean Data"):
            # Panggil fungsi preprocessing
            cleaned_data = clean_data(data, categorical_features_to_remove)

            st.write("Pada bagian ini dilakukan pembersihan dataset yang tidak memiliki relevansi terhadap faktor risiko pada penyakit hipertensi, seperti menghapus satuan yang tidak diperlukan dan menghapus noise.")
            st.dataframe(cleaned_data)

elif selected == 'Klasifikasi ERNN':
    st.write("You are at Klasifikasi ERNN")

elif selected == 'Korelasi Data':
    st.write("You are at Korelasi Data")

elif selected == 'Uji Coba':
    st.write("You are at Uji Coba")
