import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import re

# Initialize cleaned_data globally
cleaned_data = None

def clean_numeric_data(data, features_to_clean):
    for feature in features_to_clean:
        if feature in data.columns:
            data[feature] = data[feature].apply(remove_non_numeric)

    return data

def remove_non_numeric(value):
    return re.sub(r'[^0-9.]', '', str(value))

def one_hot_encode_data(data, features_to_encode):
    return pd.get_dummies(data, columns=features_to_encode, drop_first=True)

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
        
        # Specify the features to clean
        features_to_clean = ['Umur Tahun', 'Sistole', 'Diastole', 'Nafas', 'Detak Nadi']
        
        # Button to clean the data
        if st.button("Clean Data"):
            global cleaned_data  # Declare cleaned_data as global
            cleaned_data = clean_numeric_data(df, features_to_clean)
            st.write("Pada bagian ini dilakukan pembersihan dataset yang tidak memiliki relevansi terhadap faktor risiko pada penyakit hipertensi, seperti menghapus satuan yang tidak diperlukan dan menghapus noise.")
            st.dataframe(cleaned_data)

        st.markdown('<h3 style="text-align: left;"> Melakukan Transformation Data </h1>', unsafe_allow_html=True)
        # Specify the features to one-hot encode
        features_to_encode = ['Jenis Kelamin', 'Diagnosa']
        
        # Button to one-hot encode the data
        if cleaned_data is not None and st.button("One-Hot Encode"):
            encoded_data = one_hot_encode_data(cleaned_data, features_to_encode)
            st.write("One-Hot Encoded Data:")
            st.dataframe(encoded_data)
            
elif selected == 'Klasifikasi ERNN':
    st.write("You are at Klasifikasi ERNN")

elif selected == 'Korelasi Data':
    st.write("You are at Korelasi Data")

elif selected == 'Uji Coba':
    st.write("You are at Uji Coba")
