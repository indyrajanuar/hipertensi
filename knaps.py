import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

def label_encode_data(data, categorical_features):
    label_encoder = LabelEncoder()
    for feature in categorical_features:
        if feature in data.columns:
            data[feature] = label_encoder.fit_transform(data[feature])
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
    upload_file = st.sidebar.file_uploader("Masukkan file excel disini", key=1)

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
        st.markdown('<h3 style="text-align: left;"> Melakukan Transformation Data </h1>', unsafe_allow_html=True)

        # Specify the categorical features for one-hot encoding
        categorical_features = ['Jenis Kelamin', 'Diagnosa']
        # One-hot encoding
        if not st.session_state.cleaned_data.empty:
            if st.button("Label Encoding"):
                encoded_data = label_encode_data(st.session_state.cleaned_data, categorical_features)
                st.write("Label encoding completed.")
                st.dataframe(encoded_data)
                st.write(encoded_data.shape)
                st.write(encoded_data.dtypes)
                st.write(encoded_data.isnull().sum())
                
elif selected == 'Klasifikasi ERNN':
    st.write("You are at Klasifikasi ERNN")

elif selected == 'Korelasi Data':
    st.write("You are at Korelasi Data")

elif selected == 'Uji Coba':
    st.write("You are at Uji Coba")
