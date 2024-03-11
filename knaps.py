import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(data): 
    # One-hot encoding for 'Jenis Kelamin'
    one_hot_encoder = OneHotEncoder()
    encoded_features = pd.DataFrame(one_hot_encoder.fit_transform(data[['Jenis Kelamin']]))
    encoded_features.columns = one_hot_encoder.get_feature_names_out(['Jenis Kelamin'])
    data = pd.concat([data.drop('Jenis Kelamin', axis=1), encoded_features], axis=1)

    return data
    
with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Home", "PreProcessing Data", "Klasifikasi ERNN", "ERNN + Bagging", "Uji Coba"],
        icons=['house', 'table', 'boxes', 'boxes', 'check2-circle'],
        menu_icon="cast",
        default_index=1,
        orientation='vertical')

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
        st.markdown('<h3 style="text-align: left;"> Melakukan Transformation Data </h1>', unsafe_allow_html=True)
        if st.button("Preprocess Data"):
            preprocessed_data = preprocess_data(df)
            st.write("Preprocessing completed.")
            st.dataframe(preprocessed_data)
            
elif selected == 'Klasifikasi ERNN':
    st.write("Berikut merupakan hasil klasifikasi yang di dapat dari pemodelan  Elman Recurrent Neural Network (ERNN)")

elif selected == 'Klasifikasi ERNN + Bagging':
    st.write("You are at Korelasi Data")

elif selected == 'Uji Coba':
    st.write("You are at Uji Coba")
