import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler

def clean_numeric_data(data, features_to_clean):
    for feature in features_to_clean:
        if feature in data.columns:
            data[feature] = data[feature].apply(remove_non_numeric)
    return data

def remove_non_numeric(value):
    # Remove non-numeric characters using regular expression
    return re.sub(r'[^0-9.]', '', str(value))
    
def preprocess_data(df, features_to_clean, categorical_features):
    cleaned_data = clean_numeric_data(df, features_to_clean)
    encoded_data = pd.get_dummies(cleaned_data, columns=categorical_features)
    return cleaned_data, encoded_data

def normalize_data(data, features_to_normalize):
    scaler = MinMaxScaler()
    for feature in features_to_normalize:
        if feature in data.columns:
            st.write(f"Before Scaling - {feature}: Min={data[feature].min()}, Max={data[feature].max()}")
            data[feature] = scaler.fit_transform(data[[feature]])
            st.write(f"After Scaling - {feature}: Min={data[feature].min()}, Max={data[feature].max()}")
        else:
            st.warning(f"Warning: {feature} not found in DataFrame columns.")
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
    
    encoded_data = pd.DataFrame()  # Define encoded_data outside the 'One-Hot Encoding' block
    
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.dataframe(df)
        st.markdown('<h3 style="text-align: left;"> Melakukan Cleaning Data </h1>', unsafe_allow_html=True)
        
        # Specify the features to clean
        features_to_clean = ['Umur Tahun', 'Sistole', 'Diastole', 'Nafas', 'Detak Nadi']
        # Button to clean the data
        if st.button("Clean Data"):
            st.session_state.cleaned_data = clean_numeric_data(df, features_to_clean)
            st.write("Pada bagian ini dilakukan pembersihan dataset yang tidak memiliki relevansi terhadap faktor risiko pada penyakit hipertensi, seperti menghapus satuan yang tidak diperlukan dan menghapus noise.")
            st.dataframe(st.session_state.cleaned_data)

        st.markdown('<h3 style="text-align: left;"> Melakukan Transformation Data </h1>', unsafe_allow_html=True)
        # Specify the categorical features for one-hot encoding
        categorical_features = ['Jenis Kelamin', 'Diagnosa']
        # One-hot encoding
        if not st.session_state.cleaned_data.empty:
            if st.button("One-Hot Encoding"):
                encoded_data = pd.get_dummies(st.session_state.cleaned_data, columns=categorical_features)
                st.write("Pada bagian ini, dilakukan one-hot encoding untuk mengubah variabel kategorikal menjadi representasi biner.")
                st.dataframe(encoded_data)
                
            st.markdown('<h3 style="text-align: left;"> Melakukan Normalisasi Data </h1>', unsafe_allow_html=True)
            # Min-Max scaling
            features_to_normalize = ['Jenis Kelamin', 'IMT', 'Diagnosa', 'Umur Tahun', 'Sistole', 'Diastole', 'Nafas', 'Detak Nadi']
            if st.button("Min-Max Scaling"):
                st.write("Before Min-Max Scaling:")
                st.dataframe(encoded_data[features_to_normalize])
                st.session_state.normalized_data = normalize_data(encoded_data, features_to_normalize)
                st.write("After Min-Max Scaling:")
                st.dataframe(st.session_state.normalized_data[features_to_normalize])
            
elif selected == 'Klasifikasi ERNN':
    st.write("You are at Klasifikasi ERNN")

elif selected == 'Korelasi Data':
    st.write("You are at Korelasi Data")

elif selected == 'Uji Coba':
    st.write("You are at Uji Coba")
