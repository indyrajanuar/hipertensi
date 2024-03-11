import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def preprocess_data(data):
    # Replace commas with dots and convert numerical columns to floats
    numerical_columns = ['IMT']
    data[numerical_columns] = data[numerical_columns].replace({',': '.'}, regex=True).astype(float)
    
    # One-hot encoding for 'Jenis Kelamin'
    one_hot_encoder = OneHotEncoder(sparse=False)
    encoded_features = pd.DataFrame(one_hot_encoder.fit_transform(data[['Jenis Kelamin']]))
    encoded_features.columns = one_hot_encoder.get_feature_names_out(['Jenis Kelamin'])
    data = pd.concat([data.drop('Jenis Kelamin', axis=1), encoded_features], axis=1)
    
    # Transform 'Diagnosa' feature to '1' for 'YA' and '0' for 'TIDAK'
    data['Diagnosa'] = data['Diagnosa'].map({'YA': 1, 'TIDAK': 0})
    
    return data

def normalize_data(data):
    scaler = MinMaxScaler()
    data[data.columns] = scaler.fit_transform(data[data.columns])
    return data

with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Home", "PreProcessing Data", "Klasifikasi ERNN", "Klasifikasi ERNN + Bagging", "Uji Coba"],
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
            st.session_state.preprocessed_data = preprocessed_data  # Store preprocessed data in session state

        st.markdown('<h3 style="text-align: left;"> Melakukan Normalisasi Data </h1>', unsafe_allow_html=True)
        if 'preprocessed_data' in st.session_state:  # Check if preprocessed_data exists in session state
            if st.button("Normalize Data"):
                normalized_data = normalize_data(st.session_state.preprocessed_data.copy())
                st.write("Normalization completed.")
                st.dataframe(normalized_data)

elif selected == 'Klasifikasi ERNN':
    st.write("Berikut merupakan hasil klasifikasi yang di dapat dari pemodelan  Elman Recurrent Neural Network (ERNN)")
    
    # Load the trained model
    model = keras.models.load_model("your_model.h5")  # Provide the correct path to your saved model

    # Perform prediction
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5)  # Thresholding predictions

    # Convert predictions to binary
    y_pred_binary = np.round(y_pred)

    # Evaluate model
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_test, y_pred_binary)

    # Plotting confusion matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot()

    # Display classification report
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred_binary))

elif selected == 'Klasifikasi ERNN + Bagging':
    st.write("You are at Korelasi Data")

elif selected == 'Uji Coba':
    st.write("You are at Uji Coba")
