import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import keras
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_data(data): 
    # Replace commas with dots and convert numerical columns to floats
    numerical_columns = ['IMT']
    data[numerical_columns] = data[numerical_columns].replace({',': '.'}, regex=True).astype(float)
    # One-hot encoding for 'Jenis Kelamin'
    one_hot_encoder = OneHotEncoder()
    encoded_gender = one_hot_encoder.fit_transform(data[['Jenis Kelamin']].values.reshape(-1, 1))
    encoded_gender = pd.DataFrame(encoded_gender.toarray(), columns=one_hot_encoder.get_feature_names_out(['Jenis Kelamin']))    
    # Transform 'Diagnosa' feature to binary values
    data['Diagnosa'] = data['Diagnosa'].map({'YA': 1, 'TIDAK': 0})
    # Drop the original 'Jenis Kelamin' feature
    data = data.drop('Jenis Kelamin', axis=1)    
    # Concatenate encoded 'Jenis Kelamin' and transformed 'Diagnosa' with original data
    data = pd.concat([data, encoded_gender], axis=1)
    return data

def normalize_data(data):
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return normalized_data

# Function for classification using MLP (Multilayer Perceptron)
def classify_MLP(data):
    # split data fitur, target
    x = data.drop('Diagnosa', axis=1)
    y = data['Diagnosa']
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    # Convert target data to numpy array and reshape
    y_train = np.array(y_train).reshape(-1,)
    y_test = np.array(y_test).reshape(-1,)

    kf = KFold(n_splits=5)
    fold_n = 1
    max_error = 0.001

    for index, (train_idx, val_idx) in enumerate(kf.split(x_train)):
        print("######## FOLD - {} ########".format(fold_n))

        x_train_fold, x_val_fold = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(6, activation='sigmoid', input_shape=(8,)))  # Hidden layer with 6 neurons
        model.add(keras.layers.Dense(6, activation='sigmoid'))  # Context layer with 6 neurons
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(loss='mean_squared_error',
                      optimizer=keras.optimizers.Adam(learning_rate=0.1),  # Learning rate set to 0.1
                      metrics=[keras.metrics.BinaryAccuracy()])

        history = model.fit(x_train_fold, y_train_fold, validation_data=(x_val_fold, y_val_fold),
                            batch_size=32, epochs=200)

        last_val_loss = history.history['val_loss'][-1]

        if last_val_loss < max_error:
            print("Maximum error condition met, stopping training.")
            break
            
        fold_n += 1

    # Predict using the trained model on the test data
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5).astype(int)

    return y_test, y_pred

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
        if st.button("Transformation Data"):  # Check if button is clicked
            preprocessed_data = preprocess_data(df)
            st.write("Transformation completed.")
            st.dataframe(preprocessed_data)
            st.session_state.preprocessed_data = preprocessed_data  # Store preprocessed data in session state

        st.markdown('<h3 style="text-align: left;"> Melakukan Normalisasi Data </h1>', unsafe_allow_html=True)
        if 'preprocessed_data' in st.session_state:  # Check if preprocessed_data exists in session state
            if st.button("Normalize Data"):
                normalized_data = normalize_data(st.session_state.preprocessed_data.copy())
                st.write("Normalization completed.")
                st.dataframe(normalized_data)

elif selected == 'Klasifikasi ERNN':
    st.write("Berikut merupakan hasil klasifikasi yang di dapat dari pemodelan Elman Recurrent Neural Network (ERNN)")

    if upload_file is not None:
        df = pd.read_csv(upload_file)
        if 'preprocessed_data' in st.session_state:  # Check if preprocessed_data exists in session state
            normalized_data = normalize_data(st.session_state.preprocessed_data.copy())
            y_true, y_pred = classify_MLP(normalized_data)
            
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        #st.pyplot()
        st.pyplot(plt.gcf())  # Pass the current figure to st.pyplot()

        # Clear the current plot to avoid displaying it multiple times
        plt.clf()

        # Generate classification report
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppress division by zero warning
            report = classification_report(y_true, y_pred, zero_division=0)

        # Extract metrics from the classification report
        lines = report.split('\n')
        accuracy = float(lines[5].split()[1]) * 100
        precision = float(lines[2].split()[1]) * 100
        recall = float(lines[3].split()[1]) * 100

        # Display the metrics
        html_header = '<h5 style="text-align: center; word-spacing: 5em"> Accuracy Precision Recall </h5>'
        html_values = f'<div style="text-align: center; word-spacing: 7.5em;">{accuracy:.2f}% {precision:.2f}% {recall:.2f}%</div>'

        st.markdown(html_header, unsafe_allow_html=True)
        st.markdown(html_values, unsafe_allow_html=True)

elif selected == 'Klasifikasi ERNN + Bagging':
    st.write("You are at Korelasi Data")

elif selected == 'Uji Coba':
    st.write("You are at Uji Coba")
