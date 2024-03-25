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
    
    #membagi data training dan testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    #ERNN 
    kf = KFold(n_splits=5)
    fold_n = 1
    X = np.array(x_train)
    Y = np.array(y_train)
    max_error = 0.001
    for index, (train_idx, val_idx) in enumerate(kf.split(X)):
        print("######## FOLD - {} ########".format(fold_n))
        x_train, x_val = X[train_idx], X[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(6, activation='sigmoid', input_shape=(8,)))  # Hidden layer with 6 neurons
        model.add(keras.layers.Dense(6, activation='sigmoid'))  # Context layer with 6 neurons
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(loss='mean_squared_error',
                      optimizer=keras.optimizers.Adam(learning_rate=0.1),  # Learning rate set to 0.1
                      metrics=[keras.metrics.BinaryAccuracy()])

        history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                            batch_size=32, epochs=200)
        
        last_val_loss = history.history['val_loss'][-1]
        
        if last_val_loss < max_error:
            print("Maksimum error terpenuhi maka training dihentikan.")
            break
            
        model.save("model_fold_{}.h5".format(index + 1))
        fold_n += 1

        print("Processing Time: %s seconds" % (time.time() - start_time))
        
    #Evaluasi performa model terhadap data train
    model.evaluate(x_train, y_train)
    
    y_pred = model.predict(x_test) #melakukan prediksi data test dengan menggunakan model yang sudah ditraining sebelumnya
    # Memprediksi data uji dengan menggunakan fungsi threshold
    threshold = 0.5
    y_pred = (y_pred > 0.5).astype(int)
    
    #Evaluasi performa model terhadap data uji
    model.evaluate(x_test, y_test)

# Function to display confusion matrix and classification report
def display_metrics(y_test, y_pred):
    st.subheader("Confusion Matrix:")
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    # Plotting
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Visualizing confusion matrix as heatmap
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    st.write(cm)

    st.subheader("Classification Report:")
    # Confusion Matrix
    cr = classification_report(y_test, y_pred)
    st.write(cr)
    
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
    st.write("Berikut merupakan hasil klasifikasi yang di dapat dari pemodelan  Elman Recurrent Neural Network (ERNN)")
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        preprocessed_data = preprocess_data(df)
        y_test, y_pred = classify_MLP(preprocessed_data)
        display_metrics(y_test, y_pred)

elif selected == 'Klasifikasi ERNN + Bagging':
    st.write("You are at Korelasi Data")

elif selected == 'Uji Coba':
    st.write("You are at Uji Coba")
