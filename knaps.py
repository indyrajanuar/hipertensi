import streamlit as st
import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Function to preprocess data
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

# Function to normalize data
def normalize_data(data):
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return normalized_data

# Load data
upload_file = st.file_uploader("Masukkan file csv disini", key=1)
if upload_file is not None:
    df = pd.read_csv(upload_file)

# Preprocess and normalize data
preprocessed_data = preprocess_data(df)
normalized_data = normalize_data(preprocessed_data)

# Sidebar menu
with st.sidebar:
    selected = st.selectbox("Main Menu", ["Home", "PreProcessing Data", "Klasifikasi ERNN", "ERNN + Bagging", "Uji Coba"])

# Main menu
if selected == 'Home':
    st.markdown('<h1 style="text-align: center;"> Website Klasifikasi Hipertensi </h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: left;"> Hipertensi </h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: left;"> View Data </h1>', unsafe_allow_html=True)
    if upload_file is not None:
        st.write("Data yang digunakan yaitu data Penyakit Hipertensi dari UPT Puskesmas Modopuro Mojokerto.")
        st.dataframe(df)

elif selected == 'PreProcessing Data':
    st.markdown('<h3 style="text-align: left;"> Data Asli </h1>', unsafe_allow_html=True)
    st.write("Berikut merupakan data asli yang didapat dari UPT Puskesmas Modopuro Mojokerto.")
    if upload_file is not None:
        st.dataframe(df)
        st.markdown('<h3 style="text-align: left;"> Melakukan Transformation Data </h1>', unsafe_allow_html=True)
        if st.button("Transformation Data"):  # Check if button is clicked
            st.write("Transformation completed.")
            st.dataframe(preprocessed_data)

        st.markdown('<h3 style="text-align: left;"> Melakukan Normalisasi Data </h1>', unsafe_allow_html=True)
        if st.button("Normalize Data"):
            st.write("Normalization completed.")
            st.dataframe(normalized_data)

elif selected == 'Klasifikasi ERNN':
    st.write("Proses Klasifikasi menggunakan Elman Recurrent Neural Network (ERNN)")

    # Columns for features and target
    kolom_X = ['Umur Tahun', 'IMT', 'Sistole', 'Diastole', 'Nafas', 'Detak Nadi', 'Jenis Kelamin_L', 'Jenis Kelamin_P']
    kolom_y = ['Diagnosa']

    # Choosing features and target from normalized data
    x = normalized_data[kolom_X]
    y = normalized_data[kolom_y]

    # Splitting data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    kf = KFold(n_splits=5)
    fold_n = 1
    X = np.array(x_train)
    Y = np.array(y_train)
    max_error = 0.001

    for index, (train_idx, val_idx) in enumerate(kf.split(X)):
        print("######## FOLD - {} ########".format(fold_n))

        x_train_fold, x_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = Y[train_idx], Y[val_idx]

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
            print("Maksimum error terpenuhi maka training dihentikan.")
            break

        model.save("model_fold_{}.h5".format(index + 1))
        fold_n += 1

        print("Processing Time: %s seconds" % (time.time() - start_time))

    # Evaluating model performance on training data
    model.evaluate(x_train, y_train)

    # Predicting test data using the trained model
    y_pred = model.predict(x_test)

    # Applying threshold function for test data prediction
    threshold = 0.5
    y_pred = (y_pred > 0.5).astype(int)

    # Evaluating model performance on test data
    model.evaluate(x_test, y_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plotting
    st.write("Confusion Matrix:")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Visualizing confusion matrix as heatmap
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot()

    # Classification Report
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
