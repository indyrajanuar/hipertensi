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

    # Check if the dataset has sufficient samples for splitting
    if len(data) < 2:
        return None, None, "Insufficient data for classification"
    
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

    # Apply Threshold
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5).astype(int)

    # Calculate loss if applicable
    loss = None  # Placeholder for loss value
    if 'val_loss' in history.history:
        loss = history.history['val_loss'][-1]

    return y_test, y_pred, loss

def run_ernn_bagging(data):
    # split data fitur, target
    x = data.drop('Diagnosa', axis=1)
    y = data['Diagnosa']
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    # Convert target data to numpy array and reshape
    y_train = np.array(y_train).reshape(-1,)
    y_test = np.array(y_test).reshape(-1,)
    
    bagging_iterations = [1, 5, 10, 15]
    max_error = 0.001
    models = []

    # Classification Process
    accuracies_all_iterations = []  # List to store accuracies for each iteration
    for iteration in bagging_iterations:
        print("######## ITERATION - {} ########".format(iteration))
        accuracies_per_iteration = []  # List to store accuracies for the current iteration

        for i in range(iteration):
            print("Training model {} of {}...".format(i+1, iteration))
            # Randomly sample with replacement
            indices = np.random.choice(len(x_train), len(x_train), replace=True)
            x_bag = x_train.iloc[indices]
            y_bag = y_train[indices]

            model = keras.models.Sequential()
            model.add(keras.layers.Dense(6, activation='sigmoid', input_shape=(8,)))  # Hidden layer with 6 neurons
            model.add(keras.layers.Dense(6, activation='sigmoid'))  # Context layer with 6 neurons
            model.add(keras.layers.Dense(1, activation='sigmoid'))

            model.compile(loss='mean_squared_error',
                          optimizer=keras.optimizers.Adam(learning_rate=0.1),  # Learning rate set to 0.1
                          metrics=[keras.metrics.BinaryAccuracy()])

            history = model.fit(x_bag, y_bag,
                                batch_size=32, epochs=200, verbose=0)  # Set verbose=0 for less output
            models.append(model)

            print("Model {} training complete.".format(i+1))

            # Calculate accuracy for the current model
            accuracy = model.evaluate(x_test, y_test, verbose=0)[1]  # Evaluate accuracy
            accuracies_per_iteration.append(accuracy)
            
        # Calculate average accuracy for the current iteration
        avg_accuracy = np.mean(accuracies_per_iteration)
        accuracies_all_iterations.append(avg_accuracy)

    # Apply Threshold
    y_preds = []
    for model in models:
        y_pred = model.predict(x_test)
        y_pred = (y_pred > 0.5).astype(int)
        y_preds.append(y_pred)
    
    y_pred_avg = np.mean(y_preds, axis=0)  # Average predictions from all models
    
    # Plotting the accuracy
    plt.figure()
    plt.plot(bagging_iterations, accuracies_all_iterations, marker='o')
    plt.title('Accuracy vs Bagging Iterations')
    plt.xlabel('Number of Bagging Iterations')
    plt.ylabel('Average Accuracy')
    plt.grid(True)
    
    return y_test, y_pred_avg, plt.gcf(), bagging_iterations, accuracies_all_iterations  # Return accuracies along with the figure object

def main():
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
                y_true, y_pred, loss = classify_MLP(normalized_data)  # Assuming classify_MLP also returns loss
                
                # Generate confusion matrix
                cm = confusion_matrix(y_true, y_pred)
        
                # Plot confusion matrix
                plt.figure(figsize=(8, 6))
                bars = plt.bar(bagging_iterations, accuracies_all_iterations)
                plt.title('Average Accuracy vs Bagging Iterations')
                plt.xlabel('Number of Bagging Iterations')
                plt.ylabel('Average Accuracy')
                plt.xticks(bagging_iterations)
                plt.grid(axis='y')
                
                # Add text labels above each bar
                for bar, acc in zip(bars, accuracies_all_iterations):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), '{:.2f}%'.format(acc * 100),
                             ha='center', va='bottom')
                
                # Display the plot using Streamlit
                st.pyplot(plt)
                
                #plt.figure(figsize=(8, 6))
                #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                #plt.xlabel('Predicted')
                #plt.ylabel('True')
                #plt.title('Confusion Matrix')
                #st.pyplot()
                #st.pyplot(plt.gcf())  # Pass the current figure to st.pyplot()
        
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
                html_code = f"""
                <table style="margin: auto;">
                    <tr>
                        <td style="text-align: center;"><h5>Loss</h5></td>
                        <td style="text-align: center;"><h5>Accuracy</h5></td>
                        <td style="text-align: center;"><h5>Precision</h5></td>
                        <td style="text-align: center;"><h5>Recall</h5></td>
                    </tr>
                    <tr>
                        <td style="text-align: center;">{loss:.4f}</td>
                        <td style="text-align: center;">{accuracy:.2f}%</td>
                        <td style="text-align: center;">{precision:.2f}%</td>
                        <td style="text-align: center;">{recall:.2f}%</td>
                    </tr>
                </table>
                """
                
                st.markdown(html_code, unsafe_allow_html=True)
    
    elif selected == 'ERNN + Bagging':
        st.write("You are at Klasifikasi ERNN + Bagging")
        
        if upload_file is not None:
            df = pd.read_csv(upload_file)
            if 'preprocessed_data' in st.session_state:  # Check if preprocessed_data exists in session state
                normalized_data = normalize_data(st.session_state.preprocessed_data.copy())
                # Perform ERNN + Bagging classification
                y_test, y_pred, fig, bagging_iterations, accuracies_all_iterations = run_ernn_bagging(normalized_data)
                
                # Display the plot and accuracies
                st.pyplot(fig)  # Pass the figure object to st.pyplot()
                
                st.write("Average accuracies for each bagging iteration:")
                for iteration, accuracy in zip(bagging_iterations, accuracies_all_iterations):
                    st.write(f"Iteration {iteration}: {accuracy:.2f}%")
    
    elif selected == 'Uji Coba':
        st.title("Uji Coba")
        st.write("Masukkan nilai untuk pengujian:")
    
        # Input fields
        age = st.number_input("Umur", min_value=0, max_value=150, step=1, value=30)
        bmi = st.number_input("IMT", min_value=0.0, max_value=100.0, step=0.1, value=25.0)
        systole = st.number_input("Sistole", min_value=0, max_value=300, step=1, value=120)
        diastole = st.number_input("Diastole", min_value=0, max_value=200, step=1, value=80)
        breaths = st.number_input("Nafas", min_value=0, max_value=100, step=1, value=16)
        heart_rate = st.number_input("Detak Nadi", min_value=0, max_value=300, step=1, value=70)
        gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    
        # Convert gender to binary
        gender_binary = 1 if gender == "Perempuan" else 0
        
        # Button for testing
        if st.button("Hasil Uji Coba"):
            # Prepare input data for testing
            input_data = pd.DataFrame({
                "Umur": [age],
                "IMT": [bmi],
                "Sistole": [systole],
                "Diastole": [diastole],
                "Nafas": [breaths],
                "Detak Nadi": [heart_rate],
                "Jenis Kelamin": [gender_binary],
                "Diagnosa": [0]  # Placeholder value
            })
    
if __name__ == "__main__":
    main()
