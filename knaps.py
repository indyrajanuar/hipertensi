import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Function to load dataset
def load_dataset():
    url = "https://github.com/indyrajanuar/hipertensi/blob/main/datafix.csv"
    df = pd.read_csv(url)
    return df

# Function for data preprocessing
def preprocess_data(df):
    X = df.iloc[:, 1:8]  # Selecting 7 input features
    y = df.iloc[:, -1]   # Assuming the target variable is in the last column

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

# Function for classification
def classify(X_train, X_test, y_train, y_test):
    # Create and train the ERNN model
    mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, learning_rate_init=0.1, tol=0.0001, random_state=42)
    mlp.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = mlp.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return mlp, accuracy, report

# Streamlit web app
def main():
    st.title("Hipertensi Classification App")

    # Create a sidebar menu
    menu = ["Home", "Import Dataset", "Preprocessing", "Classification"]
    choice = st.sidebar.selectbox("Select Menu", menu)

    # Home page
    if choice == "Home":
        st.write("Welcome to the Hipertensi Classification App. Use the sidebar to navigate.")

    # Import Dataset page
    elif choice == "Import Dataset":
        st.subheader("Import Dataset")
        df = load_dataset()
        st.dataframe(df)

    # Preprocessing page
    elif choice == "Preprocessing":
        st.subheader("Preprocessing")
        df = load_dataset()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

        st.write("Dataset after Preprocessing:")
        st.write("X_train shape:", X_train.shape)
        st.write("X_test shape:", X_test.shape)

    # Classification page
    elif choice == "Classification":
        st.subheader("Classification")
        df = load_dataset()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        model, accuracy, report = classify(X_train, X_test, y_train, y_test)

        st.write("Model Evaluation:")
        st.write("Accuracy:", accuracy)
        st.write("Classification Report:")
        st.text(report)

        # Input form for prediction
        st.subheader("Make Prediction")
        input_data = []
        for i in range(7):
            input_data.append(st.number_input(f"Input Feature {i+1}"))

        input_data = np.array(input_data).reshape(1, -1)

        # Standardize input features
        input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)

        st.write("Prediction:", prediction[0])

if __name__ == "__main__":
    main()
