import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Function to load and preprocess dataset
def load_and_preprocess_data():
    # Load dataset
    url = "https://github.com/indyrajanuar/hipertensi/blob/main/datafix.csv"
    
    try:
        df = pd.read_csv(url, sep=';')  # Ganti dengan delimiter yang sesuai
    except Exception as e:
        st.error(f"Error reading dataset: {e}")
        return None, None, None, None, None, None

    # Clean data (handle missing values, outliers, etc.)
    # Assume the target variable is in the 'target' column
    df_cleaned = df.dropna()  # Replace with your cleaning process

    # Separate features and target
    X = df_cleaned.drop(columns=['diagnosa'])
    y = df_cleaned['diagnosa']

    # One-hot encoding for categorical variables
    categorical_features = ['categorical_column']  # Replace with your categorical column names
    numeric_features = [col for col in X.columns if col not in categorical_features]

    # Create transformers for preprocessing
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor, df_cleaned

# Function to train ERNN model
def train_ernn(X_train, y_train):
    # Create and train the ERNN model
    mlp = MLPClassifier(
        hidden_layer_sizes=(5,),
        max_iter=1000,
        learning_rate_init=0.1,
        tol=0.0001,
        random_state=42
    )
    
    mlp.fit(X_train, y_train)
    return mlp

# Streamlit web app
def main():
    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox("Select Menu", ["Home", "Dataset", "Preprocessing", "Classification"])

    if menu == "Home":
        st.title("Home")
        st.write("Deskripsi mengenai penyakit hipertensi.")

    elif menu == "Dataset":
        st.title("Dataset")
        # Load and preprocess data
        _, _, _, _, _, df_cleaned = load_and_preprocess_data()

        # Display dataset
        st.dataframe(df_cleaned)

    elif menu == "Preprocessing":
        st.title("Preprocessing")
        # Load and preprocess data
        X_train, _, _, _, _, _ = load_and_preprocess_data()

        st.write("Dataset after Preprocessing:")
        st.write("X_train shape:", X_train.shape)

    elif menu == "Classification":
        st.title("Classification")
        # Load and preprocess data
        X_train, X_test, y_train, y_test, preprocessor, _ = load_and_preprocess_data()

        # Apply preprocessing to the training and testing sets
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Train ERNN model
        ernn_model = train_ernn(X_train_processed, y_train)

        # Make predictions on the test set
        y_pred = ernn_model.predict(X_test_processed)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Display results
        st.write("Model Evaluation:")
        st.write("Accuracy:", accuracy)
        st.write("Classification Report:")
        st.text(report)

if __name__ == "__main__":
    main()
