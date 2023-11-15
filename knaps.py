
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Function to load and preprocess dataset
def load_and_preprocess_data():
    # Load dataset
    url = "https://raw.githubusercontent.com/indyrajanuar/hipertensi/main/datafix.csv"
    
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Error reading dataset: {e}")
        return None, None, None, None, None, None

    # Clean data by removing categorical columns
    # Assuming 'usia', 'sistole', 'diastole', 'nafas', 'detak_nadi' are non-categorical features
    df_cleaned = df.drop(columns=['usia', 'sistole', 'diastole', 'nafas', 'detak_nadi'])

    # Separate features and target
    X = df_cleaned.drop(columns=['jenis_kelamin', 'diagnosa'])
    y = df_cleaned['diagnosa']

    # Numeric features after removing categorical columns
    numeric_features = X.columns.tolist()

    # Create transformers for preprocessing
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_features = ['jenis_kelamin']
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

    # Apply preprocessing to the training set
    X_train_processed = preprocessor.fit_transform(X_train)

    # Display the cleaned dataset after removing categorical columns
    st.subheader("Cleaned Dataset (Categorical Columns Removed and One-Hot Encoded):")
    st.dataframe(pd.DataFrame(X_train_processed, columns=numeric_features + preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(categorical_features)))

    return X_train, X_test, y_train, y_test, preprocessor, df_cleaned

# ... (function for training and classification)

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

        # Display cleaned dataset after removing categorical columns
        st.dataframe(df_cleaned)

    elif menu == "Preprocessing":
        st.title("Preprocessing")
        # Load and preprocess data
        _, _, _, _, _, _ = load_and_preprocess_data()

        # The cleaned dataset after removing categorical columns is displayed within the load_and_preprocess_data function

    elif menu == "Classification":
        st.title("Classification")
        # Load and preprocess data
        X_train, X_test, y_train, y_test, preprocessor, _ = load_and_preprocess_data()

        # Apply preprocessing to the testing set
        X_test_processed = preprocessor.transform(X_test)

        # ... (classification code)

if __name__ == "__main__":
    main()
