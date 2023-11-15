# Import library
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
url = "datafix.csv"
df = pd.read_csv(url)

# Data preprocessing
X = df.iloc[:, 1:8]  # Selecting 7 input features
y = df.iloc[:, -1]   # Assuming the target variable is in the last column

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the ERNN model
mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, learning_rate_init=0.1, tol=0.0001, random_state=42)
mlp.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Streamlit web app
st.title("Hipertensi Classification App")

# Display dataset
st.subheader("Dataset")
st.dataframe(df)

# Display model evaluation results
st.subheader("Model Evaluation")
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
prediction = mlp.predict(input_data)

st.write("Prediction:", prediction[0])
