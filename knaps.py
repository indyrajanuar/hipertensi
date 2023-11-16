import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.title("Aplikasi Web Statis dengan Streamlit")

    # Membuat objek pilihan menu menggunakan st.sidebar.selectbox
    choose = st.sidebar.selectbox("Choose", ["Home", "Dataset", "Preprocessing", "Evaluasi", "Klasifikasi", "Help"])

    if choose == "Home":
        show_home()
    elif choose == "Dataset":
        show_dataset()
    elif choose == "Preprocessing":
        show_preprocessing()
    elif choose == "Evaluasi":
        show_evaluasi()
    elif choose == "Klasifikasi":
        show_klasifikasi()
    elif choose == "Help":
        show_help()

def show_home():
    st.write("Selamat datang di halaman Home!")
    # Tambahkan konten sesuai dengan halaman Home

# Fungsi untuk mengacak dataset
def shuffle_dataset(df):
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# Fungsi untuk menampilkan dataset dari CSV
def show_dataset():
    st.write("Halaman Dataset")

    # Impor dataset dari file CSV
    dataset_path = "https://raw.githubusercontent.com/indyrajanuar/hipertensi/main/datafix.csv"  # Gantilah dengan path sesuai dengan lokasi dataset Anda
    df = pd.read_csv(dataset_path)

    # Mengacak dataset secara otomatis saat halaman dimuat
    df = shuffle_dataset(df)

    # Tampilkan dataset setelah diacak
    #st.write("Berikut adalah tampilan beberapa baris pertama dari dataset setelah diacak:")
    st.dataframe(df)

# Fungsi untuk preprocessing data (menghapus data kategorikal pada fitur tertentu)
def preprocess_data(df):
    # Gantilah 'usia', 'sistole', 'diastole', 'nafas', 'detak nadi' dengan fitur yang sesuai dalam dataset Anda
    numerical_features = ['usia', 'sistole', 'diastole', 'nafas', 'detak nadi']

    # Menghapus data kategorikal pada fitur tertentu
    df_cleaned = df[numerical_features].dropna()

    return df_cleaned

# Fungsi untuk menampilkan halaman dataset
def show_dataset():
    st.write("Halaman Dataset")
    
    df = shuffle_dataset(df)

    # Menampilkan dataset
    st.write("Berikut adalah tampilan beberapa baris pertama dari dataset:")
    st.dataframe(df.head())

    # Tombol untuk memicu preprocessing
    if st.button("Lakukan Preprocessing"):
        df_cleaned = preprocess_data(df)
        st.write("Berikut adalah tampilan beberapa baris pertama dari dataset setelah preprocessing:")
        st.dataframe(df_cleaned.head())

if __name__ == "__main__":
    main()

def show_evaluasi():
    st.write("Halaman Evaluasi")
    # Tambahkan konten sesuai dengan halaman Evaluasi

def show_klasifikasi():
    st.write("Halaman Klasifikasi")
    # Tambahkan konten sesuai dengan halaman Klasifikasi

def show_help():
    st.write("Halaman Help")
    # Tambahkan konten sesuai dengan halaman Help

if __name__ == "__main__":
    main()
