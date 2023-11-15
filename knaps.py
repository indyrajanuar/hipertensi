import streamlit as st

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

def show_dataset():
    st.write("Halaman Dataset")
    # Tambahkan konten sesuai dengan halaman Dataset

def show_preprocessing():
    st.write("Halaman Preprocessing")
    # Tambahkan konten sesuai dengan halaman Preprocessing

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
