import streamlit as st #import modul Streamlit yang digunakan untuk membangun antarmuka pengguna
import pandas as pd #import modul pandas yang digunakan untuk analisis data
import numpy as np #import modul numpy

from streamlit_option_menu import option_menu  #pustaka yang memberikan fungsi tambahan untuk membuat menu pilihan dengan Streamlit

from sklearn.model_selection import train_test_split

with st.sidebar: #Fungsi tersebut menghasilkan objek pilihan menu
    choose = option_menu("Linear Regression (Polynomial)", ["Home", "Dataset", "Prepocessing", "Predict", "Help"],
                             icons=['house', 'table', 'cloud-upload', 'boxes','check2-circle'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding":"5!important", "background-color": "10A19D"}, #Mengatur tampilan kontainer (wadah) dari menu pilihan
            "icon": {"color": "blue", "font-size": "25px"},  #Mengatur tampilan ikon dalam menu pilihan. Properti "color" mengatur warna ikon. Properti "font-size" mengatur ukuran font ikon.
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"}, #Mengatur tampilan tautan dalam menu pilihan
            "nav-link-selected": {"background-color": "#00FFFF"}, #Mengatur tampilan tautan yang dipilih dalam menu pilihan
        }
        )
if choose=='Home':
    st.markdown('<h1 style = "text-align: center;"> Prediksi Harga Rumah</h1>', unsafe_allow_html = True)
    logo = Image.open('makam1.jpg')

    st.image(logo, use_column_width=True, caption='Rumah di Jaksel') #mengatur lebar gambar agar sesuai dengan lebar kolom
    st.write('<p style = "text-align: justify;">Rumah merupakan salah satu kebutuhan pokok manusia, selain sandang dan pangan, rumah juga berfungsi sebagai tempat tinggal dan berfungsi untuk melindungi dari gangguan iklim dan makhluk hidup lainnya. Tak kalah buruknya dengan emas, rumah pun bisa dijadikan sebagai sarana investasi masa depan karena pergerakan harga yang berubah dari waktu ke waktu, dan semakin banyak orang yang membutuhkan hunian selain kedekatan dengan tempat kerja, pusat perkantoran dan pusat bisnis, transportasi. dll tentunya akan cepat mempengaruhi harga rumah tersebut.</p>', unsafe_allow_html = True)
    st.write('<p style = "text-align: justify;">Dalam proyek ini, kami mengembangkan sebuah sistem untuk memprediksi harga rumah berdasarkan parameter luas tanah dan luas bangunan, dan output yang dihasilkan adalah prediksi harga rumah. Kami menggunakan metode regresi linear dengan fitur ekspansi (expand feature) dan melatih model menggunakan metode Stochastic Gradient Descent. Untuk mengevaluasi model, kami menggunakan metrik MSE, RMSE, dan R (Square).Diharapkan dengan adanya sistem ini, dapat membantu dalam memprediksi harga rumah sesuai dengan luas tanah dan luas bangunan yang diinginkan.</p>', unsafe_allow_html = True)
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write("Dr. Indah Agustien Siradjuddin,S.Kom.,M.Kom")

elif choose=='Dataset':
    st.markdown('<h1 style = "text-align: center;"> Data Harga Rumah </h1>', unsafe_allow_html = True) #untuk menentukan apakah Streamlit harus mengizinkan HTML dalam teks Markdown
    df = pd.read_csv('https://raw.githubusercontent.com/Shintaalya/repo/main/HARGA%20RUMAH%20JAKSEL.csv')
    df
    st.markdown('<h1 style = "text-align: center;"> Fitur Dataset: </h1><ol type = "1" style = "text-align: justify; background-color: #00FFFF; padding: 30px; border-radius: 20px;"><p>Dataset ini diambil dari kaggle.com</p><li><i><b>HARGA</b></i> = harga dari rumah</li><li><i><b>LT</b></i> = Jumlah Luas Tanah</li><li><i><b>LB</b></i> = Jumlah Luas Bangunan</li><li><i><b>JKT</b></i> = Jumlah Kamar Tidur</li><li><i><b>JKM</b></i> = Jumlah Kamar Mandi</li><li><i><b>GRS</b></i> = Ada / Tidak Ada</li></ol>', unsafe_allow_html = True)

elif choose=='Prepocessing':
    st.markdown('<h1 style = "text-align: center;"> Prediksi Harga Rumah</h1>', unsafe_allow_html = True)
    
elif choose=='Predict':
    st.markdown('<h1 style = "text-align: center;"> Prediksi Harga Rumah</h1>', unsafe_allow_html = True)
    

# Menjalankan aplikasi Streamlit
if __name__ == "__main__":
    main()
