# ML-A11.2022.14595-UAS
Judul : Sentimne Analisis Review Film menggunakan Algoritma Naive Bayes
Nama : Wahyu Aji Pamungkas
Nim : A11.2022.14595

# Proyek ini mengimplementasikan model pembelajaran mesin untuk sentimen Analis Teks Proccesing untuk mengungkap nilai positif dan negatif dalam sebuat reviw film.

# Instalasi
1. * clone Resposposi 
   ' sh
   https://github.com/Wahyuajipamungkas/ML-A11.2022.14595-UAS
   '
2. * Instal pustaka yang diperlukan 
   'sh
   pip install pandas nimpy matplotlib seaborn
   '

# Deskripsi Proyek
Proyek ini terdiri dari berbagai langkah untuk menentukan hasil analisisnya yaitu pemisahan score negatif dan positif, pemisahan kalimat dan kata dan menentukan kata yang positif dan negatif

# Tujuan
proses yang bertujuan untuk memisahkan isi dalam database yang berbentuk teks bersifat positif dan negatif. Ulasan atau pendapat khalayak umum sangat penting bagi pengembangan film selanjutnya dan bagi film itu sendiri. Algoritma yang disarankan untuk melakukan percobaan ini adalah algoritma Naïve Bayen(BA) adalah Algoritma Machine Learning untuk masalah Sentimen. 

# Tahapan
<img src="images/Screenshot 2024-07-25 202852.png" align="center" width="800" height="250">

# Latar Belakang Masalah
Dalam era digital saat ini, internet telah menjadi tempat utama bagi 
individu untuk berbagi pendapat dan pengalaman mereka terkait berbagai hal, 
termasuk film. Dengan adanya platform seperti situs web ulasan film dan media 
sosial, masyarakat memiliki akses tak terbatas untuk mengekspresikan opini 
mereka tentang film yang mereka tonton.
 Namun, memahami sentimen kolektif masyarakat terhadap suatu film bisa 
menjadi tugas yang menantang dan memakan waktu jika dilakukan secara 
manual. Oleh karena itu, penggunaan algoritma machine learning, khususnya 
algoritma Naive Bayes, dapat menjadi solusi efisien untuk menganalisis dan 
mengekstraksi sentimen dari ulasan film yang tersebar luas di internet.
 Algoritma Naive Bayes adalah salah satu metode klasifikasi yang umum 
digunakan dalam analisis sentimen. Metode ini bekerja dengan mengasumsikan 
bahwa setiap fitur (kata atau frasa) dalam sebuah dokumen independen satu 
sama lain, meskipun demikian, metode ini telah terbukti cukup efektif dalam 
banyak aplikasi analisis sentimen, termasuk analisis ulasan film.

# Permasalahan
Meskipun ulasan film tersedia dalam jumlah besar di berbagai platform 
online, memahami sentimen umum yang terkandung di dalamnya merupakan 
tantangan yang kompleks dan memakan waktu. Tanpa alat yang sesuai, 
mengidentifikasi apakah suatu ulasan bersifat positif, negatif, atau netral 
memerlukan waktu dan tenaga manusia yang signifikan.
 Oleh karena itu, diperlukan sebuah solusi otomatis yang dapat menganalisis 
ulasan film secara cepat dan efisien. Dalam hal ini, masalah yang muncul adalah 
bagaimana mengimplementasikan algoritma analisis sentimen, khususnya 
algoritma Naive Bayes, untuk mengekstraksi dan mengklasifikasikan sentimen 
dari ulasan film yang tersebar luas di internet.

# Dataset
Link dataset : https://github.com/Wahyuajipamungkas/ML-A11.2022.14595-UAS/TugasAkhir
Didalam dataset disney.csv itu terdapat 8000+ data yang dikumpulkan dalam beberapa tahun. didalam dataset tersebut terdapat atribut antara lain nama_film,film_year, author_name,review_date, Score, titlee_name, review_text, POU. disini saya menggunakan kolom score untuk dijadikan label. Saya mengambil dataset ini dari kaggle langsung dan data ini bersifat public.<br>

Untuk tahapannya/ ERDnya<br>
<img src="images/Screenshot 2024-07-25 202852.png" align="center" width="800" height="250">

## Proses Learning
