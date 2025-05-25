# Laporan Proyek Machine Learning - Aisyah Amelia Zarah Juaita

## Domain Proyek
Industri wine merupakan industri global bernilai miliaran dolar di mana kualitas produk sangat mempengaruhi harga jual dan persepsi konsumen. Penilaian kualitas wine biasanya dilakukan oleh ahli dengan metode sensorik, namun pendekatan ini subjektif dan memerlukan biaya tinggi. Oleh karena itu, penggunaan machine learning untuk memprediksi kualitas wine berdasarkan parameter kimia menjadi solusi yang efisien dan objektif. 

Masalah ini perlu diselesaikan karena:
- Produsen wine dapat melakukan kontrol kualitas secara otomatis dan cepat.
- Mengurangi ketergantungan terhadap panel uji rasa yang memerlukan biaya besar.
- Meningkatkan efisiensi dan konsistensi dalam proses produksi.
  
Referensi:
- Cortez, Paulo, et al. "Modeling wine preferences by data mining from physicochemical properties." Decision Support Systems 47.4 (2009): 547-553.
- Tawei Lo, Wine Quality Dataset – Balanced Classification Kaggle

## Business Understanding
Kualitas anggur merupakan faktor utama dalam menentukan nilai jual dan kepuasan konsumen dalam industri wine. Namun, proses penilaian kualitas secara manual oleh pakar cenderung subjektif, tidak efisien, dan mahal. Oleh karena itu, diperlukan pendekatan berbasis data untuk mengklasifikasikan kualitas anggur secara otomatis menggunakan fitur-fitur fisikokimia yang tersedia.

### Problem Statements
Pernyataan masalah yang diangkat dalam proyek ini adalah:

1. Bagaimana cara membangun sistem klasifikasi yang akurat untuk memprediksi kualitas anggur (baik atau buruk) berdasarkan data fisikokimia?

2. Algoritma machine learning mana yang paling efektif dalam melakukan klasifikasi kualitas anggur berdasarkan metrik evaluasi yang relevan?

3. Sejauh mana performa klasifikasi dapat ditingkatkan dengan penggunaan algoritma ansambel dan teknik boosting?

### Goals
Tujuan dari proyek ini adalah:

1. Mengembangkan model machine learning klasifikasi biner (good vs bad) terhadap kualitas anggur berdasarkan data numerik yang bersifat fisikokimia.

2. Membandingkan performa lima algoritma klasifikasi, yaitu:
   - Random Forest Classifier
   - Extra Trees Classifier
   - Decision Tree Classifier
   - Bagging Classifier
   - LightGBM Classifier (LGBM)

3. Mengevaluasi setiap model menggunakan metrik akurasi, precision, recall, dan F1-score untuk mengidentifikasi model terbaik dalam konteks bisnis dan data.
   
### Solution statements
Untuk menjawab tujuan di atas, solusi yang ditawarkan dalam proyek ini meliputi:

1. Baseline Model: Menggunakan Decision Tree Classifier, karena model ini sederhana dan mudah diinterpretasikan, cocok sebagai titik awal.

2. Improvement via Ensemble Methods:
   - Menerapkan Random Forest dan Extra Trees Classifier sebagai teknik bagging yang mampu 
     mengurangi overfitting dan meningkatkan akurasi.
   - Menggunakan Bagging Classifier sebagai pendekatan voting ansambel berbasis pohon keputusan.

3. Boosting Approach:
   - Menerapkan LGBMClassifier, salah satu algoritma boosting modern yang efisien dan mendukung 
     kinerja tinggi pada data besar dan kompleks.

4. Evaluation & Selection:
   - Semua model dibandingkan berdasarkan metrik klasifikasi: accuracy, precision, recall, dan 
     F1-score.
   - Model terbaik akan dipilih berdasarkan kombinasi skor evaluasi tertinggi dan generalisasi 
     terhadap data uji.

Dengan pendekatan ini, diharapkan sistem klasifikasi yang dihasilkan dapat menjadi solusi praktis dan efektif untuk prediksi kualitas anggur dalam lingkungan produksi dan kontrol kualitas industri wine.

## Data Understanding
Proyek ini menggunakan dataset Wine Quality - Classification yang tersedia di Kaggle:

🔗 https://www.kaggle.com/datasets/taweilo/wine-quality-dataset-balanced-classification

## Informasi Dataset
- Jumlah Baris: 21000
- Jumlah Kolom: 12
- Jenis Masalah: Klasifikasi Biner
- Proporsi Label: Seimbang (balanced dataset)

## Daftar Variabel

1. fixed_acidity - Keasaman tetap: Asam yang tidak menguap saat proses fermentasi, seperti asam tartarat.

2. volatile_acidity - Keasaman yang mudah menguap, seperti asam asetat (bau cuka). Terlalu tinggi = wine rusak.

3. citric_acid - Asam sitrat: Menambah rasa segar/keasaman. Jumlah kecil bisa meningkatkan kualitas wine.

4. residual_sugar - Gula yang tersisa setelah fermentasi. Wine manis memiliki nilai lebih tinggi.

5. chlorides - Kandungan garam (biasanya natrium klorida). Terlalu tinggi = rasa asin/tidak enak.

6. free_sulfur_dioxide - SO₂ bebas: Digunakan untuk mencegah pertumbuhan mikroorganisme & oksidasi.

7. total_sulfur_dioxide - Total kandungan SO₂ (bebas + terikat). Terlalu banyak = berdampak negatif pada aroma dan rasa.

8. density - Kepadatan cairan wine. Dipengaruhi oleh kadar gula, alkohol, dan komposisi kimia lain.

9. pH - Tingkat keasaman (skala 0-14). pH rendah = asam tinggi.

10. sulphates - Tambahan sulfat untuk mengawetkan dan menstabilkan wine. Bisa juga memengaruhi rasa.

11. alcohol - Persentase kandungan alkohol dalam wine. Biasanya berkisar antara 8-14%.

12. quality - Skor kualitas wine, biasanya diberikan oleh panel uji rasa (skala 0-10). Target untuk model.

## Exploratory Data Analysis (EDA)

- Mendeskripsikan Variabel dari Dataset
  
  df

Menampilkan hasil yang menampilkan variabel, jumlah kolom, dan juga jumlah baris. 

- Describe Dataset

df.describe()

Memberikan statistik deskriptif untuk setiap kolom numerik di DataFrame. Sangat berguna untuk memahami sebaran, nilai rata-rata, variasi, dan mendeteksi potensi outlier dalam dataset.

- Info Dataset

df.info()

Menampilkan struktur DataFrame, Melihat jumlah baris, kolom, dan tipe data masing-masing kolom, Mengetahui apakah ada missing values, Menampilkan penggunaan memori.

- Apakah ada Data Duplikat

df.duplicated().sum()

Ternyata ada, lalu dilakukan penghapusan data duplikat.

df_cleaned = df.drop_duplicates()

df_cleaned.duplicated().sum()

Data duplikat tidak ada lagi karena sudah dihapus sebelumnya. 

- Missing Value

df.isnull().sum()

Ternyata tidak terdapat missing value.

## Penanganan Outlier

Sudah dimasukkan code untuk penanganan outlier, menghasilkan:
- Jumlah data awal: 21000
- Jumlah data setelah menghapus outlier: 20889

## Univariate Analysis

- Kolom Numerik
  
![image](https://github.com/user-attachments/assets/001d487c-b9ab-4f70-bfd9-df1cde54800a)

Berikut adalah kode dan output statistik dari kolom numerikal, yang dimanan menghasilkan jumlah dari count, mean,std,min, 25%, 50%, 75%, dan max.

![image](https://github.com/user-attachments/assets/ddcbee16-a004-4ee2-83c8-9c67afd2af70)

![image](https://github.com/user-attachments/assets/edfed1e8-215c-4f54-a80f-0abdda632e86)

![image](https://github.com/user-attachments/assets/9ac3faeb-2b20-4f0a-8dfc-510a2ed94b44)

![image](https://github.com/user-attachments/assets/a050732e-e678-40b5-bcb7-1722635e98b5)

![image](https://github.com/user-attachments/assets/ad79db34-d090-4899-9001-ab25307e387a)

![image](https://github.com/user-attachments/assets/dafb81f4-0aef-45d6-b46a-b3e3955a2fbe)

![image](https://github.com/user-attachments/assets/93f00a9a-7180-40e4-bb16-baa12ef18576)

![image](https://github.com/user-attachments/assets/a2951d4d-3ce0-4391-9353-ad5ab185deca)

![image](https://github.com/user-attachments/assets/51506de1-d54a-475c-9530-402349d8193c)

![image](https://github.com/user-attachments/assets/f53865a2-227e-4c11-abb4-33420bccc9ce)

![image](https://github.com/user-attachments/assets/d55ac366-2ea2-4d0a-abc9-d949279112ba)

Kode dan output tersebut digunakan untuk visualisasi univariate  dari fitur numerik dalam DataFrame df, dengan menampilkan dua jenis grafik untuk setiap kolom numerik histogram dan boxplot. Histogram dan boxplot ini menghasilkan output yang Menunjukkan distribusi frekuensi data dan Menunjukkan persebaran data melalui nilai kuartil. 

- Kolom Kategori

![image](https://github.com/user-attachments/assets/b9d9e335-53f6-48d2-9440-85533b204435)

Kode dan output diatas digunakan untuk menampilkan distribusi dan proporsi nilai pada kolom kategorikal `quality`. Hasilnya menunjukkan bahwa setiap nilai kualitas dari 3 hingga 9 muncul sebanyak 3.000 kali, sehingga distribusinya merata. Proporsi masing-masing kelas adalah 14,29%, yang menunjukkan bahwa data kategori ini seimbang dan tidak perlu penyesuaian khusus sebelum analisis atau pemodelan.

![image](https://github.com/user-attachments/assets/7b97c336-f0bc-4886-ab30-305cbe499b23)

Pada kode digunakan untuk membuat visualisasi distribusi data pada kolom kategorikal menggunakan countplot dari library Seaborn. Dalam hal ini, kolom `quality` divisualisasikan untuk menunjukkan jumlah data pada setiap kategori nilai kualitas anggur. Karena data seimbang, grafik yang dihasilkan menunjukkan tinggi batang (bar) yang sama untuk setiap nilai `quality` dari 3 hingga 9. Visualisasi ini membantu dalam memahami seberapa banyak data yang dimiliki setiap kategori dan mengecek apakah terjadi ketidakseimbangan kelas.

## Multivariate Analysis

![image](https://github.com/user-attachments/assets/577e3931-8e7d-4ff1-bfc0-93d5ca5d44f4)

Korelasi antar fitur numerik dalam bentuk heatmap (peta korelasi). Korelasi ini dihitung menggunakan metode Pearson dan menunjukkan hubungan linier antara setiap pasangan fitur numerik dalam dataset. 

![image](https://github.com/user-attachments/assets/3f479555-e092-4b0f-a78c-cb5849557d63)

![image](https://github.com/user-attachments/assets/cd46f1fe-78d5-41f6-bd64-c73523687ec9)

![image](https://github.com/user-attachments/assets/2d279eb7-afce-4b75-84b5-ce66bafc4c9f)

![image](https://github.com/user-attachments/assets/21054976-e2aa-4e6a-8f21-c1bfc43250de)

![image](https://github.com/user-attachments/assets/df77a483-ff57-4490-a5f3-335e4ffc9f1e)

![image](https://github.com/user-attachments/assets/edca4649-3a17-4d32-9918-2775954b5afc)

![image](https://github.com/user-attachments/assets/b2042350-5683-4d25-9048-ca268fbc8a5d)

![image](https://github.com/user-attachments/assets/5556f657-cd02-4c10-9ad1-865959eb64aa)

![image](https://github.com/user-attachments/assets/0a5a3352-abb6-48f4-be53-80a4ad6f3aac)

![image](https://github.com/user-attachments/assets/007d9645-4c59-4b57-a663-c24e184c088b)

![image](https://github.com/user-attachments/assets/8fbe7829-2279-4df5-93a4-e06f983a3877)

Memvisualisasikan hubungan antara setiap fitur numerik dengan target variabel quality menggunakan boxplot. Boxplot ini menampilkan bagaimana distribusi nilai dari masing-masing fitur numerik berbeda di setiap level wine quality (dari 3 hingga 9). 

![image](https://github.com/user-attachments/assets/bfd3bf0f-d428-4200-ac0f-ecdc1b454826)

Berikut adalah pairplot menampilkan grafik scatterplot antara setiap pasangan fitur, sehingga kita bisa melihat hubungan dan pola antar fitur secara dua per dua. Di bagian diagonal, ditampilkan kurva kepadatan (KDE) yang memperlihatkan distribusi masing-masing fitur. Visualisasi ini membantu memahami korelasi antar fitur sekaligus bagaimana fitur-fitur tersebut berhubungan dengan kualitas anggur secara simultan.

## Data Preparation

Pada tahap ini, dilakukan serangkaian proses untuk menyiapkan data sebelum dimasukkan ke dalam algoritma machine learning. Proses ini mencakup: data cleaning, pemisahan data latih dan data uji, serta normalisasi (standardisasi).

## Data Cleaning

Pada tahap ini memisahkan data menjadi fitur dan target. Variabel `X` berisi semua kolom kecuali `quality`, yang digunakan sebagai fitur input, sedangkan `y` hanya berisi kolom `quality` sebagai target atau label yang ingin diprediksi. Dengan demikian, `X` adalah data numerik untuk analisis, dan `y` adalah nilai kualitas anggur yang menjadi fokus prediksi.

## Train-Test-Split

Pada tahap ini membagi dataset menjadi dua bagian: data latih (training) dan data uji (testing). Sebanyak 80% data (16.711 baris) digunakan untuk melatih model, dan 20% sisanya (4.178 baris) digunakan untuk menguji performa model. Pembagian ini dilakukan secara acak namun tetap menjaga proporsi kelas target (`quality`) sama pada kedua subset dengan menggunakan parameter `stratify=y`. Total data yang digunakan adalah 20.889 baris setelah pembersihan data.

- Jumlah total dataset: 20889
- Jumlah data latih: 16711
- Jumlah data uji: 4178

## Normalisasi (Standardisasi)

Di tahap ini melakukan standarisasi data fitur pada dataset latih dan uji menggunakan `StandardScaler`. Dengan standarisasi, setiap fitur diubah sehingga memiliki rata-rata nol dan standar deviasi satu. Ini penting supaya model machine learning tidak bias terhadap fitur dengan skala besar dan bisa belajar dengan lebih baik serta stabil. Proses `fit_transform` diterapkan pada data latih untuk menghitung parameter standarisasi, kemudian `transform` diterapkan ke data uji agar menggunakan skala yang sama.

Yang menghasilkan data shapes:
- x_train: (16711, 11)
- X_test: (4178, 11)
- x_train: (16711,)
- X_test: (4178,)

## Modeling
























