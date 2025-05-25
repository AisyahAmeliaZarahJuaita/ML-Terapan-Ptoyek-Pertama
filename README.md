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

![image](https://github.com/user-attachments/assets/2c8f8f88-23a4-4cc6-a050-105866b8434c)

![image](https://github.com/user-attachments/assets/e31d05ad-1ee1-42dc-9347-00b68996479c)

![image](https://github.com/user-attachments/assets/c1cc980b-3231-4a70-b906-fe4260b4d75d)

![image](https://github.com/user-attachments/assets/541fca99-3efd-4dd1-9107-f9544ce44d03)

![image](https://github.com/user-attachments/assets/613254be-a29a-42dd-98fb-9700b8582ba2)




















