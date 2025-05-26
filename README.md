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
   
### Solution Statements
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

Menampilkan hasil yang semua variabel, jumlah kolom, dan juga jumlah baris. 

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
  
![Gambar](images/27.png)

Berikut adalah output statistik dari kolom numerikal, yang dimanan menghasilkan jumlah dari count, mean,std,min, 25%, 50%, 75%, dan max.

![Gambar](images/1.png)

![Gambar](images/2.png)

![Gambar](images/3.png)

![Gambar](images/4.png)

![Gambar](images/5.png)

![Gambar](images/6.png)

![Gambar](images/7.png)

![Gambar](images/8.png)

![Gambar](images/9.png)

![Gambar](images/10.png)

![Gambar](images/11.png)

Pada output tersebut digunakan untuk visualisasi univariate  dari fitur numerik dalam DataFrame df, dengan menampilkan dua jenis grafik untuk setiap kolom numerik histogram dan boxplot. Histogram dan boxplot ini menghasilkan output yang Menunjukkan distribusi frekuensi data dan Menunjukkan persebaran data melalui nilai kuartil. 

- Kolom Kategori

![Gambar](images/28.png)

Pada output diatas digunakan untuk menampilkan distribusi dan proporsi nilai pada kolom kategorikal `quality`. Hasilnya menunjukkan bahwa setiap nilai kualitas dari 3 hingga 9 muncul sebanyak 3.000 kali, sehingga distribusinya merata. Proporsi masing-masing kelas adalah 14,29%, yang menunjukkan bahwa data kategori ini seimbang dan tidak perlu penyesuaian khusus sebelum analisis atau pemodelan.

![Gambar](images/12.png)

Pada gambar digunakan untuk membuat visualisasi distribusi data pada kolom kategorikal menggunakan countplot dari library Seaborn. Dalam hal ini, kolom `quality` divisualisasikan untuk menunjukkan jumlah data pada setiap kategori nilai kualitas anggur. Karena data seimbang, grafik yang dihasilkan menunjukkan tinggi batang (bar) yang sama untuk setiap nilai `quality` dari 3 hingga 9. Visualisasi ini membantu dalam memahami seberapa banyak data yang dimiliki setiap kategori dan mengecek apakah terjadi ketidakseimbangan kelas.

## Multivariate Analysis

![Gambar](images/13.png)

Korelasi antar fitur numerik dalam bentuk heatmap (peta korelasi). Korelasi ini dihitung menggunakan metode Pearson dan menunjukkan hubungan linier antara setiap pasangan fitur numerik dalam dataset. 

![Gambar](images/14.png)

![Gambar](images/15.png)

![Gambar](images/16.png)

![Gambar](images/17.png)

![Gambar](images/18.png)

![Gambar](images/19.png)

![Gambar](images/20.png)

![Gambar](images/21.png)

![Gambar](images/22.png)

![Gambar](images/23.png)

![Gambar](images/24.png)

Memvisualisasikan hubungan antara setiap fitur numerik dengan target variabel quality menggunakan boxplot. Boxplot ini menampilkan bagaimana distribusi nilai dari masing-masing fitur numerik berbeda di setiap level wine quality (dari 3 hingga 9). 

![Gambar](images/25.png)

Berikut adalah pairplot menampilkan grafik scatterplot antara setiap pasangan fitur, sehingga kita bisa melihat hubungan dan pola antar fitur secara dua per dua. Di bagian diagonal, ditampilkan kurva kepadatan (KDE) yang memperlihatkan distribusi masing-masing fitur. Visualisasi ini membantu memahami korelasi antar fitur sekaligus bagaimana fitur-fitur tersebut berhubungan dengan kualitas anggur secara simultan.

## Data Preparation

Pada tahap ini, dilakukan serangkaian proses untuk menyiapkan data sebelum dimasukkan ke dalam algoritma machine learning. Proses ini mencakup: data cleaning, pemisahan data latih dan data uji, serta normalisasi (standardisasi).

1. Data Cleaning

Pada tahap ini memisahkan data menjadi fitur dan target. Variabel `X` berisi semua kolom kecuali `quality`, yang digunakan sebagai fitur input, sedangkan `y` hanya berisi kolom `quality` sebagai target atau label yang ingin diprediksi. Dengan demikian, `X` adalah data numerik untuk analisis, dan `y` adalah nilai kualitas anggur yang menjadi fokus prediksi.

2. Train-Test-Split

Pada tahap ini membagi dataset menjadi dua bagian: data latih (training) dan data uji (testing). Sebanyak 80% data (16.711 baris) digunakan untuk melatih model, dan 20% sisanya (4.178 baris) digunakan untuk menguji performa model. Pembagian ini dilakukan secara acak namun tetap menjaga proporsi kelas target (`quality`) sama pada kedua subset dengan menggunakan parameter `stratify=y`. Total data yang digunakan adalah 20.889 baris setelah pembersihan data.

- Jumlah total dataset: 20889
- Jumlah data latih: 16711
- Jumlah data uji: 4178

3. Normalisasi (Standardisasi)

Di tahap ini melakukan standarisasi data fitur pada dataset latih dan uji menggunakan `StandardScaler`. Dengan standarisasi, setiap fitur diubah sehingga memiliki rata-rata nol dan standar deviasi satu. Ini penting supaya model machine learning tidak bias terhadap fitur dengan skala besar dan bisa belajar dengan lebih baik serta stabil. Proses `fit_transform` diterapkan pada data latih untuk menghitung parameter standarisasi, kemudian `transform` diterapkan ke data uji agar menggunakan skala yang sama.

Yang menghasilkan data shapes:
- x_train: (16711, 11)
- X_test: (4178, 11)
- x_train: (16711,)
- X_test: (4178,)

## Modeling

1. RandomForestClassifier
   
- Deskripsi: Model ensambel berbasis bagging yang terdiri dari banyak decision tree dan menghasilkan prediksi berdasarkan voting.

- Parameter:
  - n_estimators=100
  - max_depth=None
  - random_state=42

- Kelebihan:
  - Mengurangi overfitting dari decision tree tunggal
  - Robust terhadap noise dan outlier

- Kekurangan:
  - Kurang interpretatif
  - Waktu pelatihan lebih lama dibanding decision tree biasa
 
2. ExtraTreesClassifier
   
- Deskripsi: Model ensambel serupa dengan Random Forest, tetapi menggunakan pemilihan split secara acak dan agresif.

- Parameter:
  - n_estimators=100
  - random_state=42

- Kelebihan:
  - Eksekusi lebih cepat daripada Random Forest
  - Biasanya menghasilkan performa lebih baik dengan tuning minimal

- Kekurangan:
  - Masih termasuk model black-box
  - Kadang lebih sensitif terhadap data imbalance (meski tidak terjadi di dataset ini)

3. DecisionTreeClassifier

- Deskripsi: Model dasar yang digunakan sebagai baseline. Merupakan algoritma yang sederhana dan mudah diinterpretasikan.

- Parameter:
  - criterion='gini'
  - max_depth=None (default, tidak dibatasi)

- Kelebihan:
  - Mudah dipahami dan divisualisasikan
  - Tidak memerlukan scaling data

- Kekurangan:
  - Rentan terhadap overfitting
  - Performa kurang stabil pada data yang kompleks

4. BaggingClassifier

- Deskripsi: Model ansambel umum berbasis bagging yang dapat digunakan dengan base estimator apapun. Dalam proyek ini, digunakan dengan DecisionTree sebagai base model.

- Parameter:
  - n_estimators=50
  - random_state=42

- Kelebihan:
  - Meningkatkan stabilitas dan akurasi model dasar
  - Mengurangi variance dari base model

- Kekurangan:
  - Kurang efisien dibanding model ansambel khusus seperti Random Forest
  - Tidak secara otomatis melakukan feature selection
 
5. LGBMCClassifier

- Deskripsi: Merupakan algoritma boosting berbasis pohon yang sangat efisien untuk dataset besar dan kompleks.

- Parameter:
  - boosting_type='gbdt'
  - num_leaves=31
  - learning_rate=0.1
  - n_estimators=100

- Kelebihan:
  - Eksekusi cepat dan ringan
  - Mendukung fitur kategorikal secara native
  - Performa sangat kompetitif

- Kekurangan:
  - Rentan terhadap overfitting jika tidak dituning
  - Memerlukan pemahaman lebih dalam untuk tuning optimal

## Evaluation 

Berikut adalah evaluasi model yang digunakan:

1. Model RandomForest
- Memiliki Akurasi: 62.45%.
- Memiliki F1-score rata-rata tertimbang (weighted avg): 62%.
- Performa paling tinggi dari semua model (dalam hal akurasi)
- F1-score relatif merata di semua kelas.

2. Model ExtraTrees
- Memiliki Akurasi: 61.97%.
- Memiliki F1-score rata-rata tertimbang: 62%.
- Hampir setara dengan RandomForest  
- Kelas 3 (recall 0.67), 6(recall 0.73), dan 9(recall 0.66) memiliki recall yang cukup tinggi.

3. Model DecisionTree
- Memiliki Akurasi: 59.88%.
- Memiliki F1-score rata-rata tertimbang: 60%.
- Cukup seimbang dalam precision dan recall antar kelas.
- Performa cenderung lebih rendah dibanding model ensemble (RF, ET, Bagging).

4. Model Bagging
- Memiliki Akurasi: 61.56%.
- Memiliki F1-score rata-rata tertimbang: 61%.
- Hampir setara dengan RandomForest dan ExtraTrees.
- Bagging bisa mengurangi overfitting dari pohon tunggal.

5. Model LGBM (LightGBM)
- Memiliki Akurasi: 54.50%.
- Memiliki F1-score rata-rata tertimbang: 54%.
- Performa cukup baik di kelas 6 dan 5.

# Perbandingan Akurasi Model

![Gambar](images/26.png)

# Analisis Hasil & Relevansi terhadap Business Understanding

- Evaluasi untuk modelnya:

Random Forest menunjukkan performa terbaik secara keseluruhan di antara semua model yang diuji, dengan akurasi tertinggi sebesar 62.45% dan F1-score rata-rata tertimbang sebesar 62%. Ini menunjukkan bahwa model mampu menangkap pola dengan baik di seluruh kelas, tanpa bias yang besar terhadap kelas tertentu. Keunggulan Random Forest juga terlihat dari stabilitas performanya di semua kelas, ditandai dengan distribusi F1-score yang relatif merata. Hal ini mencerminkan bahwa model tidak hanya kuat dalam prediksi mayoritas kelas, tetapi juga cukup adil dalam menangani kelas minoritas. Sebagai model ensemble berbasis banyak pohon keputusan, Random Forest mengurangi risiko overfitting dibandingkan Decision Tree tunggal, dan meningkatkan generalisasi terhadap data yang belum pernah dilihat. Dengan keseimbangan antara akurasi, F1-score, dan distribusi antar kelas, Random Forest layak dipilih sebagai model terbaik dalam eksperimen ini.

# Relevansi Terhadap Business Understanding

1. Tujuan Proyek
   - Mengotomatisasi penilaian kualitas anggur berdasarkan data fisikokimia agar diperoleh 
     sistem klasifikasi yang akurat, efisien, dan andal.
   - Mengevaluasi dan membandingkan performa beberapa model machine learning menggunakan metrik 
     akurasi, precision, recall, dan F1-score, untuk mengidentifikasi model terbaik dalam 
     konteks bisnis dan data.

2. Model yang diuji
   - RandomForestClassifier
   - ExtraTreesClassifier
   - DecisionTreeClassifier
   - BaggingClassifier
   - LGBMClassifier

3. Hasil Evaluasi
   - RandomForestClassifier menunjukkan akurasi tertinggi sebesar 62.45% dan F1-score rata-rata 
     tertimbang sebesar 62%.
   - Model ini juga menunjukkan distribusi performa yang merata di seluruh kelas, mengurangi 
     potensi bias terhadap kelas tertentu.

4. Kesesuaian dengan tujuan
   - tujuan 1: Random Forest terbukti sebagai model yang andalan dan stabil, sehingga relevan untuk digunakan dalam proses penilaian kualitas anggur secara otomatis. Hal ini sejalan dengan tujuan pertama, yaitu membangun sistem yang akurat, efisien, dan dapat diandalkan untuk mendukung kontrol kualitas di industri wine.

- tujuan 2: Berdasarkan perbandingan metrik evaluasi, Random Forest berhasil mengungguli model baseline (Decision Tree) dan model ansambel lainnya (Extra Trees, Bagging, LGBM), menjadikannya pilihan paling optimal dalam konteks data dan kebutuhan bisnis.

5. Keunggulan Random Forest
   - Meningkatkan akurasi secara signifikan dibandingkan model dasar.
   - Menurunkan risiko overfitting melalui pendekatan ansambel.
   - Konsisten unggul dalam semua metrik utama (accuracy, precision, recall, dan F1-score).

6. Kesimpulan Bisnis
   
Implementasi RandomForestClassifier dapat membantu pelaku industri wine dalam:
- Mengambil keputusan produksi dan distribusi secara lebih cepat dan akurat.
- Menjaga standar kualitas produk dengan pendekatan objektif berbasis data.
- Mengurangi biaya dan ketergantungan pada penilaian manual yang subjektif.

# Kesimpulan

Berdasarkan hasil evaluasi, model RandomForest merupakan model terbaik di antara yang diuji, dengan akurasi tertinggi sebesar 62,45% dan F1-score rata-rata tertimbang sebesar 62%. Model ini menunjukkan performa yang relatif merata di semua kelas, menjadikannya pilihan utama untuk prediksi yang stabil dan akurat. Model ExtraTrees juga memiliki performa yang hampir setara dengan RandomForest, terutama dengan recall yang cukup tinggi pada beberapa kelas seperti kelas 3, 6, dan 9, sehingga dapat menjadi alternatif yang baik. Model Bagging juga menunjukkan hasil yang cukup kompetitif dan mampu mengurangi overfitting dibandingkan model pohon tunggal. Sementara itu, model DecisionTree memiliki performa yang lebih rendah dibandingkan model ensemble, meskipun cukup seimbang dalam precision dan recall antar kelas. Terakhir, LightGBM menunjukkan performa paling rendah di antara model-model tersebut dengan akurasi dan F1-score di bawah 55%, meskipun masih menunjukkan hasil yang cukup baik di beberapa kelas tertentu. Dengan demikian, untuk hasil terbaik secara keseluruhan, RandomForest adalah pilihan yang paling tepat, sementara ExtraTrees dan Bagging dapat dipertimbangkan sebagai alternatif yang juga efektif.


















