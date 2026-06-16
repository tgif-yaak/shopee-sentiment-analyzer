# 🛍️ Shopee App Review Sentiment Analyzer

Sistem cerdas berbasis Machine Learning untuk mendeteksi dan mengklasifikasikan sentimen kepuasan pengguna Aplikasi Shopee di Google Play Store ke dalam tiga kategori: **Positif**, **Netral**, dan **Negatif**. Proyek ini dilengkapi dengan antarmuka web interaktif menggunakan Streamlit.

---

## 📊 Hasil Eksperimen & Performa Model

Dalam proyek ini, dilakukan eksperimen menggunakan dua algoritma klasifikasi populer dengan pembobotan fitur **TF-IDF (N-gram 1-3)** dan teknik penyeimbangan data (*Oversampling*). Berikut adalah hasil perbandingan akurasinya:

| Algoritma | Akurasi | Keterangan |
| :--- | :---: | :--- |
| **Multinomial Naive Bayes** | `90.51%` | Performa baik, namun cenderung lemah pada kelas positif. |
| **Support Vector Machine (SVM RBF)** | **`95.74%`** | **Model Terbaik!** Performa sangat seimbang di semua kelas sentimen. |
| **Support Vector Machine (Linear)** | `94.55%` | Performa tinggi, sedikit di bawah kernel RBF. |

---

## 🛠️ Langkah-Langkah Pengembangan (Workflow)

1. **Data Crawling:** Mengambil 10.000 data ulasan real-time dari Google Play Store menggunakan `google-play-scraper`.
2. **Text Preprocessing:** * *Case Folding* (mengubah teks menjadi huruf kecil).
   * Pembersihan teks dari URL, angka, karakter khusus, dan tanda baca menggunakan Regular Expression (`re`).
   * *Stopwords Removal* (menghapus kata umum yang tidak bermakna) memanfaatkan pustaka `Sastrawi`.
3. **Data Balancing:** Mengatasi ketidakseimbangan kelas (*imbalanced data*) melalui metode *Random Oversampling* agar model dapat mengenali setiap sentimen secara adil.
4. **Feature Extraction:** Transformasi teks mentah menjadi bentuk vektor numerik menggunakan `TfidfVectorizer` dengan rentang fitur maksimal 10.000 kata.
5. **Model Deployment:** Menyimpan model terlatih menggunakan `joblib` dan membuat aplikasi web interaktif berbasis `Streamlit`.

---

## 📁 Struktur Repositori

```text
├── app.py                      # Kode utama antarmuka Streamlit (UI)
├── model_svm_shopee.pkl        # Model SVM terbaik yang sudah terlatih
├── vectorizer_tfidf.pkl        # File konfigurasi TF-IDF Vectorizer
├── requirements.txt            # Daftar pustaka (dependencies) untuk server
└── README.md                   # Dokumentasi proyek