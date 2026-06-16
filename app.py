import streamlit as st
import joblib
import re

# 1. SETTING HALAMAN
st.set_page_config(page_title="Shopee App Sentiment Analyzer", page_icon="🛍️", layout="centered")

# Custom CSS Total Fix: Menyejajarkan Kolom & Mengunci Bentuk Kotak Kontainer
st.markdown("""
    <style>
    /* Mengubah background luar halaman menjadi OREN Shopee */
    .stApp {
        background-color: #EE4D2D !important;
    }

    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* REVISI: Mengunci bentuk KOTAK kontainer putih agar punya batas atas yang jelas */
    .block-container {
        background-color: #FFFFFF !important;
        padding: 40px !important;
        border-radius: 20px !important;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2) !important;
        max-width: 850px !important;
        margin-bottom: 100px !important;
    }
            
    div[data-testid="stAppViewContainer"] > section {
        padding-top: 60px !important; /* Ini yang bakal maksa warna oren muncul tebal di atas kotak putih! */
    }
    
    /* REVISI MUTLAK: Memaksa kolom kiri dan kanan sejajar rata lurus dari garis paling atas */
    div[data-testid="stHorizontalBlock"] {
        align-items: flex-start !important;
    }
    
    /* Menghilangkan logo link rantai bawaan Streamlit */
    .element-container:has(h1) a {
        display: none !important;
    }
    
    /* Styling Judul Utama */
    h1 {
        color: #EE4D2D !important;
        font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
        text-align: center;
        margin-top: 0px !important;
        margin-bottom: 2px !important;
    }
    
    .sub-title {
        text-align: center;
        color: #555555 !important;
        font-size: 14px;
        font-family: 'Inter', sans-serif;
        margin-bottom: 20px;
        line-height: 1.5;
    }
    
    /* Label teks input area */
    .stTextArea label p {
        font-weight: 600 !important;
        color: #222222 !important;
        font-size: 14px !important;
        margin-bottom: 8px;
    }

    /* Mempercantik Text Area Input Box */
    textarea {
        border: 1px solid #DCDCDC !important;
        border-radius: 10px !important;
        background-color: #FAFAFA !important;
        color: #333333 !important;
        padding: 12px !important;
        font-size: 14px !important;
        height: 100px !important;
    }
    textarea:focus {
        border-color: #EE4D2D !important;
        box-shadow: 0 0 0 3px rgba(238, 77, 45, 0.12) !important;
        background-color: #FFFFFF !important;
    }
    
    /* Tombol Kiri Full Width */
    .stButton>button {
        background-color: #EE4D2D !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 0px !important;
        font-size: 15px !important;
        font-weight: bold !important;
        width: 100% !important;
        box-shadow: 0 4px 12px rgba(238, 77, 45, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background-color: #E03F1E !important;
        box-shadow: 0 6px 16px rgba(238, 77, 45, 0.3) !important;
        transform: translateY(-1px);
    }
    
    /* 1. Saat Masih Kosong (Boks Abu-abu): Beri jarak agar sejajar dengan textarea kiri */
    .placeholder-box {
        margin-top: 32px !important;
    }
    
    /* 2. Saat Hasil Keluar (Boks Alert): Setel margin-top mendekati 0 agar posisinya naik dan sejajar sempurna dengan textarea kiri */
    div[data-testid="stAlert"] {
        margin-top: 5px !important;
    }
    
    /* Kustomisasi Placeholder Box di Sebelah Kanan */
    .placeholder-box {
        border: 2px dashed #E0E0E0 !important;
        border-radius: 12px !important;
        padding: 42px 20px !important;
        text-align: center !important;
        color: #777777 !important;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        background-color: #FAFAFA !important;
    }
    
    /* Kustomisasi Alert Box Hasil Pastel Style */
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        padding: 20px !important;
        border: 1px solid transparent !important;
    }
    div[data-testid="stAlert"]:has(div[data-testid="stNotificationContentSuccess"]) {
        background-color: #EFFFFA !important;
        border-color: #BFF3E3 !important;
    }
    div[data-testid="stAlert"]:has(div[data-testid="stNotificationContentWarning"]) {
        background-color: #FFFDF0 !important;
        border-color: #FFEFA6 !important;
    }
    div[data-testid="stAlert"]:has(div[data-testid="stNotificationContentError"]) {
        background-color: #FFF5F4 !important;
        border-color: #FCD2CE !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. LOAD MODEL DAN VECTORIZER
@st.cache_resource
def load_model():
    model = joblib.load('model_svm_shopee.pkl')
    vectorizer = joblib.load('vectorizer_tfidf.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_model()
except Exception as e:
    st.error("Gagal memuat model. Pastikan file model ada di folder yang sama!")

# 3. FUNGSI PEMBERSIH TEKS
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# 4. TAMPILAN UTAMA
st.markdown("<h1>🛍️ Shopee App Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Sistem cerdas berbasis Machine Learning untuk mendeteksi sentimen kepuasan pengguna Aplikasi Shopee di Google Play Store.</p>", unsafe_allow_html=True)

# MEMBAGI HALAMAN MENJADI 2 KOLOM
kolom_kiri, kolom_kanan = st.columns(2, gap="large")

# --- KOLOM KIRI: INPUT & TOMBOL ---
with kolom_kiri:
    user_input = st.text_area("Masukkan Ulasan Pengguna:", placeholder="Ketik ulasan di sini... (Contoh: Aplikasinya bagus banget, lengkap!)")
    tombol_analisis = st.button("Analisis Sentimen")

# --- KOLOM KANAN: HASIL PREDIKSI ---
with kolom_kanan:
    
    if tombol_analisis:
        if user_input.strip() != "":
            cleaned_text = clean_text(user_input)
            vectorized_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized_text)[0]
            
            if prediction == 'positif':
                st.success(f"### 🟢 Sentimen: POSITIF  \n\n*User puas dengan fungsionalitas sistem dan performa aplikasi.* \n\n**Rating Estimasi:** \n⭐ ⭐ ⭐ ⭐ ⭐")
            elif prediction == 'netral':
                st.warning(f"### 🟡 Sentimen: NETRAL  \n\n*Ulasan bersifat netral, memberikan saran, atau kritik berimbang.* \n\n**Rating Estimasi:** \n⭐ ⭐ ⭐")
            else:
                st.error(f"### 🔴 Sentimen: NEGATIF  \n\n*User mengeluhkan masalah bug, sistem lag, atau error transaksi.* \n\n**Rating Estimasi:** \n⭐")
        else:
            st.markdown('<div class="placeholder-box">⚠️<br><br>Mohon masukkan ulasan terlebih dahulu di kolom kiri.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="placeholder-box">📊<br><br>Hasil analisis sentimen model akan muncul di sini setelah Anda mengklik tombol.</div>', unsafe_allow_html=True)