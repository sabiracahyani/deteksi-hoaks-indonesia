import streamlit as st
import joblib
import gdown
import os
import re
import nltk
from nltk.corpus import stopwords

@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)

download_nltk_data()

@st.cache_resource
def load_model():
    file_path = 'model_hoaks_with_thresh.pkl'
    if not os.path.exists(file_path):
        gdown.download(
            'https://drive.google.com/uc?id=19hmOr6EigRyYck7kseNNO1Gi5bDxktic',
            file_path,
            quiet=False
        )
    bundle = joblib.load(file_path)
    return bundle['model'], bundle['tfidf'], bundle['threshold']

@st.cache_resource
def load_stopwords():
    stop_words = set(stopwords.words('indonesian'))
    stop_words.update(stopwords.words('english'))
    stop_words.update({
        'yg', 'tdk', 'dgn', 'utk', 'krn', 'sdh', 'blm', 'jg', 'ny',
        'nya', 'saja', 'aja', 'deh', 'dong', 'nih', 'loh', 'lah',
        'mah', 'ya', 'yuk', 'wah', 'oh', 'ah', 'eh', 'iya', 'gak',
        'ga', 'nggak', 'enggak', 'gitu', 'gini', 'gimana', 'gt',
        'tp', 'jd', 'kl', 'klo', 'kalo', 'bgt', 'banget', 'emg',
        'emang', 'memang', 'juga', 'sudah', 'belum', 'bisa', 'akan',
        'ada', 'bagi', 'para', 'bila', 'maka', 'saat', 'pun'
    })
    return stop_words

def bersihkan_teks(teks, stop_words):
    teks   = str(teks)
    teks   = teks.lower()
    teks   = re.sub(r'http\S+|www\S+', '', teks)
    teks   = re.sub(r'@\w+|#\w+',      '', teks)
    teks   = re.sub(r'\d+',            '', teks)
    teks   = re.sub(r'[^\w\s]',       ' ', teks)
    teks   = re.sub(r'\s+',           ' ', teks).strip()
    tokens = teks.split()
    tokens = [k for k in tokens if k not in stop_words and len(k) > 2]
    return ' '.join(tokens)

st.set_page_config(
    page_title = "Deteksi Berita Hoaks",
    page_icon  = "🔍",
    layout     = "centered"
)

st.title("🔍 Deteksi Berita Hoaks Indonesia")
st.markdown("Masukkan **judul** dan **narasi** berita yang ingin kamu cek keasliannya.")
st.divider()

model, tfidf, THRESHOLD = load_model()
stop_words = load_stopwords()

judul  = st.text_input("📰 Judul Berita",       placeholder="Masukkan judul berita...")
narasi = st.text_area("📝 Narasi / Isi Berita", placeholder="Masukkan isi berita...", height=200)

if st.button("🔎 Cek Berita", use_container_width=True, type="primary"):
    if not judul.strip() and not narasi.strip():
        st.warning("Masukkan judul atau narasi berita terlebih dahulu.")
    else:
        teks_gabung  = judul + ' ' + narasi
        teks_bersih  = bersihkan_teks(teks_gabung, stop_words)
        teks_vektor  = tfidf.transform([teks_bersih])

        prob         = model.predict_proba(teks_vektor)[0]
        prob_hoaks   = prob[1]
        prob_valid   = prob[0]
        prediksi     = 1 if prob_hoaks > prob_valid else 0

        st.divider()

        if prediksi == 1:
            st.error("## ⚠️ HOAKS")
            st.markdown("Model memprediksi berita ini kemungkinan besar adalah **hoaks**.")
        else:
            st.success("## ✅ VALID")
            st.markdown("Model memprediksi berita ini kemungkinan besar adalah **valid**.")

        col1, col2 = st.columns(2)
        col1.metric("🔴 Probabilitas Hoaks", f"{prob_hoaks * 100:.1f}%")
        col2.metric("🟢 Probabilitas Valid",  f"{prob_valid * 100:.1f}%")
        st.progress(int(prob_hoaks * 100))

        st.divider()
        st.caption("⚠️ Hasil prediksi bersifat otomatis dan tidak menggantikan verifikasi dari sumber terpercaya.")
