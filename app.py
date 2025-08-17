import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import requests

# =========================
# Setup Aplikasi
# =========================
st.set_page_config(
    page_title="CV Screening & Job Recommendation",
    page_icon="📄",
    layout="wide"
)

# Load pre-trained embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# =========================
# Fungsi Pendukung
# =========================
def extract_text_from_pdf(uploaded_file):
    """Ekstraksi teks dari file PDF CV."""
    cv_text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                cv_text += text + "\n"
    return cv_text.strip()

def fetch_job_descriptions(query="data", location="Indonesia", limit=5):
    """Ambil job descriptions dari API eksternal (contoh dummy)."""
    API_URL = f"https://dummyjson.com/jobs/search?q={query}&limit={limit}"
    try:
        response = requests.get(API_URL, timeout=10)
        if response.status_code == 200:
            data = response.json()
            jobs = [f"{job['title']} at {job['company']} - {job['description']}" 
                    for job in data.get("jobs", [])]
            return jobs
        else:
            st.warning("⚠️ Gagal mengambil data dari API. Status:", response.status_code)
            return []
    except Exception as e:
        st.error(f"❌ Error saat mengambil data API: {e}")
        return []

def rank_jobs(cv_text, job_descriptions):
    """Hitung kesesuaian antara CV dan daftar job descriptions."""
    cv_vector = model.encode(cv_text, convert_to_tensor=True)
    job_vectors = model.encode(job_descriptions, convert_to_tensor=True)

    # Hitung similarity score
    scores = util.pytorch_cos_sim(cv_vector, job_vectors)[0].cpu().numpy()
    ranking = np.argsort(scores)[::-1]

    results = [
        {"Job Description": job_descriptions[idx], "Match Score": round(float(scores[idx]), 2)}
        for idx in ranking
    ]
    return pd.DataFrame(results)

# =========================
# UI / Tampilan
# =========================
st.title("📄 CV Screening & Job Recommendation (API Version)")
st.markdown(
    """
    Aplikasi ini membantu **screening CV** dan memberikan rekomendasi pekerjaan 
    berdasarkan kesesuaian dengan **job descriptions dari API**.
    """
)

# Sidebar untuk upload dan opsi
st.sidebar.header("📂 Upload CV")
uploaded_file = st.sidebar.file_uploader("Pilih file CV (PDF)", type="pdf")

# Sidebar query job
st.sidebar.header("🔍 Job Search Setting")
job_query = st.sidebar.text_input("Cari posisi:", value="data")
job_limit = st.sidebar.slider("Jumlah job diambil dari API:", 3, 15, 5)

# Ambil job descriptions dari API
job_descriptions = fetch_job_descriptions(query=job_query, limit=job_limit)

if uploaded_file and job_descriptions:
    with st.spinner("🔍 Memproses CV..."):
        cv_text = extract_text_from_pdf(uploaded_file)

    # Tampilkan teks hasil ekstraksi
    st.subheader("📜 CV Extracted Text")
    with st.expander("Lihat detail CV"):
        st.write(cv_text)

    # Ranking job rekomendasi
    st.subheader("🔍 Rekomendasi Pekerjaan")
    results_df = rank_jobs(cv_text, job_descriptions)

    # Tampilkan hasil sebagai tabel
    st.dataframe(results_df, use_container_width=True)

    # Progress bar untuk tiap pekerjaan
    for _, row in results_df.iterrows():
        st.write(f"**{row['Job Description']}**")
        st.progress(min(max(row['Match Score'], 0), 1))
elif not uploaded_file:
    st.info("👈 Silakan upload CV Anda di sidebar untuk memulai analisis.")
else:
    st.warning("⚠️ Tidak ada job descriptions dari API yang tersedia.")
