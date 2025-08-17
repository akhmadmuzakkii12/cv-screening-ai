import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import requests
import os
from dotenv import load_dotenv

# =========================
# Setup API Key
# =========================
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("JOOBLE_API_KEY")

url = f"https://jooble.org/api/eb18500283d6446ba0c13caf1c4f46e6"
payload = {"keywords": "data scientist", "location": "Indonesia"}

response = requests.post(url, json=payload)

print("Status:", response.status_code)
print("Response:", response.text)

# =========================
# Setup Aplikasi
# =========================
st.set_page_config(
    page_title="CV Screening & Job Recommendation",
    page_icon="📄",
    layout="wide"
)

st.title("📄 CV Screening & Job Recommendation")
st.markdown(
    """
    Aplikasi ini membantu **screening CV** dan memberikan rekomendasi pekerjaan 
    berdasarkan kesesuaian dengan **job descriptions dari API (Jooble)**.
    """
)

# =========================
# Load Model Embedding
# =========================
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


def fetch_job_descriptions(query="data", location="Indonesia", limit=5, page=1):
    """Ambil job descriptions dari Jooble API (jika ada API key), fallback ke Dummy API."""
    jobs = []
    try:
        if API_KEY:  # Pakai Jooble API
            API_URL = f"https://jooble.org/api/{API_KEY}"
            payload = {"keywords": query, "location": location, "page": page}
            response = requests.post(API_URL, json=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()
                jobs = [
                    f"{job['title']} at {job['company']} - {job['location']}. {job['snippet']}"
                    for job in data.get("jobs", [])
                ]
            else:
                st.warning(f"⚠️ Gagal ambil data Jooble API. Status: {response.status_code}")
        else:  # Pakai Dummy API
            API_URL = f"https://dummyjson.com/jobs/search?q={query}&limit={limit}"
            response = requests.get(API_URL, timeout=10)
            if response.status_code == 200:
                data = response.json()
                jobs = [
                    f"{job['title']} at {job['company']} - {job['description']}"
                    for job in data.get("jobs", [])
                ]
    except Exception as e:
        st.error(f"❌ Error saat mengambil data API: {e}")

    return jobs


def rank_jobs(cv_text, job_descriptions):
    """Hitung kesesuaian antara CV dan daftar job descriptions."""
    if not job_descriptions:
        return pd.DataFrame()

    cv_vector = model.encode(cv_text, convert_to_tensor=True)
    job_vectors = model.encode(job_descriptions, convert_to_tensor=True)

    # Hitung similarity score
    scores = util.pytorch_cos_sim(cv_vector, job_vectors)[0].cpu().numpy()
    ranking = np.argsort(scores)[::-1]

    results = [
        {
            "Job Description": job_descriptions[idx],
            "Match Score": float(scores[idx])  # simpan raw score (0–1)
        }
        for idx in ranking
    ]
    return pd.DataFrame(results)

# =========================
# UI / Tampilan
# =========================
# Sidebar
st.sidebar.header("📂 Upload CV")
uploaded_file = st.sidebar.file_uploader("Pilih file CV (PDF)", type="pdf")

st.sidebar.header("🔍 Job Search Setting")
job_query = st.sidebar.text_input("Cari posisi:", value="data scientist")
job_location = st.sidebar.text_input("Lokasi:", value="Indonesia")
job_limit = st.sidebar.slider("Jumlah job diambil:", 3, 15, 5)

# =========================
# Main Logic
# =========================
if uploaded_file:
    with st.spinner("🔍 Memproses CV..."):
        cv_text = extract_text_from_pdf(uploaded_file)

    # Tampilkan teks hasil ekstraksi
    st.subheader("📜 CV Extracted Text")
    with st.expander("Lihat detail CV"):
        st.write(cv_text)

    # Ambil job descriptions dari API
    job_descriptions = fetch_job_descriptions(query=job_query, location=job_location, limit=job_limit)

    if job_descriptions:
        st.subheader("🔍 Rekomendasi Pekerjaan")
        results_df = rank_jobs(cv_text, job_descriptions)

        if not results_df.empty:
            # Tampilkan hasil sebagai tabel
            st.dataframe(results_df, use_container_width=True)

            # Progress bar untuk tiap pekerjaan
            for _, row in results_df.iterrows():
                st.write(f"**{row['Job Description']}**")
                st.progress(min(max(row['Match Score'], 0.0), 1.0))  # normalisasi ke 0–1
        else:
            st.warning("⚠️ Tidak ada hasil ranking.")
    else:
        st.warning("⚠️ Tidak ada job descriptions dari API.")
else:
    st.info("👈 Silakan upload CV Anda di sidebar untuk memulai analisis.")

