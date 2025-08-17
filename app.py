import os
import requests
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# =========================
# Setup API Key
# =========================
load_dotenv()
API_KEY = os.getenv("JOOBLE_API_KEY")

# =========================
# Load model NLP
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =========================
# Fungsi ambil data Jooble API
# =========================
def fetch_job_descriptions(keywords="data scientist", location="Indonesia"):
    url = f"https://jooble.org/api/{API_KEY}"
    payload = {"keywords": keywords, "location": location}

    try:
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            jobs = response.json().get("jobs", [])
            return [job["snippet"] for job in jobs]
        else:
            st.warning(f"⚠️ Gagal ambil data Jooble API. Status: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"❌ Error fetch API: {e}")
        return []

# =========================
# Fungsi ekstrak teks dari CV
# =========================
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# =========================
# Fungsi ranking CV
# =========================
def rank_candidates(cv_texts, job_descriptions):
    if not job_descriptions:
        st.warning("⚠️ Tidak ada job descriptions dari API.")
        return []

    job_embeddings = model.encode(job_descriptions, convert_to_tensor=True)
    ranked = []

    for i, cv in enumerate(cv_texts):
        cv_embedding = model.encode(cv, convert_to_tensor=True)
        similarity = util.cos_sim(cv_embedding, job_embeddings).mean().item()
        ranked.append((f"CV {i+1}", similarity))

    return sorted(ranked, key=lambda x: x[1], reverse=True)

# =========================
# Streamlit UI
# =========================
st.title("📄 AI Screening CV dengan Jooble API")

# Upload CV
uploaded_files = st.file_uploader("Upload CV (PDF)", type="pdf", accept_multiple_files=True)

# Input keywords
keywords = st.text_input("Masukkan kata kunci pekerjaan", "data scientist")
location = st.text_input("Masukkan lokasi pekerjaan", "Indonesia")

if st.button("Proses Screening"):
    if uploaded_files:
        cv_texts = [extract_text_from_pdf(file) for file in uploaded_files]
        job_descriptions = fetch_job_descriptions(keywords, location)
        ranked = rank_candidates(cv_texts, job_descriptions)

        if ranked:
            st.subheader("📊 Hasil Ranking CV")
            for cv, score in ranked:
                st.write(f"{cv}: **{score:.4f}**")
    else:
        st.warning("⚠️ Silakan upload minimal 1 CV (PDF).")
