import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load model embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("📄 CV Screening & Job Recommendation")

# Upload PDF
uploaded_file = st.file_uploader("Upload CV dalam format PDF", type="pdf")

if uploaded_file is not None:
    # Ekstrak teks dari PDF
    with pdfplumber.open(uploaded_file) as pdf:
        cv_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                cv_text += text + "\n"

    st.subheader("📜 CV Extracted Text")
    st.write(cv_text)

    # Contoh job descriptions (nanti bisa diganti database/API)
    job_descriptions = [
        "Data Analyst at PT DataVision, requires Python, SQL, Tableau, and problem-solving",
        "Business Intelligence Specialist at XYZ Corp, requires SQL, Power BI, and data modeling",
        "Frontend Developer at ABC Tech, requires JavaScript, React, CSS"
    ]

    # Vectorization
    cv_vector = model.encode(cv_text, convert_to_tensor=True)
    job_vectors = model.encode(job_descriptions, convert_to_tensor=True)

    # Similarity scoring
    scores = util.pytorch_cos_sim(cv_vector, job_vectors)[0].cpu().numpy()

    # Ranking hasil
    ranking = np.argsort(scores)[::-1]

    st.subheader("🔍 Rekomendasi Pekerjaan")
    for idx in ranking:
        st.write(f"**{job_descriptions[idx]}** - Match Score: {scores[idx]:.2f}")
