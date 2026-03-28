import streamlit as st
import numpy as np
import cv2
import pandas as pd
import sys
import os
from tensorflow.keras.models import load_model
from datetime import datetime

# ---------------- PATH SETUP ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# ---------------- IMPORTS ----------------
from src.utils import (
    prepare_image,
    make_gradcam_heatmap,
    overlay_heatmap,
    get_shap_explanation,
    check_image_quality
)
from src.database import init_db, save_diagnosis, get_all_diagnoses
from src.reports import generate_pdf_report

# ---------------- INIT ----------------
init_db()

MODEL_PATH = os.path.join(PROJECT_ROOT, "models/malaria_model.h5")
IMG_SIZE = 64

@st.cache_resource
def load_ai_model():
    return load_model(MODEL_PATH)

model = load_ai_model()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Malaria AI Diagnosis", layout="wide")

# ---------------- GLOBAL STYLES ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

/* Cards */
.card {
    background: #0E1117;
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    margin-bottom: 1rem;
}

/* Diagnosis */
.diagnosis-positive {
    color: #ff4b4b;
    font-weight: bold;
    font-size: 1.5rem;
}
.diagnosis-negative {
    color: #00c853;
    font-weight: bold;
    font-size: 1.5rem;
}

/* Buttons */
.stButton>button {
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<h1>Malaria AI Diagnostic Suite</h1>
<p style='color: gray;'>Clinical-grade inference • Explainable AI • Field-ready deployment</p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select Mode", ["Single Image", "Batch Processing", "Patient Dashboard"])

st.sidebar.markdown("---")
use_shap = st.sidebar.checkbox("Enable SHAP Explainability")

# ---------------- HELPER FUNCTION ----------------
def compute_results(prediction):
    if prediction < 0.5:
        label = "Parasitized"
        confidence = 1 - prediction
    else:
        label = "Uninfected"
        confidence = prediction

    parasitized_prob = 1 - prediction
    if parasitized_prob > 0.8:
        severity = "Severe"
    elif parasitized_prob > 0.5:
        severity = "Moderate"
    elif parasitized_prob > 0.2:
        severity = "Mild"
    else:
        severity = "None"

    return label, confidence, severity

# =====================================================
# SINGLE IMAGE MODE
# =====================================================
if mode == "Single Image":
    patient_id = st.text_input("Patient ID", "Anonymous")
    uploaded_file = st.file_uploader("Upload Microscope Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        left, right = st.columns([1.2, 1])

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # -------- LEFT: IMAGE --------
        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Input Image")
            st.image(image, use_container_width=True)

            is_blurry, blur_score = check_image_quality(image)
            if is_blurry:
                st.warning(f"Blurry Image (Score: {blur_score:.2f})")

            st.markdown('</div>', unsafe_allow_html=True)

        # -------- PROCESS --------
        processed = prepare_image(image, IMG_SIZE)
        prediction = model.predict(processed)[0][0]
        label, confidence, severity = compute_results(prediction)

        # -------- RIGHT: ANALYSIS --------
        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("AI Analysis")

            if use_shap:
                with st.spinner("Generating SHAP explanation..."):
                    shap_plot = get_shap_explanation(processed, model)
                    st.image(shap_plot, use_container_width=True)
            else:
                st.image(image, caption="Analysis Complete", use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # -------- RESULTS --------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Diagnosis Summary")

        if label == "Parasitized":
            st.markdown(f"<div class='diagnosis-positive'> {label}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='diagnosis-negative'> {label}</div>", unsafe_allow_html=True)

        st.write(f"**Severity:** {severity}")
        st.write(f"**Confidence:** {confidence:.2%}")
        st.progress(float(confidence))

        st.markdown('</div>', unsafe_allow_html=True)

        # -------- ACTIONS --------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Actions")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Save Record", use_container_width=True):
                save_diagnosis(patient_id, label, float(confidence), severity)
                st.success(f"Saved for {patient_id}")

        with col2:
            pdf = generate_pdf_report(patient_id, label, float(confidence), severity)
            st.download_button(
                "Download Report",
                pdf,
                f"Report_{patient_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
            )

        st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# BATCH MODE
# =====================================================
elif mode == "Batch Processing":
    uploaded_files = st.file_uploader("Upload Multiple Images", accept_multiple_files=True)

    if uploaded_files:
        results = []

        for file in uploaded_files:
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
            pred = model.predict(prepare_image(image, IMG_SIZE))[0][0]
            label, conf, sev = compute_results(pred)

            results.append({
                "Filename": file.name,
                "Diagnosis": label,
                "Confidence": f"{conf:.2%}",
                "Severity": sev
            })

        df = pd.DataFrame(results)

        st.markdown("### 📊 Batch Overview")

        col1, col2 = st.columns(2)
        col1.metric("Total Samples", len(df))
        col2.metric("Parasitized Cases", len(df[df["Diagnosis"]=="Parasitized"]))

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "results.csv")

# =====================================================
# DASHBOARD MODE
# =====================================================
elif mode == "Patient Dashboard":
    df = get_all_diagnoses()


    if not df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Patients", len(df))
        col2.metric("Infection Rate", f"{(df['diagnosis']=='Parasitized').mean():.1%}")
        col3.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")

        st.dataframe(df, use_container_width=True)
        st.bar_chart(df['diagnosis'].value_counts())

    else:
        st.info("No patient records found.")