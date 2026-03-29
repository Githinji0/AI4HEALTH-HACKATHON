import streamlit as st
import numpy as np
import cv2
import pandas as pd
import altair as alt
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
LOGO_PATH = os.path.join(PROJECT_ROOT, "app", "assets", "logo.png")
IMG_SIZE = 64

PAGE_ICON = LOGO_PATH if os.path.exists(LOGO_PATH) else "🩺"

@st.cache_resource
def load_ai_model():
    return load_model(MODEL_PATH)

model = load_ai_model()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="NexMD", page_icon=PAGE_ICON, layout="wide")

# ---------------- GLOBAL STYLES ----------------
st.markdown("""
<style>
:root {
    --background: hsl(0 0% 98%);
    --foreground: hsl(150 10% 15%);

    --card: hsl(0 0% 100%);
    --card-foreground: hsl(150 10% 15%);

    --popover: hsl(0 0% 100%);
    --popover-foreground: hsl(150 10% 15%);

    --primary: hsl(152 60% 36%);
    --primary-foreground: hsl(0 0% 100%);

    --secondary: hsl(145 30% 94%);
    --secondary-foreground: hsl(150 10% 15%);

    --muted: hsl(140 15% 95%);
    --muted-foreground: hsl(150 5% 45%);

    --accent: hsl(160 50% 92%);
    --accent-foreground: hsl(152 60% 26%);

    --destructive: hsl(0 72% 51%);
    --destructive-foreground: hsl(0 0% 100%);

    --border: hsl(145 15% 88%);
    --input: hsl(145 15% 88%);
    --ring: hsl(152 60% 36%);

    --radius: 0.75rem;

    --sidebar-background: hsl(152 40% 18%);
    --sidebar-foreground: hsl(0 0% 95%);
    --sidebar-primary: hsl(152 60% 50%);
    --sidebar-primary-foreground: hsl(0 0% 100%);
    --sidebar-accent: hsl(152 30% 24%);
    --sidebar-accent-foreground: hsl(145 20% 90%);
    --sidebar-border: hsl(152 25% 25%);
    --sidebar-ring: hsl(152 60% 50%);

    --success: hsl(152 60% 36%);
    --success-foreground: hsl(0 0% 100%);
    --warning: hsl(38 92% 50%);
    --warning-foreground: hsl(0 0% 100%);
    --chart-parasitized: hsl(0 72% 56%);
    --chart-uninfected: hsl(152 60% 50%);
}

[data-testid="stAppViewContainer"] {
    background: var(--background);
    color: var(--foreground);
}

[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}

[data-testid="stAppToolbar"] {
    background: transparent !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

/* Sidebar collapsed toggle visibility */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    color: var(--sidebar-primary) !important;
}

[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapsedControl"] button,
button[aria-label="Toggle sidebar"] {
    color: var(--sidebar-primary) !important;
    background: var(--primary) !important;
    border: 2px solid hsl(0 0% 100%) !important;
    border-radius: 0.6rem !important;
    width: 2.4rem !important;
    height: 2.4rem !important;
    position: fixed !important;
    top: 0.85rem !important;
    left: 0.85rem !important;
    z-index: 99999 !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.22) !important;
}

[data-testid="collapsedControl"] button svg,
[data-testid="stSidebarCollapsedControl"] button svg,
button[aria-label="Toggle sidebar"] svg {
    color: var(--primary-foreground) !important;
    fill: var(--primary-foreground) !important;
}

[data-testid="collapsedControl"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover,
button[aria-label="Toggle sidebar"]:hover {
    background: hsl(152 60% 31%) !important;
    border-color: hsl(0 0% 100%) !important;
}

[data-testid="stSidebar"] {
    background: var(--sidebar-background);
    border-right: 1px solid var(--sidebar-border);
}

[data-testid="stSidebar"] * {
    color: var(--sidebar-foreground);
}

[data-testid="stSidebarUserContent"] {
    padding: 3.2rem 0.75rem 1.25rem 0.75rem;
}

[data-testid="stSidebarUserContent"] > div {
    gap: 0.85rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    margin: 0 0 0.35rem 0;
    letter-spacing: 0.01em;
}

[data-testid="stSidebar"] [data-testid="stRadio"] {
    background: hsl(152 30% 22%);
    border: 1px solid var(--sidebar-border);
    border-radius: var(--radius);
    padding: 0.85rem 0.8rem 0.75rem 0.8rem;
    margin-top: 0.15rem;
    box-shadow: inset 0 0 0 1px hsl(152 25% 28% / 0.35);
}

[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    font-size: 0.95rem;
    line-height: 1.35;
}

[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
}

[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label {
    width: 100%;
    min-height: 2.4rem;
    display: flex !important;
    align-items: center;
    border-radius: 0.6rem;
    padding: 0.45rem 0.55rem;
    background: hsl(152 28% 20%);
    border: 1px solid hsl(152 24% 30%);
    box-sizing: border-box;
    transition: background 0.2s ease, border-color 0.2s ease;
}

[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:hover {
    background: hsl(152 30% 24%);
    border-color: hsl(152 35% 40%);
}

[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
    background: hsl(152 36% 28%);
    border-color: hsl(152 60% 50%);
    box-shadow: inset 0 0 0 1px hsl(152 60% 50% / 0.25);
}

[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label p {
    width: 100%;
    margin: 0;
}

[data-testid="stSidebar"] hr {
    border: none;
    border-top: 1px solid var(--sidebar-border);
    margin: 0.95rem 0;
}

[data-testid="stSidebar"] [data-testid="stCheckbox"] {
    background: hsl(152 30% 22%);
    border: 1px solid var(--sidebar-border);
    border-radius: var(--radius);
    padding: 0.7rem 0.75rem;
    margin-top: 0.15rem;
}

.block-container {
    padding-top: 2rem;
}

/* Cards */
.card {
    background: var(--card);
    color: var(--card-foreground);
    padding: 1.5rem;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    box-shadow: 0 6px 22px rgba(35, 70, 50, 0.08);
    margin-bottom: 1rem;
}

/* Diagnosis */
.diagnosis-positive {
    color: var(--chart-parasitized);
    font-weight: bold;
    font-size: 1.5rem;
}
.diagnosis-negative {
    color: var(--chart-uninfected);
    font-weight: bold;
    font-size: 1.5rem;
}

/* Buttons */
.stButton>button {
    border-radius: var(--radius);
    height: 45px;
    font-weight: 600;
    border: 1px solid var(--primary);
    background: var(--primary);
    color: var(--primary-foreground);
}

.stButton>button:hover {
    background: hsl(152 60% 31%);
    border-color: hsl(152 60% 31%);
}

.stDownloadButton>button {
    border-radius: var(--radius);
    border: 1px solid var(--accent-foreground);
    background: var(--accent);
    color: var(--accent-foreground);
}

div[data-baseweb="input"] {
    border-color: var(--input);
}

[data-testid="stMetric"] {
    background: var(--secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.75rem;
}

[data-testid="stProgressBar"] div[role="progressbar"] {
    background: var(--primary);
}

/* Text input */
[data-testid="stTextInput"] label,
[data-testid="stTextInput"] p {
    color: var(--foreground) !important;
}

[data-testid="stTextInput"] input {
    background: var(--card) !important;
    color: var(--foreground) !important;
    border: 1px solid var(--input) !important;
    border-radius: var(--radius) !important;
    min-height: 2.5rem !important;
    line-height: 1.2 !important;
    padding: 0.55rem 0.75rem !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: var(--ring) !important;
    box-shadow: 0 0 0 0.15rem hsl(152 60% 36% / 0.18) !important;
}

[data-testid="stTextInput"] [data-testid="InputInstructions"] {
    display: block !important;
    margin-top: 0.32rem !important;
    padding-left: 0.2rem !important;
    color: var(--muted-foreground) !important;
    font-size: 0.78rem !important;
    line-height: 1.2 !important;
    text-align: left !important;
    opacity: 1 !important;
}

/* File uploader */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {
    color: var(--foreground) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: var(--foreground) !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: var(--primary) !important;
    color: var(--primary-foreground) !important;
    border: 1px solid var(--primary) !important;
    border-radius: calc(var(--radius) - 0.15rem) !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: hsl(152 60% 31%) !important;
    color: var(--primary-foreground) !important;
    border-color: hsl(152 60% 31%) !important;
}

/* Dashboard */
.section-title {
    margin: 0;
    color: var(--card-foreground);
    font-size: 1.15rem;
    font-weight: 700;
}

.section-subtitle {
    margin-top: 0.25rem;
    color: var(--muted-foreground);
    font-size: 0.92rem;
}

.metric-card {
    background: var(--secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.85rem 0.95rem;
    min-height: 5.2rem;
}

.metric-label {
    color: var(--muted-foreground);
    font-size: 0.82rem;
    margin-bottom: 0.35rem;
}

.metric-value {
    color: var(--foreground);
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: visible;
    padding-top: 0.2rem;
}

[data-testid="stDataFrame"] [data-testid="stElementToolbar"] {
    opacity: 1 !important;
    visibility: visible !important;
}

[data-testid="stDataFrame"] [role="grid"] {
    background: var(--card) !important;
}

[data-testid="stDataFrame"] [role="columnheader"] {
    background: var(--secondary) !important;
}

[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"],
[data-testid="stDataFrame"] [data-testid="stDataFrameGlideDataEditor"] {
    background: var(--card) !important;
}

[data-testid="stVegaLiteChart"] {
    background: var(--card) !important;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.35rem;
}

[data-testid="stVegaLiteChart"] > div {
    background: var(--card) !important;
}

/* Equal image heights for side-by-side analysis cards */
.equal-image img {
    width: 100% !important;
    height: 420px !important;
    object-fit: cover;
    border-radius: calc(var(--radius) - 0.1rem);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
logo_col, title_col = st.columns([0.12, 0.88])

with logo_col:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=84)

with title_col:
    st.markdown("""
    <h1>NexMD</h1>
    <p style='color: hsl(150 5% 45%);'>The next generation of Medical Diagnosis</p>
    """, unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, width=120)

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
        left, right = st.columns(2)

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # -------- LEFT: IMAGE --------
        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Input Image")
            st.markdown('<div class="equal-image">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

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
                    st.markdown('<div class="equal-image">', unsafe_allow_html=True)
                    st.image(shap_plot, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="equal-image">', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

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
        st.markdown(
            """
            <div class="card">
                <p class="section-title">Patient Dashboard</p>
                <p class="section-subtitle">Review trends, confidence, and historical diagnosis records.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        total_patients = len(df)
        infection_rate = (df['diagnosis'] == 'Parasitized').mean()
        avg_confidence = df['confidence'].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Patients</div>
                    <div class="metric-value">{total_patients}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Infection Rate</div>
                    <div class="metric-value">{infection_rate:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Confidence</div>
                    <div class="metric-value">{avg_confidence:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Build responsive time-series data for KPI charts
        trend_df = df.copy()
        trend_df['timestamp'] = pd.to_datetime(trend_df['timestamp'], errors='coerce')
        trend_df['confidence'] = pd.to_numeric(trend_df['confidence'], errors='coerce')
        trend_df = trend_df.dropna(subset=['timestamp']).sort_values('timestamp')

        if not trend_df.empty:
            daily = trend_df.groupby(trend_df['timestamp'].dt.date).agg(
                daily_patients=('id', 'count'),
                daily_parasitized=('diagnosis', lambda s: (s == 'Parasitized').sum()),
                avg_confidence=('confidence', 'mean')
            )
            daily.index = pd.to_datetime(daily.index)
            daily = daily.sort_index()

            daily['total_patients'] = daily['daily_patients'].cumsum()
            daily['infection_rate'] = daily['daily_parasitized'] / daily['daily_patients']

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Infection Rate Trend</p>', unsafe_allow_html=True)
            st.line_chart(daily[['infection_rate']], use_container_width=True, height=220)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Recent Diagnoses</p>', unsafe_allow_html=True)

        st.dataframe(df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        chart_data = df['diagnosis'].value_counts().rename_axis('Diagnosis').reset_index(name='Count')

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Diagnosis Distribution</p>', unsafe_allow_html=True)

        diagnosis_chart = (
            alt.Chart(chart_data)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X('Diagnosis:N', sort=['Parasitized', 'Uninfected']),
                y=alt.Y('Count:Q', title='Count'),
                color=alt.Color(
                    'Diagnosis:N',
                    scale=alt.Scale(
                        domain=['Parasitized', 'Uninfected'],
                        range=['#E65757', '#32C27A']
                    ),
                    legend=None
                ),
                tooltip=['Diagnosis:N', 'Count:Q']
            )
            .properties(height=280)
            .configure_axis(labelColor='#5B6661', titleColor='#2E3A34')
            .configure_view(strokeOpacity=0)
        )

        st.altair_chart(diagnosis_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("No patient records found.")