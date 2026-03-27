import streamlit as st
import numpy as np
import cv2
import pandas as pd
import sys
import os
from tensorflow.keras.models import load_model

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.utils import prepare_image, make_gradcam_heatmap, overlay_heatmap

MODEL_PATH = os.path.join(PROJECT_ROOT, "models/malaria_model.h5")
IMG_SIZE = 64

# Load model once
@st.cache_resource
def load_ai_model():
    return load_model(MODEL_PATH)

model = load_ai_model()

# Page config
st.set_page_config(page_title="Malaria AI Diagnosis", layout="wide")

# Sidebar
st.sidebar.title("Settings")
mode = st.sidebar.selectbox("Select Mode", ["Single Image", "Batch Processing"])

st.title("🦠 AI-Powered Malaria Diagnosis System")
st.markdown("### Fast • Explainable • Deployable in Low-Resource Settings")

# -----------------------------
# SINGLE IMAGE MODE
# -----------------------------
if mode == "Single Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        processed = prepare_image(image, IMG_SIZE)
        prediction = model.predict(processed)[0][0]

        # Correct Mapping: Class 0 = Parasitized, Class 1 = Uninfected
        # prediction is probability of class 1 (Uninfected)
        if prediction < 0.5:
            label = "Parasitized"
            confidence = 1 - prediction
        else:
            label = "Uninfected"
            confidence = prediction
        
        # Calculate severity based on Parasitized probability (1 - prediction)
        parasitized_prob = 1 - prediction
        if parasitized_prob > 0.8:
            severity = "Severe"
        elif parasitized_prob > 0.5:
            severity = "Moderate"
        elif parasitized_prob > 0.2:
            severity = "Mild"
        else:
            severity = "None"

        # Grad-CAM Explainability (Disabled for now)
        # heatmap = make_gradcam_heatmap(processed, model)
        # cam_image = overlay_heatmap(image, heatmap)

        with col2:
            # Show original image instead of Grad-CAM for now
            st.image(image, caption="Analysis complete", use_column_width=True)

        # Results
        st.subheader("Prediction Results")
        
        if label == "Parasitized":
            st.error(f"**Diagnosis:** {label}")
            st.warning(f"**Severity:** {severity}")
        else:
            st.success(f"**Diagnosis:** {label}")
            st.info(f"**Severity:** {severity}")

        st.write(f"**AI Confidence Score:** {confidence:.2%}")

        # Confidence bar showing the model's certainty
        st.progress(float(confidence))

# -----------------------------
# BATCH MODE
# -----------------------------
elif mode == "Batch Processing":
    uploaded_files = st.file_uploader(
        "Upload Multiple Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True
    )

    if uploaded_files:
        results = []

        for file in uploaded_files:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            processed = prepare_image(image, IMG_SIZE)
            prediction = model.predict(processed)[0][0]

            # Correct Mapping: Class 0 = Parasitized, Class 1 = Uninfected
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

            results.append({
                "Filename": file.name,
                "Diagnosis": label,
                "Confidence": f"{confidence:.2%}",
                "Severity": severity
            })

        df = pd.DataFrame(results)
        st.dataframe(df)

        # Download CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results CSV",
            csv,
            "malaria_results.csv",
            "text/csv"
        )