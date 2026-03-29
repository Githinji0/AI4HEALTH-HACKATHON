# AI4HEALTH-HACKATHON Project Report

## 1. Executive Summary
[name] is an AI-assisted malaria screening solution built around microscope blood-cell image classification. The system predicts whether a sample is Parasitized or Uninfected, estimates severity from model probability, supports explainability, stores diagnosis records, and generates downloadable clinical PDF reports.

The project combines a machine learning pipeline (training and evaluation) with an interactive Streamlit web application intended for rapid screening workflows in low-resource settings.

## 2. Problem Statement and Motivation
Malaria diagnosis often depends on microscopy and trained personnel. In many settings, high workload and limited specialist coverage can delay screening. This project addresses the need for a fast, consistent, and user-friendly triage assistant that:
- Screens blood-cell images quickly.
- Provides confidence-aware outputs.
- Adds explainability options for trust and transparency.
- Supports basic clinical workflow tasks (record storage and report export).

## 3. Project Objectives
- Build a CNN model for binary malaria image classification.
- Deploy model inference in a practical web UI.
- Add explainability support with SHAP.
- Track patient-level diagnosis history.
- Provide downloadable PDF reports for documentation.
- Deliver a themed, responsive interface for usability.

## 4. Scope and Deliverables
### In-Scope
- Training pipeline for binary classification.
- Evaluation script with core metrics.
- Streamlit application with three modes:
  - Single Image
  - Batch Processing
  - Patient Dashboard
- SQLite diagnosis storage.
- PDF report generation.
- Explainability utilities (SHAP and Grad-CAM helper utilities).

### Out-of-Scope (Current Version)
- Multi-user authentication and role-based access.
- Cloud deployment infrastructure.
- Regulatory-grade validation and calibration studies.
- Production MLOps (model registry, drift monitoring, automated retraining).

## 5. System Architecture
The system follows a modular architecture:

1. Presentation Layer
- Streamlit application handles navigation, file uploads, visualizations, dashboard, and downloads.
- Main entry: app/app.py

2. Inference Layer
- Loads pre-trained TensorFlow model and runs predictions on uploaded images.
- Model artifact: models/malaria_model.h5

3. ML Pipeline Layer
- Data loading, preprocessing, train-test split, CNN model definition, training, and evaluation scripts.
- Modules: src/data_loader.py, src/preprocessing.py, src/model.py, src/train.py, src/evaluate.py

4. Explainability and Image Utility Layer
- SHAP explanation generation and image quality checks.
- Module: src/utils.py

5. Persistence Layer
- SQLite database for diagnosis records.
- Module: src/database.py
- Database file: malaria_diagnoses.db

6. Reporting Layer
- PDF report generation per diagnosis.
- Module: src/reports.py

## 6. Data and Processing Pipeline
### Data Assumptions
The training and evaluation scripts expect a local dataset folder structure with two class directories:
- Parasitized
- Uninfected

Configured path in current code:
- E:/fs/cell_images

### Preprocessing Steps
- Image resizing to 64 x 64.
- Pixel normalization to [0, 1].
- Label extraction from folder names.
- Train-test split via scikit-learn.

### Model Architecture
The CNN is a sequential model with:
- Conv2D (32) + MaxPool
- Conv2D (64) + MaxPool
- Conv2D (128) + MaxPool
- Flatten + Dense(128) + Dropout(0.5)
- Sigmoid output for binary classification

Training settings:
- Optimizer: Adam
- Loss: Binary cross-entropy
- Metric: Accuracy
- Epochs: 10
- Batch size: 32

## 7. Application Features
### 7.1 Single Image Mode
- Upload one microscope image.
- Run preprocessing and inference.
- Output diagnosis, confidence score, and severity level.
- Optional SHAP explanation visualization.
- Blurriness detection warning using variance of Laplacian.
- Save record to database.
- Download patient PDF report.

### 7.2 Batch Processing Mode
- Upload multiple images.
- Predict each image in sequence.
- Present a tabular summary with diagnosis, confidence, and severity.
- Export results as CSV.

### 7.3 Patient Dashboard Mode
- Displays key metrics:
  - Total Patients
  - Infection Rate
  - Avg Confidence
- Shows infection-rate trend over time.
- Displays recent diagnosis records table.
- Shows diagnosis distribution chart with distinct category colors:
  - Parasitized (red tone)
  - Uninfected (green tone)

## 8. Explainability and Clinical Interpretability
The project integrates SHAP for deep-learning prediction explanations:
- Uses GradientExplainer with a simple background tensor.
- Produces a visual explanation image.

Also includes a Grad-CAM utility function for additional interpretability support, though the current UI flow primarily uses SHAP when enabled.

## 9. Database and Reporting
### Database
SQLite schema includes:
- id
- patient_id
- diagnosis
- confidence
- severity
- timestamp

Operations implemented:
- initialize table
- save diagnosis
- fetch all diagnoses
- clear all data

### PDF Reporting
Generated report includes:
- report header and timestamp
- patient identifier
- diagnosis result
- confidence percentage
- severity label
- safety disclaimer for clinical review

## 10. User Interface and Theming
The UI has been customized for clarity and consistency:
- Light, healthcare-oriented theme using Streamlit theme config and CSS tokens.
- Sidebar restyled with navigation grouping and visible collapsed toggle.
- Form controls and uploader aligned to project color palette.
- Cards, metric blocks, charts, and data table containers visually unified.
- Side-by-side input and analysis images constrained to equal height for cleaner comparison.

## 11. Dependencies
Core dependencies:
- tensorflow
- opencv-python
- numpy
- scikit-learn
- pandas
- streamlit
- matplotlib
- shap
- fpdf2

## 12. Setup and Run Guide
### 12.1 Environment Setup
1. Create virtual environment.
2. Activate environment.
3. Install dependencies from requirements.txt.

### 12.2 Model Training
Run training module:
- python -m src.train

### 12.3 Evaluation
Run evaluation module:
- python -m src.evaluate

### 12.4 Launch Application
Run Streamlit app:
- streamlit run app/app.py

### 12.5 Optional Import Sanity Check
- python test_import.py

## 13. Validation Status
- Training command has been executed successfully in the current workspace context.
- App runs via Streamlit and supports all three modes.
- Dashboard chart customization and theme consistency updates are integrated.
- Import smoke test utility is available for SHAP-related functions.

## 14. Risks, Limitations, and Gaps
1. Data path is hardcoded in training and evaluation scripts, reducing portability.
2. No built-in authentication/authorization for patient data access.
3. SQLite is single-node/local and not ideal for multi-user clinical environments.
4. Limited production hardening (no audit logging, no access control, no encryption policy in-app).
5. Model quality metrics are generated but no advanced calibration workflow is included.
6. Explainability computation may increase latency on low-resource devices.

## 15. Recommended Next Steps
1. Move dataset path and app settings to environment variables or config files.
2. Add authentication and role-based access for dashboard and records.
3. Add API/service layer and migrate database to managed backend for team use.
4. Add test coverage for data pipeline, model I/O, and key UI actions.
5. Add CI workflow for linting, tests, and packaging.
6. Expand evaluation with precision, recall, F1 tracking across versions.
7. Add model versioning and experiment tracking for reproducibility.

## 16. Ethical and Clinical Use Notes
This tool is intended for screening assistance, not final standalone diagnosis. Clinical decisions should be made by qualified professionals. Future deployment should include governance for privacy, bias checks, model monitoring, and clear human-in-the-loop protocols.

## 17. Conclusion
AI4HEALTH-HACKATHON delivers a complete prototype that connects machine learning, explainability, records management, and reporting into a practical malaria screening workflow. The current implementation is suitable for demonstration, pilot exploration, and hackathon delivery, with clear pathways to production readiness through security, portability, and MLOps improvements.
