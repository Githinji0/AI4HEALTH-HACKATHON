NexMD

AI-powered malaria screening system for rapid, explainable, and accessible diagnosis.

🚀 Overview

MalariaScope AI is a deep learning-based system that analyzes microscope blood smear images to detect malaria infection.

It combines:

🧠 AI classification (CNN)
🔍 Explainability (SHAP, Grad-CAM)
📊 Patient analytics dashboard
📄 Clinical report generation
🎯 Key Features
✅ Binary classification: Parasitized vs Uninfected
📈 Confidence-based severity estimation
🔍 Explainable AI visualizations
🗂️ Patient record storage (SQLite)
📊 Dashboard with trends & metrics
📄 Downloadable PDF reports
📦 Batch image processing
🖥️ Demo Preview

Screenshot
<img width="1360" height="634" alt="image" src="https://github.com/user-attachments/assets/e5d53e1f-2f20-44eb-b483-ebb90f0baae0" />



🏗️ System Architecture
User → Streamlit UI → Model Inference → Explainability
     → Database Storage → Dashboard → PDF Reports
⚙️ Tech Stack
TensorFlow
Python
Streamlit
OpenCV
SHAP
SQLite
FPDF
📂 Project Structure
app/
  app.py

src/
  data_loader.py
  preprocessing.py
  model.py
  train.py
  evaluate.py
  database.py
  reports.py
  utils.py

models/
  malaria_model.h5
🧪 Setup Instructions
# Clone repo
git clone https://github.com/your-username/malaria-scope-ai.git

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app/app.py
📊 Model Training
python -m src.train
📈 Evaluation
python -m src.evaluate
⚠️ Disclaimer

This tool is for screening assistance only and should not replace professional medical diagnosis.

🌍 Impact

Designed for low-resource settings to:

Reduce diagnostic delays
Support healthcare workers
Improve early malaria detection
🔮 Future Work
Cloud deployment
Mobile integration
Multi-disease detection
Real-time health surveillance


 Authors
ViraTrack
