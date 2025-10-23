# 🩺 Diabetes Risk Predictor

An interactive **Streamlit** web app that predicts a patient’s probability of diabetes using clinical features (glucose, BMI, insulin, blood pressure, age, etc.).  
The model is a **Random Forest** trained on the **Pima Indians Diabetes** dataset.

---

## Features
- **Interactive UI** – sliders/inputs for all clinical fields.  
- **Configurable decision threshold** – explore precision/recall trade-offs.  
- **Color-coded Risk Band** – Low (🟢), Moderate (🟡), Elevated (🟠), High (🔴).  
- **Feature importance chart** – see which features drive the prediction.  
- **Plain-English explanation** – quick summary of what the result means.

---

## 🌐 Live Demo
 [Try it here on Streamlit](https://diabetes-risk-analyzer.streamlit.app)


---
## Tech Stack
- **Frontend:** Streamlit  
- **ML:** scikit-learn (RandomForestClassifier)  
- **Data:** pandas, numpy  
- **Serialization:** joblib  
- **Python:** 3.10+ (3.11 recommended)

---

## Project Structure
```
diabetes-risk-predictor/
│
├── app.py # Streamlit application (main file)
├── models/
│ ├── preprocess.pkl # Fitted preprocessing pipeline (joblib)
│ └── random_forest_model.pkl # Trained Random Forest model (joblib)
├── data/
│ └── diabetes.csv # Pima Indians Diabetes dataset
├── requirements.txt # Python dependencies
└── README.md
```

---

> If `models/` is empty, you’ll need to train and export the pipeline + model first (or copy your existing `preprocess.pkl` and `random_forest_model.pkl`).

---

## Run Locally

### 1. Create & activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies
```bash 
pip install -r requirements.txt
```

### 3. Start the app
```bash
streamlit run app.py
```
Then open your browser: http://localhost:8501

---

## Configuration

Decision Threshold: Adjustable sidebar slider (0.10–0.90).
→ Lower = more sensitive (higher recall), Higher = more precise.

Quick Presets: Try “Low / Medium / High Risk” patient profiles instantly.

## Dataset: Pima Indians Diabetes

Columns used:
Pregnancies, Glucose, BloodPressure, SkinThickness, 
Insulin, BMI, DiabetesPedigreeFunction, Age

Some zero values are treated as missing by the custom transformer ZeroToNaN inside the preprocessing pipeline.
