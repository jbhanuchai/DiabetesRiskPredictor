# -------------------- Diabetes Risk Predictor (Streamlit) --------------------
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# Tiny CSS polish (headings + muted text)
st.markdown("""
    <style>
      .muted {color:#6b7280;}
      .h-title {font-weight:800; letter-spacing:.2px;}
      .section-title {font-weight:600; margin-bottom:.25rem;}
    </style>
""", unsafe_allow_html=True)

# -------------------- Paths --------------------
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_CSV = ROOT / "data" / "diabetes.csv"

# -------------------- Custom transformer (used inside saved pipeline) --------------------
class ZeroToNaN(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X.loc[X[c] == 0, c] = np.nan
        return X

# -------------------- Cached loaders --------------------
@st.cache_resource
def load_assets():
    preprocess = joblib.load(MODEL_DIR / "preprocess.pkl")
    model = joblib.load(MODEL_DIR / "random_forest_model.pkl")
    return preprocess, model

@st.cache_data
def load_df():
    return pd.read_csv(DATA_CSV)

preprocess, model = load_assets()
df = load_df()

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# -------------------- Sidebar controls --------------------
st.sidebar.header("Settings")
threshold = st.sidebar.slider(
    "Decision Threshold",
    0.10, 0.90, 0.50, 0.01,
    help="Lower values increase sensitivity (recall). Higher values increase precision."
)

preset = st.sidebar.selectbox(
    "Quick Preset (optional)",
    ("None", "Typical Low Risk", "Typical Medium Risk", "Typical High Risk")
)

st.sidebar.markdown(
    "<div class='muted'>Tip: try presets and move the threshold to see how the class flips.</div>",
    unsafe_allow_html=True
)

# -------------------- Title --------------------
st.markdown("<h1 class='h-title'>Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='muted'>Random Forest model trained on the Pima Indians Diabetes dataset. "
    "Enter values on the left, then click <b>Predict</b>. You’ll see a clear probability, class, and the top factors.</div>",
    unsafe_allow_html=True
)

# -------------------- Helper functions --------------------
def slider_default(col: str) -> float:
    return float(df[col].median())

def slider_limits(col: str) -> tuple[float, float]:
    return float(df[col].min()), float(df[col].max())

def preset_values(name: str) -> dict:
    med = {c: slider_default(c) for c in FEATURES}
    if name == "Typical Low Risk":
        med.update({"Glucose": 95, "BMI": 22, "Age": 25, "BloodPressure": 70, "DiabetesPedigreeFunction": 0.2})
    elif name == "Typical Medium Risk":
        med.update({"Glucose": 120, "BMI": 28, "Age": 40, "BloodPressure": 75, "DiabetesPedigreeFunction": 0.4})
    elif name == "Typical High Risk":
        med.update({"Glucose": 165, "BMI": 35, "Age": 55, "BloodPressure": 85, "DiabetesPedigreeFunction": 0.7})
    return med

def risk_band(p: float) -> str:
    if p < 0.20: return "Low"
    if p < 0.40: return "Moderate"
    if p < 0.60: return "Elevated"
    return "High"

def show_risk_band(proba: float):
    band = risk_band(proba)
    msg = f"Risk Band: {band}"
    if band == "Low":
        st.success(f"🟢 {msg}")
    elif band == "Moderate":
        st.warning(f"🟠 {msg}")
    else:  # "Elevated" or "High"
        st.error(f"🔴 {msg}")

def render_easy_explanation(proba: float, label: int, threshold: float):
    """Simple user-facing explanation panel right below the tabs."""
    band = risk_band(proba)
    if proba >= 0.60:
        st.warning(
            f"""
### Easy Explanation
- **Predicted Risk:** {proba*100:.1f}% — this is a probability, not a medical diagnosis.  
- **Model Prediction:** 🩸 **Diabetic**.  
- **Risk Level:** **{band}** _(Low <20%, Moderate 20–40%, Elevated 40–60%, High >60%)_  

⚠️ Please consider medical consultation for confirmation and advice.
            """
        )
    else:
        st.info(
            f"""
### Easy Explanation
- **Predicted Risk:** {proba*100:.1f}% — this is a probability, not a medical diagnosis.  
- **Model Prediction:** {'🩸 Diabetic' if label==1 else '✅ Non-Diabetic'}.  
- **Risk Level:** **{band}** _(Low <20%, Moderate 20–40%, Elevated 40–60%, High >60%)_  

💡 Tip: Move the **Decision Threshold** slider on the left to see when the result flips (currently {threshold:.2f}).
            """
        )

# -------- NEW: session state to persist results across reruns --------
if "keep_results" not in st.session_state:
    st.session_state.keep_results = False
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None
if "show_val" not in st.session_state:
    st.session_state.show_val = False

# -------------------- Layout: Inputs (left) and Results (right) --------------------
left, right = st.columns([1.1, 1.0], gap="large")

with left:
    st.markdown("<div class='section-title'>Patient Inputs</div>", unsafe_allow_html=True)

    # Native bordered container (no raw HTML) to avoid ghost boxes
    with st.container(border=True):
        defaults = preset_values(preset)

        # A form gives one clear "Predict" action
        with st.form("risk_form"):
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                _, maxv = slider_limits("Pregnancies")
                Pregnancies = st.number_input(
                    "Pregnancies", min_value=0, max_value=int(max(20, maxv)),
                    value=int(defaults["Pregnancies"]), step=1
                )

            with c2:
                _, maxv = slider_limits("Glucose")
                Glucose = st.slider(
                    "Glucose (mg/dL)", 0.0, max(250.0, maxv),
                    float(defaults["Glucose"]), step=1.0
                )

            with c3:
                _, maxv = slider_limits("BloodPressure")
                BloodPressure = st.slider(
                    "Blood Pressure (mm Hg)", 0.0, max(130.0, maxv),
                    float(defaults["BloodPressure"]), step=1.0
                )

            with c4:
                _, maxv = slider_limits("SkinThickness")
                SkinThickness = st.slider(
                    "Skin Thickness (mm)", 0.0, max(100.0, maxv),
                    float(defaults["SkinThickness"]), step=1.0
                )

            c5, c6, c7, c8 = st.columns(4)

            with c5:
                _, maxv = slider_limits("Insulin")
                Insulin = st.slider(
                    "Insulin (μU/mL)", 0.0, max(900.0, maxv),
                    float(defaults["Insulin"]), step=0.5, format="%.2f"
                )

            with c6:
                _, maxv = slider_limits("BMI")
                BMI = st.slider(
                    "BMI (kg/m²)", 0.0, max(60.0, maxv),
                    float(defaults["BMI"]), step=0.1, format="%.1f"
                )

            with c7:
                _, maxv = slider_limits("DiabetesPedigreeFunction")
                DiabetesPedigreeFunction = st.slider(
                    "Diabetes Pedigree Function", 0.0, max(3.0, round(maxv, 2)),
                    float(defaults["DiabetesPedigreeFunction"]), step=0.01, format="%.2f",
                    help="Proxy for genetic risk (family history)."
                )

            with c8:
                _, maxv = slider_limits("Age")
                Age = st.slider(
                    "Age (years)", 0, int(max(100, maxv)),
                    int(defaults["Age"]), step=1
                )

            submitted = st.form_submit_button("Predict", width="stretch")  # was use_container_width=True


# -------- helper to render results given a dict of inputs (reusable) --------
def render_results_from_inputs(inputs: dict):
    X_user = pd.DataFrame([inputs], columns=FEATURES)
    X_user_prep = preprocess.transform(X_user)
    proba = float(model.predict_proba(X_user_prep)[0, 1])
    label = int(proba >= threshold)

    with right:
        st.subheader("Result")
        # Native bordered container for the result
        with st.container(border=True):
            cA, cB = st.columns([1, 1])
            with cA:
                st.metric("Predicted Risk", f"{proba*100:.1f}%")
                st.caption(f"Decision threshold: {threshold:.2f}")
                show_risk_band(proba)
            with cB:
                (st.error if label else st.success)(
                    f"Predicted Class: {'Diabetic (1)' if label else 'Non-Diabetic (0)'}"
                )

        tabs = st.tabs(["Feature Importance", "Why this result?", "Input Summary"])

        with tabs[0]:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                imp_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances})\
                           .sort_values("Importance", ascending=False)
                st.bar_chart(imp_df.set_index("Feature"))
                st.caption("Higher bar = stronger contribution in the Random Forest model.")
            else:
                st.info("Feature importance not available for this model type.")

        with tabs[1]:
            st.write(
                "This model outputs a probability using many trees. "
                "Glucose and BMI typically have the strongest impact. "
                "If you lower the threshold in the sidebar, more cases will be flagged as positive (higher recall)."
            )

        with tabs[2]:
            st.dataframe(pd.DataFrame([inputs]), width="stretch")

        with st.container(border=False):
            render_easy_explanation(proba, label, threshold)

# -------------------- Prediction & Results control flow --------------------
if submitted:
    # save the just-submitted inputs so we can re-render after reruns
    st.session_state.keep_results = True
    st.session_state.last_inputs = {
        "Pregnancies": int(Pregnancies),
        "Glucose": float(Glucose),
        "BloodPressure": float(BloodPressure),
        "SkinThickness": float(SkinThickness),
        "Insulin": float(Insulin),
        "BMI": float(BMI),
        "DiabetesPedigreeFunction": float(DiabetesPedigreeFunction),
        "Age": int(Age),
    }
    render_results_from_inputs(st.session_state.last_inputs)

elif st.session_state.keep_results and st.session_state.last_inputs is not None:
    # no new submit this rerun, but we have prior inputs -> render them
    render_results_from_inputs(st.session_state.last_inputs)
else:
    with right:
        st.info("Adjust inputs on the left and click **Predict** to view results.")

# -------------------- Footer (optional) --------------------
st.markdown(
    """
    <hr style="margin-top:2rem; margin-bottom:.5rem;">
    <div style='text-align:center; color:gray; font-size:0.9em;'>
        © 2025 — Educational and Research Use Only.
    </div>
    """,
    unsafe_allow_html=True
)
