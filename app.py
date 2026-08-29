"""
============================================================
 CareMetrics Clinical Decision Support
 Diabetes Risk Assessment Module
============================================================
An academic ML-based clinical decision-support tool for
diabetes risk screening, built on the Pima Indians Diabetes
Dataset.

Run with:
    streamlit run app.py

Expected model artifacts (produced by the training notebook):
    models/best_model.pkl
    models/scaler.pkl
    models/feature_columns.pkl
============================================================
"""

import os
import io
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CareMetrics | Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CONSTANTS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")

APP_NAME = "CareMetrics"
APP_TAGLINE = "Clinical Decision Support · Diabetes Risk Module"

# Clinical reference ranges used purely to give the clinician
# context next to each field — NOT used by the model itself.
REFERENCE_RANGES = {
    "Glucose": {
        "unit": "mg/dL",
        "bands": [(0, 100, "Normal (fasting)"),
                  (100, 126, "Prediabetes range"),
                  (126, 999, "Diabetes range")],
    },
    "BloodPressure": {
        "unit": "mm Hg (diastolic)",
        "bands": [(0, 80, "Normal"),
                  (80, 90, "Elevated"),
                  (90, 999, "High")],
    },
    "BMI": {
        "unit": "kg/m²",
        "bands": [(0, 18.5, "Underweight"),
                  (18.5, 25, "Normal"),
                  (25, 30, "Overweight"),
                  (30, 999, "Obese")],
    },
    "Insulin": {
        "unit": "mu U/mL",
        "bands": [(0, 16, "Low"),
                  (16, 166, "Normal (2-hr serum)"),
                  (166, 9999, "Elevated")],
    },
}

FEATURE_LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose",
    "BloodPressure": "Blood Pressure",
    "SkinThickness": "Skin Thickness",
    "Insulin": "Insulin",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age",
}

# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, scaler, columns


def artifacts_ready() -> bool:
    return all(os.path.exists(p) for p in (MODEL_PATH, SCALER_PATH, COLUMNS_PATH))


# =========================================================
# GLOBAL STYLES
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    /* ---------- Header ---------- */
    .app-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding-bottom: 4px;
    }
    .app-header .logo {
        font-size: 34px;
        line-height: 1;
    }
    .app-title {
        font-size: 28px;
        font-weight: 800;
        color: #0f2b46;
        margin: 0;
    }
    .app-subtitle {
        font-size: 14px;
        color: #64748b;
        margin: 0;
        font-weight: 500;
    }
    .env-badge {
        display: inline-block;
        margin-left: 10px;
        padding: 3px 10px;
        border-radius: 999px;
        background: #eef2ff;
        color: #4338ca;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .04em;
        text-transform: uppercase;
        vertical-align: middle;
    }

    /* ---------- Section headings ---------- */
    .section-heading {
        font-size: 17px;
        font-weight: 700;
        color: #0f2b46;
        margin-top: 6px;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid #e2e8f0;
    }

    /* ---------- Result cards ---------- */
    .result-box {
        padding: 28px 26px;
        border-radius: 16px;
        text-align: left;
        min-height: 200px;
        border: 1px solid;
    }
    .low-risk {
        background: #f0fdf6;
        border-color: #86dba4;
    }
    .high-risk {
        background: #fef2f2;
        border-color: #f2a8ae;
    }
    .result-label {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .06em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 6px;
    }
    .result-text {
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .low-risk .result-text { color: #166534; }
    .high-risk .result-text { color: #991b1b; }
    .result-probability {
        font-size: 40px;
        font-weight: 800;
        margin: 4px 0 6px 0;
    }
    .low-risk .result-probability { color: #15803d; }
    .high-risk .result-probability { color: #b91c1c; }
    .result-description {
        font-size: 13.5px;
        color: #475569;
        line-height: 1.5;
    }

    /* ---------- Reference chips ---------- */
    .ref-chip {
        display: inline-block;
        font-size: 11.5px;
        font-weight: 600;
        padding: 2px 9px;
        border-radius: 999px;
        margin-top: 4px;
    }
    .ref-normal   { background: #ecfdf5; color: #047857; }
    .ref-caution  { background: #fffbeb; color: #b45309; }
    .ref-alert    { background: #fef2f2; color: #b91c1c; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        border-right: 1px solid #e2e8f0;
    }
    .sidebar-brand {
        font-size: 20px;
        font-weight: 800;
        color: #0f2b46;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding: 18px 0 4px 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================

def classify_value(feature: str, value: float):
    """Return (label, css_class) for a clinical reference band."""
    ref = REFERENCE_RANGES.get(feature)
    if not ref:
        return None, None
    for low, high, label in ref["bands"]:
        if low <= value < high:
            if label.lower().startswith(("normal", "low")):
                css = "ref-normal"
            elif "range" in label.lower() and "diabetes" in label.lower():
                css = "ref-alert"
            elif label.lower() in ("obese", "high", "elevated", "diabetes range"):
                css = "ref-alert"
            else:
                css = "ref-caution"
            return label, css
    return None, None


def reference_chip(feature: str, value: float) -> str:
    label, css = classify_value(feature, value)
    if not label:
        return ""
    return f'<span class="ref-chip {css}">{label}</span>'


def build_report_text(patient_id, assessment_date, inputs, prediction, probability) -> str:
    lines = [
        f"{APP_NAME} — Diabetes Risk Assessment Report",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Patient ID: {patient_id or 'Not provided'}",
        f"Assessment Date: {assessment_date}",
        "",
        "Clinical Measurements",
        "-" * 50,
    ]
    for key, label in FEATURE_LABELS.items():
        lines.append(f"{label}: {inputs[key]}")
    lines += [
        "",
        "Model Output",
        "-" * 50,
        f"Predicted class: {'Higher predicted risk' if prediction == 1 else 'Lower predicted risk'}",
        f"Estimated probability (positive class): {probability:.1%}",
        "",
        "Disclaimer",
        "-" * 50,
        "This report is generated by a machine-learning model built for an",
        "academic project. It is NOT a medical diagnosis and must not be used",
        "as a substitute for professional clinical judgement.",
    ]
    return "\n".join(lines)


# =========================================================
# LOAD MODEL (guarded)
# =========================================================

model = scaler = columns = None
model_error = None

if artifacts_ready():
    try:
        model, scaler, columns = load_artifacts()
    except Exception as e:
        model_error = str(e)
else:
    model_error = "Model artifacts not found."

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown('<div class="sidebar-brand">🩺 CareMetrics</div>', unsafe_allow_html=True)
    st.caption("Clinical Decision Support Platform")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Risk Assessment",
            "📋 Patient Information",
            "🤖 Model Information",
            "📚 About Project",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    if model is not None:
        st.success("Model status: Online")
        st.caption(f"Active model: **{type(model).__name__}**")
    else:
        st.error("Model status: Unavailable")

    st.caption("Build: Academic ML Project · v1.0")
    st.caption(f"Session date: {date.today().isoformat()}")

    st.divider()
    st.markdown(
        "<div style='font-size:12px;color:#94a3b8;'>"
        "For screening support only. Not a certified medical device."
        "</div>",
        unsafe_allow_html=True,
    )

# =========================================================
# HEADER (shared across pages)
# =========================================================

st.markdown(
    f"""
    <div class="app-header">
        <div class="logo">🩺</div>
        <div>
            <p class="app-title">{APP_NAME}
                <span class="env-badge">Academic Build</span>
            </p>
            <p class="app-subtitle">{APP_TAGLINE}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# =========================================================
# GUARD: MODEL NOT LOADED
# =========================================================

if model is None and page == "🏠 Risk Assessment":
    st.error("⚠️ Model files could not be loaded. The risk assessment tool is unavailable.")
    with st.expander("Details"):
        st.write("Expected files:")
        st.code("models/best_model.pkl\nmodels/scaler.pkl\nmodels/feature_columns.pkl")
        if model_error:
            st.write(f"Error: {model_error}")
    st.stop()

# =========================================================
# PAGE: RISK ASSESSMENT
# =========================================================

if page == "🏠 Risk Assessment":

    st.info(
        "Enter the patient's diagnostic measurements below to generate a "
        "machine-learning-based diabetes risk estimate. All fields reflect "
        "standard clinical intake values."
    )

    # ---------------- Patient details ----------------
    st.markdown('<div class="section-heading">Patient Details</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        patient_id = st.text_input("Patient ID / MRN", placeholder="e.g. PT-00142")
    with c2:
        assessment_date = st.date_input("Assessment Date", value=date.today())
    with c3:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=33, step=1)

    # ---------------- Clinical measurements ----------------
    st.markdown('<div class="section-heading">Clinical Measurements</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    with col2:
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120, step=1,
                                   help="Fasting plasma glucose concentration.")
        st.markdown(reference_chip("Glucose", glucose), unsafe_allow_html=True)
    with col3:
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70, step=1,
                                          help="Diastolic blood pressure.")
        st.markdown(reference_chip("BloodPressure", blood_pressure), unsafe_allow_html=True)
    with col4:
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1,
                                          help="Triceps skin fold thickness.")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        insulin = st.number_input("Insulin (mu U/mL)", min_value=0, max_value=900, value=80, step=1,
                                   help="2-hour serum insulin.")
        st.markdown(reference_chip("Insulin", insulin), unsafe_allow_html=True)
    with col6:
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=28.0, step=0.1)
        st.markdown(reference_chip("BMI", bmi), unsafe_allow_html=True)
    with col7:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01,
                               help="Likelihood of diabetes based on family history.")
    with col8:
        st.metric("Model Features", "8")
        st.caption("Random Forest (tuned)")

    # Soft clinical warnings for physiologically implausible zero values
    zero_flags = [name for name, val in
                  [("Glucose", glucose), ("Blood Pressure", blood_pressure), ("BMI", bmi)]
                  if val == 0]
    if zero_flags:
        st.warning(
            "⚠️ The following fields are set to 0, which is not physiologically "
            f"typical and may reduce prediction reliability: {', '.join(zero_flags)}."
        )

    st.divider()
    predict = st.button("🔍 Run Risk Assessment", use_container_width=True, type="primary")

    # ---------------- Prediction ----------------
    if predict:
        input_data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }

        try:
            with st.spinner("Scoring patient data…"):
                input_array = np.array([[input_data[c] for c in columns]])
                input_scaled = scaler.transform(input_array)
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]

            st.divider()
            st.markdown('<div class="section-heading">Assessment Result</div>', unsafe_allow_html=True)

            result_col1, result_col2 = st.columns([1.6, 1])

            with result_col1:
                risk_class = "high-risk" if prediction == 1 else "low-risk"
                risk_label = "Higher Predicted Risk" if prediction == 1 else "Lower Predicted Risk"
                icon = "⚠️" if prediction == 1 else "✅"
                st.markdown(
                    f"""
                    <div class="result-box {risk_class}">
                        <div class="result-label">Model Output</div>
                        <div class="result-text">{icon} {risk_label}</div>
                        <div class="result-probability">{probability:.1%}</div>
                        <div class="result-description">
                            Estimated probability of the positive diabetes class,
                            based on the {8} clinical measurements provided above.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with result_col2:
                st.subheader("Risk Probability")
                st.metric("Estimated Probability", f"{probability:.1%}")
                st.progress(float(probability))
                if probability < 0.30:
                    st.success("Lower probability range")
                elif probability < 0.70:
                    st.warning("Intermediate probability range")
                else:
                    st.error("Higher probability range")

            st.warning(
                "⚠️ This result is generated by a machine-learning model for "
                "academic screening purposes only. It is **not** a medical "
                "diagnosis and does not replace professional clinical evaluation."
            )

            # ---------------- Summary ----------------
            st.divider()
            st.markdown('<div class="section-heading">Assessment Summary</div>', unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Patient ID", patient_id if patient_id else "Not provided")
            s2.metric("Assessment Date", str(assessment_date))
            s3.metric("Age", f"{age} yrs")
            s4.metric("BMI", f"{bmi:.1f}")

            with st.expander("📋 View Complete Clinical Measurements"):
                st.table(pd.DataFrame({
                    "Measurement": list(FEATURE_LABELS.values()),
                    "Value": [
                        pregnancies, f"{glucose} mg/dL", f"{blood_pressure} mm Hg",
                        f"{skin_thickness} mm", f"{insulin} mu U/mL", f"{bmi:.1f}",
                        f"{dpf:.2f}", f"{age} years",
                    ],
                }))

            # ---------------- Downloadable report ----------------
            report_text = build_report_text(patient_id, assessment_date, input_data, prediction, probability)
            st.download_button(
                "⬇️ Download Assessment Report (.txt)",
                data=report_text,
                file_name=f"diabetes_assessment_{patient_id or 'patient'}_{assessment_date}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        except Exception as e:
            st.error("Prediction could not be completed.")
            st.exception(e)

# =========================================================
# PAGE: PATIENT INFORMATION
# =========================================================

elif page == "📋 Patient Information":

    st.markdown('<div class="section-heading">Feature Reference Guide</div>', unsafe_allow_html=True)
    st.write("The prediction model uses eight numerical diagnostic features, listed below with typical clinical context.")

    st.dataframe(
        pd.DataFrame({
            "Feature": list(FEATURE_LABELS.values()),
            "Description": [
                "Number of pregnancies",
                "Plasma glucose concentration (fasting)",
                "Diastolic blood pressure",
                "Triceps skin fold thickness",
                "Two-hour serum insulin",
                "Body Mass Index (weight / height²)",
                "Diabetes likelihood based on family history",
                "Age in years",
            ],
            "Typical Unit": ["count", "mg/dL", "mm Hg", "mm", "mu U/mL", "kg/m²", "score", "years"],
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown('<div class="section-heading">Clinical Reference Bands</div>', unsafe_allow_html=True)
    for feature, ref in REFERENCE_RANGES.items():
        st.markdown(f"**{FEATURE_LABELS[feature]}** ({ref['unit']})")
        band_text = "  ·  ".join(
            f"{low}–{high if high < 999 else '∞'}: {label}" for low, high, label in ref["bands"]
        )
        st.caption(band_text)

# =========================================================
# PAGE: MODEL INFORMATION
# =========================================================

elif page == "🤖 Model Information":

    st.markdown('<div class="section-heading">Model Overview</div>', unsafe_allow_html=True)
    st.success(f"Active model: **{type(model).__name__}**")

    st.markdown('<div class="section-heading">Machine Learning Pipeline</div>', unsafe_allow_html=True)
    st.code(
        "Pima Indians Diabetes Dataset\n"
        "        │\n"
        "Data Cleaning\n"
        "        │\n"
        "Invalid Zero Handling\n"
        "        │\n"
        "Median Imputation\n"
        "        │\n"
        "Train / Test Split\n"
        "        │\n"
        "StandardScaler\n"
        "        │\n"
        "SMOTE (class balancing)\n"
        "        │\n"
        "Model Comparison\n"
        "        │\n"
        "Random Forest Hyperparameter Tuning\n"
        "        │\n"
        "Final Model Selection\n"
        "        │\n"
        "Joblib Export\n"
        "        │\n"
        "Streamlit Deployment",
        language=None,
    )

    st.markdown('<div class="section-heading">Technologies Used</div>', unsafe_allow_html=True)
    tech = ["Python", "Pandas", "NumPy", "Scikit-learn", "XGBoost", "SMOTE", "Joblib", "Streamlit", "Jupyter"]
    st.write(" · ".join(tech))

    st.markdown('<div class="section-heading">Saved Model Artifacts</div>', unsafe_allow_html=True)
    st.code("models/\n├── best_model.pkl\n├── scaler.pkl\n└── feature_columns.pkl")

# =========================================================
# PAGE: ABOUT PROJECT
# =========================================================

elif page == "📚 About Project":

    st.markdown('<div class="section-heading">Project Objective</div>', unsafe_allow_html=True)
    st.write(
        "This project applies supervised machine learning to estimate diabetes "
        "risk from routine patient diagnostic measurements, presented through a "
        "clinical-style decision-support interface."
    )

    st.markdown('<div class="section-heading">Dataset</div>', unsafe_allow_html=True)
    st.write(
        "The project uses the Pima Indians Diabetes Dataset, containing 768 "
        "patient records and eight numerical diagnostic features."
    )

    st.markdown('<div class="section-heading">Models Compared</div>', unsafe_allow_html=True)
    st.write("Logistic Regression, Decision Tree, Random Forest, KNN, SVM, and XGBoost.")

    st.markdown('<div class="section-heading">Final Model</div>', unsafe_allow_html=True)
    st.write("The deployed application uses the saved, tuned model produced by the training notebook.")

    st.markdown('<div class="section-heading">⚠️ Medical Disclaimer</div>', unsafe_allow_html=True)
    st.error(
        "This application is an academic machine-learning project. It is not a "
        "certified medical device and must not be used for diagnosis or "
        "treatment. Healthcare decisions must always be made by qualified "
        "medical professionals."
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    f"""
    <div class="footer">
        🩺 {APP_NAME} — Diabetes Risk Assessment &nbsp;|&nbsp;
        Academic Machine Learning Project &nbsp;|&nbsp;
        Not for Clinical Diagnosis
    </div>
    """,
    unsafe_allow_html=True,
)
