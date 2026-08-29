
import streamlit as st
import numpy as np
import joblib
import os
from datetime import date

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------------------------------------
# MODEL FILE PATHS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_artifacts():

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columns = joblib.load(COLUMNS_PATH)

    return model, scaler, columns


try:
    model, scaler, columns = load_artifacts()

except Exception as e:

    st.error("Model files could not be loaded.")

    st.write("Make sure your GitHub repository contains:")

    st.code(
        "models/best_model.pkl\n"
        "models/scaler.pkl\n"
        "models/feature_columns.pkl"
    )

    st.error(str(e))

    st.stop()


# ---------------------------------------------------------
# CUSTOM STYLE
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
    }

    .subtitle {
        font-size: 17px;
        color: #666666;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }

    .low-risk {
        background-color: #eaf7ee;
        border: 1px solid #b7dfc2;
    }

    .high-risk {
        background-color: #fff0f0;
        border: 1px solid #e5b5b5;
    }

    .result-text {
        font-size: 30px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.title("🏥 Diabetes Care")

    st.write("Machine Learning Risk Assessment")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Risk Assessment",
            "📋 Patient Information",
            "🤖 Model Information",
            "📚 About Project"
        ]
    )

    st.divider()

    st.success("Model Loaded")

    st.caption("Academic project")


# =========================================================
# RISK ASSESSMENT
# =========================================================

if page == "🏠 Risk Assessment":

    st.markdown(
        '<div class="main-title">🏥 Diabetes Risk Assessment</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine-learning based diabetes screening support system'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.info(
        "Enter the patient's diagnostic measurements below "
        "to generate a machine-learning risk prediction."
    )

    # -----------------------------------------------------
    # PATIENT DETAILS
    # -----------------------------------------------------

    st.subheader("👤 Patient Details")

    c1, c2, c3 = st.columns(3)

    with c1:

        patient_id = st.text_input(
            "Patient ID",
            placeholder="Example: PT-001"
        )

    with c2:

        assessment_date = st.date_input(
            "Assessment Date",
            value=date.today()
        )

    with c3:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=33
        )

    # -----------------------------------------------------
    # CLINICAL MEASUREMENTS
    # -----------------------------------------------------

    st.subheader("🩺 Clinical Measurements")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

    with c2:

        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=0,
            max_value=300,
            value=120
        )

    with c3:

        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)",
            min_value=0,
            max_value=200,
            value=70
        )

    with c4:

        skin_thickness = st.number_input(
            "Skin Thickness (mm)",
            min_value=0,
            max_value=100,
            value=20
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        insulin = st.number_input(
            "Insulin (mu U/mL)",
            min_value=0,
            max_value=900,
            value=80
        )

    with c2:

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=28.0,
            step=0.1
        )

    with c3:

        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01
        )

    with c4:

        st.metric(
            "Model Features",
            "8"
        )

    st.divider()

    # -----------------------------------------------------
    # PREDICTION BUTTON
    # -----------------------------------------------------

    predict = st.button(
        "🔍 Assess Diabetes Risk",
        use_container_width=True
    )

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if predict:

        input_data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }

        try:

            input_array = np.array(
                [[input_data[column] for column in columns]]
            )

            input_scaled = scaler.transform(input_array)

            prediction = model.predict(input_scaled)[0]

            probability = model.predict_proba(
                input_scaled
            )[0][1]

            st.divider()

            st.subheader("📊 Assessment Result")

            if prediction == 1:

                st.markdown(
                    f"""
                    <div class="result-box high-risk">

                    <div class="result-text">
                    ⚠️ Higher Predicted Risk
                    </div>

                    <h2>{probability:.1%}</h2>

                    <p>
                    Estimated model probability for the
                    positive diabetes class.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="result-box low-risk">

                    <div class="result-text">
                    ✅ Lower Predicted Risk
                    </div>

                    <h2>{probability:.1%}</h2>

                    <p>
                    Estimated model probability for the
                    positive diabetes class.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.progress(
                float(probability)
            )

            st.subheader("📋 Assessment Summary")

            s1, s2, s3, s4 = st.columns(4)

            with s1:
                st.metric(
                    "Patient ID",
                    patient_id if patient_id else "Not provided"
                )

            with s2:
                st.metric(
                    "Assessment Date",
                    str(assessment_date)
                )

            with s3:
                st.metric(
                    "Age",
                    f"{age} years"
                )

            with s4:
                st.metric(
                    "BMI",
                    f"{bmi:.1f}"
                )

            st.warning(
                "⚠️ This is an academic machine-learning "
                "prediction and not a medical diagnosis. "
                "Clinical decisions must be made by qualified "
                "healthcare professionals."
            )

        except Exception as e:

            st.error(
                "Prediction could not be completed."
            )

            st.exception(e)


# =========================================================
# PATIENT INFORMATION
# =========================================================

elif page == "📋 Patient Information":

    st.title("📋 Patient Information")

    st.write(
        "The model uses eight numerical diagnostic features."
    )

    st.table(
        {
            "Feature": [
                "Pregnancies",
                "Glucose",
                "Blood Pressure",
                "Skin Thickness",
                "Insulin",
                "BMI",
                "Diabetes Pedigree Function",
                "Age"
            ],

            "Description": [
                "Number of pregnancies",
                "Plasma glucose concentration",
                "Diastolic blood pressure",
                "Triceps skin fold thickness",
                "Two-hour serum insulin",
                "Body Mass Index",
                "Diabetes family-history related measure",
                "Age in years"
            ]
        }
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

elif page == "🤖 Model Information":

    st.title("🤖 Model Information")

    st.success(
        f"Loaded model: {type(model).__name__}"
    )

    st.subheader("Machine Learning Workflow")

    st.code(
        """
Dataset
   ↓
Data Cleaning
   ↓
Missing / Invalid Value Handling
   ↓
Train-Test Split
   ↓
StandardScaler
   ↓
SMOTE
   ↓
Model Comparison
   ↓
Random Forest Hyperparameter Tuning
   ↓
Final Model
   ↓
Streamlit Prediction
        """
    )

    st.subheader("Technologies Used")

    st.write(
        "Python • Pandas • NumPy • Scikit-learn • "
        "XGBoost • SMOTE • Joblib • Streamlit • Jupyter"
    )

    st.subheader("Saved Model Files")

    st.code(
        """
models/
├── best_model.pkl
├── scaler.pkl
└── feature_columns.pkl
        """
    )


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "📚 About Project":

    st.title("📚 About the Project")

    st.subheader("Project Objective")

    st.write(
        "This project uses machine learning to estimate "
        "diabetes risk from eight diagnostic measurements."
    )

    st.subheader("Dataset")

    st.write(
        "The project uses the Pima Indians Diabetes Dataset "
        "containing 768 patient records and eight numerical "
        "features."
    )

    st.subheader("Models Compared")

    st.write(
        "Logistic Regression, Decision Tree, Random Forest, "
        "KNN, SVM, and XGBoost."
    )

    st.subheader("Final Model")

    st.write(
        "The project uses a tuned Random Forest model after "
        "model comparison and hyperparameter tuning."
    )

    st.subheader("⚠️ Medical Disclaimer")

    st.error(
        "This application is an academic machine-learning "
        "project. It is not a medical device and must not be "
        "used for diagnosis or treatment. Always consult a "
        "qualified healthcare professional for medical decisions."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "🏥 Diabetes Risk Assessment | "
    "Academic Machine Learning Project | "
    "Not for clinical diagnosis"
)

