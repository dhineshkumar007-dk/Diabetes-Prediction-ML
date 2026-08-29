
import streamlit as st
import numpy as np
import joblib
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(135deg, #e8f4ff, #f5f9ff);
    border: 1px solid #d9e8f5;
    margin-bottom: 25px;
}

.hero h1 {
    margin-bottom: 5px;
}

.card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
    background-color: #ffffff;
    margin-bottom: 15px;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin: 15px 0;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #f7f9fc;
    border: 1px solid #e2e6eb;
    text-align: center;
}

.small-text {
    color: #666666;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR, "models", "scaler.pkl"
)

COLUMNS_PATH = os.path.join(
    BASE_DIR, "models", "feature_columns.pkl"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columns = joblib.load(COLUMNS_PATH)

    return model, scaler, columns


try:

    model, scaler, columns = load_artifacts()

except Exception as e:

    st.error("Unable to load the machine learning model.")

    st.code(
        "models/best_model.pkl\n"
        "models/scaler.pkl\n"
        "models/feature_columns.pkl"
    )

    st.write("Error:", e)

    st.stop()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🩺 Diabetes AI")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Prediction",
            "📊 Model Information",
            "📚 About Project"
        ]
    )

    st.markdown("---")

    st.subheader("⚠️ Disclaimer")

    st.caption(
        "This application is an academic machine-learning "
        "project and is not a medical diagnostic device. "
        "Predictions should not be used for medical decisions."
    )

# =========================================================
# PREDICTION PAGE
# =========================================================

if page == "🏠 Prediction":

    st.markdown(
        """
        <div class="hero">
            <h1>🩺 Diabetes Risk Predictor</h1>
            <p>
            Machine learning based diabetes risk prediction
            using patient diagnostic measurements.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("👤 Patient Information")

    st.write(
        "Enter the patient's measurements below and click "
        "**Predict Diabetes Risk**."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=20,
                value=1,
                step=1,
                help="Number of times the patient has been pregnant."
            )

            glucose = st.number_input(
                "Glucose Level (mg/dL)",
                min_value=0,
                max_value=300,
                value=120,
                help="Plasma glucose concentration."
            )

            blood_pressure = st.number_input(
                "Blood Pressure (mm Hg)",
                min_value=0,
                max_value=200,
                value=70,
                help="Diastolic blood pressure."
            )

            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0,
                max_value=100,
                value=20,
                help="Triceps skin fold thickness."
            )

        with col2:

            insulin = st.number_input(
                "Insulin (mu U/mL)",
                min_value=0,
                max_value=900,
                value=80,
                help="2-Hour serum insulin level."
            )

            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=70.0,
                value=28.0,
                step=0.1,
                help="Body Mass Index."
            )

            dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                step=0.01,
                help="Diabetes pedigree function value."
            )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=33,
                step=1
            )

        submitted = st.form_submit_button(
            "🔍 Predict Diabetes Risk",
            use_container_width=True
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    if submitted:

        input_dict = {

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

            input_arr = np.array(
                [[input_dict[c] for c in columns]]
            )

            input_scaled = scaler.transform(input_arr)

            prediction = model.predict(
                input_scaled
            )[0]

            probability = model.predict_proba(
                input_scaled
            )[0][1]

            # =============================================
            # RESULT
            # =============================================

            st.divider()

            st.subheader("📋 Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                if prediction == 1:

                    st.error(
                        "⚠️ Higher Predicted Risk"
                    )

                    st.write(
                        "The model classified this input "
                        "as the higher-risk class."
                    )

                else:

                    st.success(
                        "✅ Lower Predicted Risk"
                    )

                    st.write(
                        "The model classified this input "
                        "as the lower-risk class."
                    )

            with result_col2:

                st.metric(
                    "Estimated Probability",
                    f"{probability:.1%}"
                )

                st.progress(
                    min(float(probability), 1.0)
                )

            st.caption(
                "The probability is produced by the trained "
                "machine-learning model and is not a clinical diagnosis."
            )

            # =============================================
            # PATIENT SUMMARY
            # =============================================

            st.divider()

            st.subheader("📊 Patient Input Summary")

            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

            with summary_col1:

                st.metric(
                    "Glucose",
                    f"{glucose} mg/dL"
                )

            with summary_col2:

                st.metric(
                    "BMI",
                    f"{bmi:.1f}"
                )

            with summary_col3:

                st.metric(
                    "Age",
                    age
                )

            with summary_col4:

                st.metric(
                    "Blood Pressure",
                    f"{blood_pressure} mm Hg"
                )

            with st.expander("View all patient measurements"):

                st.write(
                    input_dict
                )

        except Exception as e:

            st.error(
                "Prediction could not be completed."
            )

            st.exception(e)

# =========================================================
# MODEL INFORMATION PAGE
# =========================================================

elif page == "📊 Model Information":

    st.title("📊 Model Information")

    st.write(
        "This section provides information about the "
        "machine-learning model used by the application."
    )

    model_name = type(model).__name__

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            '<div class="metric-card">'
            '<h3>🤖 Algorithm</h3>'
            f'<p>{model_name}</p>'
            '</div>',
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            '<div class="metric-card">'
            '<h3>🎯 Task</h3>'
            '<p>Binary Classification</p>'
            '</div>',
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            '<div class="metric-card">'
            '<h3>📥 Features</h3>'
            '<p>8 Patient Measurements</p>'
            '</div>',
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("🔄 Machine Learning Workflow")

    st.markdown(
        """
        **1. Dataset Collection**

        Pima Indians Diabetes Dataset

        **2. Data Preprocessing**

        Data cleaning, feature preparation and scaling.

        **3. Exploratory Data Analysis**

        Analysis of distributions and relationships between
        patient measurements.

        **4. Model Training**

        Classification algorithms were trained using
        the prepared dataset.

        **5. Model Evaluation**

        Models were evaluated using classification metrics.

        **6. Final Model**

        The selected trained model is saved as
        `best_model.pkl`.

        **7. Deployment**

        The trained model is integrated with Streamlit
        for interactive prediction.
        """
    )

    st.divider()

    st.subheader("📥 Input Features")

    feature_data = {
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
            "Serum insulin level",
            "Body Mass Index",
            "Diabetes family-history function",
            "Age of patient"
        ]
    }

    st.table(feature_data)

# =========================================================
# ABOUT PROJECT PAGE
# =========================================================

elif page == "📚 About Project":

    st.title("📚 About the Project")

    st.subheader("🎯 Problem Statement")

    st.write(
        "The objective of this project is to develop a machine "
        "learning classification system that predicts the "
        "diabetes risk class from patient diagnostic measurements."
    )

    st.subheader("📊 Dataset")

    st.write(
        "The project uses the Pima Indians Diabetes Dataset. "
        "The dataset contains medical diagnostic measurements "
        "and a binary outcome indicating the diabetes class."
    )

    st.subheader("🧠 Machine Learning")

    st.write(
        "The project follows a complete machine-learning workflow "
        "including preprocessing, exploratory data analysis, "
        "model training, evaluation, model improvement and deployment."
    )

    st.subheader("💻 Technologies Used")

    technologies = [
        "Python",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Matplotlib",
        "Seaborn",
        "Joblib",
        "Jupyter Notebook",
        "Streamlit",
        "GitHub"
    ]

    for technology in technologies:

        st.markdown(
            f"• {technology}"
        )

    st.subheader("🔄 Project Pipeline")

    st.code(
        """
Dataset
   ↓
Data Preprocessing
   ↓
EDA & Visualization
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Improvement
   ↓
Save Model
   ↓
Streamlit Application
   ↓
Deployment
        """
    )

    st.divider()

    st.info(
        "⚠️ Educational purpose only. This application "
        "does not provide medical diagnosis or medical advice."
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🩺 Diabetes Risk Predictor | Machine Learning Mini Project"
)
