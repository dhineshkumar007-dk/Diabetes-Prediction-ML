
import streamlit as st
import numpy as np
import joblib
import os
from datetime import date

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🏥",
    layout="wide"
)

# =========================================================
# MODEL FILE PATHS
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

    st.error("Model files could not be loaded.")

    st.write("Please make sure these files exist:")

    st.code(
        "models/best_model.pkl\n"
        "models/scaler.pkl\n"
        "models/feature_columns.pkl"
    )

    st.error(str(e))

    st.stop()

# =========================================================
# PROFESSIONAL CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main page */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Main title */

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #777777;
        margin-bottom: 15px;
    }

    /* Result cards */

    .result-box {
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        margin-top: 10px;
        min-height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }

    /* Dark green card */

    .low-risk {
        background: linear-gradient(
            135deg,
            #10271b,
            #173d29
        );

        border: 1px solid #3d8b5b;
        color: white;
    }

    /* Dark red card */

    .high-risk {
        background: linear-gradient(
            135deg,
            #2b1215,
            #471a1f
        );

        border: 1px solid #a94b55;
        color: white;
    }

    .result-text {
        font-size: 30px;
        font-weight: 700;
        color: white;
        margin-bottom: 8px;
    }

    .result-probability {
        font-size: 48px;
        font-weight: 800;
        color: white;
        margin: 5px 0 10px 0;
    }

    .result-description {
        font-size: 15px;
        color: #dddddd;
        line-height: 1.5;
    }

    /* Sidebar */

    [data-testid="stSidebar"] {
        border-right: 1px solid #dddddd;
    }

    /* Section headings */

    .section-heading {
        font-size: 22px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Footer */

    .footer {
        text-align: center;
        color: #777777;
        font-size: 13px;
        padding: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🏥 Diabetes Care")

    st.caption(
        "Machine Learning Risk Assessment"
    )

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

    st.caption(
        f"Model: {type(model).__name__}"
    )

    st.caption(
        "Academic project"
    )

# =========================================================
# RISK ASSESSMENT
# =========================================================

if page == "🏠 Risk Assessment":

    st.markdown(
        '<div class="main-title">'
        '🏥 Diabetes Risk Assessment'
        '</div>',
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

    st.markdown(
        '<div class="section-heading">'
        '👤 Patient Details'
        '</div>',
        unsafe_allow_html=True
    )

    patient_col1, patient_col2, patient_col3 = st.columns(3)

    with patient_col1:

        patient_id = st.text_input(
            "Patient ID",
            placeholder="Example: PT-001"
        )

    with patient_col2:

        assessment_date = st.date_input(
            "Assessment Date",
            value=date.today()
        )

    with patient_col3:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=33,
            step=1
        )

    # -----------------------------------------------------
    # CLINICAL MEASUREMENTS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-heading">'
        '🩺 Clinical Measurements'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

    with col2:

        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=0,
            max_value=300,
            value=120,
            step=1
        )

    with col3:

        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)",
            min_value=0,
            max_value=200,
            value=70,
            step=1
        )

    with col4:

        skin_thickness = st.number_input(
            "Skin Thickness (mm)",
            min_value=0,
            max_value=100,
            value=20,
            step=1
        )

    col5, col6, col7, col8 = st.columns(4)

    with col5:

        insulin = st.number_input(
            "Insulin (mu U/mL)",
            min_value=0,
            max_value=900,
            value=80,
            step=1
        )

    with col6:

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=28.0,
            step=0.1
        )

    with col7:

        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01
        )

    with col8:

        st.metric(
            "Model Features",
            "8"
        )

    st.divider()

    # -----------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------

    predict = st.button(
        "🔍 Assess Diabetes Risk",
        use_container_width=True
    )

    # =====================================================
    # PREDICTION
    # =====================================================

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

            # Arrange features in training order

            input_array = np.array(
                [
                    [
                        input_data[column]
                        for column in columns
                    ]
                ]
            )

            # Scale input

            input_scaled = scaler.transform(
                input_array
            )

            # Prediction

            prediction = model.predict(
                input_scaled
            )[0]

            # Probability

            probability = model.predict_proba(
                input_scaled
            )[0][1]

            # -------------------------------------------------
            # ASSESSMENT RESULT
            # -------------------------------------------------

            st.divider()

            st.markdown(
                '<div class="section-heading">'
                '📊 Assessment Result'
                '</div>',
                unsafe_allow_html=True
            )

            result_col1, result_col2 = st.columns(
                [1.6, 1]
            )

            with result_col1:

                if prediction == 1:

                    st.markdown(
                        f"""
                        <div class="result-box high-risk">

                            <div class="result-text">
                                ⚠️ Higher Predicted Risk
                            </div>

                            <div class="result-probability">
                                {probability:.1%}
                            </div>

                            <div class="result-description">
                                Estimated model probability for
                                the positive diabetes class.
                            </div>

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

                            <div class="result-probability">
                                {probability:.1%}
                            </div>

                            <div class="result-description">
                                Estimated model probability for
                                the positive diabetes class.
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # -------------------------------------------------
            # PROBABILITY PANEL
            # -------------------------------------------------

            with result_col2:

                st.subheader("Risk Probability")

                st.metric(
                    "Estimated Probability",
                    f"{probability:.1%}"
                )

                st.progress(
                    float(probability)
                )

                if probability < 0.30:

                    st.success(
                        "Lower probability range"
                    )

                elif probability < 0.70:

                    st.warning(
                        "Intermediate probability range"
                    )

                else:

                    st.error(
                        "Higher probability range"
                    )

            # -------------------------------------------------
            # DISCLAIMER
            # -------------------------------------------------

            st.warning(
                "⚠️ This result is generated by a machine-learning "
                "model for academic screening purposes. It is not "
                "a medical diagnosis and should not replace "
                "professional clinical evaluation."
            )

            # -------------------------------------------------
            # PATIENT SUMMARY
            # -------------------------------------------------

            st.divider()

            st.markdown(
                '<div class="section-heading">'
                '📋 Assessment Summary'
                '</div>',
                unsafe_allow_html=True
            )

            summary1, summary2, summary3, summary4 = st.columns(4)

            with summary1:

                st.metric(
                    "Patient ID",
                    patient_id
                    if patient_id
                    else "Not provided"
                )

            with summary2:

                st.metric(
                    "Assessment Date",
                    str(assessment_date)
                )

            with summary3:

                st.metric(
                    "Age",
                    f"{age} years"
                )

            with summary4:

                st.metric(
                    "BMI",
                    f"{bmi:.1f}"
                )

            # -------------------------------------------------
            # COMPLETE MEASUREMENTS
            # -------------------------------------------------

            with st.expander(
                "📋 View Complete Clinical Measurements"
            ):

                st.table(
                    {
                        "Measurement": [
                            "Pregnancies",
                            "Glucose",
                            "Blood Pressure",
                            "Skin Thickness",
                            "Insulin",
                            "BMI",
                            "Diabetes Pedigree Function",
                            "Age"
                        ],

                        "Value": [
                            pregnancies,
                            f"{glucose} mg/dL",
                            f"{blood_pressure} mm Hg",
                            f"{skin_thickness} mm",
                            f"{insulin} mu U/mL",
                            f"{bmi:.1f}",
                            f"{dpf:.2f}",
                            f"{age} years"
                        ]
                    }
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
        "The prediction model uses eight numerical "
        "diagnostic features."
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
        f"Loaded Model: {type(model).__name__}"
    )

    st.subheader(
        "Machine Learning Workflow"
    )

    st.code(
        """
Pima Indians Diabetes Dataset
            ↓
Data Cleaning
            ↓
Invalid Zero Handling
            ↓
Median Imputation
            ↓
Train / Test Split
            ↓
StandardScaler
            ↓
SMOTE
            ↓
Model Comparison
            ↓
Random Forest Tuning
            ↓
Final Model
            ↓
Joblib Export
            ↓
Streamlit Deployment
        """
    )

    st.subheader(
        "Technologies Used"
    )

    st.write(
        "Python • Pandas • NumPy • Scikit-learn • "
        "XGBoost • SMOTE • Joblib • Streamlit • Jupyter"
    )

    st.subheader(
        "Saved Model Files"
    )

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

    st.subheader(
        "Project Objective"
    )

    st.write(
        "This project applies machine learning to estimate "
        "diabetes risk from patient diagnostic measurements."
    )

    st.subheader(
        "Dataset"
    )

    st.write(
        "The project uses the Pima Indians Diabetes Dataset "
        "containing 768 patient records and eight numerical "
        "features."
    )

    st.subheader(
        "Models Compared"
    )

    st.write(
        "Logistic Regression, Decision Tree, Random Forest, "
        "KNN, SVM, and XGBoost."
    )

    st.subheader(
        "Final Model"
    )

    st.write(
        "The final application uses the saved tuned model "
        "generated during the machine-learning workflow."
    )

    st.subheader(
        "⚠️ Medical Disclaimer"
    )

    st.error(
        "This application is an academic machine-learning "
        "project. It is not a medical device and must not "
        "be used for diagnosis or treatment. Healthcare "
        "decisions should be made by qualified professionals."
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">

    🏥 Diabetes Risk Assessment |
    Academic Machine Learning Project |
    Not for Clinical Diagnosis

    </div>
    """,
    unsafe_allow_html=True
)

