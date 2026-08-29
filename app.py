```python
import streamlit as st
import numpy as np
import joblib
import os
from datetime import date

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PROFESSIONAL CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.header {
    padding: 25px 30px;
    border-radius: 16px;
    background: linear-gradient(135deg, #eaf4ff, #f8fbff);
    border: 1px solid #d7e7f5;
    margin-bottom: 25px;
}

.header-title {
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 5px;
}

.header-subtitle {
    font-size: 16px;
    color: #5f6b76;
}

.section-title {
    font-size: 22px;
    font-weight: 650;
    margin-top: 15px;
    margin-bottom: 12px;
}

.info-card {
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e1e7ed;
    background: #ffffff;
    margin-bottom: 15px;
}

.metric-card {
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e1e7ed;
    background: #f8fafc;
    text-align: center;
}

.risk-low {
    padding: 28px;
    border-radius: 16px;
    background: #eefaf2;
    border: 1px solid #b9e6c8;
    text-align: center;
}

.risk-high {
    padding: 28px;
    border-radius: 16px;
    background: #fff1f1;
    border: 1px solid #f0bcbc;
    text-align: center;
}

.risk-title {
    font-size: 28px;
    font-weight: 700;
}

.risk-probability {
    font-size: 42px;
    font-weight: 750;
    margin-top: 8px;
}

.footer {
    text-align: center;
    color: #777;
    font-size: 13px;
    padding-top: 20px;
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
def load_model():

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columns = joblib.load(COLUMNS_PATH)

    return model, scaler, columns


try:

    model, scaler, columns = load_model()

except Exception as e:

    st.error("The machine-learning model could not be loaded.")

    st.code(
        "models/best_model.pkl\n"
        "models/scaler.pkl\n"
        "models/feature_columns.pkl"
    )

    st.exception(e)
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🏥 Diabetes Care")

    st.caption("Machine Learning Risk Assessment")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Risk Assessment",
            "📋 Patient Information",
            "🤖 Model & Technology",
            "📚 About Diabetes"
        ]
    )

    st.divider()

    st.markdown("### System Status")

    st.success("Model loaded")

    st.caption(
        "Academic machine-learning project"
    )

# =========================================================
# RISK ASSESSMENT
# =========================================================

if page == "🏠 Risk Assessment":

    st.markdown("""
    <div class="header">

    <div class="header-title">
    🏥 Diabetes Risk Assessment
    </div>

    <div class="header-subtitle">
    Machine-learning based screening support using
    patient diagnostic measurements.
    </div>

    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Enter the available patient measurements below. "
        "The system will estimate the model's predicted risk class."
    )

    st.markdown(
        '<div class="section-title">👤 Patient Assessment</div>',
        unsafe_allow_html=True
    )

    with st.form("assessment_form"):

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

        st.markdown(
            '<div class="section-title">🩺 Clinical Measurements</div>',
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
                value=120
            )

        with col3:

            blood_pressure = st.number_input(
                "Blood Pressure (mm Hg)",
                min_value=0,
                max_value=200,
                value=70
            )

        with col4:

            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0,
                max_value=100,
                value=20
            )

        col5, col6, col7, col8 = st.columns(4)

        with col5:

            insulin = st.number_input(
                "Insulin (mu U/mL)",
                min_value=0,
                max_value=900,
                value=80
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
                "Diabetes Pedigree",
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                step=0.01
            )

        with col8:

            st.write("")
            st.write("")
            st.caption("8 diagnostic features used by the model.")

        st.divider()

        submitted = st.form_submit_button(
            "🔍 Assess Diabetes Risk",
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

            input_array = np.array(
                [[input_dict[c] for c in columns]]
            )

            input_scaled = scaler.transform(
                input_array
            )

            prediction = model.predict(
                input_scaled
            )[0]

            probability = model.predict_proba(
                input_scaled
            )[0][1]

            st.divider()

            st.markdown(
                '<div class="section-title">📊 Assessment Result</div>',
                unsafe_allow_html=True
            )

            result_col1, result_col2 = st.columns(
                [1.5, 1]
            )

            with result_col1:

                if prediction == 1:

                    st.markdown(
                        f"""
                        <div class="risk-high">

                        <div class="risk-title">
                        ⚠️ Higher Predicted Risk
                        </div>

                        <div class="risk-probability">
                        {probability:.1%}
                        </div>

                        <p>
                        Estimated model probability for
                        the positive diabetes class.
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f"""
                        <div class="risk-low">

                        <div class="risk-title">
                        ✅ Lower Predicted Risk
                        </div>

                        <div class="risk-probability">
                        {probability:.1%}
                        </div>

                        <p>
                        Estimated model probability for
                        the positive diabetes class.
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with result_col2:

                st.markdown(
                    '<div class="metric-card">',
                    unsafe_allow_html=True
                )

                st.metric(
                    "Model Probability",
                    f"{probability:.1%}"
                )

                st.progress(
                    min(max(float(probability), 0.0), 1.0)
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

            st.warning(
                "This result is a machine-learning prediction "
                "for academic screening purposes. It is not a "
                "medical diagnosis and should not replace "
                "professional clinical evaluation."
            )

            # =================================================
            # PATIENT SUMMARY
            # =================================================

            st.divider()

            st.markdown(
                '<div class="section-title">📋 Patient Summary</div>',
                unsafe_allow_html=True
            )

            summary1, summary2, summary3, summary4 = st.columns(4)

            with summary1:

                st.metric(
                    "Patient ID",
                    patient_id if patient_id else "Not provided"
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

            with st.expander(
                "View complete measurement summary"
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
                            age
                        ]
                    }
                )

# =========================================================
# PATIENT INFORMATION
# =========================================================

elif page == "📋 Patient Information":

    st.title("📋 Patient Information")

    st.write(
        "The prediction system uses eight numerical features "
        "from the Pima Indians Diabetes Dataset."
    )

    feature_info = {

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

    st.table(feature_info)

    st.divider()

    st.subheader("🔄 Assessment Workflow")

    st.markdown("""
    **1. Patient measurements**

    The user enters the available diagnostic measurements.

    **2. Data preparation**

    The values are arranged in the same feature order
    used during model training.

    **3. Feature scaling**

    The saved StandardScaler transforms the measurements.

    **4. Machine-learning prediction**

    The saved tuned model generates the predicted class
    and probability.

    **5. Risk presentation**

    The application presents the model result clearly
    for educational screening purposes.
    """)

# =========================================================
# MODEL & TECHNOLOGY
# =========================================================

elif page == "🤖 Model & Technology":

    st.title("🤖 Model & Technology")

    model_name = type(model).__name__

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">

            <h3>Algorithm</h3>

            <h2>{model_name}</h2>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="metric-card">

            <h3>Problem Type</h3>

            <h2>Binary Classification</h2>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="metric-card">

            <h3>Input Features</h3>

            <h2>8</h2>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("🧠 Machine Learning Pipeline")

    st.code("""
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
SMOTE on Training Data
            ↓
Model Comparison
            ↓
Random Forest Hyperparameter Tuning
            ↓
Final Model
            ↓
Joblib Model Export
            ↓
Streamlit Deployment
    """)

    st.subheader("🛠️ Technologies")

    technologies = [
        "Python",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "XGBoost",
        "imbalanced-learn / SMOTE",
        "Joblib",
        "Jupyter Notebook",
        "Streamlit",
        "GitHub"
    ]

    tech_cols = st.columns(2)

    for i, tech in enumerate(technologies):

        with tech_cols[i % 2]:

            st.markdown(
                f"• **{tech}**"
            )

    st.divider()

    st.subheader("📦 Model Files")

    st.code("""
models/
├── best_model.pkl
├── scaler.pkl
└── feature_columns.pkl
    """)

# =========================================================
# ABOUT DIABETES
# =========================================================

elif page == "📚 About Diabetes":

    st.title("📚 About Diabetes")

    st.subheader("What is diabetes?")

    st.write(
        "Diabetes is a chronic metabolic condition associated "
        "with elevated blood glucose levels. Diabetes management "
        "and diagnosis require appropriate clinical assessment."
    )

    st.subheader("Why prediction models can be useful")

    st.write(
        "Machine-learning models can identify patterns in "
        "historical data and provide risk estimates that may "
        "be useful for educational screening and research."
    )

    st.subheader("Dataset")

    st.write(
        "This project uses the Pima Indians Diabetes Dataset, "
        "containing 768 records with eight numerical diagnostic "
        "features and a binary Outcome variable."
    )

    st.subheader("Important limitations")

    st.markdown("""
    - The dataset is relatively small.
    - The dataset represents a specific population.
    - Machine-learning predictions may contain errors.
    - Model probability is not equivalent to clinical probability.
    - This application has not been validated as a medical device.
    - A healthcare professional should perform actual diagnosis.
    """)

    st.divider()

    st.error(
        "⚠️ MEDICAL DISCLAIMER\n\n"
        "This application is an academic machine-learning "
        "project. It is not intended to diagnose, treat, cure, "
        "or prevent diabetes or any other medical condition. "
        "Do not make healthcare decisions based solely on this "
        "application."
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    🏥 Diabetes Risk Assessment System |
    Academic Machine Learning Project

    <br>

    For educational and research purposes only.

    </div>
    """,
    unsafe_allow_html=True
)
```
