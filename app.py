```python
import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# Model files are inside the "models" folder
# located in the same folder as this app.py file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, scaler, columns


st.title("🩺 Diabetes Risk Predictor")

st.write(
    "Enter the patient's diagnostic measurements below. "
    "The model predicts the likelihood of diabetes based on "
    "the Pima Indians Diabetes Dataset."
)

st.caption(
    "⚠️ This is an academic project, not a medical device. "
    "Do not use for actual diagnosis."
)

try:
    model, scaler, columns = load_artifacts()

except FileNotFoundError as e:
    st.error("Model files could not be found.")
    st.write("Please make sure these files exist inside the models folder:")
    st.code(
        "models/best_model.pkl\n"
        "models/scaler.pkl\n"
        "models/feature_columns.pkl"
    )
    st.write(f"Error: {e}")
    st.stop()


with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

        glucose = st.number_input(
            "Glucose Level (mg/dL)",
            min_value=0,
            max_value=300,
            value=120
        )

        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)",
            min_value=0,
            max_value=200,
            value=70
        )

        skin_thickness = st.number_input(
            "Skin Thickness (mm)",
            min_value=0,
            max_value=100,
            value=20
        )

    with col2:
        insulin = st.number_input(
            "Insulin (mu U/mL)",
            min_value=0,
            max_value=900,
            value=80
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=28.0,
            step=0.1
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=33,
            step=1
        )

    submitted = st.form_submit_button("Predict")


if submitted:

    input_dict = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    input_arr = np.array([
        [input_dict[c] for c in columns]
    ])

    input_scaled = scaler.transform(input_arr)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()

    if prediction == 1:

        st.error(
            f"⚠️ Higher risk of diabetes — "
            f"estimated probability: **{probability:.1%}**"
        )

    else:

        st.success(
            f"✅ Lower risk of diabetes — "
            f"estimated probability: **{probability:.1%}**"
        )

    st.progress(
        min(int(probability * 100), 100)
    )

    st.caption(
        "Probability reflects the model's prediction, "
        "not a clinical diagnosis."
    )
```
