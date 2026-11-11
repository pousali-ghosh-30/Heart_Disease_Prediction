
import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("Logistic_Regression_heart.pkl")
preprocessor = joblib.load("preprocessor.pkl")
columns = joblib.load("columns.pkl")

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# Application heading
st.title("❤️ Heart Disease Prediction")
st.write("Enter the details below to get a model prediction.")

st.divider()

# Input form
with st.form("prediction_form"):

    st.subheader("Personal Information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=40
    )

    sex = st.selectbox("Sex", [0, 1])

    st.subheader("Medical Information")

    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])

    trtbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=600,
        value=200
    )

    fbs = st.selectbox("Fasting Blood Sugar", [0, 1])

    restecg = st.selectbox("Resting ECG", [0, 1, 2])

    thalachh = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

    exng = st.selectbox(
        "Exercise Induced Angina",
        [0, 1]
    )

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slp = st.selectbox("Slope", [0, 1, 2])

    caa = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3, 4]
    )

    thall = st.selectbox(
        "Thalassemia",
        [0, 1, 2, 3]
    )

    submit = st.form_submit_button("Predict")

# Prediction
if submit:

    input_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trtbps": trtbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalachh": thalachh,
        "exng": exng,
        "oldpeak": oldpeak,
        "slp": slp,
        "caa": caa,
        "thall": thall
    }

    # Arrange columns in the original training order
    input_df = pd.DataFrame([input_data])
    input_df = input_df[columns]

    # Apply the saved preprocessing
    processed_data = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(processed_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("The model predicts a high risk of heart disease.")
    else:
        st.success("The model predicts a low risk of heart disease.")

    st.caption(
        "Educational demonstration only. "
        "This prediction is not a medical diagnosis."
    )
