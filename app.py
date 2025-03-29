import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os

# Google Drive file ID for credit card model
file_id = "1e3K7NAowcpfxWc7sfqpk3YHVeyzEDl4J"
model_filename = "credit_default_model.pkl"

# Check if the model file exists, otherwise download it
if not os.path.exists(model_filename):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_filename, quiet=False)

# Load the trained model
model = joblib.load(model_filename)

# Load the scaler (already available)
scaler = joblib.load("scaler.pkl")

# Define the Streamlit app
st.title("Credit Card Default Prediction")

# Input fields
limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=0, value=50000)
sex = st.selectbox("Sex", ["Male", "Female"])
education = st.selectbox("Education", ["Graduate School", "University", "High School", "Others"])
marriage = st.selectbox("Marriage", ["Married", "Single", "Others"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
pay_status = st.slider("Payment Status (Months Delayed)", min_value=-2, max_value=8, value=0)
bill_amount = st.number_input("Average Bill Amount", min_value=0, value=50000)
pay_amount = st.number_input("Average Payment Amount", min_value=0, value=10000)

# Preprocess input data
input_data = pd.DataFrame({
    "LIMIT_BAL": [limit_bal],
    "SEX": [1 if sex == "Male" else 2],
    "EDUCATION": [1 if education == "Graduate School" else 2 if education == "University" else 3 if education == "High School" else 4],
    "MARRIAGE": [1 if marriage == "Married" else 2 if marriage == "Single" else 3],
    "AGE": [age],
    "PAY_0": [pay_status],
    "avg_bill_amt": [bill_amount],
    "avg_pay_amt": [pay_amount],
    "bill_payment_diff": [bill_amount - pay_amount],
    "credit_utilization": [bill_amount / limit_bal]
})

# Scale numerical features
numerical_cols = ["LIMIT_BAL", "avg_bill_amt", "avg_pay_amt", "bill_payment_diff", "credit_utilization"]
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# Make predictions
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# Display results
if prediction == 1:
    st.error(f"High Probability of Default: {probability:.2%}")
else:
    st.success(f"Low Probability of Default: {probability:.2%}")
