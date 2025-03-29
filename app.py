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

# Input fields for all 23 features
st.header("Enter Customer Details")

limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=0, value=50000)
pay_0 = st.slider("PAY_0 (Most Recent Payment Status)", min_value=-2, max_value=8, value=0)
pay_2 = st.slider("PAY_2 (Payment Status 2 months ago)", min_value=-2, max_value=8, value=0)
pay_3 = st.slider("PAY_3 (Payment Status 3 months ago)", min_value=-2, max_value=8, value=0)
pay_4 = st.slider("PAY_4 (Payment Status 4 months ago)", min_value=-2, max_value=8, value=0)
pay_5 = st.slider("PAY_5 (Payment Status 5 months ago)", min_value=-2, max_value=8, value=0)
pay_6 = st.slider("PAY_6 (Payment Status 6 months ago)", min_value=-2, max_value=8, value=0)

bill_amt1 = st.number_input("BILL_AMT1 (Latest Bill Amount)", min_value=0, value=50000)
bill_amt2 = st.number_input("BILL_AMT2 (Bill Amount 2 months ago)", min_value=0, value=48000)
bill_amt3 = st.number_input("BILL_AMT3 (Bill Amount 3 months ago)", min_value=0, value=46000)
bill_amt4 = st.number_input("BILL_AMT4 (Bill Amount 4 months ago)", min_value=0, value=44000)
bill_amt5 = st.number_input("BILL_AMT5 (Bill Amount 5 months ago)", min_value=0, value=42000)
bill_amt6 = st.number_input("BILL_AMT6 (Bill Amount 6 months ago)", min_value=0, value=40000)

pay_amt1 = st.number_input("PAY_AMT1 (Last Payment Amount)", min_value=0, value=10000)
pay_amt2 = st.number_input("PAY_AMT2 (Payment 2 months ago)", min_value=0, value=9000)
pay_amt3 = st.number_input("PAY_AMT3 (Payment 3 months ago)", min_value=0, value=8000)
pay_amt4 = st.number_input("PAY_AMT4 (Payment 4 months ago)", min_value=0, value=7000)
pay_amt5 = st.number_input("PAY_AMT5 (Payment 5 months ago)", min_value=0, value=6000)
pay_amt6 = st.number_input("PAY_AMT6 (Payment 6 months ago)", min_value=0, value=5000)

# Create a dataframe with user input
input_data = pd.DataFrame([[
    limit_bal, pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
    bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
    pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6
]], columns=[
    "LIMIT_BAL", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
])

# Scale numerical features
input_data[input_data.columns] = scaler.transform(input_data)

# Make predictions
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# Display results
if prediction == 1:
    st.error(f"High Probability of Default: {probability:.2%}")
else:
    st.success(f"Low Probability of Default: {probability:.2%}")
