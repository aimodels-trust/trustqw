# -*- coding: utf-8 -*-
"""app.py"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# Title of the App
st.title("Trust & Transparency in Business Systems Using XAI")

# Sidebar for Domain Selection
domain = st.sidebar.selectbox("Select Domain", ["Finance", "Healthcare", "Customer Service"])

# Load Datasets
@st.cache_data
def load_data(domain):
    if domain == "Finance":
        # Load UCI Credit Card Default Dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
        data = pd.read_excel(url, skiprows=1, engine="xlrd")  # Use xlrd for .xls files
        return data

    elif domain == "Healthcare":
        # Load Heart Disease UCI Dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        data = pd.read_csv(url, names=columns, na_values="?")
        return data

    elif domain == "Customer Service":
        # Load Amazon Customer Reviews Dataset
        url = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
        data = pd.read_json(url, lines=True)
        return data

# Train a Simple Model
def train_model(data, target_column):
    if target_column not in data.columns:
        st.error(f"Target column '{target_column}' not found in the dataset.")
        return None, None, None, None, None

    # Drop rows with missing values
    data = data.dropna()

    # Split dataset
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

# Generate SHAP Explanations
def generate_shap_explanation(model, X_train, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values

# Generate LIME Explanations
def generate_lime_explanation(model, X_train, X_test, feature_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        np.array(X_train), feature_names=feature_names, class_names=["Class 0", "Class 1"], verbose=True, mode="classification"
    )
    explanation = explainer.explain_instance(np.array(X_test.iloc[0]), model.predict_proba, num_features=len(feature_names))
    return explanation

# Main Application Logic
if domain == "Finance":
    st.header("Finance: Credit Scoring and Fraud Detection")
    data = load_data("Finance")
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Train Model
    target_column = "default payment next month"
    model, X_train, X_test, y_train, y_test = train_model(data, target_column)

    if model is not None:
        # SHAP Explanation
        st.subheader("SHAP Explanation")
        explainer, shap_values = generate_shap_explanation(model, X_train, X_test)
        st.pyplot(shap.summary_plot(shap_values[1], X_test))  # Use shap_values[1] for binary classification

        # LIME Explanation
        st.subheader("LIME Explanation")
        feature_names = X_train.columns.tolist()
        lime_explanation = generate_lime_explanation(model, X_train, X_test, feature_names)
        st.pyplot(lime_explanation.as_pyplot_figure())

elif domain == "Healthcare":
    st.header("Healthcare: Disease Prediction")
    data = load_data("Healthcare")
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Train Model
    target_column = "target"
    model, X_train, X_test, y_train, y_test = train_model(data, target_column)

    if model is not None:
        # SHAP Explanation
        st.subheader("SHAP Explanation")
        explainer, shap_values = generate_shap_explanation(model, X_train, X_test)
        st.pyplot(shap.summary_plot(shap_values[1], X_test))  # Use shap_values[1] for binary classification

        # LIME Explanation
        st.subheader("LIME Explanation")
        feature_names = X_train.columns.tolist()
        lime_explanation = generate_lime_explanation(model, X_train, X_test, feature_names)
        st.pyplot(lime_explanation.as_pyplot_figure())

elif domain == "Customer Service":
    st.header("Customer Service: Sentiment Analysis")
    data = load_data("Customer Service")
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Preprocess Data (Example: Sentiment Analysis)
    data["sentiment"] = data["overall"].apply(lambda x: "Positive" if x > 3 else "Negative")
    target_column = "sentiment"

    # Train Model
    model, X_train, X_test, y_train, y_test = train_model(data, target_column)

    if model is not None:
        # SHAP Explanation
        st.subheader("SHAP Explanation")
        explainer, shap_values = generate_shap_explanation(model, X_train, X_test)
        st.pyplot(shap.summary_plot(shap_values[1], X_test))  # Use shap_values[1] for binary classification

        # LIME Explanation
        st.subheader("LIME Explanation")
        feature_names = X_train.columns.tolist()
        lime_explanation = generate_lime_explanation(model, X_train, X_test, feature_names)
        st.pyplot(lime_explanation.as_pyplot_figure())
