import os

import pandas as pd
import numpy as np
import requests
import joblib
import streamlit as st

try:
    import shap
except ImportError:
    shap = None

from src.data_preprocessing import load_data, feature_engineering

# Configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')

# Load data and engineer features
data_X, data_y = load_data()
X_engineered = feature_engineering(data_X)
FEATURE_NAMES = X_engineered.columns.tolist()

# Load model and components for local prediction path
model = joblib.load('models/best_model.joblib')
scaler = joblib.load('models/scaler.joblib')
selector = joblib.load('models/feature_selector.joblib')
SELECTED_FEATURES = np.array(FEATURE_NAMES)[selector.get_support()]

st.set_page_config(page_title="Cancer Prediction", layout="wide")

st.title("Cancer Prediction Model")

use_api = st.sidebar.checkbox("Use backend API (FastAPI)", value=False)

st.write("Enter the feature values to get a prediction.")

# Input fields for selected features
inputs = {}
for feature in SELECTED_FEATURES:
    inputs[feature] = st.number_input(f"{feature}", value=float(X_engineered[feature].mean()))

if st.button("Predict"):
    # Build full feature vector using defaults for unselected features
    full_row = X_engineered.mean().to_dict()
    full_row.update(inputs)
    features_list = [full_row[name] for name in FEATURE_NAMES]

    if use_api:
        try:
            resp = requests.post(
                f"{BACKEND_URL}/predict",
                json={"features": features_list},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            probability = data.get('probability')
            prediction = data.get('prediction')
        except Exception as e:
            st.error(f"Failed to call backend: {e}")
            st.stop()
    else:
        x = np.array(features_list).reshape(1, -1)
        x_scaled = scaler.transform(x)
        x_selected = selector.transform(x_scaled)
        probability = float(model.predict_proba(x_selected)[0][1])
        prediction = int(model.predict(x_selected)[0])

    # Interpret prediction in human-friendly terms
    cancer_label = "No Cancer" if prediction == 1 else "Cancer"
    cancer_prob = 1 - probability  # probability of the malignant class

    if prediction == 1:
        st.success(f"{cancer_label} (No Cancer probability: {probability:.2f}, Cancer probability: {cancer_prob:.2f})")
    else:
        st.error(f"{cancer_label} (Cancer probability: {cancer_prob:.2f}, No Cancer probability: {probability:.2f})")

    st.subheader("Feature Importance")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_selected)
        shap.summary_plot(shap_values, x_selected, feature_names=SELECTED_FEATURES, show=False)
        st.pyplot()
    except Exception:
        st.write("SHAP plot not available for this model type.")
