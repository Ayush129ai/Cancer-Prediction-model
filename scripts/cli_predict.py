"""CLI tool to make a single cancer prediction from user-entered feature values.

This script mirrors the Streamlit app input process:
- Loads feature names from training data
- Loads the scaler, selector, and model artifacts
- Prompts the user for values for the selected features
- Builds a full feature vector using mean values for non-selected features
- Prints a clear prediction result (Cancer / No Cancer + probabilities)

Usage:
    python scripts/cli_predict.py
"""

import os
import sys

# Ensure repository root is on sys.path so `src` can be imported when running as a script.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import joblib
import numpy as np

from src.data_preprocessing import load_data, feature_engineering


def _prompt_float(prompt: str, default: float) -> float:
    """Prompt the user for a float value, with a default."""
    while True:
        try:
            raw = input(f"{prompt} [{default:.4f}]: ").strip()
            if raw == "":
                return default
            return float(raw)
        except ValueError:
            print("Invalid value, please enter a number.")


def main():
    # Load preprocessing artifacts and model
    try:
        model = joblib.load('models/best_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        selector = joblib.load('models/feature_selector.joblib')
    except FileNotFoundError as e:
        print(f"Missing artifact: {e}")
        print("Run `python train_model.py` first to generate model artifacts.")
        sys.exit(1)

    # Get feature names from training data
    X, _ = load_data()
    X_engineered = feature_engineering(X)
    feature_names = X_engineered.columns.tolist()

    # Determine which features the selector uses
    selected_mask = selector.get_support()
    selected_features = np.array(feature_names)[selected_mask].tolist()

    print("\n=== Cancer Prediction CLI ===\n")
    print("Enter values for the selected features below (press Enter to use the default mean value).\n")

    # Use mean values for all features; override with user input for selected features
    full_row = X_engineered.mean().to_dict()

    for feat in selected_features:
        full_row[feat] = _prompt_float(feat, full_row[feat])

    # Build feature vector in the original order
    feature_vector = np.array([full_row[f] for f in feature_names]).reshape(1, -1)

    # Apply preprocessing and prediction
    x_scaled = scaler.transform(feature_vector)
    x_selected = selector.transform(x_scaled)

    proba = model.predict_proba(x_selected)[0]
    prob_no_cancer = float(proba[1])
    prob_cancer = float(proba[0])

    pred = int(model.predict(x_selected)[0])
    label = "No Cancer" if pred == 1 else "Cancer"

    print("\n=== Prediction ===")
    print(f"Label: {label}")
    print(f"No Cancer probability: {prob_no_cancer:.4f}")
    print(f"Cancer probability:    {prob_cancer:.4f}\n")


if __name__ == "__main__":
    main()
