"""Demonstrate a prediction run with explicit user-provided feature values.

This script mimics a user entering feature values (partial overrides of the dataset mean)
and prints the model output as a clear "Cancer / No Cancer" recommendation.

Run:
    python scripts/demo_prediction.py
"""

import os
import sys

# Ensure project root is on sys.path so `src` can be imported when run as a script.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import joblib
import numpy as np

from src.data_preprocessing import load_data, feature_engineering


def main():
    # Load model + preprocess artifacts
    model = joblib.load('models/best_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    selector = joblib.load('models/feature_selector.joblib')

    # Get feature order and defaults
    X, y = load_data()
    X_eng = feature_engineering(X)
    feature_names = X_eng.columns.tolist()
    defaults = X_eng.mean().to_dict()

    # Simulate user input: start from defaults, then override a few values
    user_input = defaults.copy()
    # Example user changes (these represent the "user added details")
    user_input['mean radius'] = defaults['mean radius'] * 1.10
    user_input['mean texture'] = defaults['mean texture'] * 0.85
    user_input['worst concavity'] = defaults['worst concavity'] * 1.20

    print("User-provided feature values (sample override):")
    for k in ['mean radius', 'mean texture', 'worst concavity']:
        print(f"  {k}: {user_input[k]:.4f}")
    print("\nRunning prediction...\n")

    # Build feature vector in correct order
    feature_vector = np.array([user_input[name] for name in feature_names]).reshape(1, -1)

    # Apply preprocessing
    x_scaled = scaler.transform(feature_vector)
    x_selected = selector.transform(x_scaled)

    # Predict
    proba = model.predict_proba(x_selected)[0]
    pred = int(model.predict(x_selected)[0])

    label = "No Cancer" if pred == 1 else "Cancer"
    prob_no_cancer = proba[1]
    prob_cancer = proba[0]

    print("Model output:")
    print(f"  Suggested label: {label}")
    print(f"  Probability (No Cancer): {prob_no_cancer:.4f}")
    print(f"  Probability (Cancer):    {prob_cancer:.4f}")


if __name__ == "__main__":
    main()
