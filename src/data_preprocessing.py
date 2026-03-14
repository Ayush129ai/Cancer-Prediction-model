import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data():
    """Load the Breast Cancer Wisconsin dataset."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y

def feature_engineering(X):
    """Create additional features."""
    X_engineered = X.copy()
    X_engineered['mean compactness'] = X_engineered['mean perimeter']**2 / X_engineered['mean area']
    X_engineered['worst compactness'] = X_engineered['worst perimeter']**2 / X_engineered['worst area']
    X_engineered['mean symmetry ratio'] = X_engineered['mean symmetry'] / X_engineered['mean texture']
    X_engineered['worst symmetry ratio'] = X_engineered['worst symmetry'] / X_engineered['worst texture']
    X_engineered['fractal dim diff'] = X_engineered['worst fractal dimension'] - X_engineered['mean fractal dimension']
    X_engineered['mean concavity area ratio'] = X_engineered['mean concavity'] / X_engineered['mean area']
    X_engineered['worst concavity area ratio'] = X_engineered['worst concavity'] / X_engineered['worst area']
    return X_engineered

def preprocess_data(X, y):
    """Preprocess the data: feature engineering, scaling."""
    X_engineered = feature_engineering(X)
    X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, 'models/scaler.joblib')

    return X_train_scaled, X_test_scaled, y_train, y_test, X_engineered.columns

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(X, y)
    print("Data loaded and preprocessed.")
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)