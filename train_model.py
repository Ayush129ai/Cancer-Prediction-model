from src.data_preprocessing import load_data, preprocess_data
from src.feature_selection import select_features
from src.model_training import train_models, evaluate_model
import joblib

def main():
    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(X, y)

    # Feature selection
    X_train_selected, selected_features = select_features(X_train, y_train, feature_names)

    # Apply feature selection to test set
    selector = joblib.load('models/feature_selector.joblib')
    X_test_selected = selector.transform(X_test)

    # Train models
    model = train_models(X_train_selected, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_test_selected, y_test)

    print("Model training and evaluation complete.")

if __name__ == "__main__":
    main()