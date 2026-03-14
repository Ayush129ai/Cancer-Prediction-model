from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
import numpy as np
import joblib

def select_features(X_train, y_train, feature_names):
    """Perform feature selection using LASSO."""
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_train, y_train)

    print("Best alpha:", lasso.alpha_)

    # Select features
    selector = SelectFromModel(lasso, prefit=True)
    X_train_selected = selector.transform(X_train)
    selected_features = feature_names[selector.get_support()]

    print("Selected features:", selected_features)
    print("Number of selected features:", len(selected_features))

    # Save selector
    joblib.dump(selector, 'models/feature_selector.joblib')

    return X_train_selected, selected_features

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    X, y = load_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(X, y)
    X_train_selected, selected_features = select_features(X_train, y_train, feature_names)