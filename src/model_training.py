from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.calibration import calibration_curve
import xgboost as xgb
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_models(X_train, y_train):
    """Train multiple models and select the best one."""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        mean_score = np.mean(scores)
        print(f"{name} CV AUC: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_model = model

    # Train best model on full training data
    best_model.fit(X_train, y_train)

    # Save model
    joblib.dump(best_model, 'models/best_model.joblib')

    # Also save as pickle for compatibility
    import pickle
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    print(f"Best model: {best_model.__class__.__name__}")
    return best_model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test set."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba)
    }

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Classification Report
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.savefig('models/confusion_matrix.png')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('models/roc_curve.png')
    plt.show()

    # Calibration Plot
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Plot')
    plt.savefig('models/calibration_plot.png')
    plt.show()

    return metrics

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    from feature_selection import select_features
    X, y = load_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(X, y)
    X_train_selected, selected_features = select_features(X_train, y_train, feature_names)
    # Apply to test
    selector = joblib.load('models/feature_selector.joblib')
    X_test_selected = selector.transform(X_test)
    model = train_models(X_train_selected, y_train)
    evaluate_model(model, X_test_selected, y_test)
    # Note: For simplicity, using selected features, but need to apply to test too
    # In full pipeline, apply selector to test
    selector = joblib.load('models/feature_selector.joblib')
    X_test_selected = selector.transform(X_test)
    model = train_models(X_train_selected, y_train)
    evaluate_model(model, X_test_selected, y_test)