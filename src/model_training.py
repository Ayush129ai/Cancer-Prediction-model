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
import mlflow
import mlflow.sklearn

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train multiple models, log with MLflow, and evaluate the best one."""
    mlflow.set_experiment("Breast_Cancer_Prediction")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=f"Train_{name}"):
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            mean_score = np.mean(scores)
            print(f"{name} CV AUC: {mean_score:.4f}")
            
            mlflow.log_param("model_type", name)
            mlflow.log_metric("cv_auc", mean_score)

            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("roc_auc", auc)
            
            # Log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure()
            sns.heatmap(cm, annot=True, fmt='d')
            plt.title(f'Confusion Matrix - {name}')
            cm_path = f'models/cm_{name.replace(" ", "_")}.png'
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)

            # Log model to MLflow (which gives us model versioning)
            mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=f"BreastCancer_{name.replace(' ', '')}")

            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_name = name

    print(f"Best model: {best_name}")
    # Also save best locally for existing compat
    joblib.dump(best_model, 'models/best_model.joblib')
    import pickle
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
        
    return best_model

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    from feature_selection import select_features
    X, y = load_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(X, y)
    X_train_selected, selected_features = select_features(X_train, y_train, feature_names)
    # Apply to test
    selector = joblib.load('models/feature_selector.joblib')
    X_test_selected = selector.transform(X_test)
    
    # Train and evaluate with MLflow
    model = train_and_evaluate(X_train_selected, y_train, X_test_selected, y_test)