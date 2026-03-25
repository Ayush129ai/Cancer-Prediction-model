import os
import sys
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from sqlalchemy import desc

# Add root folder to sys.path so we can import from src
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_preprocessing import load_data, feature_engineering
from flask_ui.models import db, PredictionHistory
from prometheus_flask_exporter import PrometheusMetrics
import shap

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Configurations
app.config['SECRET_KEY'] = 'dev-secret-key-12345'
db_path = os.path.join(os.path.dirname(__file__), 'predictions.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

@app.route('/health')
def health():
    return {'status': 'healthy', 'dependencies': 'ok'}, 200

@app.route('/ready')
def ready():
    # If the model is loaded, we are ready to serve traffic
    if model is not None:
        return {'status': 'ready'}, 200
    return {'status': 'not ready'}, 503

# --- ML Artifact Loading ---
models_dir = os.path.join(BASE_DIR, 'models')
model = joblib.load(os.path.join(models_dir, 'best_model.joblib'))
scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
selector = joblib.load(os.path.join(models_dir, 'feature_selector.joblib'))

# Get features metadata
X, y = load_data()
X_engineered = feature_engineering(X)
FEATURE_NAMES = X_engineered.columns.tolist()
# Identifying which ones are selected for showing in form if needed
SELECTED_FEATURES = np.array(FEATURE_NAMES)[selector.get_support()].tolist()
DEFAULTS = X_engineered.mean().to_dict()

# Calculate bounds. We use min/max of the dataset, and give ~10% padding so we don't overly strict users 
# on extreme but theoretically possible values, while clamping lower bounds to 0 for physical measurements.
MIN_VALS = {feat: max(0.0, float(X_engineered[feat].min() * 0.9)) for feat in FEATURE_NAMES}
MAX_VALS = {feat: float(X_engineered[feat].max() * 1.1) for feat in FEATURE_NAMES}

# Initialize SHAP explainer
background_data = selector.transform(scaler.transform(X_engineered.values[:100]))
explainer = shap.Explainer(model, background_data)

# Initialize DB
with app.app_context():
    db.create_all()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Reconstruct full feature array using defaults initially
            input_features = DEFAULTS.copy()
            
            # The form should provide the inputs for SELECTED_FEATURES
            for feature in SELECTED_FEATURES:
                val_str = request.form.get(feature)
                if val_str is not None and val_str.strip() != "":
                    input_features[feature] = float(val_str)
            
            # Create ordered array
            feature_vector = np.array([input_features[name] for name in FEATURE_NAMES]).reshape(1, -1)
            
            # Transform
            x_scaled = scaler.transform(feature_vector)
            x_selected = selector.transform(x_scaled)
            
            # Predict
            proba = model.predict_proba(x_selected)[0]
            pred = int(model.predict(x_selected)[0])
            
            label = "No Cancer" if pred == 1 else "Cancer"
            prob_no_cancer = float(proba[1])
            prob_cancer = float(proba[0])
            
            # --- SHAP Explainability ---
            shap_values = explainer(x_selected)
            shaps = shap_values.values[0]
            # Some explainers return shape (features, classes), just take the positive class ( कैंसर / 0 is malignant usually in breast_cancer, let's just take the main output )
            if len(shaps.shape) > 1:
                shaps = shaps[:, 0]  # Assuming class 0 is cancer

            feature_impacts = []
            for i, feat in enumerate(SELECTED_FEATURES):
                feature_impacts.append((feat, float(shaps[i])))
            
            # Sort by absolute magnitude and get top 3
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            top_features = feature_impacts[:3]
            explanation_str = ", ".join([f"{f} ({'+' if v>0 else ''}{v:.2f})" for f, v in top_features])

            # Save to Database
            history_record = PredictionHistory(
                user_id=request.form.get('user_id', 'Anonymous'),
                prediction=pred,
                probability_no_cancer=prob_no_cancer,
                probability_cancer=prob_cancer,
                label=label
            )
            history_record.set_features(input_features)
            db.session.add(history_record)
            db.session.commit()
            
            flash(f"Prediction successful! Label: {label} "
                  f"(Cancer Probability: {prob_cancer:.2%}) | Top driving factors: {explanation_str}", "success")
            return redirect(url_for('index'))
            
        except Exception as e:
            flash(f"Error processing prediction: {str(e)}", "danger")

    # Fetch history
    history = PredictionHistory.query.order_by(desc(PredictionHistory.timestamp)).all()
    
    return render_template(
        'index.html',
        features=SELECTED_FEATURES,
        defaults=DEFAULTS,
        feature_mins=MIN_VALS,
        feature_maxs=MAX_VALS,
        history=history
    )

if __name__ == '__main__':
    app.run(debug=False, port=5000)
