from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import json

from backend.supabase_client import log_prediction
from src.data_preprocessing import load_data, feature_engineering

app = FastAPI(title="Cancer Prediction API")

# Load model and preprocessing objects
model = joblib.load('models/best_model.joblib')
scaler = joblib.load('models/scaler.joblib')
selector = joblib.load('models/feature_selector.joblib')

# Load feature names from training data
X, y = load_data()
X_engineered = feature_engineering(X)
FEATURE_NAMES = X_engineered.columns.tolist()
SELECTED_FEATURES = np.array(FEATURE_NAMES)[selector.get_support()].tolist()
DEFAULTS = X_engineered.mean().to_dict()

# Compute model metrics (on training data for demo, in practice use test set)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
X_scaled = scaler.transform(X_engineered)
X_selected = selector.transform(X_scaled)
y_pred = model.predict(X_selected)
y_proba = model.predict_proba(X_selected)[:, 1]
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc = roc_auc_score(y, y_proba)

class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="List of feature values matching the training order")
    user_id: Optional[str] = Field(None, description="Optional user id for logging")

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    label: str

@app.get('/', response_class=HTMLResponse)
def homepage():
    # Render a simple form to submit feature values and see predictions
    input_fields = "\n".join(
        f"<div><label>{feat}: <input name=\"{feat}\" type=\"number\" step=\"any\" value=\"0\" style=\"width:220px\"/></label></div>"
        for feat in SELECTED_FEATURES
    )

    html = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Cancer Prediction</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: #333;
    }}
    .container {{
      max-width: 800px;
      margin: 0 auto;
      background: rgba(255, 255, 255, 0.9);
      padding: 20px;
      border-radius: 10px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    h1 {{
      color: #4a4a4a;
      text-align: center;
    }}
    h2 {{
      color: #666;
      border-bottom: 2px solid #667eea;
      padding-bottom: 5px;
    }}
    input {{
      margin: 4px 0;
      padding: 8px;
      width: calc(100% - 16px);
      border: 1px solid #ccc;
      border-radius: 4px;
    }}
    button {{
      padding: 10px 16px;
      margin-top: 12px;
      background: #667eea;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 16px;
    }}
    button:hover {{
      background: #5a67d8;
    }}
    .result {{
      margin-top: 16px;
      font-weight: bold;
      padding: 10px;
      border-radius: 4px;
      background: #e8f5e8;
      border: 1px solid #4caf50;
    }}
    .metrics {{
      margin-top: 20px;
      padding: 10px;
      background: #f0f8ff;
      border-radius: 4px;
      border: 1px solid #2196f3;
    }}
    .metric {{
      display: inline-block;
      margin: 5px 10px;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Cancer Prediction</h1>
    <p>Fill in the values below, then click <strong>Predict</strong>.</p>
    <form id="predict-form">
      {input_fields}
      <button type="submit">Predict</button>
    </form>
    <div id="result" class="result"></div>

    <div class="metrics">
      <h2>Model Performance Metrics</h2>
      <div class="metric">Accuracy: {accuracy:.3f}</div>
      <div class="metric">Precision: {precision:.3f}</div>
      <div class="metric">Recall: {recall:.3f}</div>
      <div class="metric">F1-Score: {f1:.3f}</div>
      <div class="metric">ROC AUC: {auc:.3f}</div>
    </div>
  </div>

  <script>
    const defaults = {defaults_json};
    const featureNames = {feature_names_json};
    const form = document.getElementById('predict-form');
    const resultDiv = document.getElementById('result');

    form.addEventListener('submit', async (e) => {{
      e.preventDefault();
      const data = new FormData(form);
      const features = [];

      // Start with defaults for all features
      for (const name of featureNames) {{
        features.push(defaults[name] || 0);
      }}

      // Override with form values for selected features
      for (const [key, value] of data.entries()) {{
        const val = parseFloat(value);
        if (!isNaN(val)) {{
          const idx = featureNames.indexOf(key);
          if (idx !== -1) {{
            features[idx] = val;
          }}
        }}
      }}

      try {{
        const resp = await fetch('/predict', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ features }}),
        }});

        if (!resp.ok) {{
          const err = await resp.json();
          resultDiv.textContent = 'Error: ' + (err.detail || resp.statusText);
          return;
        }}

        const dataResp = await resp.json();
        resultDiv.textContent = `Prediction: ${{dataResp.label}} (No Cancer: ${{dataResp.probability.toFixed(4)}} / Cancer: ${{(1 - dataResp.probability).toFixed(4)}})`;
      }} catch (err) {{
        resultDiv.textContent = 'Request failed: ' + err.message;
      }}
    }});
  </script>
</body>
</html>
""".format(
    input_fields=input_fields,
    defaults_json=json.dumps(DEFAULTS),
    feature_names_json=json.dumps(FEATURE_NAMES),
    accuracy=accuracy,
    precision=precision,
    recall=recall,
    f1=f1,
    auc=auc
)

    return HTMLResponse(content=html, status_code=200)

@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != len(FEATURE_NAMES):
        raise HTTPException(status_code=400, detail=f"Expected {len(FEATURE_NAMES)} features")

    x = np.array(req.features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    x_selected = selector.transform(x_scaled)

    proba = model.predict_proba(x_selected)[0]
    prob_benign = float(proba[1])
    prob_cancer = float(proba[0])

    pred = int(model.predict(x_selected)[0])
    label = "No Cancer" if pred == 1 else "Cancer"

    if req.user_id:
        try:
            log_prediction(req.user_id, dict(zip(FEATURE_NAMES, req.features)), pred, prob_benign)
        except Exception:
            # Supabase logging is optional; do not fail prediction if logging fails
            pass

    return PredictResponse(prediction=pred, probability=prob_benign, label=label)
