import os
import sys
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import sqlite3
import json

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_preprocessing import load_data, feature_engineering

def generate_drift_report():
    # 1. Load reference (training) data distribution
    print("Loading reference data...")
    X, _ = load_data()
    X_engineered = feature_engineering(X)
    reference_data = X_engineered

    # 2. Load current (production) data from SQLite
    db_path = os.path.join(BASE_DIR, 'flask_ui', 'predictions.db')
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}. No production data to analyze.")
        return

    print("Loading production data...")
    conn = sqlite3.connect(db_path)
    
    # Read predictions history
    # We stored features as JSON string in 'features' column
    df_history = pd.read_sql_query("SELECT features FROM prediction_history", conn)
    conn.close()

    if df_history.empty:
        print("No prediction data collected yet.")
        return

    # Extract JSON into dataframe
    current_data = pd.DataFrame([json.loads(x) for x in df_history['features']])
    
    # Ensure current data only has columns that match reference
    cols = [col for col in reference_data.columns if col in current_data.columns]
    reference_data = reference_data[cols]
    current_data = current_data[cols]

    # 3. Generate Report
    print(f"Comparing Reference ({len(reference_data)} rows) vs Current ({len(current_data)} rows)...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data, current_data=current_data)
    
    # 4. Save report
    report_path = os.path.join(BASE_DIR, 'reports')
    os.makedirs(report_path, exist_ok=True)
    out_file = os.path.join(report_path, 'data_drift_report.html')
    drift_report.save_html(out_file)
    print(f"Drift report generated and saved to {out_file}")

if __name__ == "__main__":
    generate_drift_report()