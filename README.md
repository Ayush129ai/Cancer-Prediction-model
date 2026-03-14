# Cancer Prediction Model

This project builds an advanced machine learning model to predict cancer diagnosis using the Breast Cancer Wisconsin dataset with engineered features. The model uses multiple algorithms including Logistic Regression, Random Forest, SVM, and XGBoost, with feature selection via LASSO and additional feature engineering.

## Features

- Data preprocessing with feature engineering (compactness, symmetry ratios, etc.)
- Exploratory Data Analysis (EDA) with visualizations
- Feature selection using LASSO regression
- Model training and evaluation with cross-validation
- Multiple models: Logistic Regression, Random Forest, SVM, XGBoost
- Advanced evaluation: Accuracy, Precision, Recall, F1-Score, ROC AUC, Calibration Plots
- Model interpretability using SHAP
- Web interface using Streamlit for predictions

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd cancer-prediction-model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1) Set up Environment

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Train / Save Model

Run the training script to create the model and preprocessing artifacts:

```bash
python train_model.py
```

This will create the following files under `models/`:
- `best_model.joblib` (and `best_model.pkl`)
- `scaler.joblib`
- `feature_selector.joblib`

### 3) Run FastAPI Backend

Start the backend API on port 8000:

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

The API endpoint is:
- `POST /predict` (expects JSON `{ "features": [...], "user_id": "optional" }`)

Example curl test (replace `...` with actual feature values):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [...], "user_id": "tester"}'
```

### 4) Run Streamlit Frontend

Start the Streamlit app:

```bash
streamlit run app/app.py
```

In the Streamlit UI you can toggle between running predictions locally (direct model inference) and calling the backend API.

### 5) Run Desktop GUI (Tkinter)

Start the desktop GUI, which allows you to enter feature values and get a prediction without opening a browser:

```bash
python gui.py
```

This GUI uses the same trained model, scaler, and feature selector as the Streamlit app.

### 5) (Optional) Using Supabase Logging

1. Copy `.env.example` to `.env` and set your Supabase URL and service key.
2. The backend will automatically log predictions to a `predictions` table if `user_id` is provided.

### 6) Docker (Optional)

Run the full stack using Docker Compose:

```bash
docker compose up --build
```

- API available at `http://localhost:8000`
- Streamlit UI available at `http://localhost:8501`

## EDA Notebook

Open `notebooks/eda.ipynb` in Jupyter to explore the data and models.

## Project Structure

- `data/`: Raw and processed data
- `src/`: Source code for preprocessing, training, and evaluation
- `models/`: Saved trained models and plots
- `notebooks/`: Jupyter notebooks for EDA
- `app/`: Streamlit web application
- `backend/`: FastAPI backend and Supabase integration

## Extras (Optional Enhancements)

- **User authentication**: Add login via Supabase Auth (JWT) and enforce protected API routes.
- **Explainability**: The Streamlit UI uses SHAP for tabular explainability. For image models, implement Grad-CAM.
- **Admin dashboard**: Build a separate Streamlit page or React dashboard that queries Supabase prediction logs.
- **PDF reporting**: Use `reportlab` or `pdfkit` to generate PDF summaries of predictions.
- **Docker**: Use `docker compose up --build` to run both API and UI containers.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC
- Confusion Matrix
- Calibration Plot

## Clinical Impact

This model can help prioritize patients for screening and guide treatment decisions by providing probabilistic predictions of breast cancer malignancy based on clinical and engineered features. The calibration plot ensures reliable probability estimates for clinical use.

## License

MIT License
- `notebooks/`: Jupyter notebooks for EDA
- `app/`: Streamlit web application

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC

## Clinical Impact

This model can assist in prioritizing patients for screening or guiding treatment decisions by providing probabilistic predictions of cancer diagnosis based on clinical features.

## License

MIT License