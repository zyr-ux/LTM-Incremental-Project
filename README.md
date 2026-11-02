# RetailMind AI

RetailMind AI is an end-to-end retail AI platform built incrementally across 35 engineering and data science modules. The platform introduces applied machine learning, deep learning, operational tooling, and scalable system design to solve high-impact retail problems—including customer churn prediction, demand forecasting, inventory optimization, and customer personalization.

Daily progress, incremental architecture notes, and iteration experiments are tracked in the [Developer Logs Index](docs/devlogs/INDEX.md).

---

## 🎯 Customer Churn Prediction Module

The initial platform service establishes RetailMind's machine learning backbone by shipping a working, persisted customer churn prediction model exposed as an inference service to flag at-risk customers proactively.

- **Dataset**: [Telco Customer Churn (WA_Fn-UseC_-Telco-Customer-Churn.csv)](data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv)
- **Core Libraries**: `pandas`, `scikit-learn`, `joblib`
- **Workflow**: Ingest $\rightarrow$ Preprocess $\rightarrow$ Train Baseline $\rightarrow$ Evaluate $\rightarrow$ Persist pipeline $\rightarrow$ Serve predictions.

---

## 🏗 Repository Structure

```text
LTM Incremental Project/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── devlogs/
├── models/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   └── service.py
├── .gitignore
└── README.md
```

---

## 📊 Baseline Model Performance

*Baseline classifier: Logistic Regression (`class_weight='balanced'`, `random_state=42`)*  
*Evaluated on 20% stratified holdout test set ($N=1409$)*

| Metric | Baseline Score | Notes |
| :--- | :---: | :--- |
| **Accuracy** | **73.81%** | Overall classification accuracy |
| **Precision (Churn=1)** | **50.43%** | Precision on positive churn class |
| **Recall (Churn=1)** | **78.34%** | High retention catch rate for at-risk customers |
| **F1-Score (Churn=1)** | **61.36%** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | **84.15%** | Strong threshold-independent discriminative ability |

> For detailed metric interpretations, classification reports, and iterative findings, see the [Developer Logs Index](docs/devlogs/INDEX.md).

---

## 🚀 Quickstart & Usage

### 1. Requirements
Install the core dependencies:
```bash
pip install pandas scikit-learn joblib
```

### 2. Run Training & Evaluation Pipeline
Ensure `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in `data/raw/` and execute:
```bash
python src/pipeline.py
```
This script will:
1. Ingest and sanitize the data (handling missing whitespace values in `TotalCharges`).
2. Save cleaned data to `data/processed/telco_churn_clean.csv`.
3. Train-test split using stratification.
4. Apply preprocessing (`OneHotEncoder` for categoricals, `StandardScaler` for numerical features).
5. Train the baseline Logistic Regression model.
6. Print evaluation metrics and persist the full pipeline to `models/churn_pipeline_baseline.joblib`.

### 3. Inference via Churn Service
Use the standalone service helper in `src/service.py`:
```python
from src.service import predict_churn

sample_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

result = predict_churn(sample_customer)
print(result)
# Output: {'churn_prediction': 1, 'churn_probability': 0.8048, 'risk_level': 'High'}
```
