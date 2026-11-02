# RetailMind AI

RetailMind AI is an end-to-end retail AI platform built incrementally across 35 engineering and data science modules. The platform introduces applied machine learning, deep learning, operational tooling, and scalable system design to solve high-impact retail problems—including customer churn prediction, demand forecasting, inventory optimization, and customer personalization.

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
│   ├── compare_models.py
│   └── service.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Model Evaluation & Benchmarking

All models are trained with identical preprocessing (`StandardScaler` + `OneHotEncoder`) on an 80/20 stratified train/test split.

### Multi-Model Comparison

| Model | Accuracy | Precision (Churn=1) | Recall (Churn=1) | F1-Score (Churn=1) | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 73.81% | 50.43% | **78.34%** | 61.36% | **84.15%** | Baseline reference |
| **Decision Tree** | 72.75% | 48.67% | 48.93% | 48.80% | 65.17% | High variance / overfits |
| **Random Forest** | **76.51%** | **55.13%** | 61.76% | 58.26% | 82.28% | Balanced ensemble |
| **SVM (RBF Kernel)** | 74.52% | 51.33% | **77.27%** | **61.69%** | 82.13% | **Champion Candidate** |

> **Champion Selection**: **SVM (RBF Kernel)** was selected as the non-linear champion for its high recall (**77.27%**) and strongest non-linear F1 score (**61.69%**), which minimizes missed customer churn opportunities.

---

## 📝 Developer Logs

Detailed iteration notes, architectural decisions, data audits, and experiment analyses are recorded daily.
Browse all daily logs in the [Devlogs Index](docs/devlogs/INDEX.md).

---

## 🚀 Quickstart & Usage

### 1. Environment Setup (venv)
Create and activate a virtual environment, then install the dependencies:

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Training & Evaluation Pipelines
- To train the baseline Logistic Regression model:
  ```bash
  python src/pipeline.py
  ```
- To train and compare Decision Tree, Random Forest, and SVM models:
  ```bash
  python src/compare_models.py
  ```
  This trains all three candidate models with consistent preprocessing, evaluates them against the holdout set, and saves the champion pipeline to `models/churn_pipeline_champion.joblib`.

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
