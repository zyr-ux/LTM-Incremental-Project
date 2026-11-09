# RetailMind AI

RetailMind AI is an end-to-end retail AI platform built incrementally across 35 engineering and data science modules. The platform introduces applied machine learning, deep learning, operational tooling, and scalable system design to solve high-impact retail problems—including customer churn prediction, demand forecasting, inventory optimization, and customer personalization.

---

## 🎯 Customer Churn Prediction Module

The initial platform service establishes RetailMind's machine learning backbone by shipping a working, persisted customer churn prediction model exposed as an inference service to flag at-risk customers proactively.

- **Dataset**: [Telco Customer Churn (WA_Fn-UseC_-Telco-Customer-Churn.csv)](data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv)
- **Core Libraries**: `pandas`, `scikit-learn`, `xgboost`, `joblib`
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
│   ├── churn_pipeline_baseline.joblib
│   ├── churn_pipeline_champion.joblib
│   ├── churn_v3.joblib
│   └── churn_v4.joblib
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── compare_models.py
│   ├── tune_and_diagnose.py
│   ├── feature_engineering_and_ensembles.py
│   └── service.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Model Evaluation & Benchmarking

All models are trained with identical preprocessing (`StandardScaler` + `OneHotEncoder`) on an 80/20 stratified train/test split.

### Multi-Model Comparison (Day 2 Untuned vs Day 3 Tuned)

#### Day 3: Systematic Hyperparameter Tuning (`GridSearchCV` 5-Fold Cross-Validation)

| Model | CV ROC-AUC | Train ROC-AUC | Generalization Gap | Optimal Hyperparameters | Bias / Variance Diagnosis |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Logistic Regression** | 0.8464 | 0.8500 | **0.0036** | `C=10.0`, `class_weight=None` | Minimal variance; slight linear bias |
| **Decision Tree** | 0.8293 | 0.8541 | 0.0248 | `max_depth=5`, `min_samples_split=20` | Depth constraint contained overfit |
| **Random Forest (Champion)** | **0.8474** | 0.9057 | 0.0583 | `n_estimators=200`, `max_depth=8`, `min_samples_split=10` | **Tuned Champion**: Best discriminative AUC |
| **SVM (RBF Kernel)** | 0.8423 | 0.8488 | 0.0065 | `C=1.0`, `gamma=0.01`, `class_weight='balanced'` | Strong generalization; balanced margins |

#### Holdout Test Set Benchmark ($N = 1,409$)

| Model | Accuracy | Precision (Churn=1) | Recall (Churn=1) | F1-Score (Churn=1) | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 73.81% | 50.43% | 78.34% | 61.36% | 84.15% | Day 1 Reference |
| **Logistic Regression (Tuned)** | 80.41% | 65.41% | 55.61% | 60.12% | 84.12% | Tuned Linear |
| **Decision Tree (Tuned)** | 79.84% | 63.47% | 56.68% | 59.89% | 82.97% | Pruned Tree |
| **Random Forest (Tuned)** | 80.13% | 66.91% | 49.73% | 57.06% | 84.33% | Day 3 Champion (`churn_v3.joblib`) |
| **SVM (RBF Kernel, Tuned)** | 70.48% | 46.80% | **82.09%** | 59.61% | 83.34% | High Recall Specialist |
| **Random Forest (Day 4 FE)** | 79.84% | 65.73% | 50.27% | 56.97% | 84.29% | Bagging + Domain FE |
| **Random Forest + PCA (15 comps)** | 80.06% | 67.82% | 47.33% | 55.75% | 84.14% | Bagging + PCA Reduction |
| **XGBoost + PCA (15 comps)** | 81.05% | **68.38%** | 53.21% | **59.85%** | 84.77% | Boosting + PCA Reduction |
| **XGBoost (Day 4 FE Champion)** | **80.55%** | 67.48% | 51.60% | 58.48% | **84.82%** | **Phase-1 Production Champion (`churn_v4.joblib`)** |
| **Day 5 ANN (Keras Feed-Forward)** | 78.85% | 60.67% | 57.75% | 59.18% | 83.95% | Deep Learning Propensity Model (`churn_ann_v5.keras`) |

> **Phase-1 Production Champion Lock**: **XGBoost** with engineered domain features (`n_estimators=100`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`) is locked as the **Day 4 & Phase-1 deliverable champion** (Milestone Assessment 2) and persisted to `models/churn_v4.joblib` (and active `models/churn_pipeline_champion.joblib`). It delivers the highest holdout ROC-AUC (**84.82%**) and 5-fold CV ROC-AUC (**0.8495**), with a minimal generalization gap ($0.0198$), superior precision ($67.48\%$), and fully frozen end-to-end preprocessing.

---

## 🧠 Deep Learning vs Classical ML: When to Use Which?

In Day 5, we trained a feed-forward Artificial Neural Network (Keras `Dense` layers with ReLU, Dropout, and Sigmoid output) and evaluated it directly against the Phase-1 XGBoost champion:

| Decision Factor | Classical ML (XGBoost / LightGBM) | Deep Learning (ANNs / DNNs) |
| :--- | :--- | :--- |
| **Tabular Data Performance** | **Superior** (ROC-AUC 84.82% vs 83.95%). Tree algorithms partition tabular, heterogeneous feature spaces natively. | Competitive but susceptible to plateauing without massive sample volume. |
| **Sample Efficiency** | High; strong performance on 5,000–50,000 rows. | Lower; requires larger datasets ($>10^5-10^6$ rows) to outperform boosting. |
| **Training & Hyperparameters** | Fast, robust convergence with standard hyperparameter grids. | Sensitive to architecture, learning rates, batch size, and regularization. |
| **Multimodal & Unstructured** | Weak; cannot natively process raw images, text, or graph embeddings without heavy feature extraction. | **Dominant**; seamlessly combines tabular metadata, text tokens, embeddings, and time series sequences. |
| **Recommendation for RetailMind AI** | **Keep XGBoost as Phase-1 Champion** for tabular churn prediction. | **Deploy Neural Networks** when RetailMind expands to unstructured multimodal data (product imagery, customer review NLP, clickstream sequence embeddings). |

---

## 📈 Learning Curve Diagnostics (Bias vs Variance)

A 5-fold cross-validated learning curve analysis on the locked champion confirms model behavior as sample size scales:

| Training Samples | Mean Train ROC-AUC | Mean Validation ROC-AUC | Generalization Gap |
| :---: | :---: | :---: | :---: |
| **563** | 0.9584 | 0.8295 | 0.1289 |
| **1,831** | 0.9385 | 0.8428 | 0.0957 |
| **3,098** | 0.9192 | 0.8434 | 0.0758 |
| **4,366** | 0.9059 | 0.8439 | 0.0620 |
| **5,634** | **0.8985** | **0.8458** | **0.0528** |

**Diagnostic Takeaway**:
- The generalization gap contracts significantly (from $0.1289 \rightarrow 0.0528$) as sample size scales.
- Validation score plateaus near $\approx 0.846$, showing that further improvements require **feature engineering** (Day 4+) rather than simply accumulating more rows.

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
  This trains candidate models with consistent preprocessing, evaluates them against the holdout set, and saves the candidate pipeline.
- To execute Day 3 Multi-Model Grid Search, Bias-Variance Diagnostics, and Learning Curves:
  ```bash
  python src/tune_and_diagnose.py
  ```
  This tunes hyperparameter grids across all models using 5-fold cross-validation, runs learning curve diagnostics, and locks the champion model to `models/churn_v3.joblib`.
- To execute Day 4 Feature Engineering, PCA Analysis, Ensemble Boosting/Bagging & Finalize Phase-1 Model:
  ```bash
  python src/feature_engineering_and_ensembles.py
  ```
  This computes tenure buckets, financial charge velocity ratios, and contract/service interaction flags, benchmarks PCA dimensionality reduction against bagging and gradient boosting, and locks the finalized production model to `models/churn_v4.joblib` and `models/churn_pipeline_champion.joblib`.
- To run Day 5 Keras MNIST Warm-Up Demo:
  ```bash
  python src/mnist_demo.py
  ```
- To train Day 5 Feed-Forward ANN & Benchmark Against Phase-1 Champion:
  ```bash
  python src/ann_churn_model.py
  ```
  This builds a normalized feed-forward neural network with early stopping, saves `models/churn_ann_v5.keras`, and runs a head-to-head benchmark against the Phase-1 champion.
- To run Day 6 Fashion-MNIST CNN Prototype:
  ```bash
  python src/fashion_mnist_cnn.py
  ```
- To train Day 6 Catalog Image Classification CNN & Confusion Matrix:
  ```bash
  python src/catalog_cnn.py
  ```
  This trains the image classifier with real-time data augmentation on product imagery, saves `models/product_catalog_cnn_v6.keras`, and outputs confusion matrix diagnostics.

---

## 🖼 Product Catalog Computer Vision Module (Day 6)

Day 6 introduces automated catalog tagging to reduce manual entry and power downstream visual search.

- **Architecture**: Conv2D ($3\times3$) $\rightarrow$ MaxPool $\rightarrow$ Conv2D ($3\times3$) $\rightarrow$ MaxPool $\rightarrow$ Dropout ($0.25$) $\rightarrow$ Dense ($64$) $\rightarrow$ Dense ($4$, Softmax)
- **Data Augmentation**: `RandomFlip`, `RandomRotation(0.05)`, `RandomZoom(0.05)`
- **Holdout Test Accuracy**: **94.10%** ($N = 1,000$ product images)
- **Per-Category Performance**:
  - `Apparel`: 96% Precision, 96% Recall
  - `Footwear`: 98% Precision, 96% Recall
  - `Accessories`: 88% Precision, 92% Recall
  - `Personal Care`: 90% Precision, 84% Recall
- **Inference Service**: `src/catalog_service.py` provides `classify_product_image()` for instant catalog tagging on image upload.

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
