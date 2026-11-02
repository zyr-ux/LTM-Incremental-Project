# Day 1 DevLog: End-to-End Baseline ML Pipeline & ML-as-a-Service

**Date**: 2026-11-02  
**Platform**: RetailMind AI  
**Increment**: 01 / 35  
**Focus**: Pipeline Architecture, Baseline Modeling, Serialization & Inference Service  
**Stack**: `pandas`, `scikit-learn`, `joblib`  

---

## 1. Objectives & Architectural Context
RetailMind AI requires an operational backbone to flag at-risk customers from Day 1. Rather than building ad-hoc scripts, the goal today was to establish a modular, reproducible pipeline architecture that will persist across future increments:
- Clean modular repository layout (`data/`, `src/`, `models/`, `docs/`, `notebooks/`).
- End-to-end ingestion and preprocessing with robust handling of schema quirks (e.g. stringified blanks in numeric columns).
- Training and evaluation of an interpretable baseline classifier.
- Serialization via `joblib` that bundles preprocessing and model weights into one cohesive artifact to eliminate training-serving skew.
- A clean, thin prediction helper exposed as the platform's initial ML service interface.

---

## 2. Dataset Exploration & Key Observations (Telco Churn)
- **Target Variable**: `Churn` (`Yes` / `No` mapped to `1` / `0`). Class distribution is moderately imbalanced (~73% non-churn vs ~27% churn), requiring stratified train/test splitting and metrics focused beyond raw accuracy (Precision, Recall, ROC-AUC).
- **Identifier**: `customerID` dropped during modeling as it holds no generalizable predictive value.
- **Type Anomaly**: `TotalCharges` contains empty whitespace strings (`" "`) for new customers where `tenure == 0`. Handled by coercing to numeric and imputing with `0.0` or median.
- **Feature Partitioning**:
  - **Categorical Features**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.
  - **Numerical Features**: `tenure`, `MonthlyCharges`, `TotalCharges`.

---

## 3. Implementation Details

### Pipeline Design (`src/pipeline.py`)
1. **Preprocessing**:
   - `ColumnTransformer` with `OneHotEncoder(handle_unknown='ignore')` for categorical columns and `StandardScaler()` for continuous numerical columns.
2. **Estimator**:
   - Baseline: `LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)`
3. **Artifact Persistence**:
   - The fitted pipeline (preprocessor + model) is serialized directly to `models/churn_pipeline_baseline.joblib`.

### Service Layer (`src/service.py`)
- Provides `get_model()` with in-memory caching to avoid repeated disk reads.
- Provides `predict_churn(customer_data)` supporting a single customer dict, list of dicts, or pandas DataFrame, returning the predicted churn flag, churn probability, and risk tiers (`Low`, `Medium`, `High`).

---

## 4. Evaluation & Metrics Summary

*Baseline results recorded from stratified test holdout ($N = 1,409$ customers; 1,035 Non-Churn, 374 Churn):*

| Metric | Baseline Score | Practical Retail Significance |
| :--- | :---: | :--- |
| **Accuracy** | **73.81%** (`0.7381`) | Balanced benchmark baseline across all customers |
| **Precision (Churn=1)** | **50.43%** (`0.5043`) | 1 in 2 proactive retention offers reach a real churner |
| **Recall (Churn=1)** | **78.34%** (`0.7834`) | Captures ~78.3% of true churners before departure |
| **F1-Score (Churn=1)** | **61.36%** (`0.6136`) | Solid initial balance of sensitivity vs false alarms |
| **ROC-AUC** | **84.15%** (`0.8415`) | Strong ranking power across custom risk thresholds |

### Classification Report Details
```text
              precision    recall  f1-score   support

   0 (Stay)       0.90      0.72      0.80      1035
  1 (Churn)       0.50      0.78      0.61       374

    accuracy                           0.74      1409
   macro avg       0.70      0.75      0.71      1409
weighted avg       0.80      0.74      0.75      1409
```

---

## 5. Challenges & Decisions
- **Training-Serving Skew Prevention**: Combining preprocessing steps directly inside `sklearn.pipeline.Pipeline` guarantees that the exact transformations (scaling parameters, one-hot category mappings) applied during training are identically applied during inference in `service.py`.
- **Handling Zero Tenure**: Rather than discarding new customer rows with empty `TotalCharges`, we impute or set to `0.0` to preserve data integrity.

---

## 6. Next Steps (Day 2 Preview)
- Explore tree-based and non-linear algorithms: Decision Trees, Random Forests, and Support Vector Machines.
- Perform hyperparameter tuning and cross-validation to establish a champion model over this Day 1 baseline.

