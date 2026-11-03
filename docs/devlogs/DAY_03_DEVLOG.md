# Day 3 DevLog: Multi-Model Tuning, Bias-Variance Diagnostics & Champion Lock

**Date**: 2026-11-03  
**Platform**: RetailMind AI  
**Increment**: 03 / 35  
**Focus**: Hyperparameter Tuning (`GridSearchCV`), Cross-Validation (`StratifiedKFold`), Bias-Variance Diagnostics, Learning Curves, Model Versioning  
**Stack**: `scikit-learn`  

---

## 1. Objectives & Context

In Day 1 and Day 2, we built an end-to-end churn pipeline and compared candidate models using default or untuned settings. Untuned models carry intrinsic risks:
1. **High Variance (Overfitting)**: Trees and ensembles can memorize training noise, creating wide gaps between train and validation scores.
2. **High Bias (Underfitting)**: Constrained linear models or heavily penalized models might under-represent complex retail consumer behaviors.

The Day 3 increment focuses on systematic hyperparameter tuning, bias-variance diagnostics, and locking a validated, production-grade churn champion.

### Key Deliverables:
- Define exhaustive hyperparameter search grids for candidate architectures.
- Run `GridSearchCV` with 5-fold stratified cross-validation and `return_train_score=True` to observe generalization gaps.
- Analyze learning curves across increasing training sample sizes to guide data and regularization decisions.
- Version the locked champion model to `models/churn_v3.joblib`.
- Integrate the tuned champion seamlessly into `src/service.py`.

---

## 2. Hyperparameter Search Grids

Each candidate algorithm was evaluated across a targeted hyperparameter grid:

1. **Logistic Regression**:
   - `C`: `[0.01, 0.1, 1.0, 10.0]` (Regularization strength)
   - `class_weight`: `['balanced', None]`
   - `solver`: `['lbfgs']`

2. **Decision Tree Classifier**:
   - `max_depth`: `[3, 5, 8, 12, None]` (Tree complexity constraint)
   - `min_samples_split`: `[2, 10, 20]` (Minimum node split threshold)
   - `class_weight`: `['balanced', None]`

3. **Random Forest Classifier**:
   - `n_estimators`: `[50, 100, 200]` (Ensemble scale)
   - `max_depth`: `[5, 8, 12]` (Tree depth regularizer)
   - `min_samples_split`: `[2, 5, 10]`
   - `class_weight`: `['balanced', None]`

4. **Support Vector Machine (RBF Kernel)**:
   - `C`: `[0.1, 1.0, 5.0]` (Margin softness)
   - `gamma`: `['scale', 0.01]` (Kernel coefficient)
   - `class_weight`: `['balanced']`

---

## 3. Systematic 5-Fold Cross-Validation & Diagnostics

Cross-validation optimization was guided by **ROC-AUC** to balance discrimination across varying risk thresholds:

| Model | Best CV ROC-AUC | Mean Train ROC-AUC | Gap (Variance) | Optimal Hyperparameters | Bias / Variance Diagnosis |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Logistic Regression** | 0.8464 | 0.8500 | **0.0036** | `C=10.0`, `class_weight=None` | Minimal variance, slight bias (linear boundary limit) |
| **Decision Tree** | 0.8293 | 0.8541 | 0.0248 | `max_depth=5`, `min_samples_split=20` | Pruned depth resolved Day 2 severe overfit |
| **Random Forest** | **0.8474** | 0.9057 | 0.0583 | `n_estimators=200`, `max_depth=8`, `min_samples_split=10` | **Champion**: Peak discriminative power |
| **SVM (RBF Kernel)** | 0.8423 | 0.8488 | 0.0065 | `C=1.0`, `gamma=0.01`, `class_weight='balanced'` | Low variance, stable high-dimensional boundary |

---

## 4. Holdout Test Set Benchmarking (N = 1,409)

Evaluating best estimators on the untouched holdout test partition:

| Model | Accuracy | Precision (Churn=1) | Recall (Churn=1) | F1-Score (Churn=1) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Tuned)** | 80.41% | 65.41% | 55.61% | 60.12% | 84.12% |
| **Decision Tree (Tuned)** | 79.84% | 63.47% | 56.68% | 59.89% | 82.97% |
| **Random Forest (Tuned)** | **80.13%** | **66.91%** | 49.73% | 57.06% | **84.33%** |
| **SVM (RBF Kernel, Tuned)** | 70.48% | 46.80% | **82.09%** | 59.61% | 83.34% |

---

## 5. Learning Curve & Bias-Variance Analysis

A learning curve was computed for the champion model across training subsets ($10\%$ to $100\%$):

| Training Samples | Mean Train ROC-AUC | Mean Validation ROC-AUC | Generalization Gap |
| :---: | :---: | :---: | :---: |
| **563** | 0.9584 | 0.8295 | 0.1289 |
| **1,831** | 0.9385 | 0.8428 | 0.0957 |
| **3,098** | 0.9192 | 0.8434 | 0.0758 |
| **4,366** | 0.9059 | 0.8439 | 0.0620 |
| **5,634** | **0.8985** | **0.8458** | **0.0528** |

### Insights from the Curve:
1. **Variance Reduction**: As training sample size grew from 563 to 5,634, the generalization gap shrank from $12.89\%$ to $5.28\%$.
2. **Convergence**: The validation ROC-AUC plateaued at $\approx 0.846$. This indicates that feeding additional samples without new behavioral features will yield diminishing returns.
3. **Actionable Next Steps**: In upcoming modules, performance will benefit most from **feature engineering** (e.g. usage velocity, contract tenure ratios) rather than raw sample collection alone.

---

## 6. Champion Locking & Versioning

- **Locked Champion**: **Random Forest** (`n_estimators=200, max_depth=8, min_samples_split=10, class_weight=None`)
- **Model Artifact**: Serialized to `models/churn_v3.joblib` and mirrored to `models/churn_pipeline_champion.joblib`.
- **Inference Service**: `src/service.py` is upgraded to load `churn_v3.joblib` with graceful fallback.

