# Day 2 DevLog: Non-Linear Models & Champion Selection

**Date**: 2026-11-03  
**Platform**: RetailMind AI  
**Increment**: 02 / 35  
**Focus**: Decision Trees, Random Forests, Support Vector Machines (SVM), Champion Selection  
**Stack**: `scikit-learn`  

---

## 1. Objectives & Context
In Day 1, we established a linear baseline with Logistic Regression. However, customer churn behavior exhibits complex non-linear feature interactions (such as tenure crossed with contract type and monthly spend).

The Day 2 goal is to:
- Reuse the Day 1 preprocessing pipeline (`StandardScaler` + `OneHotEncoder`) to ensure fair, identical feature representation across all candidates.
- Train and evaluate three diverse non-linear algorithms: **Decision Tree**, **Random Forest**, and **Support Vector Machine (SVM with RBF kernel)**.
- Fix a random seed (`random_state=42`) across train/test splitting and models for reproducible comparison.
- Prioritize **Recall (Churn=1)** as the primary selection criterion: in retail churn management, failing to detect an at-risk customer is significantly costlier than reaching out to an unthreatened customer.
- Persist the champion pipeline to `models/churn_pipeline_champion.joblib` alongside Day 1's baseline.

---

## 2. Experimental Results & Model Comparison

All models were evaluated on the identical 20% stratified holdout set ($N = 1,409$; 1,035 Non-Churn, 374 Churn):

| Model | Accuracy | Precision (Churn=1) | Recall (Churn=1) | F1-Score (Churn=1) | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Logistic Regression)** | 73.81% | 50.43% | **78.34%** | 61.36% | 84.15% | Day 1 Reference |
| **Decision Tree** | 72.75% | 48.67% | 48.93% | 48.80% | 65.17% | Prone to overfitting |
| **Random Forest** | **76.51%** | **55.13%** | 61.76% | 58.26% | **82.28%** | Best accuracy & precision balance |
| **SVM (RBF Kernel)** | 74.52% | 51.33% | **77.27%** | **61.69%** | 82.13% | **Champion Candidate** |

---

## 3. Analysis & Champion Decision

### Decision Tree Analysis
- An unconstrained decision tree suffered from high variance and memorization of training paths, resulting in poor generalizability (ROC-AUC 65.17%, Recall 48.93%).
- Without pruning (`max_depth`, `min_samples_split`), single decision trees struggle with noisy boundary cases in tabular retail data.

### Random Forest Analysis
- Ensemble bagging significantly reduced variance over the single decision tree, raising ROC-AUC to 82.28% and accuracy to 76.51%.
- Offers highest precision (55.13%), but caught only 61.76% of churners under balanced sample weights.

### Support Vector Machine (RBF) Analysis & Champion Selection
- **Champion**: **SVM (RBF Kernel)** is selected as the Day 2 champion.
- **Why it won**:
  1. **High Recall (77.27%)**: Detects nearly 4 out of 5 at-risk churners.
  2. **Top F1-Score (61.69%)**: Maintains the best harmonic balance between precision and recall among the non-linear candidates.
  3. **Effective Margin Boundary**: The RBF kernel maps customer features into high-dimensional space where complex relationship boundaries between tenure and billing types become linearly separable.

---

## 4. Artifacts & Code Delivery
- Comparison script: `src/compare_models.py` runs all three models, logs formatted tables, and exports the winner.
- Model artifact: `models/churn_pipeline_champion.joblib` saved alongside `models/churn_pipeline_baseline.joblib`.
- Inference service: `src/service.py` automatically prefers the champion model when present.
