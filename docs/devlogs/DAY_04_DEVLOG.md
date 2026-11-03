# Day 4 DevLog: Feature Engineering, PCA Analysis & Ensemble Boosting Champion Lock

**Date**: 2026-11-03  
**Platform**: RetailMind AI  
**Increment**: 04 / 35  
**Deliverable**: Phase-1 Production Model Deliverable (Milestone Assessment 2)  
**Focus**: Domain Feature Engineering, Principal Component Analysis (PCA), Bagging vs Boosting Ensembles (Random Forest vs XGBoost), Pipeline Freezing, Production Inference  
**Stack**: `scikit-learn`, `xgboost`, `pandas`, `joblib`  

---

## 1. Objectives & Scenario Context

Day 3 diagnostics showed that model performance with raw features had reached a plateau ($\text{CV ROC-AUC} \approx 0.846 - 0.847$), and learning curve analysis revealed that accumulating more raw data without feature enrichment yielded diminishing returns.

In Day 4, we finalized the Phase-1 retail churn model by:
1. **Engineering domain-specific features**:
   - Customer tenure life-cycle cohorts (`tenure_group`).
   - Charge velocity and billing ratios (`monthly_to_total_ratio`, `charge_ratio`, `avg_monthly_diff`).
   - High-leverage contract, service count, and payment method flags (`is_month_to_month`, `has_fiber_optic`, `electronic_check`, `total_services`, `has_streaming`, `has_security_or_backup`).
2. **Applying PCA Dimensionality Reduction**:
   - Assessed orthogonal component projection (15 components capturing $>90\%$ variance) against full feature representations to evaluate information density and regularization trade-offs.
3. **Benchmarking Advanced Ensembles**:
   - Bagging: Tuned `RandomForestClassifier` with domain features and PCA.
   - Gradient Boosting: `XGBClassifier` with tree depth constraints (`max_depth=3`, shrinkage `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`).
4. **Finalizing & Freezing Production Pipeline**:
   - Enforcing custom scikit-learn transformer encapsulation (`RetailFeatureEngineer`) within a single serialized pipeline (`churn_v4.joblib` and `churn_pipeline_champion.joblib`).
   - Updating production inference service (`src/service.py`) and documentation (`README.md`).

---

## 2. Engineered Domain Features & Rationales

| Feature | Type | Definition & Mathematical Rationale |
| :--- | :--- | :--- |
| `tenure_group` | Categorical Bucket | Discretized customer lifecycle into `0-12`, `12-24`, `24-48`, `48-60`, `60+` months. Churn is heavily concentrated in the first 12 months ($\approx 47.4\%$). |
| `monthly_to_total_ratio` | Continuous Ratio | $\frac{\text{MonthlyCharges}}{\text{TotalCharges} + 1.0}$; measures initial contract tenure weight and billing acceleration. High ratios indicate brand-new or high-friction customers. |
| `charge_ratio` | Continuous Ratio | $\frac{\text{TotalCharges}}{(\text{MonthlyCharges} \times \text{tenure}) + \epsilon}$; tracks whether actual lifetime spend matches expected spend (identifying unbilled credits or mid-term plan changes). |
| `avg_monthly_diff` | Continuous Diff | $\text{MonthlyCharges} - \frac{\text{TotalCharges}}{\text{tenure} + 1.0}$; indicates price escalations or promotional discounts rolling off. |
| `total_services` | Integer Count | Sum of active add-on subscriptions (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`). Customers with higher service stickiness churn significantly less. |
| `has_streaming` | Binary Flag | Indicator for active multimedia streaming subscriptions (`StreamingTV` or `StreamingMovies`). |
| `has_security_or_backup`| Binary Flag | Indicator for cyber protection services (`OnlineSecurity` or `OnlineBackup`). |
| `is_month_to_month` | Binary Flag | Flag for month-to-month contracts (primary churn driver). |
| `has_fiber_optic` | Binary Flag | Flag for fiber optic internet service (historically correlated with higher churn complaints). |
| `electronic_check` | Binary Flag | Flag for electronic check payments (highest-churn payment channel). |

---

## 3. PCA & Ensemble Evaluation Benchmark

All models evaluated under 5-fold Stratified Cross-Validation ($N=5,634$) and Holdout Test Set ($N=1,409$):

| Architecture & Pipeline | 5-Fold CV ROC-AUC | Train ROC-AUC | Generalization Gap | Holdout Acc | Holdout Precision | Holdout Recall | Holdout F1 | Holdout ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Day-3 Champion (RF Baseline)** | 0.8474 | 0.8988 | 0.0514 | 80.13% | 66.91% | 49.73% | 57.06% | 84.33% | Previous Champion |
| **Random Forest (Day 4 FE)** | 0.8465 | 0.9074 | 0.0609 | 79.84% | 65.73% | 50.27% | 56.97% | 84.29% | Bagging + Domain FE |
| **Random Forest + PCA (15 comps)** | 0.8401 | 0.9210 | 0.0809 | 80.06% | 67.82% | 47.33% | 55.75% | 84.14% | Bagging + PCA |
| **XGBoost + PCA (15 comps)** | 0.8432 | 0.8711 | 0.0279 | **81.05%** | **68.38%** | **53.21%** | **59.85%** | 84.77% | Boosting + PCA |
| **XGBoost (Day 4 FE Champion)** | **0.8495** | 0.8693 | **0.0198** | 80.55% | 67.48% | 51.60% | 58.48% | **84.82%** | **Phase-1 Production Champion** |

### Empirical Findings:
1. **Boosting vs Bagging**: Gradient boosting (`XGBoost`) outperforms bagging (`Random Forest`) across discriminative AUC ($84.82\%$ vs $84.29\%$), while displaying a substantially smaller generalization gap ($0.0198$ vs $0.0609$).
2. **Impact of PCA**: PCA condensed 31 features into 15 components while retaining strong test performance ($84.77\%$ AUC, $68.38\%$ precision). However, full engineered features with XGBoost delivered peak cross-validated discrimination ($0.8495$ CV AUC) and preserved full domain interpretability.
3. **Feature Importances**: The top predictive features in the champion XGBoost model are `Contract_Month-to-month` ($31.1\%$), `is_month_to_month` ($22.0\%$), `has_fiber_optic` ($5.4\%$), `OnlineSecurity_No` ($3.5\%$), `TechSupport_No` ($3.3\%$), and `avg_monthly_diff` ($1.9\%$).

---

## 4. Pipeline Freezing & Production Service

To eliminate train-serve skew and protect production stability:
- All transformations are embedded within a `sklearn.pipeline.Pipeline` starting with `RetailFeatureEngineer`, flowing through `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`), and terminating in `XGBClassifier`.
- Frozen pipeline saved to `models/churn_v4.joblib` and mirrored to active pointer `models/churn_pipeline_champion.joblib`.
- Inference engine `src/service.py` handles unpickling seamlessly and serves single or batch customer payloads with predicted churn probabilities and calibrated risk categorizations (`Low`, `Medium`, `High`).

