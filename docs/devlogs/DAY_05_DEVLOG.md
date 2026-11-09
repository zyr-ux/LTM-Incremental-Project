# Day 5 DevLog: Neural Networks & Feed-Forward ANN Propensity Model (Keras)

**Date**: 2026-11-09  
**Platform**: RetailMind AI  
**Increment**: 05 / 35  
**Deliverable**: Deep Learning Customer Propensity Model & Classical ML vs Deep Learning Benchmark  
**Focus**: Keras Sequential API, Dense feed-forward layers, Dropout regularization, EarlyStopping, MNIST warm-up, Tabular Deep Learning Trade-offs  
**Stack**: `tensorflow`, `keras`, `scikit-learn`, `xgboost`, `pandas`, `joblib`  

---

## 1. Objectives & Context

Day 5 introduces deep learning to RetailMind AI:
1. **Internalize NN Core Mechanics via MNIST Warm-up**:
   - Built a feed-forward neural network on the classic MNIST dataset to master layers, activations (`relu`, `softmax`), categorical loss, and mini-batch gradient descent.
2. **Build an ANN Customer Propensity Model**:
   - Leveraged normalized continuous inputs and one-hot encoded categorical features.
   - Built a structured feed-forward ANN (`Dense(32, relu)` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Dense(16, relu)` $\rightarrow$ `Dense(1, sigmoid)`).
   - Applied `EarlyStopping` monitoring validation loss to prevent overfitting on tabular data.
3. **Head-to-Head Benchmark Against Phase-1 Champion**:
   - Rigorously compared the Day 5 ANN against the locked Phase-1 production champion (`XGBoost` with domain feature engineering from Day 4).
   - Answered the core engineering question: *Does neural network depth provide an advantage on this tabular retail dataset?*
4. **Model Artifact Persistence**:
   - Persisted the trained neural network to `models/churn_ann_v5.keras` and its fitted data preprocessor to `models/ann_preprocessor_v5.joblib`.

---

## 2. Architecture & Design

### A. Warm-up: MNIST Image Classifier (`src/mnist_demo.py`)
- **Input**: 784 flattened normalized pixel values ($28 \times 28$).
- **Hidden Layers**: Dense(128, ReLU) + Dropout(0.2) + Dense(64, ReLU).
- **Output**: Dense(10, Softmax) with `sparse_categorical_crossentropy`.
- **Outcome**: Achieved **97.42%** test accuracy in 5 epochs, validating gradient propagation and batch optimization.

### B. Retail Propensity ANN (`src/ann_churn_model.py`)
- **Feature Matrix**: 46 preprocessed inputs (StandardScaler on continuous features + OneHotEncoder on categoricals).
- **Hidden Layers**:
  - `Dense(32, activation='relu')`
  - `Dropout(0.2)` for regularization
  - `Dense(16, activation='relu')`
- **Output Layer**: `Dense(1, activation='sigmoid')` predicting $P(\text{Churn} = 1)$.
- **Loss & Optimizer**: `binary_crossentropy` with `Adam(learning_rate=0.001)`.
- **Early Stopping**: Monitored `val_loss` with patience of 8 epochs and `restore_best_weights=True`. Stopped training at epoch 29.

---

## 3. Head-to-Head Benchmark: Classical ML vs ANN

Both models evaluated on the identical holdout test set ($N = 1,409$, 80/20 stratified split):

| Metric | Phase-1 Champion (XGBoost + Domain FE) | Day 5 ANN (Keras Feed-Forward) | Winner / Delta |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **80.55%** | 78.85% | XGBoost (+1.70%) |
| **Precision** | **67.48%** | 60.67% | XGBoost (+6.81%) |
| **Recall** | 51.60% | **57.75%** | ANN (+6.15%) |
| **F1-Score** | 58.48% | **59.18%** | ANN (+0.70%) |
| **ROC-AUC** | **84.82%** | 83.95% | XGBoost (+0.87%) |

---

## 4. Honest Assessment: Did Depth Help Here?

### Did depth help?
**No, deeper representation learning did not outperform gradient-boosted trees on this tabular dataset.**
- While the ANN achieved a higher recall (57.75% vs 51.60%) and a slight F1 gain, **XGBoost maintained superior overall discriminative power (ROC-AUC 84.82% vs 83.95%)** and noticeably higher precision (67.48% vs 60.67%).
- Tabular retail datasets have heterogeneous data types (discrete flags, life-cycle categories, skewed monetary metrics) with sharp decision boundaries that tree ensembles partition naturally. Dense layers smooth these boundaries, requiring more extensive tuning and regularization to avoid underperforming tree-based models.

### When to Use Which:
- **Use Classical ML (Gradient Boosted Trees / XGBoost / LightGBM)**:
  - Tabular business datasets with structured columns and sample sizes under hundreds of thousands.
  - When fast inference, sample efficiency, explainability (SHAP / feature importances), and minimal hyperparameter fragility are critical.
- **Use Deep Learning (ANNs / DNNs)**:
  - Large-scale datasets ($>10^6$ records) where manual feature engineering reaches a ceiling.
  - Multimodal scenarios (e.g., combining tabular customer metadata with unstructured text reviews, support chat embeddings, or clickstream sequences).
  - Online continuous learning or unified embedding representations across multiple downstream retail tasks.

