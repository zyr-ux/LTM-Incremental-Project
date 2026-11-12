# RetailMind AI

RetailMind AI is an end-to-end retail AI platform built incrementally across 35 engineering and data science modules. The platform introduces applied machine learning, deep learning, operational tooling, and scalable system design to solve high-impact retail problems—including customer churn prediction, demand forecasting, inventory optimization, and customer personalization.

---

## 🎯 Platform Modules

### 1. Customer Churn Propensity Module (Days 1–5)
- **Dataset**: [Telco Customer Churn (WA_Fn-UseC_-Telco-Customer-Churn.csv)](data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv)
- **Core Libraries**: `pandas`, `scikit-learn`, `xgboost`, `tensorflow`, `joblib`
- **Champion**: XGBoost with domain feature engineering (`models/churn_v4.joblib` / `churn_pipeline_champion.joblib`), reaching **84.82% ROC-AUC**.
- **Inference Service**: `src/service.py` (`predict_churn`).

### 2. Product Catalog Vision Module (Days 6–8)
- **Datasets**: Fashion-MNIST & [Fashion Product Images](https://huggingface.co/datasets/ashraq/fashion-product-images-small) (auto-downloaded)
- **Core Libraries**: `keras`, `tensorflow`, `opencv-python`, `ultralytics`, `pillow`, `pyarrow`
- **Champion**: MobileNetV2 Transfer Learning (`models/product_catalog_transfer_v7.keras`), reaching **97.70% Accuracy** + YOLOv8n object detection (`models/yolov8n.pt`).
- **Segmentation & Background Removal**: `src/catalog_segmenter.py` (product masking & uniform background compositing via OpenCV).
- **Explainability**: `src/catalog_explainability.py` (Grad-CAM heatmaps & overlay generation via TensorFlow gradients).
- **Unified CV Service**: `src/vision_module.py` (`process_catalog_item`, `analyze_catalog_image`).

### 3. Demand Forecasting & Inventory Optimization Module (Days 9–10)
- **Dataset**: [Store Item Demand Forecasting](data/raw/store_item_demand.csv) (auto-mirrored from Kaggle challenge)
- **Core Libraries**: `keras`, `pandas`, `scikit-learn`, `joblib`, `numpy`, `matplotlib`, `seaborn`
- **Day 9 Baseline Champion**: Deep LSTM Recurrent Neural Network (`models/demand_lstm_v9.keras`), reaching **18.01 MAE** and **13.77% MAPE** for 1-week ahead demand.
- **Day 10 Multi-Horizon Champion**: Sequence-to-Sequence (Seq2Seq) with Bahdanau Additive Attention (`models/demand_seq2seq_attention_v10.keras`), reaching **14.08 MAE** on next-week demand (**+21.8% accuracy boost**) and **21.45 MAE** across a full 4-week multi-horizon forecast.
- **Attention Diagnostics**: `docs/demand_attention_weights.png` (temporal attention alignment matrix showing look-back focus per forecast horizon step).
- **Inference & Procurement Service**: `src/demand_service.py` (`predict_multi_step_demand` for 4-week multi-horizon replenishment schedules and `predict_demand` for backward-compatible rollouts).

---

## 🏗 Repository Structure

```text
LTM Incremental Project/
├── data/
│   ├── raw/
│   │   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   │   ├── store_item_demand.csv
│   │   └── fashion_products/ (gitignored, auto-downloaded)
│   └── processed/
│       └── telco_churn_clean.csv
├── docs/
│   ├── demand_attention_weights.png
│   ├── demand_forecast_evaluation.png
│   ├── demand_seq2seq_evaluation.png
│   └── devlogs/
│       ├── DAY_01_DEVLOG.md
│       ├── DAY_02_DEVLOG.md
│       ├── DAY_03_DEVLOG.md
│       ├── DAY_04_DEVLOG.md
│       ├── DAY_05_DEVLOG.md
│       ├── DAY_06_DEVLOG.md
│       ├── DAY_07_DEVLOG.md
│       ├── DAY_08_DEVLOG.md
│       ├── DAY_09_DEVLOG.md
│       ├── DAY_10_DEVLOG.md
│       └── INDEX.md
├── models/
│   ├── ann_preprocessor_v5.joblib
│   ├── churn_ann_v5.keras
│   ├── churn_pipeline_baseline.joblib
│   ├── churn_pipeline_champion.joblib
│   ├── churn_v3.joblib
│   ├── churn_v4.joblib
│   ├── demand_lstm_v9.keras
│   ├── demand_scaler_metadata_v9.json
│   ├── demand_scaler_v10.joblib
│   ├── demand_scaler_v9.joblib
│   ├── demand_seq2seq_attention_v10.keras
│   ├── demand_seq2seq_metadata_v10.json
│   ├── product_catalog_classes_v6.json
│   ├── product_catalog_classes_v7.json
│   ├── product_catalog_cnn_v6.keras
│   ├── product_catalog_transfer_v7.keras
│   └── yolov8n.pt
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── pipeline.py
│   ├── compare_models.py
│   ├── tune_and_diagnose.py
│   ├── feature_engineering_and_ensembles.py
│   ├── service.py
│   ├── mnist_demo.py
│   ├── ann_churn_model.py
│   ├── fashion_mnist_cnn.py
│   ├── catalog_cnn.py
│   ├── catalog_service.py
│   ├── transfer_classifier.py
│   ├── catalog_detector.py
│   ├── catalog_segmenter.py
│   ├── catalog_explainability.py
│   ├── vision_module.py
│   ├── demand_forecast_lstm.py
│   ├── demand_forecast_seq2seq.py
│   └── demand_service.py
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

> **Phase-1 Production Champion Lock**: **XGBoost** with engineered domain features (`n_estimators=100`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`) is locked as the **Day 4 & Phase-1 deliverable champion** and persisted to `models/churn_v4.joblib` (and active `models/churn_pipeline_champion.joblib`). It delivers the highest holdout ROC-AUC (**84.82%**) and 5-fold CV ROC-AUC (**0.8495**), with a minimal generalization gap (**0.0198**), superior precision (**67.48\%**), and fully frozen end-to-end preprocessing.

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

## 🖼 Product Catalog Computer Vision Modules (Days 6–8)

Days 6 to 8 establish RetailMind AI's complete computer vision platform for catalog auto-tagging, quality assurance, background standardization, and model explainability.

### Transfer Learning Benchmark ($N = 1,000$ Holdout Images)

| Architecture / Model | Overall Accuracy | Apparel F1 | Accessories F1 | Footwear F1 | Personal Care F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Day 6 From-Scratch CNN** | 94.10% | 0.96 | 0.89 | 0.97 | 0.87 | Baseline CNN |
| **Day 7 Transfer Learning (MobileNetV2)** | **97.70%** | **0.99** | **0.96** | **0.99** | **0.93** | **Production Vision Champion** |

### Computer Vision Capabilities (Phase 2 Close)
1. **Transfer Learning Classification**: Fast MobileNetV2 backbone classifying items into 4 retail categories (`Apparel`, `Accessories`, `Footwear`, `Personal Care`) at 97.70% accuracy.
2. **Object Detection**: YOLOv8n identifies objects to verify catalog uploads and flag invalid supplier images.
3. **Foreground Segmentation & Background Cleaning**: OpenCV-powered masking removes noisy backgrounds and composites products onto clean, uniform catalog backgrounds.
4. **Grad-CAM Explainability**: Visual heatmaps show catalog review teams exactly which image regions drove the model's decision.
3. **Shelf & Planogram Analytics**: Counts facings on store shelves and spots out-of-stock gaps.

---

## 📝 Developer Logs

Detailed iteration notes, architectural decisions, data audits, and experiment analyses are recorded daily.
Browse all daily logs in the [Devlogs Index](docs/devlogs/INDEX.md).

---

## 🚀 Quickstart & Usage

### 1. Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Running Modules

All training pipelines and inference services live under `src/` and can be executed directly. For eg. :

```bash
python src/feature_engineering_and_ensembles.py  # Churn tabular champion (XGBoost)
python src/transfer_classifier.py                # Catalog vision champion (MobileNetV2)
python src/demand_forecast_lstm.py               # Demand forecast champion (LSTM)
```
