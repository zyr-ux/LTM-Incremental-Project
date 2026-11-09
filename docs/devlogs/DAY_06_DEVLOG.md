# Day 6 DevLog: Convolutional Neural Networks (CNN) & Computer Vision Catalog Classifier

**Date**: 2026-11-09  
**Platform**: RetailMind AI  
**Increment**: 06 / 35  
**Deliverable**: Retail Computer Vision Module 1 — Product Catalog Auto-Classification  
**Focus**: CNN Architecture (Conv2D, MaxPooling2D, Flatten, Dense), Real-time Data Augmentation, Confusion Matrix Error Diagnostics, Model Persistence & Inference Service  
**Stack**: `tensorflow`, `keras`, `pillow`, `pyarrow`, `scikit-learn`, `numpy`  

---

## 1. Objectives & Context

Day 6 introduces Computer Vision (CV) to RetailMind AI:
1. **Prototype on Fashion-MNIST**:
   - Built and validated a 2-stage Conv2D + MaxPooling2D architecture on $28 \times 28$ grayscale apparel silhouettes to confirm convergence, reaching **89.78%** test accuracy.
2. **Scale to Real Catalog Images (Fashion Product Images)**:
   - Built an image-loading and data augmentation pipeline (`RandomFlip`, `RandomRotation`, `RandomZoom`) handling $60 \times 60 \times 3$ RGB product images.
3. **Train Catalog Classifier**:
   - Classifies products into core retail categories (`Apparel`, `Accessories`, `Footwear`, `Personal Care`).
   - Achieved **94.10% holdout test accuracy** with `EarlyStopping`.
4. **Error Analysis via Confusion Matrix**:
   - Diagnosed inter-class confusion (e.g., small overlap between Apparel and Accessories like scarves/belts).
5. **Production Inference Service**:
   - Created `src/catalog_service.py` to auto-tag product catalog images on upload.

---

## 2. Architecture & Design

### A. Fashion-MNIST Prototype (`src/fashion_mnist_cnn.py`)
- Input: $28 \times 28 \times 1$
- Layers: Conv2D(32, 3x3) $\rightarrow$ MaxPool(2x2) $\rightarrow$ Conv2D(64, 3x3) $\rightarrow$ MaxPool(2x2) $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(64, relu) $\rightarrow$ Dense(10, softmax)
- Result: **89.78% test accuracy** (loss: 0.2811) in 5 epochs.

### B. Catalog Auto-Tagger CNN (`src/catalog_cnn.py`)
- **Input**: $60 \times 60 \times 3$ RGB product imagery.
- **Data Augmentation**:
  - `RandomFlip("horizontal")`
  - `RandomRotation(0.05)`
  - `RandomZoom(0.05)`
- **Feature Extraction & Classification**:
  - `Conv2D(32, (3, 3), activation='relu')`
  - `MaxPooling2D((2, 2))`
  - `Conv2D(64, (3, 3), activation='relu')`
  - `MaxPooling2D((2, 2))`
  - `Dropout(0.25)`
  - `Flatten()`
  - `Dense(64, activation='relu')`
  - `Dropout(0.3)`
  - `Dense(4, activation='softmax')`

---

## 3. Evaluation & Diagnostics

Evaluated on 1,000 holdout test images ($N = 1,000$):

### Classification Report
| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Apparel** | 0.96 | 0.96 | 0.96 | 487 |
| **Accessories** | 0.88 | 0.92 | 0.89 | 260 |
| **Footwear** | 0.98 | 0.96 | 0.97 | 202 |
| **Personal Care** | 0.90 | 0.84 | 0.87 | 51 |
| **Overall Accuracy** | — | — | **94.10%** | **1,000** |

### Confusion Matrix
```text
Actual \ Predicted       Apparel    Accessories    Footwear    Personal Care
----------------------------------------------------------------------------
Apparel                      466             18           1                2
Accessories                   17            238           2                3
Footwear                       0              8         194                0
Personal Care                  0              8           0               43
```

**Confusion Insights**:
- The primary source of misclassification occurs between **Apparel** and **Accessories** (18 Apparel items classified as Accessories, 17 Accessories classified as Apparel). This reflects ambiguous domain boundaries (e.g., caps, ties, scarves, belts, gloves).
- Footwear and Personal Care exhibit extremely clean separation (98% precision for Footwear).

---

## 4. Artifacts & Integration

- Model weights: `models/product_catalog_cnn_v6.keras`
- Class labels: `models/product_catalog_classes_v6.json`
- Service endpoint: `src/catalog_service.py` (`classify_product_image`)

