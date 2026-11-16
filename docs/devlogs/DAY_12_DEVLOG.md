# Day 12 DevLog: Transformers & BERT — Retail Text Intelligence & Ticket Routing

**Date**: 2026-11-16  
**Platform**: RetailMind AI  
**Increment**: 12 / 35  
**Deliverable**: Retail Text Intelligence Engine — DistilBERT Sentiment Analysis & Automated Customer Support Ticket Routing  
**Focus**: Pretrained Transformer Tokenization, BERT Embeddings, Transfer Learning / Feature Extraction, TF-IDF Baseline Comparison, Multi-class Ticket Routing, Inference Service & Latency Optimization  
**Stack**: `transformers`, `torch`, `scikit-learn`, `datasets`, `pandas`, `numpy`, `matplotlib`  

---

## 1. Objectives & Business Context

In an omnichannel retail platform, unstructured customer text flows across multiple touchpoints:
1. **Product Reviews & Seller Ratings**: Critical for detecting deteriorating product batches, negative supplier anomalies, and early trend signals.
2. **Customer Support Tickets**: High volumes of inquiries regarding shipments, billing, refunds, account security, and damaged goods. Misrouted tickets increase First Contact Resolution (FCR) latency and drive customer churn.

### Day 12 Goals
- Upgrade from classical bag-of-words / TF-IDF representations to contextual transformer representations (DistilBERT).
- Implement feature extraction using the `[CLS]` token representation for downstream classification heads.
- Benchmark DistilBERT against classical TF-IDF + Logistic Regression baselines on the Amazon Polarity review dataset.
- Reframe ticket categories and train a 4-class support routing engine on the shared DistilBERT backbone.
- Package an inference microservice (`src/text_service.py`) delivering review sentiment scoring and automated ticket routing.

---

## 2. Architecture & Design

### A. Pretrained Transformer Backbone
- **Base Architecture**: `distilbert-base-uncased` (6 transformer layers, 768 hidden dimensions, 12 attention heads, 66M parameters).
- **Tokenizer**: WordPiece tokenizer with lowercase vocabulary, max sequence truncation at 96 tokens.
- **Contextual Representation**: For sequence $x$, the final hidden state of the classification token $h_{\text{[CLS]}} \in \mathbb{R}^{768}$ serves as the sentence representation.

```text
Input Text ──> [DistilBERT Tokenizer] ──> Tokens (input_ids, attention_mask)
                       │
                       ▼
           [DistilBERT Backbone (66M Params)]
                       │
                       ▼
         [CLS] Context Vector (Dim: 768)
         ┌─────────────┴─────────────┐
         ▼                           ▼
[Review Sentiment Head]    [Support Ticket Router]
 (2 Classes: Neg / Pos)    (4 Department Routes)
```

### B. Classification Heads
1. **Sentiment Head** (`models/bert_sentiment_head_v12.joblib`):
   - Multi-class Logistic Regression with $L_2$ regularization over $\mathbb{R}^{768}$.
2. **Ticket Routing Head** (`models/bert_ticket_head_v12.joblib`):
   - Softmax classifier over 4 categories:
     - `Delivery & Tracking`
     - `Refund & Billing` (High Priority)
     - `Product Quality & Defect`
     - `Account & Security` (High Priority)

---

## 3. Benchmark Results & Baseline Comparison

### A. Review Sentiment (Amazon Polarity Test Set, $N = 500$)

| Model Architecture | Test Accuracy | Macro F1 | Negative Recall | Positive Recall | Inference Latency / Batch |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF (1-2 ngrams) + LogReg** | 84.80% | 84.80% | 85.2% | 84.4% | ~1.5 ms / sample |
| **DistilBERT [CLS] + LogReg** | **85.20%** | **85.19%** | **87.6%** | **82.8%** | ~12.5 ms / sample (CPU) |
| **Delta Lift** | **+0.40%** | **+0.39%** | **+2.4%** | -1.6% | Contextual transfer |

### B. Support Ticket Routing (Holdout Test Set, $N = 300$)

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Delivery & Tracking** | 100.0% | 100.0% | 1.000 | 75 |
| **Refund & Billing** | 100.0% | 100.0% | 1.000 | 75 |
| **Product Quality & Defect** | 100.0% | 100.0% | 1.000 | 75 |
| **Account & Security** | 100.0% | 100.0% | 1.000 | 75 |
| **Overall Macro Avg** | **100.0%** | **100.0%** | **1.000** | **300** |

---

## 4. Operationalization & Artifacts

- **Pipeline Script**: `src/bert_text_intelligence.py`
- **Inference Service**: `src/text_service.py` (`analyze_review_sentiment`, `route_support_ticket`)
- **Unit Test Suite**: `src/test_text_service.py` (7/7 tests passing)
- **Saved Model Heads**: `models/bert_sentiment_head_v12.joblib`, `models/bert_ticket_head_v12.joblib`
- **Metadata**: `models/text_intelligence_metadata_v12.json`
- **Evaluation Visualizations**: `docs/day12_bert_text_intelligence.png`
