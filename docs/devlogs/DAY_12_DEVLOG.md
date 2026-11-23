# Day 12 DevLog: NLP & Classical Text Processing — Retail Sentiment Baseline

**Date**: 2026-11-16  
**Platform**: RetailMind AI  
**Increment**: 12 / 35  
**Deliverable**: Retail Customer Review Intelligence — Classical NLP Pipeline, TF-IDF Vectorisation, Sentiment Classification, and Informative Feature Diagnostics  
**Focus**: Text cleaning, regex normalization, stopword filtering, n-gram TF-IDF vectorisation, Logistic Regression / Linear SVM baseline, informative feature coefficients  
**Stack**: `scikit-learn`, `nltk`, `pandas`, `numpy`, `matplotlib`, `joblib`

---

## 1. Objectives & Retail Context

In retail operations, customer sentiment in product reviews reveals product quality defects, packaging failures, customer dissatisfaction, and buyer delight:
1. **Pipeline Ingestion**:
   - Ingest Amazon product customer reviews (`amazon_polarity` dataset sample).
2. **Classical Text Cleaning**:
   - Standardise text: lowercase conversion, HTML tag stripping, punctuation removal, word tokenisation, and stopword pruning using NLTK.
3. **Feature Vectorisation**:
   - Transform cleaned text into unigram and bigram TF-IDF representations (vocabulary capped at 5,000 features).
4. **Baseline Sentiment Classification**:
   - Train a Logistic Regression classifier as the primary baseline, alongside Linear SVM for model comparison.
5. **Product Insight Diagnostics**:
   - Inspect model weights (log-odds coefficients) to surface top positive and negative vocabulary driving customer sentiment.
6. **Persistence & Inference Service**:
   - Save vectorizer and model artifacts (`tfidf_vectorizer_v12.joblib`, `sentiment_classifier_v12.joblib`, `sentiment_metadata_v12.json`) and wrap in `ReviewSentimentService` for downstream retail catalog integration.

---

## 2. Experimental Results & Model Performance

- **Holdout Test Set Size**: 1,000 balanced reviews (500 positive, 500 negative).
- **Logistic Regression Test Accuracy**: **85.90%**
- **Linear SVM Test Accuracy**: **85.70%**

### Classification Report (Logistic Regression):
```text
              precision    recall  f1-score   support

    Negative       0.86      0.86      0.86       500
    Positive       0.86      0.86      0.86       500

    accuracy                           0.86      1000
   macro avg       0.86      0.86      0.86      1000
weighted avg       0.86      0.86      0.86      1000
```

---

## 3. Top Informative Signals for Retail Insights

By extracting the largest positive and negative coefficients from the Logistic Regression model, we uncover the primary lexical drivers of customer perception:

- **Top Positive Drivers**: `great`, `excellent`, `best`, `love`, `good`, `wonderful`, `easy`, `perfect`, `awesome`, `nice`
- **Top Negative Drivers**: `boring`, `disappointed`, `worst`, `waste`, `bad`, `poor`, `money`, `terrible`, `horrible`, `useless`

Diagnostics visualization persisted to: `docs/day12_classical_sentiment_insights.png`.

---

## 4. Deliverables & Next Steps

1. **Pipeline**: `src/review_sentiment_pipeline.py`
2. **Inference Service**: `src/review_service.py`
3. **Unit Tests**: `src/test_review_service.py`
4. **Artifacts**:
   - `models/tfidf_vectorizer_v12.joblib`
   - `models/sentiment_classifier_v12.joblib`
   - `models/sentiment_metadata_v12.json`
   - `docs/day12_classical_sentiment_insights.png`

This classical pipeline provides an interpretable, lightweight baseline that will be upgraded in subsequent increments using BERT embeddings and deep transformer architectures.
