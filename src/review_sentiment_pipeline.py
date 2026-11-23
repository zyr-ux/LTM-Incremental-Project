import re
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
DOCS_DIR = ROOT_DIR / "docs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

STOP_WORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    clean_tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
    return " ".join(clean_tokens)

def load_and_preprocess_data(data_path: Path):
    df = pd.read_parquet(data_path)
    df["clean_text"] = df["text"].apply(clean_text)
    return df

def get_top_keywords(vectorizer: TfidfVectorizer, model: LogisticRegression, top_n: int = 15):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    
    top_negative_indices = np.argsort(coefs)[:top_n]
    top_positive_indices = np.argsort(coefs)[-top_n:][::-1]
    
    negative_terms = [(feature_names[i], round(float(coefs[i]), 4)) for i in top_negative_indices]
    positive_terms = [(feature_names[i], round(float(coefs[i]), 4)) for i in top_positive_indices]
    
    return positive_terms, negative_terms

def plot_sentiment_insights(positive_terms, negative_terms, output_path: Path):
    pos_words = [t[0] for t in positive_terms][::-1]
    pos_scores = [t[1] for t in positive_terms][::-1]
    neg_words = [t[0] for t in negative_terms][::-1]
    neg_scores = [t[1] for t in negative_terms][::-1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].barh(neg_words, neg_scores, color="#e63946")
    axes[0].set_title("Top Negative Informative Terms (Log-Odds)")
    axes[0].set_xlabel("Coefficient")
    
    axes[1].barh(pos_words, pos_scores, color="#2a9d8f")
    axes[1].set_title("Top Positive Informative Terms (Log-Odds)")
    axes[1].set_xlabel("Coefficient")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def main():
    parquet_path = DATA_DIR / "reviews_sentiment_sample.parquet"
    print("Loading and cleaning Amazon reviews...")
    df = load_and_preprocess_data(parquet_path)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    
    print("Vectorising with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    
    print("Training Logistic Regression baseline...")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_vec, y_train)
    
    print("Training Linear SVM baseline for comparison...")
    linear_svm = LinearSVC(max_iter=2000, random_state=42)
    linear_svm.fit(X_train_vec, y_train)
    
    lr_preds = log_reg.predict(X_test_vec)
    svm_preds = linear_svm.predict(X_test_vec)
    
    lr_acc = float(accuracy_score(y_test, lr_preds))
    svm_acc = float(accuracy_score(y_test, svm_preds))
    
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    print(f"Linear SVM Accuracy: {svm_acc:.4f}")
    print("\nLogistic Regression Report:")
    print(classification_report(y_test, lr_preds, target_names=["Negative", "Positive"]))
    
    pos_terms, neg_terms = get_top_keywords(vectorizer, log_reg, top_n=15)
    print("Top Positive Signals:", [w for w, _ in pos_terms[:7]])
    print("Top Negative Signals:", [w for w, _ in neg_terms[:7]])
    
    plot_path = DOCS_DIR / "day12_classical_sentiment_insights.png"
    plot_sentiment_insights(pos_terms, neg_terms, plot_path)
    print(f"Saved sentiment keywords plot to {plot_path}")
    
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer_v12.joblib"
    model_path = MODELS_DIR / "sentiment_classifier_v12.joblib"
    metadata_path = MODELS_DIR / "sentiment_metadata_v12.json"
    
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(log_reg, model_path)
    
    metadata = {
        "model_type": "LogisticRegression + TFIDF",
        "vocabulary_size": len(vectorizer.vocabulary_),
        "ngram_range": [1, 2],
        "accuracy": round(lr_acc, 4),
        "linear_svm_accuracy": round(svm_acc, 4),
        "top_positive_terms": pos_terms,
        "top_negative_terms": neg_terms,
        "label_mapping": {"0": "Negative", "1": "Positive"},
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("Pipeline artifacts saved successfully.")

if __name__ == "__main__":
    main()
