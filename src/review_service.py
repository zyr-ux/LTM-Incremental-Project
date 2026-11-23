import re
import json
from pathlib import Path
from typing import Any, Dict
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
VENV_NLTK = ROOT_DIR / "venv" / "nltk_data"

if VENV_NLTK.exists() and str(VENV_NLTK) not in nltk.data.path:
    nltk.data.path.insert(0, str(VENV_NLTK))

STOP_WORDS = set(stopwords.words("english"))

class ReviewSentimentService:
    def __init__(self, vectorizer_path=None, model_path=None, metadata_path=None):
        self.vectorizer_path = vectorizer_path or (MODELS_DIR / "tfidf_vectorizer_v12.joblib")
        self.model_path = model_path or (MODELS_DIR / "sentiment_classifier_v12.joblib")
        self.metadata_path = metadata_path or (MODELS_DIR / "sentiment_metadata_v12.json")

        self.vectorizer = joblib.load(self.vectorizer_path)
        self.model = joblib.load(self.model_path)
        
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = word_tokenize(text)
        clean_tokens = [str(w) for w in tokens if w not in STOP_WORDS and len(w) > 2]
        result_str = " ".join(clean_tokens)
        assert isinstance(result_str, str)
        return result_str

    def analyze_review(self, review_text: str) -> Dict[str, Any]:
        cleaned = self.clean_text(review_text)
        features = self.vectorizer.transform([cleaned])
        pred_label = int(self.model.predict(features)[0])
        probs = self.model.predict_proba(features)[0]

        sentiment_label = "Positive" if pred_label == 1 else "Negative"
        confidence = float(probs[pred_label])

        return {
            "original_text": review_text,
            "cleaned_text": cleaned,
            "sentiment": sentiment_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "Negative": round(float(probs[0]), 4),
                "Positive": round(float(probs[1]), 4)
            }
        }
