import json
from pathlib import Path
from typing import Dict, List, Union
import joblib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
METADATA_PATH = MODELS_DIR / "text_intelligence_metadata_v13.json"
SENTIMENT_HEAD_PATH = MODELS_DIR / "bert_sentiment_head_v13.joblib"
TICKET_HEAD_PATH = MODELS_DIR / "bert_ticket_head_v13.joblib"

_tokenizer = None
_backbone = None
_sentiment_head = None
_ticket_head = None
_metadata = None

def _load_text_intelligence_resources():
    global _tokenizer, _backbone, _sentiment_head, _ticket_head, _metadata
    if _metadata is None:
        with open(METADATA_PATH, "r") as f:
            _metadata = json.load(f)
    if _tokenizer is None or _backbone is None:
        model_name = _metadata.get("model_name", "distilbert-base-uncased")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _backbone = AutoModel.from_pretrained(model_name)
        _backbone.eval()
    if _sentiment_head is None:
        _sentiment_head = joblib.load(SENTIMENT_HEAD_PATH)
    if _ticket_head is None:
        _ticket_head = joblib.load(TICKET_HEAD_PATH)
    return _tokenizer, _backbone, _sentiment_head, _ticket_head, _metadata

def _embed_text_batch(texts: List[str]) -> np.ndarray:
    tokenizer, backbone, _, _, metadata = _load_text_intelligence_resources()
    max_length = metadata.get("max_length", 96)
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = backbone(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embeddings

def analyze_review_sentiment(review_input: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
    single_input = isinstance(review_input, str)
    texts = [review_input] if single_input else list(review_input)
    _, _, sentiment_head, _, metadata = _load_text_intelligence_resources()
    classes = metadata.get("sentiment_classes", ["Negative", "Positive"])
    embeddings = _embed_text_batch(texts)
    probs = sentiment_head.predict_proba(embeddings)
    preds = np.argmax(probs, axis=1)
    results = [
        {
            "text": text,
            "sentiment": classes[pred_idx],
            "confidence": round(float(probs[i, pred_idx]), 4),
            "probabilities": {classes[c]: round(float(probs[i, c]), 4) for c in range(len(classes))}
        }
        for i, (text, pred_idx) in enumerate(zip(texts, preds))
    ]
    return results[0] if single_input else results

def route_support_ticket(ticket_input: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
    single_input = isinstance(ticket_input, str)
    texts = [ticket_input] if single_input else list(ticket_input)
    _, _, _, ticket_head, metadata = _load_text_intelligence_resources()
    categories = metadata.get("ticket_categories", [])
    embeddings = _embed_text_batch(texts)
    probs = ticket_head.predict_proba(embeddings)
    preds = np.argmax(probs, axis=1)
    results = [
        {
            "ticket_text": text,
            "routed_department": categories[pred_idx],
            "confidence": round(float(probs[i, pred_idx]), 4),
            "priority": "High" if categories[pred_idx] in ["Refund & Billing", "Account & Security"] else "Standard",
            "department_probabilities": {categories[c]: round(float(probs[i, c]), 4) for c in range(len(categories))}
        }
        for i, (text, pred_idx) in enumerate(zip(texts, preds))
    ]
    return results[0] if single_input else results
