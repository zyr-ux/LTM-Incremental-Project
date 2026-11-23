import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "t5-small"

class ReviewSummarizer:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=True)
            self.model.eval()

    def summarize_reviews(self, reviews: List[str], max_length: int = 60, min_length: int = 15) -> str:
        if not reviews:
            return "No reviews available to summarize."
        self._load_model()
        combined_text = " ".join([r.strip() for r in reviews if r.strip()])
        truncated_text = combined_text[:1000]
        input_text = f"summarize: {truncated_text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=2,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    def extract_review_themes(self, reviews: List[str]) -> Dict[str, List[str]]:
        pos_keywords = ["love", "great", "excellent", "comfortable", "perfect", "good", "nice", "durable"]
        neg_keywords = ["poor", "bad", "small", "tight", "broke", "terrible", "cheap", "waste", "disappointed"]
        pos_aspects = []
        neg_aspects = []
        for rev in reviews:
            rev_lower = rev.lower()
            for kw in pos_keywords:
                if kw in rev_lower and kw not in pos_aspects:
                    pos_aspects.append(kw)
            for kw in neg_keywords:
                if kw in rev_lower and kw not in neg_aspects:
                    neg_aspects.append(kw)
        return {
            "strengths": pos_aspects[:4],
            "concerns": neg_aspects[:4]
        }

    def summarize_product_reviews(self, product_name: str, reviews: List[str]) -> Dict:
        summary_text = self.summarize_reviews(reviews)
        themes = self.extract_review_themes(reviews)
        return {
            "product_name": product_name,
            "total_reviews": len(reviews),
            "theme_summary": summary_text,
            "detected_themes": themes
        }
