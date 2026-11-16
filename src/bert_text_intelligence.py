import json
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 96
BATCH_SIZE = 64
RANDOM_STATE = 42

def load_reviews_dataset(sample_size: int = 2000) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    data_path = Path("data/processed/reviews_sentiment_sample.parquet")
    df = pd.read_parquet(data_path)
    pos_df = df[df["label"] == 1].head(sample_size // 2)
    neg_df = df[df["label"] == 0].head(sample_size // 2)
    sample_df = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    texts = sample_df["text"].tolist()
    labels = sample_df["label"].to_numpy()
    train_texts, test_texts, train_y, test_y = train_test_split(
        texts, labels, test_size=0.25, random_state=RANDOM_STATE, stratify=labels
    )
    return train_texts, test_texts, train_y, test_y

def extract_bert_embeddings(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH
) -> np.ndarray:
    model.eval()
    all_embeddings = []
    total_samples = len(texts)
    for start_idx in range(0, total_samples, batch_size):
        batch_texts = texts[start_idx : start_idx + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)
    return np.vstack(all_embeddings)

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str]
) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "report": report
    }

def generate_ticket_dataset(size_per_category: int = 300) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, List[str]]:
    categories = ["Delivery & Tracking", "Refund & Billing", "Product Quality & Defect", "Account & Security"]
    templates = {
        "Delivery & Tracking": [
            "Where is my package? The tracking number {id} has shown no updates for five days.",
            "My shipment was marked delivered but there is nothing on my front porch or mailbox.",
            "Can I change the delivery address for order {id}? The courier has not picked it up yet.",
            "The courier delayed my package by two weeks and customer service cannot locate it.",
            "Package arrived completely damaged and soaked during transit. Tracking code {id}."
        ],
        "Refund & Billing": [
            "I was double-charged on my credit card for invoice {id}. Please refund the duplicate.",
            "Returned the order last Monday but my refund has not been credited to my bank account.",
            "Unauthorized transaction appeared on my monthly statement from your retail store.",
            "The promo discount code was not applied during checkout and I need a price adjustment.",
            "Cancel my subscription and issue a full credit refund for this recurring billing cycle."
        ],
        "Product Quality & Defect": [
            "The blender stopped working after two uses and the motor smells burnt.",
            "Item arrived broken with glass shards inside the box. Not fit for use.",
            "The fabric stitching on this jacket unraveled on the first wash. Very cheap material.",
            "Missing hardware and screws required to assemble the dining table.",
            "The color and dimensions do not match the photos or specifications advertised online."
        ],
        "Account & Security": [
            "I am locked out of my account because two-factor SMS authentication is not sending codes.",
            "Received an alert that my account password was changed from an unrecognized device.",
            "Please update my registered email address and phone number for security verification.",
            "Someone placed an order using my stored credit card without my authorization.",
            "Cannot log in to my account dashboard despite resetting my password multiple times."
        ]
    }
    rng = np.random.RandomState(RANDOM_STATE)
    all_texts = []
    all_labels = []
    for cat_idx, cat_name in enumerate(categories):
        tpls = templates[cat_name]
        for item_idx in range(size_per_category):
            tpl = rng.choice(tpls)
            ticket_id = f"RM-{rng.randint(100000, 999999)}"
            text = tpl.format(id=ticket_id)
            all_texts.append(text)
            all_labels.append(cat_idx)
    labels = np.array(all_labels)
    train_texts, test_texts, train_y, test_y = train_test_split(
        all_texts, labels, test_size=0.25, random_state=RANDOM_STATE, stratify=labels
    )
    return train_texts, test_texts, train_y, test_y, categories

def plot_confusion_matrices(
    y_sentiment_true: np.ndarray,
    y_sentiment_pred: np.ndarray,
    sentiment_classes: List[str],
    y_ticket_true: np.ndarray,
    y_ticket_pred: np.ndarray,
    ticket_classes: List[str],
    output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm_sent = confusion_matrix(y_sentiment_true, y_sentiment_pred)
    im0 = axes[0].imshow(cm_sent, interpolation="nearest", cmap=plt.cm.Blues)
    axes[0].set_title("DistilBERT Review Sentiment (Holdout Test)")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(sentiment_classes)
    axes[0].set_yticklabels(sentiment_classes)
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    for r in range(cm_sent.shape[0]):
        for c in range(cm_sent.shape[1]):
            val = cm_sent[r, c]
            color = "white" if val > cm_sent.max() / 2 else "black"
            axes[0].text(c, r, str(val), ha="center", va="center", color=color, fontweight="bold")
    cm_ticket = confusion_matrix(y_ticket_true, y_ticket_pred)
    im1 = axes[1].imshow(cm_ticket, interpolation="nearest", cmap=plt.cm.Greens)
    axes[1].set_title("DistilBERT Support Ticket Routing (Holdout Test)")
    axes[1].set_xticks(range(len(ticket_classes)))
    axes[1].set_yticks(range(len(ticket_classes)))
    axes[1].set_xticklabels(ticket_classes, rotation=25, ha="right")
    axes[1].set_yticklabels(ticket_classes)
    axes[1].set_xlabel("Predicted Route")
    axes[1].set_ylabel("True Category")
    for r in range(cm_ticket.shape[0]):
        for c in range(cm_ticket.shape[1]):
            val = cm_ticket[r, c]
            color = "white" if val > cm_ticket.max() / 2 else "black"
            axes[1].text(c, r, str(val), ha="center", va="center", color=color, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

def main():
    print("Initializing DistilBERT Tokenizer and Backbone...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert_backbone = AutoModel.from_pretrained(MODEL_NAME)

    print("Loading review sentiment dataset...")
    train_rev, test_rev, train_rev_y, test_rev_y = load_reviews_dataset(sample_size=2000)

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    train_rev_tfidf = tfidf.fit_transform(train_rev)
    test_rev_tfidf = tfidf.transform(test_rev)
    tfidf_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    tfidf_clf.fit(train_rev_tfidf, train_rev_y)
    tfidf_rev_preds = tfidf_clf.predict(test_rev_tfidf)
    tfidf_sentiment_results = evaluate_predictions(test_rev_y, tfidf_rev_preds, ["Negative", "Positive"])

    print("Extracting DistilBERT embeddings for sentiment...")
    train_rev_emb = extract_bert_embeddings(train_rev, tokenizer, bert_backbone)
    test_rev_emb = extract_bert_embeddings(test_rev, tokenizer, bert_backbone)

    bert_sentiment_head = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    bert_sentiment_head.fit(train_rev_emb, train_rev_y)
    bert_rev_preds = bert_sentiment_head.predict(test_rev_emb)
    bert_sentiment_results = evaluate_predictions(test_rev_y, bert_rev_preds, ["Negative", "Positive"])

    print("Generating and routing support tickets...")
    train_tkt, test_tkt, train_tkt_y, test_tkt_y, tkt_categories = generate_ticket_dataset(size_per_category=300)

    train_tkt_tfidf = tfidf.fit_transform(train_tkt)
    test_tkt_tfidf = tfidf.transform(test_tkt)
    tfidf_tkt_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    tfidf_tkt_clf.fit(train_tkt_tfidf, train_tkt_y)
    tfidf_tkt_preds = tfidf_tkt_clf.predict(test_tkt_tfidf)
    tfidf_ticket_results = evaluate_predictions(test_tkt_y, tfidf_tkt_preds, tkt_categories)

    print("Extracting DistilBERT embeddings for ticket routing...")
    train_tkt_emb = extract_bert_embeddings(train_tkt, tokenizer, bert_backbone)
    test_tkt_emb = extract_bert_embeddings(test_tkt, tokenizer, bert_backbone)

    bert_ticket_head = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    bert_ticket_head.fit(train_tkt_emb, train_tkt_y)
    bert_tkt_preds = bert_ticket_head.predict(test_tkt_emb)
    bert_ticket_results = evaluate_predictions(test_tkt_y, bert_tkt_preds, tkt_categories)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(bert_sentiment_head, models_dir / "bert_sentiment_head_v12.joblib")
    joblib.dump(bert_ticket_head, models_dir / "bert_ticket_head_v12.joblib")

    metadata = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "sentiment_classes": ["Negative", "Positive"],
        "ticket_categories": tkt_categories,
        "sentiment_comparison": {
            "tfidf_baseline_accuracy": tfidf_sentiment_results["accuracy"],
            "tfidf_baseline_f1_macro": tfidf_sentiment_results["f1_macro"],
            "bert_accuracy": bert_sentiment_results["accuracy"],
            "bert_f1_macro": bert_sentiment_results["f1_macro"],
            "lift_accuracy_pct": round((bert_sentiment_results["accuracy"] - tfidf_sentiment_results["accuracy"]) * 100, 2),
            "lift_f1_pct": round((bert_sentiment_results["f1_macro"] - tfidf_sentiment_results["f1_macro"]) * 100, 2),
        },
        "ticket_routing_comparison": {
            "tfidf_baseline_accuracy": tfidf_ticket_results["accuracy"],
            "tfidf_baseline_f1_macro": tfidf_ticket_results["f1_macro"],
            "bert_accuracy": bert_ticket_results["accuracy"],
            "bert_f1_macro": bert_ticket_results["f1_macro"],
            "lift_accuracy_pct": round((bert_ticket_results["accuracy"] - tfidf_ticket_results["accuracy"]) * 100, 2),
            "lift_f1_pct": round((bert_ticket_results["f1_macro"] - tfidf_ticket_results["f1_macro"]) * 100, 2),
        },
        "detailed_reports": {
            "sentiment_bert": bert_sentiment_results["report"],
            "ticket_bert": bert_ticket_results["report"]
        }
    }
    with open(models_dir / "text_intelligence_metadata_v12.json", "w") as f:
        json.dump(metadata, f, indent=2)

    plot_confusion_matrices(
        test_rev_y,
        bert_rev_preds,
        ["Negative", "Positive"],
        test_tkt_y,
        bert_tkt_preds,
        tkt_categories,
        Path("docs/day12_bert_text_intelligence.png")
    )
    print("Execution complete. Saved models, metadata, and evaluation plots.")

if __name__ == "__main__":
    main()
