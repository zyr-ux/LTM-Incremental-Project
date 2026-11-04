# Model Training Pipeline

from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

from src.preprocessing import (
    get_train_test_data,
    get_preprocessor,
    BASE_DIR,
)

MODEL_PATH = BASE_DIR / "models" / "churn_pipeline_baseline.joblib"

def train():
    X_train, X_test, y_train, y_test = get_train_test_data()

    preprocessor = get_preprocessor(handle_unknown="ignore", impute_strategy="median")
    clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print("Baseline Model Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, preds))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Saved model to", MODEL_PATH)

if __name__ == "__main__":
    train()
