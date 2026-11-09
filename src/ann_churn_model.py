import sys
from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import get_train_test_data, get_preprocessor, BASE_DIR

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.random.set_seed(42)
np.random.seed(42)

ANN_MODEL_PATH = BASE_DIR / "models" / "churn_ann_v5.keras"
PREPROCESSOR_PATH = BASE_DIR / "models" / "ann_preprocessor_v5.joblib"
CHAMPION_PATH = BASE_DIR / "models" / "churn_v4.joblib"

def build_ann_model(input_dim: int) -> keras.Sequential:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model

def train_and_benchmark():
    X_train, X_test, y_train, y_test = get_train_test_data()

    preprocessor = get_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    model = build_ann_model(X_train_proc.shape[1])

    history = model.fit(
        X_train_proc,
        y_train.values,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    ann_probs = model.predict(X_test_proc, verbose=0).flatten()
    ann_preds = (ann_probs >= 0.5).astype(int)

    ann_metrics = {
        "Accuracy": accuracy_score(y_test, ann_preds),
        "Precision": precision_score(y_test, ann_preds),
        "Recall": recall_score(y_test, ann_preds),
        "F1": f1_score(y_test, ann_preds),
        "ROC-AUC": roc_auc_score(y_test, ann_probs)
    }

    champion_model = joblib.load(CHAMPION_PATH)
    champ_preds = champion_model.predict(X_test)
    champ_probs = champion_model.predict_proba(X_test)[:, 1]

    champ_metrics = {
        "Accuracy": accuracy_score(y_test, champ_preds),
        "Precision": precision_score(y_test, champ_preds),
        "Recall": recall_score(y_test, champ_preds),
        "F1": f1_score(y_test, champ_preds),
        "ROC-AUC": roc_auc_score(y_test, champ_probs)
    }

    print("\n" + "=" * 60)
    print("DAY 5 HEAD-TO-HEAD BENCHMARK: ANN vs Phase-1 Champion")
    print("=" * 60)
    print(f"{'Metric':<15} | {'Phase-1 Champion (XGB)':<22} | {'Day 5 ANN (Keras)':<18}")
    print("-" * 60)
    for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        print(f"{metric:<15} | {champ_metrics[metric]:<22.4f} | {ann_metrics[metric]:<18.4f}")
    print("=" * 60)

    ANN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(ANN_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"\nSaved ANN model to: {ANN_MODEL_PATH}")
    print(f"Saved ANN preprocessor to: {PREPROCESSOR_PATH}")

    return ann_metrics, champ_metrics

if __name__ == "__main__":
    train_and_benchmark()

