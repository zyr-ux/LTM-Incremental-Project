import sys
from pathlib import Path
import os
import io
import json
import urllib.request
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.random.set_seed(42)
np.random.seed(42)

PARQUET_PATH = ROOT_DIR / "data" / "raw" / "fashion_products" / "fashion_products_sample.parquet"
DATASET_URL = "https://huggingface.co/datasets/ashraq/fashion-product-images-small/resolve/main/data/train-00000-of-00002-6cff4c59f91661c3.parquet"
MODEL_PATH = ROOT_DIR / "models" / "product_catalog_cnn_v6.keras"
METADATA_PATH = ROOT_DIR / "models" / "product_catalog_classes_v6.json"

CATEGORIES = ["Apparel", "Accessories", "Footwear", "Personal Care"]
LABEL_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}
IDX_TO_LABEL = {idx: cat for idx, cat in enumerate(CATEGORIES)}

def ensure_dataset_exists():
    if PARQUET_PATH.exists():
        return
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading fashion products dataset (parquet)...")
    req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(PARQUET_PATH, "wb") as f:
        f.write(response.read())
    print("Download complete.")

def load_catalog_data(sample_size: int = 5000):
    ensure_dataset_exists()
    table = pq.read_table(PARQUET_PATH)
    df = table.select(["masterCategory", "image"]).to_pandas()
    df = df[df["masterCategory"].isin(CATEGORIES)].reset_index(drop=True)

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    images = []
    labels = []
    for _, row in df.iterrows():
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB").resize((60, 60))
        images.append(np.array(img, dtype=np.float32) / 255.0)
        labels.append(LABEL_TO_IDX[row["masterCategory"]])

    X = np.stack(images)
    y = np.array(labels, dtype=np.int32)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def build_catalog_cnn(input_shape=(60, 60, 3), num_classes=4) -> keras.Sequential:
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
    ])

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        augmentation,
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_catalog_data(sample_size=5000)

    model = build_catalog_cnn()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1,
    )

    y_probs = model.predict(X_test, verbose=0)
    y_preds = np.argmax(y_probs, axis=1)

    acc = accuracy_score(y_test, y_preds)
    cm = confusion_matrix(y_test, y_preds)

    print("\n" + "=" * 55)
    print("DAY 6 FASHION PRODUCT CATALOG CNN EVALUATION")
    print("=" * 55)
    print(f"Overall Test Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_preds, target_names=CATEGORIES))

    print("Confusion Matrix:")
    header = f"{'Actual \\ Predicted':<20}" + "".join(f"{c:>15}" for c in CATEGORIES)
    print(header)
    print("-" * len(header))
    for idx, row in enumerate(cm):
        row_str = f"{CATEGORIES[idx]:<20}" + "".join(f"{count:>15}" for count in row)
        print(row_str)
    print("=" * 55)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(IDX_TO_LABEL, f, indent=2)

    print(f"\nSaved CNN model to: {MODEL_PATH}")
    print(f"Saved class metadata to: {METADATA_PATH}")

    return acc, cm

if __name__ == "__main__":
    train_and_evaluate()

