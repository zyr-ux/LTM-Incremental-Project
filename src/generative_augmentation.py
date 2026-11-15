import sys
from pathlib import Path
import os
import io
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
import matplotlib.pyplot as plt

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
VAE_ENCODER_PATH = ROOT_DIR / "models" / "vae_encoder_personal_care.keras"
VAE_DECODER_PATH = ROOT_DIR / "models" / "vae_decoder_personal_care.keras"
BASELINE_MODEL_PATH = ROOT_DIR / "models" / "product_catalog_cnn_v6.keras"
AUGMENTED_MODEL_PATH = ROOT_DIR / "models" / "product_catalog_cnn_augmented_v11.keras"
EVALUATION_PLOT_PATH = ROOT_DIR / "docs" / "day11_synthetic_augmentation_results.png"
METRICS_REPORT_PATH = ROOT_DIR / "docs" / "day11_augmentation_metrics.json"

CATEGORIES = ["Apparel", "Accessories", "Footwear", "Personal Care"]
LABEL_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}
IDX_TO_LABEL = {idx: cat for idx, cat in enumerate(CATEGORIES)}
TARGET_CLASS = "Personal Care"
TARGET_CLASS_IDX = LABEL_TO_IDX[TARGET_CLASS]

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class ProductVAE(keras.Model):
    def __init__(self, encoder, decoder, kl_weight=0.0001, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + (self.kl_weight * kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def build_vae(input_shape=(60, 60, 3), latent_dim=32):
    encoder_inputs = keras.Input(shape=input_shape, name="encoder_input")
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
    x = layers.Dense(15 * 15 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((15, 15, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    decoder_outputs = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    vae = ProductVAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
    return vae, encoder, decoder

def load_catalog_data(sample_size: int = 5000):
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

def build_classifier(input_shape=(60, 60, 3), num_classes=4):
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

def filter_realistic_samples(generated_images, std_threshold=0.08, mean_bounds=(0.70, 0.98)):
    realistic_samples = []
    for img in generated_images:
        img_std = float(np.std(img))
        img_mean = float(np.mean(img))
        if img_std >= std_threshold and mean_bounds[0] <= img_mean <= mean_bounds[1]:
            realistic_samples.append(img)
    return np.array(realistic_samples, dtype=np.float32)

def generate_quality_samples(decoder, num_required=150, latent_dim=32):
    accepted = []
    while len(accepted) < num_required:
        batch_z = np.random.normal(0.0, 0.9, size=(num_required * 2, latent_dim))
        candidates = decoder.predict(batch_z, verbose=0)
        filtered = filter_realistic_samples(candidates)
        for img in filtered:
            accepted.append(img)
            if len(accepted) >= num_required:
                break
    return np.array(accepted, dtype=np.float32)

def run_experiment(vae_epochs=12, classifier_epochs=15, num_synthetic=150):
    print("Loading catalog dataset (5,000 samples)...")
    X_train, X_test, y_train, y_test = load_catalog_data(sample_size=5000)

    print("\nTraining Baseline Classifier without generative augmentation...")
    baseline_model = build_classifier()
    early_stop_baseline = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    baseline_model.fit(
        X_train, y_train,
        epochs=classifier_epochs,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop_baseline],
        verbose=1,
    )

    y_pred_base = np.argmax(baseline_model.predict(X_test, verbose=0), axis=1)
    acc_base = float(accuracy_score(y_test, y_pred_base))
    report_base = classification_report(y_test, y_pred_base, target_names=CATEGORIES, output_dict=True)
    cm_base = confusion_matrix(y_test, y_pred_base)

    target_train_mask = (y_train == TARGET_CLASS_IDX)
    X_target_train = X_train[target_train_mask]
    print(f"\nTraining VAE on under-represented class '{TARGET_CLASS}' ({len(X_target_train)} training samples)...")
    vae, encoder, decoder = build_vae(latent_dim=32)
    vae.fit(X_target_train, epochs=vae_epochs, batch_size=32, verbose=1)

    VAE_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    encoder.save(VAE_ENCODER_PATH)
    decoder.save(VAE_DECODER_PATH)

    print(f"\nGenerating and quality-filtering {num_synthetic} realistic synthetic samples...")
    final_synthetic = generate_quality_samples(decoder, num_required=num_synthetic, latent_dim=32)
    print(f"Retained {len(final_synthetic)} realistic synthetic samples for '{TARGET_CLASS}'.")

    synthetic_labels = np.full(shape=(len(final_synthetic),), fill_value=TARGET_CLASS_IDX, dtype=np.int32)
    X_train_augmented = np.concatenate([X_train, final_synthetic], axis=0)
    y_train_augmented = np.concatenate([y_train, synthetic_labels], axis=0)

    shuffle_indices = np.random.permutation(len(X_train_augmented))
    X_train_augmented = X_train_augmented[shuffle_indices]
    y_train_augmented = y_train_augmented[shuffle_indices]

    print("\nTraining Augmented Classifier with synthetic images folded in...")
    augmented_model = build_classifier()
    early_stop_aug = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    augmented_model.fit(
        X_train_augmented, y_train_augmented,
        epochs=classifier_epochs,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop_aug],
        verbose=1,
    )

    y_pred_aug = np.argmax(augmented_model.predict(X_test, verbose=0), axis=1)
    acc_aug = float(accuracy_score(y_test, y_pred_aug))
    report_aug = classification_report(y_test, y_pred_aug, target_names=CATEGORIES, output_dict=True)
    cm_aug = confusion_matrix(y_test, y_pred_aug)

    augmented_model.save(AUGMENTED_MODEL_PATH)

    print("\n" + "=" * 60)
    print("DAY 11 GENERATIVE AUGMENTATION RESULTS COMPARISON")
    print("=" * 60)
    print(f"Overall Accuracy: Baseline = {acc_base * 100:.2f}% | Augmented = {acc_aug * 100:.2f}%")
    print("-" * 60)
    for cat in CATEGORIES:
        base_rec = report_base[cat]["recall"] * 100
        aug_rec = report_aug[cat]["recall"] * 100
        base_f1 = report_base[cat]["f1-score"] * 100
        aug_f1 = report_aug[cat]["f1-score"] * 100
        print(f"{cat:<15} Recall: {base_rec:5.1f}% -> {aug_rec:5.1f}% | F1: {base_f1:5.1f}% -> {aug_f1:5.1f}%")
    print("=" * 60)

    metrics_payload = {
        "overall_accuracy_baseline": acc_base,
        "overall_accuracy_augmented": acc_aug,
        "target_class": TARGET_CLASS,
        "baseline_metrics": {c: report_base[c] for c in CATEGORIES},
        "augmented_metrics": {c: report_aug[c] for c in CATEGORIES},
        "confusion_matrix_baseline": cm_base.tolist(),
        "confusion_matrix_augmented": cm_aug.tolist(),
        "num_synthetic_samples_added": len(final_synthetic),
    }
    with open(METRICS_REPORT_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    create_comparison_plot(final_synthetic[:8], cm_base, cm_aug, report_base, report_aug)
    return metrics_payload

def create_comparison_plot(sample_images, cm_base, cm_aug, report_base, report_aug):
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    for i in range(min(4, len(sample_images))):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(sample_images[i])
        ax.set_title(f"Synthetic Sample {i+1}", fontsize=10)
        ax.axis("off")

    ax_cm1 = fig.add_subplot(gs[1, :2])
    ax_cm1.imshow(cm_base, cmap="Blues")
    ax_cm1.set_title("Baseline Confusion Matrix (Day 6)")
    ax_cm1.set_xticks(range(len(CATEGORIES)))
    ax_cm1.set_yticks(range(len(CATEGORIES)))
    ax_cm1.set_xticklabels(CATEGORIES, rotation=25, ha="right", fontsize=8)
    ax_cm1.set_yticklabels(CATEGORIES, fontsize=8)
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            ax_cm1.text(j, i, str(cm_base[i, j]), ha="center", va="center", color="black" if cm_base[i, j] < cm_base.max()/2 else "white")

    ax_cm2 = fig.add_subplot(gs[1, 2:])
    ax_cm2.imshow(cm_aug, cmap="Greens")
    ax_cm2.set_title("Augmented Confusion Matrix (Day 11 VAE)")
    ax_cm2.set_xticks(range(len(CATEGORIES)))
    ax_cm2.set_yticks(range(len(CATEGORIES)))
    ax_cm2.set_xticklabels(CATEGORIES, rotation=25, ha="right", fontsize=8)
    ax_cm2.set_yticklabels(CATEGORIES, fontsize=8)
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            ax_cm2.text(j, i, str(cm_aug[i, j]), ha="center", va="center", color="black" if cm_aug[i, j] < cm_aug.max()/2 else "white")

    ax_bar = fig.add_subplot(gs[2, :])
    x = np.arange(len(CATEGORIES))
    width = 0.35
    base_recalls = [report_base[c]["recall"] * 100 for c in CATEGORIES]
    aug_recalls = [report_aug[c]["recall"] * 100 for c in CATEGORIES]
    ax_bar.bar(x - width/2, base_recalls, width, label="Baseline Recall", color="#4a90e2")
    ax_bar.bar(x + width/2, aug_recalls, width, label="VAE Augmented Recall", color="#50e3c2")
    ax_bar.set_ylabel("Recall (%)")
    ax_bar.set_title("Per-Class Recall Comparison (Fairness Lift on Under-represented Class)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(CATEGORIES)
    ax_bar.set_ylim(0, 110)
    ax_bar.legend()
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)

    EVALUATION_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(EVALUATION_PLOT_PATH, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved evaluation plot to: {EVALUATION_PLOT_PATH}")

if __name__ == "__main__":
    run_experiment()
