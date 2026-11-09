import sys
from pathlib import Path
import json
import numpy as np
from PIL import Image
import keras
from src.catalog_detector import detect_catalog_objects

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "product_catalog_transfer_v7.keras"
METADATA_PATH = ROOT_DIR / "models" / "product_catalog_classes_v7.json"

_classifier = None
_classes = None

def get_classifier():
    global _classifier, _classes
    if _classifier is None:
        _classifier = keras.models.load_model(MODEL_PATH)
    if _classes is None:
        with open(METADATA_PATH, "r") as f:
            _classes = {int(k): v for k, v in json.load(f).items()}
    return _classifier, _classes

def classify_product(image_input) -> dict:
    classifier, classes = get_classifier()

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input.astype("uint8")).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    img_resized = img.resize((96, 96))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    batch = np.expand_dims(img_array, axis=0)

    probs = classifier.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_category": classes.get(pred_idx, "Unknown"),
        "confidence": round(float(probs[pred_idx]), 4),
        "all_probabilities": {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))},
    }

def analyze_catalog_image(image_input) -> dict:
    classification = classify_product(image_input)
    detections = detect_catalog_objects(image_input)
    return {
        "classification": classification,
        "detections": detections,
        "qa_status": "Passed" if len(detections) > 0 or classification["confidence"] > 0.8 else "Needs Review",
    }

