from pathlib import Path
import json
import numpy as np
from PIL import Image
import keras

from src.catalog_detector import detect_catalog_objects
from src.catalog_segmenter import segment_product, composite_on_background
from src.catalog_explainability import generate_gradcam, overlay_gradcam

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
        "class_id": pred_idx,
        "confidence": round(float(probs[pred_idx]), 4),
        "all_probabilities": {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))},
    }

# Unified CV pipeline combining classification, detection, segmentation, and explainability
def process_catalog_item(image_input, bg_color=(255, 255, 255)) -> dict:
    classification = classify_product(image_input)
    detections = detect_catalog_objects(image_input)
    mask = segment_product(image_input)
    clean_image = composite_on_background(image_input, mask, bg_color=bg_color)
    gradcam_heatmap = generate_gradcam(image_input, class_idx=classification["class_id"])
    explainability_overlay = overlay_gradcam(image_input, gradcam_heatmap)

    qa_passed = len(detections) > 0 or classification["confidence"] > 0.8

    return {
        "classification": classification,
        "detections": detections,
        "segmentation_mask": mask,
        "clean_image": clean_image,
        "gradcam_heatmap": gradcam_heatmap,
        "explainability_overlay": explainability_overlay,
        "qa_status": "Passed" if qa_passed else "Needs Review",
    }

# Backward-compatible lightweight inspection
def analyze_catalog_image(image_input) -> dict:
    result = process_catalog_item(image_input)
    return {
        "classification": result["classification"],
        "detections": result["detections"],
        "qa_status": result["qa_status"],
    }
