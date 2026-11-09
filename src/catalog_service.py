import sys
from pathlib import Path
import json
import numpy as np
from PIL import Image
import keras

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "product_catalog_cnn_v6.keras"
METADATA_PATH = ROOT_DIR / "models" / "product_catalog_classes_v6.json"

_model = None
_classes = None

def get_model_and_classes():
    global _model, _classes
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH)
    if _classes is None:
        with open(METADATA_PATH, "r") as f:
            _classes = {int(k): v for k, v in json.load(f).items()}
    return _model, _classes

def classify_product_image(image_input) -> dict:
    model, classes = get_model_and_classes()

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input.astype("uint8")).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    img_resized = img.resize((60, 60))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    batch = np.expand_dims(img_array, axis=0)

    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    predicted_category = classes.get(pred_idx, "Unknown")

    class_probs = {classes[i]: float(probs[i]) for i in range(len(classes))}

    return {
        "predicted_category": predicted_category,
        "confidence": round(confidence, 4),
        "all_probabilities": class_probs,
    }

