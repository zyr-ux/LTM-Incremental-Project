import sys
from pathlib import Path
import json
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
from src.multimodal_search import MultimodalCatalogSearch

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "product_catalog_cnn_augmented_v11.keras"
FALLBACK_MODEL_PATH = ROOT_DIR / "models" / "product_catalog_cnn_v6.keras"
MODEL_PATH = DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else FALLBACK_MODEL_PATH
METADATA_PATH = ROOT_DIR / "models" / "product_catalog_classes_v6.json"

_model = None
_classes = None

def get_model_and_classes():
    global _model, _classes
    if _model is None:
        import keras
        _model = keras.models.load_model(MODEL_PATH)
    if _classes is None:
        with open(METADATA_PATH, "r") as f:
            _classes = {int(k): v for k, v in json.load(f).items()}
    return _model, _classes

def classify_product_image(image_input) -> Dict[str, Any]:
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


_multimodal_search: Optional[MultimodalCatalogSearch] = None


def get_multimodal_search() -> MultimodalCatalogSearch:
    global _multimodal_search
    if _multimodal_search is None:
        _multimodal_search = MultimodalCatalogSearch()
        if not _multimodal_search.load_index():
            _multimodal_search.build_index_from_parquet(limit=100, save=True)
    return _multimodal_search


def search_catalog_by_text(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    searcher = get_multimodal_search()
    return searcher.search_by_text(query=query, top_k=top_k)


def search_catalog_by_image(image_input: Any, top_k: int = 5) -> List[Dict[str, Any]]:
    searcher = get_multimodal_search()
    return searcher.search_by_image(image_input=image_input, top_k=top_k)

