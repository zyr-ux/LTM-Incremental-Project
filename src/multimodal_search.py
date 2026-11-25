import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch

try:
    from transformers import CLIPModel, CLIPProcessor
    CLIP_AVAILABLE = True
except ImportError:
    CLIPModel = None
    CLIPProcessor = None
    CLIP_AVAILABLE = False

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_PARQUET_PATH = ROOT_DIR / "data" / "raw" / "fashion_products" / "fashion_products_sample.parquet"
DEFAULT_EMBEDDINGS_PATH = ROOT_DIR / "data" / "processed" / "clip_catalog_embeddings.npz"
DEFAULT_METADATA_PATH = ROOT_DIR / "data" / "processed" / "clip_catalog_metadata.json"
EMBEDDING_DIM = 512


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32).flatten()
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def _load_pil_image(image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, (str, Path)):
        return Image.open(image_input).convert("RGB")
    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def _mock_text_embedding(text: str) -> np.ndarray:
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    tokens = text.lower().split()
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "little") % EMBEDDING_DIM
        vec[idx] += 1.0
    if not tokens:
        vec[0] = 1.0
    return _normalize_vector(vec)


def _mock_image_embedding(image: Image.Image) -> np.ndarray:
    small = image.resize((16, 16))
    pixels = np.asarray(small, dtype=np.float32).flatten() / 255.0
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    copy_len = min(len(pixels), EMBEDDING_DIM)
    vec[:copy_len] = pixels[:copy_len]
    return _normalize_vector(vec)


class MultimodalCatalogSearch:
    def __init__(
        self,
        model_id: str = DEFAULT_CLIP_MODEL,
        embeddings_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        use_mock: bool = False,
    ):
        self.model_id = model_id
        self.embeddings_path = embeddings_path or DEFAULT_EMBEDDINGS_PATH
        self.metadata_path = metadata_path or DEFAULT_METADATA_PATH
        self.use_mock = use_mock
        self.model: Any = None
        self.processor: Any = None
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []

    def _load_model(self) -> None:
        if self.model is not None or self.use_mock or not CLIP_AVAILABLE:
            return
        if CLIPModel is None or CLIPProcessor is None:
            return
        try:
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            self.model = CLIPModel.from_pretrained(self.model_id)
            self.model.eval()
        except Exception:
            self.model = None
            self.processor = None

    def embed_text(self, text: str) -> np.ndarray:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Search text cannot be empty.")

        self._load_model()
        if self.model is not None and self.processor is not None and not self.use_mock:
            inputs = self.processor(text=[clean_text], return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
            return _normalize_vector(features[0].cpu().numpy())

        return _mock_text_embedding(clean_text)

    def embed_image(self, image_input: Union[str, Path, bytes, Image.Image]) -> np.ndarray:
        image = _load_pil_image(image_input)
        self._load_model()
        if self.model is not None and self.processor is not None and not self.use_mock:
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            return _normalize_vector(features[0].cpu().numpy())

        return _mock_image_embedding(image)

    def build_index(self, products: List[Dict[str, Any]], save: bool = True) -> int:
        if not products:
            raise ValueError("Product list cannot be empty.")

        vectors: List[np.ndarray] = []
        clean_metadata: List[Dict[str, Any]] = []

        for item in products:
            name = str(item.get("productDisplayName", "Unknown Product"))
            category = str(item.get("masterCategory", ""))
            colour = str(item.get("baseColour", ""))
            description = f"{name} {category} {colour}".strip()

            text_vec = self.embed_text(description)
            if item.get("image") is not None:
                img_vec = self.embed_image(item["image"])
                combined = _normalize_vector((text_vec + img_vec) / 2.0)
            else:
                combined = text_vec

            vectors.append(combined)
            clean_metadata.append({
                "id": item.get("id"),
                "productDisplayName": name,
                "masterCategory": category,
                "subCategory": item.get("subCategory", ""),
                "articleType": item.get("articleType", ""),
                "baseColour": colour,
            })

        self.embeddings = np.vstack(vectors).astype(np.float32)
        self.metadata = clean_metadata

        if save:
            self.save_index()
        return len(self.metadata)

    def build_index_from_parquet(
        self,
        parquet_path: Optional[Path] = None,
        limit: int = 200,
        save: bool = True,
    ) -> int:
        source_path = parquet_path or DEFAULT_PARQUET_PATH
        if not source_path.exists():
            raise FileNotFoundError(f"Catalog parquet not found: {source_path}")

        df = pd.read_parquet(source_path).head(limit)
        products: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            img_field = row.get("image")
            img_bytes = img_field.get("bytes") if isinstance(img_field, dict) else img_field
            products.append({
                "id": int(row["id"]) if pd.notna(row.get("id")) else None,
                "productDisplayName": row.get("productDisplayName", ""),
                "masterCategory": row.get("masterCategory", ""),
                "subCategory": row.get("subCategory", ""),
                "articleType": row.get("articleType", ""),
                "baseColour": row.get("baseColour", ""),
                "image": img_bytes,
            })
        return self.build_index(products, save=save)

    def save_index(self) -> None:
        if self.embeddings is None:
            raise ValueError("No embeddings to save.")
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.embeddings_path, embeddings=self.embeddings)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def load_index(self) -> bool:
        if not self.embeddings_path.exists() or not self.metadata_path.exists():
            return False
        data = np.load(self.embeddings_path)
        self.embeddings = data["embeddings"].astype(np.float32)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        return True

    def _cosine_search(self, query_vec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        if self.embeddings is None or not self.metadata:
            if not self.load_index():
                raise ValueError("Catalog index is empty. Build or load an index first.")

        assert self.embeddings is not None
        scores = self.embeddings @ query_vec
        k = min(max(top_k, 1), len(self.metadata))
        top_indices = np.argsort(scores)[::-1][:k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            item = dict(self.metadata[int(idx)])
            item["score"] = round(float(scores[int(idx)]), 4)
            results.append(item)
        return results

    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vec = self.embed_text(query)
        return self._cosine_search(query_vec, top_k=top_k)

    def search_by_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        query_vec = self.embed_image(image_input)
        return self._cosine_search(query_vec, top_k=top_k)
