import argparse
import csv
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = ROOT_DIR / "data" / "raw" / "fashion_products" / "styles.csv"
DEFAULT_POLICIES_DIR = ROOT_DIR / "data" / "raw" / "policies"
DEFAULT_STORE_DIR = ROOT_DIR / "data" / "processed" / "retail_knowledge_base"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than chunk size")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            search_start = max(start + chunk_size // 2, start + overlap + 1)
            space = text.rfind(" ", search_start, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap

    return chunks


def product_to_text(product: dict[str, str]) -> str:
    fields = [
        ("Product", "productDisplayName"),
        ("Department", "gender"),
        ("Category", "masterCategory"),
        ("Subcategory", "subCategory"),
        ("Type", "articleType"),
        ("Color", "baseColour"),
        ("Season", "season"),
        ("Year", "year"),
        ("Usage", "usage"),
    ]
    details = [f"{label}: {product[key]}" for label, key in fields if product.get(key)]
    return ". ".join(details)


def load_catalog_documents(catalog_path: Path) -> list[dict]:
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog CSV not found at {catalog_path}. Download styles.csv from "
            "fashion-product-images-small and place it there, or pass --catalog."
        )

    documents = []
    with catalog_path.open("r", encoding="utf-8", newline="") as catalog_file:
        for row in csv.DictReader(catalog_file):
            text = product_to_text(row)
            if not text:
                continue
            documents.append({
                "text": text,
                "metadata": {
                    "source": str(catalog_path.name),
                    "source_type": "product",
                    "product_id": row.get("id", ""),
                    "product_name": row.get("productDisplayName", ""),
                    "category": row.get("masterCategory", ""),
                    "subcategory": row.get("subCategory", ""),
                    "article_type": row.get("articleType", ""),
                    "color": row.get("baseColour", ""),
                },
            })
    return documents


def load_policy_documents(policies_dir: Path) -> list[dict]:
    documents = []
    for policy_path in sorted(policies_dir.glob("*.md")):
        text = policy_path.read_text(encoding="utf-8").strip()
        if text:
            documents.append({
                "text": text,
                "metadata": {
                    "source": policy_path.name,
                    "source_type": "policy",
                },
            })
    return documents


class RetailKnowledgeBase:
    def __init__(
        self,
        store_dir: Path = DEFAULT_STORE_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.store_dir = Path(store_dir)
        self.index_path = self.store_dir / "knowledge_base.faiss"
        self.metadata_path = self.store_dir / "chunks.json"
        self.model_name = model_name
        self.model = None

    def _get_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise ImportError(
                    "Install the Day 18 dependencies with: pip install -r requirements.txt"
                ) from error
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def build(
        self,
        catalog_path: Path = DEFAULT_CATALOG_PATH,
        policies_dir: Path = DEFAULT_POLICIES_DIR,
        policies_only: bool = False,
    ) -> int:
        documents = load_policy_documents(Path(policies_dir))
        if not policies_only:
            documents.extend(load_catalog_documents(Path(catalog_path)))
        if not documents:
            raise ValueError("No product or policy documents were found to index")

        chunks = []
        for document in documents:
            for chunk_number, text_chunk in enumerate(chunk_text(document["text"])):
                chunks.append({
                    "text": text_chunk,
                    "metadata": {
                        **document["metadata"],
                        "chunk": chunk_number,
                    },
                })
        if not chunks:
            raise ValueError("The source documents did not contain any text to index")

        try:
            import faiss
            import numpy as np
        except ImportError as error:
            raise ImportError(
                "Install the Day 18 dependencies with: pip install -r requirements.txt"
            ) from error

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self._get_model().encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip():
            raise ValueError("Search query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Knowledge base is missing. Build it first with --build.")

        try:
            import faiss
            import numpy as np
        except ImportError as error:
            raise ImportError(
                "Install the Day 18 dependencies with: pip install -r requirements.txt"
            ) from error

        index = faiss.read_index(str(self.index_path))
        chunks = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        query_vector = self._get_model().encode([query], convert_to_numpy=True)
        query_vector = np.asarray(query_vector, dtype="float32")
        faiss.normalize_L2(query_vector)
        scores, result_ids = index.search(query_vector, min(top_k, len(chunks)))

        results = []
        for score, chunk_id in zip(scores[0], result_ids[0]):
            if chunk_id < 0:
                continue
            results.append({
                "score": float(score),
                "text": chunks[chunk_id]["text"],
                "metadata": chunks[chunk_id]["metadata"],
            })
        return results


def main():
    parser = argparse.ArgumentParser(description="Build and search the RetailMind knowledge base")
    parser.add_argument("--build", action="store_true", help="Create or replace the saved FAISS index")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH, help="Path to Kaggle styles.csv")
    parser.add_argument("--policies-only", action="store_true", help="Build using policy files without a catalog CSV")
    parser.add_argument("--query", help="Search the saved knowledge base")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()

    knowledge_base = RetailKnowledgeBase()
    if args.build:
        count = knowledge_base.build(catalog_path=args.catalog, policies_only=args.policies_only)
        print(f"Saved {count} chunks to {knowledge_base.store_dir}")
    if args.query:
        for result in knowledge_base.search(args.query, top_k=args.top_k):
            print(f"\nScore: {result['score']:.3f} | {result['metadata']['source']}")
            print(result["text"])
    if not args.build and not args.query:
        parser.print_help()


if __name__ == "__main__":
    main()
