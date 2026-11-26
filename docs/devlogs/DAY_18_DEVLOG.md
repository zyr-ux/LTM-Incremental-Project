# Day 18 DevLog: RAG Foundations — Vector Knowledge Base

**Date**: 2026-11-26  
**Platform**: RetailMind AI  
**Increment**: 18 / 35  
**Deliverable**: Searchable product and policy knowledge base  
**Focus**: Text chunking, sentence-transformer embeddings, FAISS similarity search, and persistence  
**Stack**: `sentence-transformers`, `faiss-cpu`, Python standard library

---

## Objectives

Create a persisted corpus for later grounded assistant responses, combining the Fashion Product Images `styles.csv` catalog metadata with RetailMind policy and FAQ documents.

## Implementation

- `src/knowledge_base.py` reads catalog rows and Markdown policy files, then creates chunks with a 700-character target and 100-character overlap.
- Chunks are encoded with `sentence-transformers/all-MiniLM-L6-v2`. FAISS inner-product search over normalized vectors ranks semantic matches.
- The FAISS index and JSON source metadata are saved under `data/processed/retail_knowledge_base/`.
- Policy source documents are stored in `data/raw/policies/`; the catalog CSV is expected at `data/raw/fashion_products/styles.csv` by default.

## Build and Search

```bash
python -m src.knowledge_base --build
python -m src.knowledge_base --query "How long does express shipping take?" --top-k 5
```

If the Kaggle CSV is not available, `--build --policies-only` creates a policy-only index. The first build downloads the embedding model through Hugging Face if it is not cached locally.

## Verification

`tests/test_knowledge_base.py` covers text chunking, invalid chunk settings, catalog text formatting, and metadata loading without requiring model downloads. Full semantic-search verification requires the Day 18 dependencies and catalog CSV.
