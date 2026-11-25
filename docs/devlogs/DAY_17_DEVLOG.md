# Day 17 DevLog: TTS Voice & Multimodal Search

**Date**: 2026-11-25  
**Platform**: RetailMind AI  
**Increment**: 17 / 35  
**Deliverable**: TTS Voice Support Replies & CLIP Multimodal Catalog Search (Generation Track Complete)  
**Focus**: Hugging Face Text-to-Speech (`facebook/mms-tts-eng`), CLIP Image-Text Embeddings (`openai/clip-vit-base-patch32`), Cosine Similarity Search, Assistant & Catalog Integration  
**Stack**: `transformers`, `torch`, `Pillow`, `numpy`, `pandas`, `wave`  

---

## 1. Objectives & Retail Context

Accessible and engaging retail experiences combine natural voice responses with visual product discovery:
1. **Voice Support Replies (TTS)**: Customers using the RetailMind support assistant can receive spoken `.wav` audio replies alongside text responses.
2. **Multimodal Catalog Search (CLIP)**: Shoppers can search the fashion catalog using either natural language descriptions ("red running shoes") or a reference product image, powered by shared CLIP image-text embeddings.

### Day 17 Key Deliverables
- **TTS Voice Pipeline (`src/tts_voice.py`)**: Wraps Hugging Face `pipeline("text-to-speech", model="facebook/mms-tts-eng")` with standard-library `wave` `.wav` export to `data/processed/voice_replies/` and a deterministic offline fallback tone generator.
- **CLIP Multimodal Search (`src/multimodal_search.py`)**: Embeds catalog images and text metadata from `data/raw/fashion_products/fashion_products_sample.parquet` into a shared 512-d L2-normalized CLIP space (`openai/clip-vit-base-patch32`), persisting index files in `data/processed/` (`clip_catalog_embeddings.npz` and `clip_catalog_metadata.json`).
- **Search-by-Text & Search-by-Image**: Cosine similarity ranking over the shared CLIP embedding index.
- **Assistant & Catalog UX Wiring**:
  - `src/assistant_service.py`: Added `speak_reply`, `speak_assistant_reply`, and `with_voice=True` support on `chat` / `chat_with_assistant`.
  - `src/catalog_service.py`: Added `get_multimodal_search`, `search_catalog_by_text`, and `search_catalog_by_image`.

---

## 2. Architecture & Design

```text
  Customer Text Query / Product Image
                  │
                  ▼
     [CLIP Encoder (transformers)]
     (Text & Image -> 512-d Space)
                  │
                  ▼
   [Cosine Similarity vs Catalog Index]
   (data/processed/clip_catalog_*.npz/json)
                  │
                  ▼
       Top-K Matching Products

  Support Assistant Reply Text
                  │
                  ▼
     [HF TTS Pipeline / Fallback]
                  │
                  ▼
  data/processed/voice_replies/*.wav
```

---

## 3. Verification & Testing

Unit tests in `src/test_multimodal_and_tts.py` verify:
1. `test_voice_synthesizer_generates_wav`: Generates valid `.wav` audio files from support text.
2. `test_assistant_chat_with_voice`: Confirms `RetailAssistantModule.chat(..., with_voice=True)` returns both text and audio metadata.
3. `test_multimodal_search_text_and_image`: Verifies CLIP index creation, `.npz`/`.json` persistence, reload, search-by-text, and search-by-image.
4. `test_catalog_service_multimodal_helpers`: Verifies `search_catalog_by_text` and `search_catalog_by_image` integration.

All Day 17 tests, existing assistant tests (`src/test_assistant_service.py`), and `ty check` pass.
