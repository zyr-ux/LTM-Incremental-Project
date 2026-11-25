from pathlib import Path
import numpy as np
from PIL import Image
from src.assistant_service import RetailAssistantModule
from src.catalog_service import search_catalog_by_image, search_catalog_by_text
from src.multimodal_search import MultimodalCatalogSearch
from src.tts_voice import VoiceSynthesizer


def test_voice_synthesizer_generates_wav():
    out_path = Path("data/processed/voice_replies/test_reply.wav")
    tts = VoiceSynthesizer(use_mock=True)
    result = tts.speak("Your return is eligible within 30 days.", output_path=str(out_path))

    assert result["status"] == "Generated"
    assert result["sample_rate"] == 16000
    assert out_path.exists()
    assert out_path.stat().st_size > 44
    if out_path.exists():
        out_path.unlink()


def test_assistant_chat_with_voice():
    out_path = Path("data/processed/voice_replies/test_chat_reply.wav")
    assistant = RetailAssistantModule(tts=VoiceSynthesizer(use_mock=True))
    result = assistant.chat(
        "What is your return policy?",
        with_voice=True,
        audio_output_path=str(out_path),
    )

    assert "response" in result
    assert "audio" in result
    assert result["audio"]["status"] == "Generated"
    assert out_path.exists()
    if out_path.exists():
        out_path.unlink()


def test_multimodal_search_text_and_image():
    emb_path = Path("data/processed/test_clip_embeddings.npz")
    meta_path = Path("data/processed/test_clip_metadata.json")

    blue_img = Image.new("RGB", (64, 64), color=(20, 60, 200))
    red_img = Image.new("RGB", (64, 64), color=(220, 30, 30))

    products = [
        {
            "id": 1,
            "productDisplayName": "Navy Blue Denim Jacket",
            "masterCategory": "Apparel",
            "subCategory": "Topwear",
            "articleType": "Jackets",
            "baseColour": "Blue",
            "image": blue_img,
        },
        {
            "id": 2,
            "productDisplayName": "Crimson Red Running Shoes",
            "masterCategory": "Footwear",
            "subCategory": "Shoes",
            "articleType": "Sports Shoes",
            "baseColour": "Red",
            "image": red_img,
        },
    ]

    searcher = MultimodalCatalogSearch(
        embeddings_path=emb_path,
        metadata_path=meta_path,
        use_mock=True,
    )
    count = searcher.build_index(products, save=True)
    assert count == 2
    assert emb_path.exists()
    assert meta_path.exists()

    text_hits = searcher.search_by_text("red running shoes", top_k=2)
    assert len(text_hits) == 2
    assert text_hits[0]["id"] == 2

    image_hits = searcher.search_by_image(blue_img, top_k=2)
    assert len(image_hits) == 2
    assert image_hits[0]["id"] == 1

    reloaded = MultimodalCatalogSearch(
        embeddings_path=emb_path,
        metadata_path=meta_path,
        use_mock=True,
    )
    assert reloaded.load_index()
    assert reloaded.embeddings is not None
    assert searcher.embeddings is not None
    assert np.allclose(reloaded.embeddings, searcher.embeddings)

    if emb_path.exists():
        emb_path.unlink()
    if meta_path.exists():
        meta_path.unlink()


def test_catalog_service_multimodal_helpers():
    import src.catalog_service as cs

    mock_searcher = MultimodalCatalogSearch(use_mock=True)
    blue_img = Image.new("RGB", (64, 64), color=(20, 60, 200))
    mock_searcher.build_index(
        [
            {
                "id": 101,
                "productDisplayName": "Blue Cotton Shirt",
                "masterCategory": "Apparel",
                "baseColour": "Blue",
                "image": blue_img,
            }
        ],
        save=False,
    )
    original = cs._multimodal_search
    cs._multimodal_search = mock_searcher
    try:
        text_results = search_catalog_by_text("blue shirt", top_k=1)
        image_results = search_catalog_by_image(blue_img, top_k=1)
        assert text_results[0]["id"] == 101
        assert image_results[0]["id"] == 101
    finally:
        cs._multimodal_search = original


if __name__ == "__main__":
    test_voice_synthesizer_generates_wav()
    test_assistant_chat_with_voice()
    test_multimodal_search_text_and_image()
    test_catalog_service_multimodal_helpers()
    print("All Day 17 TTS & Multimodal Search tests passed!")
