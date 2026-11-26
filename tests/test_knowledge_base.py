import csv

from src.knowledge_base import chunk_text, load_catalog_documents, product_to_text


def test_chunk_text_keeps_all_text_and_adds_overlap():
    text = "one two three four five six seven eight nine ten"
    chunks = chunk_text(text, chunk_size=20, overlap=5)

    assert len(chunks) > 1
    assert all(len(chunk) <= 20 for chunk in chunks)
    assert "one" in chunks[0]
    assert "ten" in chunks[-1]
    assert chunks[0].split()[-1] in chunks[1].split()


def test_chunk_text_rejects_invalid_overlap():
    try:
        chunk_text("sample text", chunk_size=10, overlap=10)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("Expected invalid overlap to raise ValueError")


def test_product_to_text_includes_catalog_details():
    product = {
        "productDisplayName": "Blue Cotton Shirt",
        "masterCategory": "Apparel",
        "baseColour": "Blue",
    }

    text = product_to_text(product)

    assert "Product: Blue Cotton Shirt" in text
    assert "Category: Apparel" in text
    assert "Color: Blue" in text


def test_load_catalog_documents_keeps_search_metadata(tmp_path):
    catalog_path = tmp_path / "styles.csv"
    with catalog_path.open("w", encoding="utf-8", newline="") as catalog_file:
        writer = csv.DictWriter(
            catalog_file,
            fieldnames=["id", "productDisplayName", "masterCategory", "baseColour"],
        )
        writer.writeheader()
        writer.writerow({
            "id": "123",
            "productDisplayName": "Blue Cotton Shirt",
            "masterCategory": "Apparel",
            "baseColour": "Blue",
        })

    documents = load_catalog_documents(catalog_path)

    assert len(documents) == 1
    assert documents[0]["metadata"]["product_id"] == "123"
    assert documents[0]["metadata"]["source_type"] == "product"
