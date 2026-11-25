import numpy as np
import pytest
from PIL import Image
from src.catalog_segmenter import composite_on_background, segment_product


def test_segment_product_with_pil_and_numpy():
    img = Image.new("RGB", (60, 60), color=(255, 255, 255))
    arr = np.array(img)
    arr[15:45, 15:45] = [30, 80, 180]

    mask_from_pil = segment_product(Image.fromarray(arr))
    mask_from_np = segment_product(arr)

    assert mask_from_pil.shape == (60, 60)
    assert np.array_equal(mask_from_pil, mask_from_np)
    assert mask_from_pil[30, 30] == 255
    assert mask_from_pil[0, 0] == 0


def test_segment_product_invalid_input_raises():
    with pytest.raises(ValueError):
        segment_product(12345)


def test_composite_on_background_replaces_background():
    arr = np.full((40, 40, 3), 255, dtype=np.uint8)
    arr[10:30, 10:30] = [50, 100, 150]
    mask = segment_product(arr)

    composited = composite_on_background(arr, mask, bg_color=(10, 20, 30))
    assert composited.shape == (40, 40, 3)
    assert tuple(composited[0, 0]) == (10, 20, 30)
    assert tuple(composited[20, 20]) == (50, 100, 150)
