from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent

# Segment foreground object using thresholding and morphological cleanup
def segment_product(image_input, threshold_val: int = 240) -> np.ndarray:
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# Replace background with uniform color using product segmentation mask
def composite_on_background(image_input, mask: np.ndarray, bg_color=(255, 255, 255)) -> np.ndarray:
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    bg = np.full_like(img, bg_color, dtype=np.uint8)
    return (img.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

