from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import keras

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "product_catalog_transfer_v7.keras"

_gradcam_submodels = None

# Extract sub-models for feature map and classification head
def _get_gradcam_submodels():
    global _gradcam_submodels
    if _gradcam_submodels is None:
        model = keras.models.load_model(MODEL_PATH)
        base = model.get_layer("mobilenetv2_1.00_96")
        gap = model.get_layer("global_average_pooling2d")
        dense1 = model.get_layer("dense")
        dense2 = model.get_layer("dense_1")

        conv_submodel = keras.Model(inputs=base.input, outputs=base.get_layer("out_relu").output)

        head_input = keras.Input(shape=base.get_layer("out_relu").output.shape[1:])
        x = gap(head_input)
        x = dense1(x)
        head_output = dense2(x)
        head_submodel = keras.Model(inputs=head_input, outputs=head_output)

        _gradcam_submodels = (conv_submodel, head_submodel)
    return _gradcam_submodels

# Compute Grad-CAM heatmap highlighting visual regions driving prediction
def generate_gradcam(image_input, class_idx: int = None) -> np.ndarray:
    conv_model, head_model = _get_gradcam_submodels()

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    orig_size = img.size
    img_resized = img.resize((96, 96))
    img_array = np.expand_dims(np.array(img_resized, dtype=np.float32) / 255.0, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)
        predictions = head_model(conv_outputs)
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1).numpy()

    heatmap = np.maximum(heatmap, 0.0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return cv2.resize(heatmap, orig_size)

# Blend Grad-CAM heatmap over image as RGB overlay
def overlay_gradcam(image_input, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    if heatmap.shape[:2] != img.shape[:2]:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(img, 1.0 - alpha, colored_heatmap, alpha, 0)
    return blended

