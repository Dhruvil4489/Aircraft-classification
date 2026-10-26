"""Grad-CAM explainability module for VGG16 aircraft damage classification.

Computes genuine Gradient-weighted Class Activation Mapping (Grad-CAM)
on the last convolutional layer (block5_conv3) of the VGG16 backbone.
Produces high-resolution attention heatmaps and transparently blended
inspection overlays for visual explainability.
"""

import base64
import io
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

from aircraft_ai import config
from aircraft_ai.inference import get_model, preprocess_image
from aircraft_ai.logging_setup import setup_logger

logger = setup_logger("aircraft_ai.gradcam")

_GRADCAM_LOCK = threading.Lock()
_CONV_MODEL: Optional[tf.keras.Model] = None
_HEAD_MODEL: Optional[tf.keras.Model] = None


def get_gradcam_submodels(model: Any = None) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Extract or lazily initialize the conv and head submodels for Grad-CAM."""
    global _CONV_MODEL, _HEAD_MODEL

    if _CONV_MODEL is not None and _HEAD_MODEL is not None:
        return _CONV_MODEL, _HEAD_MODEL

    with _GRADCAM_LOCK:
        if _CONV_MODEL is None or _HEAD_MODEL is None:
            active_model = model if model is not None else get_model()
            base_model = active_model.get_layer("vgg16_feature_extractor")
            last_conv = base_model.get_layer("block5_conv3")

            # 1. Feature extractor up to last conv layer
            conv_model = tf.keras.Model(
                base_model.input,
                last_conv.output,
                name="gradcam_conv_feature_extractor",
            )

            # 2. Classifier head from last conv output to final sigmoid
            block5_pool = base_model.get_layer("block5_pool")
            flatten = base_model.get_layer("flatten")

            head_input = tf.keras.Input(shape=last_conv.output.shape[1:])
            x = block5_pool(head_input)
            x = flatten(x)
            for layer in active_model.layers[1:-1]:
                x = layer(x)

            # Pre-activation logit linear layer (avoids sigmoid vanishing gradient saturation)
            final_layer = active_model.layers[-1]
            dense_linear = tf.keras.layers.Dense(1, name="gradcam_logit_linear")
            logit = dense_linear(x)

            head_model = tf.keras.Model(head_input, logit, name="gradcam_head_classifier")
            dense_linear.set_weights(final_layer.get_weights())

            _CONV_MODEL = conv_model
            _HEAD_MODEL = head_model
            logger.info("Initialized Grad-CAM submodels for block5_conv3 (pre-activation logit head).")

    return _CONV_MODEL, _HEAD_MODEL


def determine_focus_area(heatmap_2d: np.ndarray) -> str:
    """Analyze spatial concentration of the 2D attention activation."""
    h, w = heatmap_2d.shape
    mid_h, mid_w = h // 2, w // 2

    q_top_left = float(np.mean(heatmap_2d[:mid_h, :mid_w]))
    q_top_right = float(np.mean(heatmap_2d[:mid_h, mid_w:]))
    q_bot_left = float(np.mean(heatmap_2d[mid_h:, :mid_w]))
    q_bot_right = float(np.mean(heatmap_2d[mid_h:, mid_w:]))
    center = float(np.mean(heatmap_2d[mid_h // 2 : mid_h + mid_h // 2, mid_w // 2 : mid_w + mid_w // 2]))

    quads = {
        "upper-left surface region": q_top_left,
        "upper-right surface region": q_top_right,
        "lower-left surface region": q_bot_left,
        "lower-right surface region": q_bot_right,
        "central panel region": center,
    }
    top_region = max(quads, key=quads.get)  # type: ignore
    return top_region


def compute_gradcam(
    image_input: Union[str, Path, bytes, Image.Image],
    target_class: str = "dent",
    model: Any = None,
    alpha: float = 0.45,
) -> Dict[str, Any]:
    """Compute genuine Grad-CAM attention heatmap and blended inspection overlay.

    Args:
        image_input: Input image file path, bytes, or PIL Image.
        target_class: Target class ('dent' or 'crack') to compute gradients for.
        model: Optional compiled VGG16 model.
        alpha: Heatmap blend opacity on the original image (0.0 to 1.0).

    Returns:
        Structured dictionary containing base64 data URLs for heatmap and overlay,
        activation statistics, spatial focus region, and engineering disclaimers.
    """
    try:
        conv_model, head_model = get_gradcam_submodels(model)
        processed_batch = preprocess_image(image_input)

        with tf.GradientTape() as tape:
            conv_out = conv_model(processed_batch)
            tape.watch(conv_out)
            logit_out = head_model(conv_out)
            logit_val = logit_out[0][0]

            # Pre-sigmoid linear logit target score avoids sigmoid vanishing gradients
            if target_class == "dent":
                target_score = logit_val
            else:
                target_score = -logit_val

        grads = tape.gradient(target_score, conv_out)
        if grads is None:
            raise ValueError("Gradient calculation returned None for conv activations.")

        # Global average pooling of gradients: channel importance weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the conv activations by importance
        cam = tf.reduce_sum(conv_out[0] * pooled_grads, axis=-1)

        # Apply ReLU to retain only positive evidence
        cam = tf.maximum(cam, 0.0)

        # Normalize to [0, 1]
        max_val = tf.reduce_max(cam)
        if max_val > 0:
            cam = cam / max_val
        cam_np = cam.numpy()

        focus_region = determine_focus_area(cam_np)

        # Resize CAM to 224x224 matching input dimensions
        cam_img = Image.fromarray((cam_np * 255.0).astype(np.uint8)).resize(
            (config.IMG_ROWS, config.IMG_COLS),
            resample=Image.Resampling.BILINEAR,
        )
        cam_normalized = np.array(cam_img, dtype=np.float32) / 255.0

        # Apply Jet colormap
        cmap = plt.get_cmap("jet")
        rgba_heatmap = cmap(cam_normalized)
        rgb_heatmap = (rgba_heatmap[:, :, :3] * 255.0).astype(np.uint8)
        pil_heatmap = Image.fromarray(rgb_heatmap)

        # Create blended overlay on the 224x224 RGB image
        orig_img_224 = Image.fromarray((processed_batch[0] * 255.0).astype(np.uint8))
        orig_np = np.array(orig_img_224, dtype=np.float32)
        blended_np = (1.0 - alpha) * orig_np + alpha * rgb_heatmap
        blended_np = np.clip(blended_np, 0, 255).astype(np.uint8)
        pil_blended = Image.fromarray(blended_np)

        # Encode to Base64 PNG
        buf_hm = io.BytesIO()
        pil_heatmap.save(buf_hm, format="PNG")
        b64_hm = "data:image/png;base64," + base64.b64encode(buf_hm.getvalue()).decode("utf-8")

        buf_ov = io.BytesIO()
        pil_blended.save(buf_ov, format="PNG")
        b64_ov = "data:image/png;base64," + base64.b64encode(buf_ov.getvalue()).decode("utf-8")

        # Encode 224x224 original RGB preview
        buf_orig = io.BytesIO()
        orig_img_224.save(buf_orig, format="PNG")
        b64_orig = "data:image/png;base64," + base64.b64encode(buf_orig.getvalue()).decode("utf-8")

        return {
            "available": True,
            "method": "Grad-CAM (VGG16 block5_conv3)",
            "target_class": target_class,
            "focus_region": focus_region,
            "mean_activation": round(float(np.mean(cam_np)), 4),
            "max_activation": round(float(np.max(cam_np)), 4),
            "heatmap_data_url": b64_hm,
            "overlay_data_url": b64_ov,
            "preprocessed_image_data_url": b64_orig,
            "disclaimer": (
                "Grad-CAM attention heatmap highlights receptive field features in VGG16 conv layers "
                "that contributed to the predicted class score. It represents model explainability, "
                "not certified structural damage boundaries or NDT measurements."
            ),
        }

    except Exception as exc:
        logger.warning("Grad-CAM generation failed: %s", exc)
        return {
            "available": False,
            "method": "Grad-CAM",
            "target_class": target_class,
            "error": str(exc),
            "disclaimer": (
                "Attention heatmap could not be generated. Defect classification was performed "
                "at the whole-image level."
            ),
        }
