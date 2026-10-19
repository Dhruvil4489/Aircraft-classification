"""Integration tests verifying real ImageNet VGG16 loading, BLIP inference, and Keras layer execution."""

from pathlib import Path

import numpy as np
import pytest

from aircraft_ai import config
from aircraft_ai.blip_layer import generate_text, get_blip_components
from aircraft_ai.captioning import describe_image
from aircraft_ai.classifier import build_vgg16_classifier


@pytest.mark.slow
def test_real_imagenet_vgg16_load():
    """Verify ImageNet-pretrained VGG16 downloads/loads cleanly and builds feature extractor."""
    model = build_vgg16_classifier(weights="imagenet", freeze_backbone=True)
    assert model is not None
    # Verify input shape
    assert model.input_shape == (None, 224, 224, 3)
    # Output shape should be (None, 1)
    assert model.output_shape == (None, 1)


@pytest.mark.slow
def test_real_blip_load_and_inference(synthetic_image_rgb):
    """Verify real Hugging Face BLIP model loads and produces caption and summary."""
    # Caption
    caption_res = describe_image(synthetic_image_rgb, task="caption")
    assert "raw_generated_text" in caption_res
    assert len(caption_res["raw_generated_text"]) > 0
    assert caption_res["task"] == "caption"

    # Summary
    summary_res = describe_image(synthetic_image_rgb, task="summary")
    assert "raw_generated_text" in summary_res
    assert len(summary_res["raw_generated_text"]) > 0
    assert summary_res["task"] == "summary"


@pytest.mark.slow
def test_blip_layer_and_generate_text(synthetic_image_rgb):
    """Verify custom Keras layer BlipCaptionSummaryLayer and generate_text helper.
    
    Academic specification:
    - Returns a tf.string tensor.
    - Decoded text is non-empty.
    - Does NOT return the error sentinel 'Error processing image'.
    """
    import tensorflow as tf

    path_tensor = tf.constant(str(synthetic_image_rgb.resolve()))
    task_tensor = tf.constant("caption")

    result_tensor = generate_text(path_tensor, task_tensor)
    assert isinstance(result_tensor, tf.Tensor)

    decoded = bytes(result_tensor.numpy()).decode("utf-8")
    assert len(decoded) > 0
    assert decoded != "Error processing image", "Layer returned error sentinel for a valid image!"
