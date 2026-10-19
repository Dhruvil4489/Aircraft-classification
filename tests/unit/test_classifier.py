"""Unit tests for VGG16 classifier architecture, class mapping, and preprocessing."""

from typing import Any

import numpy as np
import pytest
from PIL import Image

from aircraft_ai import config
from aircraft_ai.classifier import build_vgg16_classifier, compile_classifier
from aircraft_ai.inference import predict_damage, preprocess_image


@pytest.mark.fast
def test_class_mapping_specification():
    """Verify class mapping strictly satisfies academic specification."""
    assert config.CLASS_INDICES == {"crack": 0, "dent": 1}
    assert config.INVERTED_CLASS_INDICES == {0: "crack", 1: "dent"}


@pytest.mark.fast
def test_preprocessing_shape_and_scaling(synthetic_image_rgb):
    """Test preprocessing produces (1, 224, 224, 3) float32 in range [0, 1]."""
    tensor = preprocess_image(synthetic_image_rgb)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


@pytest.mark.fast
def test_preprocessing_rgb_conversion(tmp_path):
    """Test that grayscale or RGBA images are converted to 3-channel RGB."""
    # Grayscale image
    gray_path = tmp_path / "gray.png"
    Image.new("L", (100, 100), color=128).save(gray_path)
    tensor = preprocess_image(gray_path)
    assert tensor.shape == (1, 224, 224, 3)

    # RGBA image
    rgba_path = tmp_path / "rgba.png"
    Image.new("RGBA", (100, 100), color=(10, 20, 30, 255)).save(rgba_path)
    tensor = preprocess_image(rgba_path)
    assert tensor.shape == (1, 224, 224, 3)


@pytest.mark.fast
def test_model_architecture_fast():
    """Test VGG16 model structure and frozen weights using weights=None for fast unit test."""
    model = build_vgg16_classifier(weights=None, freeze_backbone=True)
    compile_classifier(model)

    # Verify base model exists and layers are frozen
    base_model = model.layers[0]
    for layer in base_model.layers:
        assert layer.trainable is False

    # Verify head architecture: Sequential([base, Dense(512), Dropout(0.3), Dense(512), Dropout(0.3), Dense(1)])
    layers = model.layers
    assert len(layers) == 6
    assert layers[1].__class__.__name__ == "Dense"
    assert layers[1].units == 512
    assert layers[1].activation.__name__ == "relu"

    assert layers[2].__class__.__name__ == "Dropout"
    assert pytest.approx(layers[2].rate) == 0.3

    assert layers[3].__class__.__name__ == "Dense"
    assert layers[3].units == 512
    assert layers[3].activation.__name__ == "relu"

    assert layers[4].__class__.__name__ == "Dropout"
    assert pytest.approx(layers[4].rate) == 0.3

    assert layers[5].__class__.__name__ == "Dense"
    assert layers[5].units == 1
    assert layers[5].activation.__name__ == "sigmoid"

    # Forward pass with random tensor
    synthetic_batch = np.random.rand(2, 224, 224, 3).astype(np.float32)
    v_zero: Any = 0
    output = model.predict(synthetic_batch, verbose=v_zero)
    assert output.shape == (2, 1)
    assert np.all(output >= 0.0) and np.all(output <= 1.0)
