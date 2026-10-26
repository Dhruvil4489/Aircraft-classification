"""Unit tests for Grad-CAM explainability module."""

import pytest
from PIL import Image

from aircraft_ai import config
from aircraft_ai.gradcam import compute_gradcam, determine_focus_area, get_gradcam_submodels
from aircraft_ai.inference import get_model


@pytest.fixture
def real_dent_image():
    dent_candidates = list(config.TEST_DIR.glob("dent/*.jpg"))
    assert len(dent_candidates) > 0
    return dent_candidates[0]


@pytest.fixture
def real_crack_image():
    crack_candidates = list(config.TEST_DIR.glob("crack/*.jpg"))
    assert len(crack_candidates) > 0
    return crack_candidates[0]


def test_gradcam_submodels_initialization():
    """Verify that Grad-CAM submodels initialize correctly from VGG16."""
    model = get_model()
    conv_model, head_model = get_gradcam_submodels(model)
    assert conv_model is not None
    assert head_model is not None
    assert conv_model.layers[-1].name == "block5_conv3"
    assert conv_model.output.shape[1:] == (14, 14, 512)


def test_compute_gradcam_on_dent(real_dent_image):
    """Verify Grad-CAM produces valid heatmaps and overlays for a dent image."""
    result = compute_gradcam(real_dent_image, target_class="dent")
    assert result["available"] is True
    assert result["method"] == "Grad-CAM (VGG16 block5_conv3)"
    assert result["target_class"] == "dent"
    assert result["focus_region"] is not None
    assert 0.0 <= result["mean_activation"] <= 1.0
    assert 0.0 <= result["max_activation"] <= 1.0
    assert result["heatmap_data_url"].startswith("data:image/png;base64,")
    assert result["overlay_data_url"].startswith("data:image/png;base64,")
    assert result["preprocessed_image_data_url"].startswith("data:image/png;base64,")
    assert "disclaimer" in result


def test_compute_gradcam_on_crack(real_crack_image):
    """Verify Grad-CAM produces valid heatmaps and overlays for a crack image."""
    result = compute_gradcam(real_crack_image, target_class="crack")
    assert result["available"] is True
    assert result["target_class"] == "crack"
    assert len(result["heatmap_data_url"]) > 100
    assert len(result["overlay_data_url"]) > 100


def test_gradcam_graceful_error_handling():
    """Verify compute_gradcam handles non-image input gracefully without crashing."""
    result = compute_gradcam(b"corrupted_non_image_bytes", target_class="dent")
    assert result["available"] is False
    assert "error" in result
