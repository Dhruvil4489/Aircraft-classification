"""Unit tests for CLIP-based domain relevance screening and out-of-distribution rejection."""

import pytest
from PIL import Image, ImageDraw

from aircraft_ai import config
from aircraft_ai.domain_screening import screen_image_domain, load_pil_image_safely


@pytest.fixture
def signature_image():
    """Create a synthetic high-contrast handwritten signature fixture."""
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line([(30, 100), (80, 40), (120, 150), (160, 60), (200, 120), (250, 50), (320, 160), (370, 90)], fill=(10, 20, 120), width=3)
    draw.line([(50, 140), (350, 130)], fill=(10, 20, 120), width=2)
    return img


@pytest.fixture
def document_image():
    """Create a synthetic text document / invoice fixture."""
    img = Image.new("RGB", (300, 400), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)
    for y in range(40, 360, 25):
        draw.line([(40, y), (260, y)], fill=(40, 40, 40), width=2)
    return img


@pytest.fixture
def transparent_image():
    """Create a transparent RGBA image fixture."""
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    return img


def test_signature_is_rejected_as_out_of_domain(signature_image):
    """Verify that a handwritten signature is rejected and classified as non-aircraft."""
    result = screen_image_domain(signature_image, threshold=config.DOMAIN_ACCEPTANCE_THRESHOLD)
    
    assert result["is_aircraft_domain"] is False, "Signature should be rejected as out-of-domain"
    assert result["in_domain_score"] < config.DOMAIN_ACCEPTANCE_THRESHOLD
    assert "Signature" in result["matched_category"] or "Drawing" in result["matched_category"]
    assert result["rejection_reason"] is not None


def test_document_is_rejected_as_out_of_domain(document_image):
    """Verify that a printed text document is rejected."""
    result = screen_image_domain(document_image, threshold=config.DOMAIN_ACCEPTANCE_THRESHOLD)
    
    assert result["is_aircraft_domain"] is False, "Document should be rejected as out-of-domain"
    assert result["in_domain_score"] < config.DOMAIN_ACCEPTANCE_THRESHOLD


def test_real_aircraft_test_images_are_accepted():
    """Verify that genuine aircraft structural test images pass domain screening."""
    dent_candidates = list(config.TEST_DIR.glob("dent/*.jpg"))
    crack_candidates = list(config.TEST_DIR.glob("crack/*.jpg"))
    
    assert len(dent_candidates) > 0, "Dent test images must exist"
    assert len(crack_candidates) > 0, "Crack test images must exist"
    
    dent_sample = dent_candidates[0]
    crack_sample = crack_candidates[0]
    
    dent_res = screen_image_domain(dent_sample)
    assert dent_res["is_aircraft_domain"] is True, f"Real dent image {dent_sample} must pass domain screening"
    assert dent_res["in_domain_score"] >= config.DOMAIN_ACCEPTANCE_THRESHOLD
    
    crack_res = screen_image_domain(crack_sample)
    assert crack_res["is_aircraft_domain"] is True, f"Real crack image {crack_sample} must pass domain screening"
    assert crack_res["in_domain_score"] >= config.DOMAIN_ACCEPTANCE_THRESHOLD


def test_safe_image_loader_transparency(transparent_image):
    """Verify that transparent RGBA images are safely composited to 3-channel RGB."""
    rgb_img = load_pil_image_safely(transparent_image)
    assert rgb_img.mode == "RGB"
    assert rgb_img.size == (100, 100)
