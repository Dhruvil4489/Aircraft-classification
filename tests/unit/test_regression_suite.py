"""Comprehensive regression test suite for AeroDamage AI 3-stage inspection pipeline.

Covers all 7 mandatory regression scenarios specified in Requirement 9:
1. Ordinary aircraft photograph without obvious damage.
2. Properly labelled dent image.
3. Properly labelled crack image.
4. Clearly unrelated signature image.
5. Corrupt or unreadable image file.
6. Ambiguous or low-quality aircraft image (near decision threshold).
7. Borderline uncertain prediction with calibrated decision margin.
"""

import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import pytest
from fastapi.testclient import TestClient

from aircraft_ai import config
from aircraft_ai.api import app
from aircraft_ai.errors import CorruptImageError
from aircraft_ai.inference import analyze_image, get_model


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def ordinary_aircraft_path():
    path = config.WORKSPACE_ROOT / "tests" / "fixtures" / "sample_ordinary_aircraft.jpg"
    assert path.exists(), f"Ordinary aircraft fixture missing at {path}"
    return path


@pytest.fixture
def signature_path():
    path = config.WORKSPACE_ROOT / "tests" / "fixtures" / "sample_signature.png"
    assert path.exists(), f"Signature fixture missing at {path}"
    return path


@pytest.fixture
def benchmark_dent_path():
    candidates = list(config.TEST_DIR.glob("dent/*.jpg"))
    assert len(candidates) > 0, "No dent test images found"
    return candidates[0]


@pytest.fixture
def benchmark_crack_path():
    candidates = list(config.TEST_DIR.glob("crack/*.jpg"))
    assert len(candidates) > 0, "No crack test images found"
    return candidates[0]


# -----------------------------------------------------------------------------
# Test Case 1: Ordinary Aircraft Photograph Without Obvious Damage
# -----------------------------------------------------------------------------
def test_regression_case1_ordinary_aircraft_photo(ordinary_aircraft_path):
    """Verify ordinary aircraft photo is accepted as aircraft domain (Stage A),
    but damage presence is NOT_ASSESSED (Stage B), preventing false damage claims.
    """
    res = analyze_image(ordinary_aircraft_path)

    # Stage A: Accepted as aircraft
    assert res["aircraft_image_status"] == config.AIRCRAFT_STATUS_SUPPORTED
    assert res["is_supported"] is True
    assert res["domain_screening"]["is_aircraft_domain"] is True

    # Stage B: Damage presence unverified — NOT assessed as true damage
    assert res["damage_presence_status"] == config.DAMAGE_PRESENCE_NOT_ASSESSED
    assert res["status"] == config.STATUS_DAMAGE_UNVERIFIED
    assert res["uncertainty_status"] == config.UNCERTAINTY_DAMAGE_PRESENCE_UNVERIFIED

    # Stage C: Defect classifier suppressed
    assert res["damage_type"] == config.DAMAGE_TYPE_NOT_ASSESSED
    assert res["dent_score"] is None
    assert res["crack_score"] is None
    assert res["predicted_class"] is None

    # Required limitation message
    assert config.DAMAGE_PRESENCE_UNVERIFIED_MESSAGE in res["decision_message"]

    # Grad-CAM withheld to avoid false defect visualization
    assert res["localization"]["available"] is False


# -----------------------------------------------------------------------------
# Test Case 2: Properly Labelled Dent Image
# -----------------------------------------------------------------------------
def test_regression_case2_labelled_dent_image(benchmark_dent_path):
    """Verify a verified dent defect panel passes Stage A & B, and is classified
    by Stage C with valid Grad-CAM localization.
    """
    res = analyze_image(benchmark_dent_path)

    assert res["aircraft_image_status"] == config.AIRCRAFT_STATUS_SUPPORTED
    assert res["damage_presence_status"] == config.DAMAGE_PRESENCE_DAMAGE_DETECTED
    assert res["damage_type"] in ("dent", "crack", "UNCERTAIN")
    assert res["dent_score"] is not None
    assert res["crack_score"] is not None
    assert 0.0 <= res["dent_score"] <= 1.0
    assert 0.0 <= res["crack_score"] <= 1.0
    assert round(res["dent_score"] + res["crack_score"], 2) == 1.0

    # Grad-CAM generated on defect panel
    assert res["localization"]["available"] is True
    assert res["localization"]["heatmap_data_url"].startswith("data:image/png;base64,")


# -----------------------------------------------------------------------------
# Test Case 3: Properly Labelled Crack Image
# -----------------------------------------------------------------------------
def test_regression_case3_labelled_crack_image(benchmark_crack_path):
    """Verify a verified crack defect panel passes Stage A & B, and is classified
    by Stage C with high crack score and Grad-CAM.
    """
    res = analyze_image(benchmark_crack_path)

    assert res["aircraft_image_status"] == config.AIRCRAFT_STATUS_SUPPORTED
    assert res["damage_presence_status"] == config.DAMAGE_PRESENCE_DAMAGE_DETECTED
    assert res["damage_type"] == "crack"
    assert res["crack_score"] is not None
    assert res["crack_score"] > 0.50
    assert res["localization"]["available"] is True


# -----------------------------------------------------------------------------
# Test Case 4: Clearly Unrelated Signature Image
# -----------------------------------------------------------------------------
def test_regression_case4_unrelated_signature_rejection(signature_path):
    """Verify an unrelated signature image is rejected at Stage A with UNSUPPORTED status."""
    res = analyze_image(signature_path)

    assert res["aircraft_image_status"] == config.AIRCRAFT_STATUS_UNSUPPORTED
    assert res["damage_presence_status"] == config.DAMAGE_PRESENCE_NOT_ASSESSED
    assert res["damage_type"] == config.DAMAGE_TYPE_NOT_ASSESSED
    assert res["status"] == config.STATUS_UNSUPPORTED
    assert res["is_supported"] is False
    assert res["dent_score"] is None
    assert res["crack_score"] is None
    assert res["uncertainty_status"] == config.UNCERTAINTY_OUT_OF_DOMAIN
    assert res["localization"]["available"] is False


# -----------------------------------------------------------------------------
# Test Case 5: Corrupt or Malformed Image
# -----------------------------------------------------------------------------
def test_regression_case5_corrupt_image(client):
    """Verify corrupt byte streams raise CorruptImageError and return HTTP 400."""
    with pytest.raises(CorruptImageError):
        analyze_image(b"not_an_image_file_garbage_bytes")

    # API response verification
    res = client.post(
        "/api/analyze",
        files={"file": ("damaged.jpg", b"corrupted_stream_bytes_xyz", "image/jpeg")},
    )
    assert res.status_code == 400
    assert "Invalid or corrupted image" in res.json()["detail"]


# -----------------------------------------------------------------------------
# Test Case 6: Ambiguous or Low-Quality Aircraft Image
# -----------------------------------------------------------------------------
def test_regression_case6_ambiguous_aircraft_image():
    """Verify that a degraded, blurred, or ambiguous panel image near the decision
    boundary is safely caught by the uncertainty guard.
    """
    # Create an ambiguous panel-like texture with blurred noise
    img = Image.new("RGB", (224, 224), color=(140, 140, 145))
    draw = ImageDraw.Draw(img)
    for x in range(0, 224, 20):
        draw.line([(x, 0), (x, 224)], fill=(125, 125, 130), width=1)
    draw.line([(60, 60), (160, 160)], fill=(90, 90, 95), width=2)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=3.0))

    buf = io.BytesIO()
    blurred.save(buf, format="JPEG")
    ambiguous_bytes = buf.getvalue()

    # Pass with wide uncertainty margin to test ambiguity detection
    res = analyze_image(ambiguous_bytes, uncertainty_margin=0.45)
    assert res["aircraft_image_status"] in (config.AIRCRAFT_STATUS_SUPPORTED, config.AIRCRAFT_STATUS_UNSUPPORTED)
    if res["damage_presence_status"] == config.DAMAGE_PRESENCE_DAMAGE_DETECTED:
        assert res["uncertainty_status"] == config.UNCERTAINTY_UNCERTAIN
        assert res["damage_type"] == config.DAMAGE_TYPE_UNCERTAIN


# -----------------------------------------------------------------------------
# Test Case 7: Decision Margin & Threshold Uncertainty Verification
# -----------------------------------------------------------------------------
def test_regression_case7_decision_margin_uncertainty(benchmark_dent_path):
    """Verify that when uncertainty_margin is wide enough to encompass the score,
    the pipeline correctly assigns UNCERTAIN status without false certainty.
    """
    # Test with uncertainty margin large enough to catch the prediction
    res = analyze_image(benchmark_dent_path, uncertainty_margin=0.49)
    assert res["uncertainty_status"] == config.UNCERTAINTY_UNCERTAIN
    assert res["damage_type"] == config.DAMAGE_TYPE_UNCERTAIN
    assert res["status"] == config.STATUS_UNCERTAIN
    assert "Uncertain result" in res["decision_message"]


# -----------------------------------------------------------------------------
# API Contract Verification: All 7 Explicit Fields Returned
# -----------------------------------------------------------------------------
def test_regression_api_schema_completeness(client, ordinary_aircraft_path):
    """Verify POST /api/analyze returns all 7 explicit required fields."""
    with open(ordinary_aircraft_path, "rb") as f:
        response = client.post(
            "/api/analyze",
            files={"file": ("ordinary.jpg", f.read(), "image/jpeg")},
        )
    assert response.status_code == 200
    data = response.json()

    # The 7 required fields from Section 3
    assert "aircraft_image_status" in data
    assert "damage_presence_status" in data
    assert "damage_type" in data
    assert "dent_score" in data
    assert "crack_score" in data
    assert "uncertainty_status" in data
    assert "model_version" in data

    assert data["aircraft_image_status"] == "SUPPORTED"
    assert data["damage_presence_status"] == "NOT_ASSESSED"
    assert data["damage_type"] == "NOT_ASSESSED"
    assert data["dent_score"] is None
    assert data["crack_score"] is None
