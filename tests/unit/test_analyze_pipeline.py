"""Unit and integration tests for the multi-stage analysis pipeline and API endpoints."""

import io
from PIL import Image, ImageDraw
import pytest
from fastapi.testclient import TestClient

from aircraft_ai import config
from aircraft_ai.api import app
from aircraft_ai.inference import analyze_image


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def signature_bytes():
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line([(30, 100), (80, 40), (120, 150), (160, 60), (200, 120), (250, 50), (320, 160), (370, 90)], fill=(10, 20, 120), width=3)
    draw.line([(50, 140), (350, 130)], fill=(10, 20, 120), width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def real_dent_bytes():
    dent_candidates = list(config.TEST_DIR.glob("dent/*.jpg"))
    assert len(dent_candidates) > 0
    with open(dent_candidates[0], "rb") as f:
        return f.read()


def test_analyze_image_rejects_signature(signature_bytes):
    """Verify that analyze_image returns UNSUPPORTED_IMAGE status for a signature."""
    res = analyze_image(signature_bytes)
    assert res["status"] == config.STATUS_UNSUPPORTED
    assert res["is_supported"] is False
    assert res["predicted_class"] is None
    assert "Unsupported image" in res["decision_message"]
    assert res["domain_screening"]["is_aircraft_domain"] is False


def test_analyze_image_accepts_aircraft(real_dent_bytes):
    """Verify that analyze_image accepts real aircraft image and classifies damage."""
    res = analyze_image(real_dent_bytes)
    assert res["is_supported"] is True
    assert res["status"] in (config.STATUS_COMPLETED, config.STATUS_UNCERTAIN)
    assert res["raw_dent_score"] is not None
    assert res["raw_crack_score"] is not None
    assert res["domain_screening"]["is_aircraft_domain"] is True


def test_api_analyze_endpoint_signature_rejection(client, signature_bytes):
    """Verify POST /api/analyze returns structured UNSUPPORTED_IMAGE response for signature."""
    response = client.post(
        "/api/analyze",
        files={"file": ("signature.png", signature_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == config.STATUS_UNSUPPORTED
    assert data["is_supported"] is False
    assert data["predicted_class"] is None


def test_api_model_status_and_info(client):
    """Verify GET /api/model/status and /api/model/info return real model metadata."""
    status_res = client.get("/api/model/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert "model_trained" in status_data
    assert "VGG16" in status_data["backbone"]

    info_res = client.get("/api/model/info")
    assert info_res.status_code == 200
    info_data = info_res.json()
    assert info_data["backbone"]["name"] == "VGG16"
    assert info_data["class_mapping"] == {"crack": 0, "dent": 1}
    assert info_data["threshold"] == 0.5
