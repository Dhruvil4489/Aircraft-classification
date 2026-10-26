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
    # Verify new schema fields
    assert "inspection_id" in res
    assert res["inspection_id"].startswith("INS-")
    assert "processing_time_ms" in res
    assert res["processing_time_ms"] > 0
    assert "recommended_action" in res
    assert "localization" in res
    assert res["localization"]["available"] is True
    assert res["localization"]["heatmap_data_url"].startswith("data:image/png;base64,")


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
    assert "recommended_action" in data


def test_api_analyze_corrupt_image(client):
    """Verify POST /api/analyze returns 400 for corrupted image upload."""
    response = client.post(
        "/api/analyze",
        files={"file": ("corrupt.jpg", b"not_an_image_file_content", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Invalid or corrupted image" in response.json()["detail"]


def test_api_inspection_report_html_generation(client):
    """Verify POST /api/inspection/report/html produces a printable HTML report."""
    payload = {
        "inspection_id": "INS-TEST-123",
        "timestamp": "2026-10-25T20:00:00Z",
        "status": "COMPLETED",
        "predicted_class": "dent",
        "model_score": 0.8842,
        "decision_message": "Defect classified as DENT based on structural visual features.",
        "recommended_action": "Log dent depth per SRM Chapter 51.",
        "processing_time_ms": 120.5,
        "localization": {
            "heatmap_data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "overlay_data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        },
    }
    response = client.post("/api/inspection/report/html", json=payload)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html_text = response.text
    assert "INS-TEST-123" in html_text
    assert "DENT" in html_text
    assert "AERODAMAGE AI" in html_text


def test_api_benchmark_examples(client):
    """Verify GET /api/examples returns benchmark images including crack_example_2."""
    response = client.get("/api/examples")
    assert response.status_code == 200
    examples = response.json()["examples"]
    assert "dent_example_1.jpg" in examples
    assert "crack_example_2.jpg" in examples

    img_res = client.get("/api/examples/crack_example_2.jpg")
    assert img_res.status_code == 200
    assert img_res.headers["content-type"] == "image/jpeg"


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

