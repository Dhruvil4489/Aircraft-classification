"""Unit tests for FastAPI endpoints, status codes, and input validation."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from aircraft_ai.api import app


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.mark.fast
def test_health_endpoint(client):
    """Test /health returns 200 and expected status keys."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_trained" in data
    assert data["class_mapping"] == {"crack": 0, "dent": 1}


@pytest.mark.fast
def test_class_mapping_endpoint(client):
    """Test /api/class-mapping returns 200 and academic mapping."""
    response = client.get("/api/class-mapping")
    assert response.status_code == 200
    data = response.json()
    assert data["class_indices"] == {"crack": 0, "dent": 1}
    assert data["threshold"] == 0.5


@pytest.mark.fast
def test_classify_invalid_file_type(client):
    """Test /api/classify rejects non-image text uploads with 400."""
    fake_file = io.BytesIO(b"This is just plain text, not an image")
    response = client.post(
        "/api/classify",
        files={"file": ("notes.txt", fake_file, "text/plain")},
    )
    # Rejects as 400 or 503 if untrained; if reached validator returns 400
    assert response.status_code in [400, 503]


@pytest.mark.fast
def test_caption_corrupt_file_handling(client):
    """Test /api/caption rejects corrupted image with 400."""
    fake_file = io.BytesIO(b"GARBAGE_HEADER_CORRUPT_BYTES")
    response = client.post(
        "/api/caption",
        files={"file": ("corrupt.jpg", fake_file, "image/jpeg")},
    )
    assert response.status_code == 400


@pytest.mark.fast
def test_figures_allow_list_security(client):
    """Test that requesting an unlisted or path-traversal figure returns 404."""
    response = client.get("/api/results/figures/../../etc/passwd")
    assert response.status_code == 404

    response = client.get("/api/results/figures/arbitrary_file.png")
    assert response.status_code == 404


@pytest.mark.fast
def test_examples_endpoint(client):
    """Test /api/examples returns list of benchmark images."""
    response = client.get("/api/examples")
    assert response.status_code == 200
    data = response.json()
    assert "examples" in data
    assert len(data["examples"]) > 0
