"""Pytest fixtures and configuration."""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aircraft_ai import config


@pytest.fixture
def synthetic_image_rgb(tmp_path: Path) -> Path:
    """Create a temporary synthetic valid RGB image fixture for fast unit testing."""
    img_path = tmp_path / "valid_damage_synthetic.jpg"
    img = Image.new("RGB", (300, 300), color=(120, 150, 180))
    img.save(img_path, format="JPEG")
    return img_path


@pytest.fixture
def synthetic_image_bytes() -> bytes:
    """Return raw bytes of a valid JPEG image."""
    buffer = io.BytesIO()
    img = Image.new("RGB", (224, 224), color=(200, 50, 50))
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def corrupt_image_file(tmp_path: Path) -> Path:
    """Create a corrupted/unreadable image file."""
    corrupt_path = tmp_path / "corrupt_image.jpg"
    corrupt_path.write_bytes(b"NOT_A_VALID_JPEG_HEADER_CONTENT_GARBAGE")
    return corrupt_path


@pytest.fixture
def oversized_image_bytes() -> bytes:
    """Return an empty buffer representing oversized file metadata."""
    # 11 MB dummy buffer
    return b"0" * (11 * 1024 * 1024)
