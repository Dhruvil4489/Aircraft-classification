"""Unit tests for BLIP captioning, prompt conditioning, error handling, and custom Keras layer."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aircraft_ai.captioning import describe_image
from aircraft_ai.errors import CorruptImageError, InvalidTaskError


@pytest.mark.fast
def test_invalid_task_rejection(synthetic_image_rgb):
    """Test that describe_image rejects tasks other than 'caption' or 'summary'."""
    with pytest.raises(InvalidTaskError) as exc_info:
        describe_image(synthetic_image_rgb, task="segmentation")
    assert "Task must be 'caption' or 'summary'" in str(exc_info.value)


@pytest.mark.fast
def test_missing_image_file_handling():
    """Test describe_image raises FileNotFoundError for non-existent image paths."""
    with pytest.raises(FileNotFoundError):
        describe_image(Path("data/non_existent_image_12345.jpg"), task="caption")


@pytest.mark.fast
def test_corrupt_image_handling(corrupt_image_file):
    """Test describe_image raises CorruptImageError for unreadable images."""
    with pytest.raises(CorruptImageError):
        describe_image(corrupt_image_file, task="caption")
