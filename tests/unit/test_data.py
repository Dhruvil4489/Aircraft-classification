"""Unit tests for dataset download, safe extraction, validation, and leakage detection."""

import io
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from aircraft_ai.data import (
    calculate_file_sha256,
    detect_data_leakage,
    is_archive_safe,
    is_dataset_valid,
    safe_extract_tar,
    validate_and_inspect_dataset,
)
from aircraft_ai.errors import UnsafeArchiveError


@pytest.mark.fast
def test_safe_tar_extraction_valid(tmp_path: Path):
    """Test safe extraction of a normal tar archive."""
    tar_file = tmp_path / "valid.tar"
    dest_dir = tmp_path / "extracted"

    # Create dummy tar
    with tarfile.open(tar_file, "w") as tar:
        data = b"Hello aircraft AI"
        info = tarfile.TarInfo(name="test_split/dent/img1.jpg")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    safe_extract_tar(tar_file, dest_dir)
    extracted_file = dest_dir / "test_split" / "dent" / "img1.jpg"
    assert extracted_file.exists()
    assert extracted_file.read_bytes() == data


@pytest.mark.fast
def test_safe_tar_rejection_path_traversal(tmp_path: Path):
    """Test that archive with '..' component is rejected."""
    tar_file = tmp_path / "malicious_traversal.tar"
    dest_dir = tmp_path / "extracted"

    with tarfile.open(tar_file, "w") as tar:
        data = b"Malicious payload"
        info = tarfile.TarInfo(name="../../outside.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    with pytest.raises(UnsafeArchiveError) as exc_info:
        safe_extract_tar(tar_file, dest_dir)
    assert "path traversal" in str(exc_info.value).lower()


@pytest.mark.fast
def test_safe_tar_rejection_absolute_path(tmp_path: Path):
    """Test that archive with leading '/' is rejected."""
    tar_file = tmp_path / "malicious_abs.tar"
    dest_dir = tmp_path / "extracted"

    with tarfile.open(tar_file, "w") as tar:
        data = b"Malicious payload"
        info = tarfile.TarInfo(name="/etc/evil.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    with pytest.raises(UnsafeArchiveError) as exc_info:
        safe_extract_tar(tar_file, dest_dir)
    assert "absolute path" in str(exc_info.value).lower()


@pytest.mark.fast
def test_dataset_structural_validation(tmp_path: Path):
    """Test is_dataset_valid on complete vs incomplete directories."""
    dataset_dir = tmp_path / "mock_dataset"

    # Incomplete initially
    assert is_dataset_valid(dataset_dir) is False

    # Create 6 required directories
    for split in ["train", "valid", "test"]:
        for cls in ["crack", "dent"]:
            cls_dir = dataset_dir / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            # Create a small valid image
            img = Image.new("RGB", (10, 10), color="white")
            img.save(cls_dir / "sample.jpg")

    assert is_dataset_valid(dataset_dir) is True


@pytest.mark.fast
def test_corrupt_image_detection(tmp_path: Path):
    """Test validate_and_inspect_dataset properly identifies corrupt images without deleting."""
    dataset_dir = tmp_path / "mock_dataset"
    for split in ["train", "valid", "test"]:
        for cls in ["crack", "dent"]:
            cls_dir = dataset_dir / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            # Valid image
            img = Image.new("RGB", (10, 10), color="blue")
            img.save(cls_dir / "valid.jpg")

    # Inject one corrupted file
    corrupt_file = dataset_dir / "train" / "dent" / "corrupt.jpg"
    corrupt_file.write_bytes(b"NOT_AN_IMAGE_DATA_CORRUPT")

    report = validate_and_inspect_dataset(dataset_dir)
    assert len(report["corrupt_images"]) == 1
    assert "corrupt.jpg" in report["corrupt_images"][0]["path"]
    # Ensure file was reported but NOT deleted
    assert corrupt_file.exists()


@pytest.mark.fast
def test_leakage_detection(tmp_path: Path):
    """Test detect_data_leakage identifies cross-split base IDs and exact duplicates."""
    dataset_dir = tmp_path / "mock_dataset"
    train_dent = dataset_dir / "train" / "dent"
    valid_dent = dataset_dir / "valid" / "dent"
    train_dent.mkdir(parents=True, exist_ok=True)
    valid_dent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (10, 10), color="green")
    # Base ID: "photo123" shared across train and valid
    img.save(train_dent / "photo123.rf.hash1.jpg")
    img.save(valid_dent / "photo123.rf.hash2.jpg")

    leakage = detect_data_leakage(dataset_dir)
    assert leakage["roboflow_base_id_overlaps_count"] == 1
    assert leakage["roboflow_base_id_overlaps"][0]["base_id"] == "photo123"
    assert "train" in leakage["roboflow_base_id_overlaps"][0]["splits"]
    assert "valid" in leakage["roboflow_base_id_overlaps"][0]["splits"]
