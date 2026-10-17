"""Dataset utilities: download, safe extraction, validation, and leakage detection."""

import hashlib
import os
import shutil
import tarfile
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from PIL import Image

from aircraft_ai import config
from aircraft_ai.errors import (
    CorruptImageError,
    DatasetError,
    DatasetNotFoundError,
    UnsafeArchiveError,
)
from aircraft_ai.logging_setup import setup_logger

logger = setup_logger("aircraft_ai.data", log_file=str(config.LOGS_DIR / "data_preparation.log"))

MAX_EXTRACT_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB limit
MAX_MEMBER_COUNT = 5000


def calculate_file_sha256(filepath: Path | str) -> str:
    """Calculate SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def is_archive_safe(tar_ref: tarfile.TarFile, target_dir: Path) -> Tuple[bool, str]:
    """Inspect tar archive members to ensure no path traversal or symlinks exist."""
    total_size = 0
    resolved_target = target_dir.resolve()

    for idx, member in enumerate(tar_ref.getmembers()):
        if idx > MAX_MEMBER_COUNT:
            return False, f"Archive member count exceeds limit ({MAX_MEMBER_COUNT})."

        # Check for path traversal components
        if member.name.startswith("/") or member.name.startswith("\\"):
            return False, f"Member '{member.name}' has absolute path."
        if ".." in Path(member.name).parts:
            return False, f"Member '{member.name}' contains '..' path traversal."

        # Check for special file types
        if member.issym() or member.islnk():
            return False, f"Member '{member.name}' is a symbolic or hard link."
        if member.isdev() or member.ischr() or member.isblk() or member.isfifo():
            return False, f"Member '{member.name}' is a device or FIFO."

        total_size += member.size
        if total_size > MAX_EXTRACT_SIZE_BYTES:
            return False, f"Archive unpacked size exceeds limit ({MAX_EXTRACT_SIZE_BYTES} bytes)."

        # Ensure destination path stays within target_dir
        dest_path = (target_dir / member.name).resolve()
        try:
            dest_path.relative_to(resolved_target)
        except ValueError:
            return False, f"Member '{member.name}' resolves outside target directory."

    return True, "Safe"


def safe_extract_tar(tar_path: Path, extract_dir: Path) -> None:
    """Safely extract tar file with validation against traversal attacks."""
    logger.info("Verifying archive safety: %s", tar_path)
    with tarfile.open(tar_path, "r") as tar_ref:
        safe, reason = is_archive_safe(tar_ref, extract_dir)
        if not safe:
            raise UnsafeArchiveError(f"Rejected malicious or malformed archive: {reason}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        tar_ref.extractall(path=extract_dir)
    logger.info("Safely extracted %s to %s", tar_path, extract_dir)


def is_dataset_valid(dataset_dir: Path = config.DATASET_DIR) -> bool:
    """Check if the dataset directory has all splits and classes with at least 1 image each."""
    if not dataset_dir.exists():
        return False

    for split in config.SPLITS:
        split_path = dataset_dir / split
        if not split_path.is_dir():
            return False
        for cls in config.CLASSES:
            cls_path = split_path / cls
            if not cls_path.is_dir():
                return False
            # Check at least one image exists
            images = [f for f in cls_path.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            if len(images) == 0:
                return False

    return True


def download_dataset(
    url: str = config.DATASET_URL,
    target_dir: Path = config.DATA_ROOT,
    force: bool = False,
) -> Path:
    """Download the dataset tar archive to target_dir if not already present."""
    tar_path = target_dir / config.TAR_FILENAME
    final_dataset_dir = target_dir / config.DATASET_DIR_NAME

    if not force and is_dataset_valid(final_dataset_dir):
        logger.info("Valid extracted dataset already exists at %s. Skipping download.", final_dataset_dir)
        return final_dataset_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    temp_tar_path = target_dir / f"{config.TAR_FILENAME}.temp"

    if tar_path.exists() and not force:
        logger.info("Found existing tar archive at %s", tar_path)
    else:
        logger.info("Downloading dataset from %s", url)
        try:
            with urllib.request.urlopen(url) as response, open(temp_tar_path, "wb") as out_file:
                total_size = int(response.info().get("Content-Length", -1))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1 MB chunks
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if int(percent) % 20 == 0:
                            logger.info("Download progress: %.1f%% (%d / %d bytes)", percent, downloaded, total_size)
            if temp_tar_path.exists():
                shutil.move(str(temp_tar_path), str(tar_path))
            logger.info("Download completed successfully.")
        except urllib.error.URLError as e:
            if temp_tar_path.exists():
                temp_tar_path.unlink()
            error_msg = (
                f"Failed to download dataset from {url}: {e}.\n"
                f"Manual Resolution: Please download the file '{config.TAR_FILENAME}' manually "
                f"and place it at: {tar_path.resolve()}"
            )
            logger.error(error_msg)
            raise DatasetError(error_msg) from e

    sha256_checksum = calculate_file_sha256(tar_path)
    logger.info("Downloaded tar archive SHA-256: %s (unverified against third-party source)", sha256_checksum)

    safe_extract_tar(tar_path, target_dir)

    # Clean up temp files if any
    if temp_tar_path.exists():
        temp_tar_path.unlink()

    if not is_dataset_valid(final_dataset_dir):
        raise DatasetError(f"Extracted dataset at {final_dataset_dir} failed structural validation.")

    return final_dataset_dir


def validate_and_inspect_dataset(dataset_dir: Path = config.DATASET_DIR) -> Dict[str, Any]:
    """Validate all images with PIL, count splits and classes, and inspect for corrupt files."""
    if not dataset_dir.exists():
        raise DatasetNotFoundError(f"Dataset directory not found: {dataset_dir}")

    report: Dict[str, Any] = {
        "dataset_path": str(dataset_dir.resolve()),
        "splits": {},
        "total_images": 0,
        "corrupt_images": [],
        "unsupported_images": [],
    }

    total_count = 0

    for split in config.SPLITS:
        split_dir = dataset_dir / split
        report["splits"][split] = {
            "classes": {},
            "split_total": 0,
        }

        for cls in config.CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                report["splits"][split]["classes"][cls] = {"count": 0, "status": "missing_directory"}
                continue

            image_files = sorted([f for f in cls_dir.iterdir() if f.is_file()])
            valid_count = 0

            for img_file in image_files:
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                    report["unsupported_images"].append(str(img_file))
                    continue

                try:
                    with Image.open(img_file) as img:
                        img.verify()
                    # Re-open to verify readable pixel data
                    with Image.open(img_file) as img:
                        img.load()
                    valid_count += 1
                except Exception as e:
                    logger.warning("Corrupt or unreadable image found: %s (%s)", img_file, e)
                    report["corrupt_images"].append({"path": str(img_file), "error": str(e)})

            report["splits"][split]["classes"][cls] = {
                "count": valid_count,
                "status": "ok" if valid_count > 0 else "empty",
            }
            report["splits"][split]["split_total"] += valid_count
            total_count += valid_count

    report["total_images"] = total_count
    return report


def detect_data_leakage(dataset_dir: Path = config.DATASET_DIR) -> Dict[str, Any]:
    """Detect potential data leakage across splits.
    
    1. Base image identifier overlap (the prefix before .rf. in Roboflow format).
    2. Exact duplicate files across splits via SHA-256 hash.
    """
    logger.info("Scanning dataset for cross-split leakage and duplicate images...")
    base_id_map: Dict[str, Set[str]] = defaultdict(set)
    hash_map: Dict[str, Set[str]] = defaultdict(set)

    for split in config.SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for cls in config.CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            for img_file in cls_dir.iterdir():
                if not img_file.is_file() or img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                # Roboflow base id extraction: <base_id>.rf.<hash>.jpg
                filename = img_file.name
                if ".rf." in filename:
                    base_id = filename.split(".rf.")[0]
                else:
                    base_id = img_file.stem

                location_tag = f"{split}/{cls}/{filename}"
                base_id_map[base_id].add(split)

                # File content hash for exact duplicates
                file_hash = calculate_file_sha256(img_file)
                hash_map[file_hash].add(location_tag)

    # Analyze overlaps
    base_id_overlaps = []
    for base_id, splits in base_id_map.items():
        if len(splits) > 1:
            base_id_overlaps.append({"base_id": base_id, "splits": sorted(list(splits))})

    exact_duplicates = []
    for f_hash, locations in hash_map.items():
        if len(locations) > 1:
            # Check if duplicates span across different splits
            splits_involved = {loc.split("/")[0] for loc in locations}
            if len(splits_involved) > 1:
                exact_duplicates.append({"sha256": f_hash, "locations": sorted(list(locations))})

    leakage_report = {
        "roboflow_base_id_overlaps_count": len(base_id_overlaps),
        "roboflow_base_id_overlaps": base_id_overlaps[:20],  # sample first 20 if large
        "cross_split_exact_duplicates_count": len(exact_duplicates),
        "cross_split_exact_duplicates": exact_duplicates,
        "note": "Reported for diagnostic analysis. No dataset files modified.",
    }
    logger.info(
        "Leakage check complete: %d base ID overlaps, %d exact duplicates across splits.",
        len(base_id_overlaps),
        len(exact_duplicates),
    )
    return leakage_report
