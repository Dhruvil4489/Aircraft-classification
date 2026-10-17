"""Data preparation and validation script.

Downloads the dataset if needed, verifies archive safety, validates directory structure,
tests image integrity with PIL, checks for data leakage and duplicates, and writes
artifacts/results/dataset_report.json.
"""

import json
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aircraft_ai import config
from aircraft_ai.data import (
    detect_data_leakage,
    download_dataset,
    validate_and_inspect_dataset,
)
from aircraft_ai.logging_setup import setup_logger

logger = setup_logger("scripts.prepare_data", log_file=str(config.LOGS_DIR / "data_preparation.log"))


def main() -> None:
    logger.info("Starting dataset preparation and validation pipeline...")

    # Step 1: Download and extract safely
    try:
        dataset_dir = download_dataset(url=config.DATASET_URL, target_dir=config.DATA_ROOT, force=False)
        logger.info("Dataset located at: %s", dataset_dir)
    except Exception as e:
        logger.error("Failed to prepare dataset: %s", e)
        sys.exit(1)

    # Step 2: Validate image readability and counts
    inspection_report = validate_and_inspect_dataset(dataset_dir)
    logger.info("Validation complete. Total images: %d", inspection_report["total_images"])

    # Step 3: Check for data leakage and duplicates
    leakage_report = detect_data_leakage(dataset_dir)

    # Step 4: Assemble comprehensive dataset report
    final_report = {
        "dataset_name": "Aircraft Damage Dataset v1",
        "original_source": "Roboflow Universe (Aircraft Damage Detection by youssef-donia-fhktl)",
        "source_url": config.DATASET_URL,
        "roboflow_url": "https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk",
        "license": "CC BY 4.0",
        "academic_citation": "Roboflow Aircraft Damage Detection dataset (youssef-donia-fhktl, CC BY 4.0)",
        "splits_summary": {
            split: {
                "crack": inspection_report["splits"][split]["classes"].get("crack", {}).get("count", 0),
                "dent": inspection_report["splits"][split]["classes"].get("dent", {}).get("count", 0),
                "total": inspection_report["splits"][split]["split_total"],
            }
            for split in config.SPLITS
        },
        "total_images": inspection_report["total_images"],
        "class_mapping": config.CLASS_INDICES,
        "corrupt_images": inspection_report["corrupt_images"],
        "unsupported_images": inspection_report["unsupported_images"],
        "leakage_analysis": leakage_report,
    }

    # Step 5: Save dataset report
    config.DATASET_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.DATASET_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    logger.info("Saved dataset report to %s", config.DATASET_REPORT_PATH)
    print("\n=== Dataset Preparation Summary ===")
    print(f"Total Images: {final_report['total_images']}")
    for split, counts in final_report["splits_summary"].items():
        print(f"  {split.upper()}: Total={counts['total']}, Crack={counts['crack']}, Dent={counts['dent']}")
    print(f"Report saved to: {config.DATASET_REPORT_PATH}")


if __name__ == "__main__":
    main()
