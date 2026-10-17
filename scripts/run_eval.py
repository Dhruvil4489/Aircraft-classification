"""Script to run model evaluation on the test split."""

import json
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aircraft_ai.evaluate import run_evaluation
from aircraft_ai.logging_setup import setup_logger

logger = setup_logger("scripts.run_eval")


def main() -> None:
    logger.info("Executing evaluation script...")
    metrics = run_evaluation()
    print("\n=== Evaluation Results ===")
    print("Academic Notebook Evaluation:")
    nb = metrics.get("notebook_faithful_evaluation", {})
    print(f"  Samples: {nb.get('samples')}, Batch Size: {nb.get('batch_size')}, Steps: {nb.get('steps')}")
    print(f"  Test Loss: {nb.get('test_loss')}, Test Accuracy: {nb.get('test_accuracy')}")
    if nb.get("warning"):
        print(f"  Warning: {nb.get('warning')}")

    print("\nFull Supplementary Evaluation:")
    full = metrics.get("full_supplementary_evaluation", {})
    print(f"  Total Samples: {full.get('full_samples_count')}")
    print(f"  Test Loss: {full.get('test_loss')}, Test Accuracy: {full.get('test_accuracy')}")
    print(f"  Positive Class (Dent) F1: {full.get('positive_class_dent_metrics', {}).get('f1_score')}")
    print(f"  Negative Class (Crack) F1: {full.get('negative_class_crack_metrics', {}).get('f1_score')}")
    print("Full metrics saved to: artifacts/results/metrics.json")


if __name__ == "__main__":
    main()
