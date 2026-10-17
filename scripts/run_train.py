"""Script to run VGG16 model training on aircraft damage dataset."""

import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aircraft_ai.logging_setup import setup_logger
from aircraft_ai.train import run_training

logger = setup_logger("scripts.run_train")


def main() -> None:
    logger.info("Executing training script...")
    result = run_training()
    print("\n=== Training Completed Successfully ===")
    print(f"Model saved to: {result['model_path']}")
    print(f"Persistence parity max diff: {result['persistence_max_diff']:.2e}")
    print("History saved to: artifacts/results/history.json")
    print("Plots saved to: artifacts/results/")


if __name__ == "__main__":
    main()
