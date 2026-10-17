"""Script to execute BLIP inference on academic benchmark images and record real outputs."""

import json
import os
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import transformers
import tensorflow as tf

from aircraft_ai import config
from aircraft_ai.blip_layer import generate_text
from aircraft_ai.captioning import describe_image
from aircraft_ai.logging_setup import setup_logger

logger = setup_logger("scripts.run_blip_examples", log_file=str(config.LOGS_DIR / "blip_execution.log"))

# Academic benchmark image paths
IMAGE_TASK9_10 = config.DATASET_DIR / "test" / "dent" / "149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg"
IMAGE_SECTION22 = config.DATASET_DIR / "test" / "dent" / "144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg"


def main():
    logger.info("Starting BLIP benchmark inference...")

    if not IMAGE_TASK9_10.exists():
        logger.error("Task 9/10 image not found at %s. Please run scripts/prepare_data.py first.", IMAGE_TASK9_10)
        sys.exit(1)

    if not IMAGE_SECTION22.exists():
        logger.error("Section 2.2 image not found at %s. Please run scripts/prepare_data.py first.", IMAGE_SECTION22)
        sys.exit(1)

    results = {
        "model_id": config.BLIP_MODEL_ID,
        "device": config.BLIP_DEVICE,
        "libraries": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tensorflow": tf.__version__,
        },
        "examples": [],
    }

    test_cases = [
        {"name": "Task 9 (Image 149_22)", "path": IMAGE_TASK9_10, "task": "caption"},
        {"name": "Task 10 (Image 149_22)", "path": IMAGE_TASK9_10, "task": "summary"},
        {"name": "Section 2.2 (Image 144_10)", "path": IMAGE_SECTION22, "task": "caption"},
        {"name": "Section 2.2 (Image 144_10)", "path": IMAGE_SECTION22, "task": "summary"},
    ]

    for tc in test_cases:
        logger.info("Running BLIP on %s with task '%s'...", tc["name"], tc["task"])
        
        # Direct describe_image
        direct_out = describe_image(tc["path"], task=tc["task"])

        # Keras Layer / generate_text test
        path_tensor = tf.constant(str(tc["path"].resolve()))
        task_tensor = tf.constant(tc["task"])
        keras_tensor = generate_text(path_tensor, task_tensor)
        keras_decoded = keras_tensor.numpy().decode("utf-8")

        results["examples"].append({
            "test_case": tc["name"],
            "image_path": str(tc["path"]),
            "task": tc["task"],
            "prompt_used": direct_out["prompt_used"],
            "raw_output": direct_out["raw_generated_text"],
            "cleaned_output": direct_out["generated_text_without_prompt"],
            "keras_layer_decoded_output": keras_decoded,
            "layer_error_sentinel_returned": keras_decoded == "Error processing image",
        })

    config.BLIP_EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.BLIP_EXAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved BLIP benchmark examples to %s", config.BLIP_EXAMPLES_PATH)

    print("\n=== BLIP Inference Results ===")
    for ex in results["examples"]:
        print(f"\n[{ex['test_case']} | Task: {ex['task']}]")
        print(f"  Prompt: \"{ex['prompt_used']}\"")
        print(f"  Output: \"{ex['raw_output']}\"")
        print(f"  Keras Layer Output: \"{ex['keras_layer_decoded_output']}\"")


if __name__ == "__main__":
    main()
