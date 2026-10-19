"""Model training pipeline replicating the academic notebook specification exactly."""

import json
import random
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from aircraft_ai import config
from aircraft_ai.classifier import get_compiled_model
from aircraft_ai.errors import DatasetNotFoundError
from aircraft_ai.logging_setup import setup_logger

logger = setup_logger("aircraft_ai.train", log_file=str(config.LOGS_DIR / "training.log"))


def set_reproducibility_seeds(seed: int = config.SEED_VALUE) -> None:
    """Set seeds for random, numpy, and tensorflow."""
    import tensorflow as tf
    logger.info("Setting reproducibility seeds: %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_generators():
    """Create train, valid, and test ImageDataGenerators exactly matching notebook tasks."""
    import tensorflow as tf
    ImageDataGenerator: Any = getattr(tf.keras.preprocessing, "image").ImageDataGenerator

    train_dir = str(config.TRAIN_DIR)
    valid_dir = str(config.VALID_DIR)
    test_dir = str(config.TEST_DIR)

    for path_str in [train_dir, valid_dir, test_dir]:
        if not Path(path_str).exists():
            raise DatasetNotFoundError(
                f"Directory not found: {path_str}. Please run scripts/prepare_data.py first."
            )

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Training generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(config.IMG_ROWS, config.IMG_COLS),
        batch_size=config.BATCH_SIZE,
        seed=config.SEED_VALUE,
        class_mode="binary",
        shuffle=True,
    )

    # Task 1: valid_generator
    valid_generator = valid_datagen.flow_from_directory(
        directory=valid_dir,
        class_mode="binary",
        seed=config.SEED_VALUE,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        target_size=(config.IMG_ROWS, config.IMG_COLS),
    )

    # Task 2: test_generator
    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        class_mode="binary",
        seed=config.SEED_VALUE,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        target_size=(config.IMG_ROWS, config.IMG_COLS),
    )

    # Class mapping assertion
    logger.info("Detected class indices: %s", train_generator.class_indices)
    assert train_generator.class_indices == config.CLASS_INDICES, (
        f"Expected class mapping {config.CLASS_INDICES}, got {train_generator.class_indices}"
    )

    return train_generator, valid_generator, test_generator


def plot_and_save_training_curves(history: Dict[str, list]) -> None:
    """Generate and save loss and accuracy plots matching notebook figures."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Training Loss (notebook style)
    plt.figure()
    plt.title("Training Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(history["loss"], color="#2563eb", linewidth=2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(config.TRAINING_LOSS_PLOT_PATH, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved training loss plot to %s", config.TRAINING_LOSS_PLOT_PATH)

    # 2. Validation Loss (notebook style)
    plt.figure()
    plt.title("Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(history["val_loss"], color="#dc2626", linewidth=2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(config.VALIDATION_LOSS_PLOT_PATH, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved validation loss plot to %s", config.VALIDATION_LOSS_PLOT_PATH)

    # 3. Task 6: Accuracy Curve (figsize=(5, 5))
    plt.figure(figsize=(5, 5))
    plt.plot(history["accuracy"], label="Training Accuracy", color="#2563eb", linewidth=2)
    plt.plot(history["val_accuracy"], label="Validation Accuracy", color="#16a34a", linewidth=2)
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(config.ACCURACY_CURVE_PLOT_PATH, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved Task 6 accuracy curve plot to %s", config.ACCURACY_CURVE_PLOT_PATH)


def verify_model_persistence(original_model: Any, saved_path: Path, test_generator: Any) -> float:
    """Reload model and verify prediction equality with np.allclose."""
    import keras
    logger.info("Reloading model from %s to verify persistence parity...", saved_path)
    reloaded_model: Any = keras.models.load_model(str(saved_path))

    # Take a small batch
    test_generator.reset()
    batch_images, _ = next(test_generator)
    orig_m: Any = original_model
    pred_orig = orig_m.predict(batch_images, verbose=0)
    pred_reload = reloaded_model.predict(batch_images, verbose=0)

    max_diff = float(np.max(np.abs(pred_orig - pred_reload)))
    logger.info("Maximum prediction diff after reload: %.2e (target atol 1e-5)", max_diff)
    assert np.allclose(pred_orig, pred_reload, atol=1e-5), f"Reloaded predictions differ! max diff={max_diff}"
    return max_diff


def run_training() -> Dict[str, Any]:
    """Execute the full training pipeline (Tasks 1-6)."""
    set_reproducibility_seeds(config.SEED_VALUE)

    logger.info("Initializing dataset generators...")
    train_gen, valid_gen, test_gen = create_generators()
    logger.info(
        "Generators ready: train samples=%d, valid samples=%d, test samples=%d",
        train_gen.samples,
        valid_gen.samples,
        test_gen.samples,
    )

    # Build and compile model (Tasks 3 and 4)
    model = get_compiled_model(weights="imagenet")
    model.summary(print_fn=lambda x: logger.info("%s", x))

    # Task 5: Train the model
    logger.info(
        "Starting training for %d epochs with batch size %d...",
        config.N_EPOCHS,
        config.BATCH_SIZE,
    )
    fit_history = model.fit(
        train_gen,
        epochs=config.N_EPOCHS,
        validation_data=valid_gen,
    )

    train_history = model.history.history
    logger.info("Training complete. Final epochs metrics: %s", {k: v[-1] for k, v in train_history.items()})

    # Save training history
    with open(config.HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(train_history, f, indent=2)
    logger.info("Saved history to %s", config.HISTORY_PATH)

    # Generate and save plots (Task 6)
    plot_and_save_training_curves(train_history)

    # Persist model
    config.MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(config.MODEL_SAVE_PATH))
    logger.info("Saved trained Keras model to %s", config.MODEL_SAVE_PATH)

    # Persist class indices and training config
    with open(config.CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    training_config = {
        "seed_value": config.SEED_VALUE,
        "batch_size": config.BATCH_SIZE,
        "n_epochs": config.N_EPOCHS,
        "img_rows": config.IMG_ROWS,
        "img_cols": config.IMG_COLS,
        "input_shape": list(config.INPUT_SHAPE),
        "learning_rate": config.LEARNING_RATE,
        "train_samples": train_gen.samples,
        "valid_samples": valid_gen.samples,
        "test_samples": test_gen.samples,
        "class_indices": train_gen.class_indices,
    }
    with open(config.TRAINING_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)

    # Verification of persistence
    max_diff = verify_model_persistence(model, config.MODEL_SAVE_PATH, test_gen)

    return {
        "status": "success",
        "history": train_history,
        "model_path": str(config.MODEL_SAVE_PATH),
        "persistence_max_diff": max_diff,
    }
