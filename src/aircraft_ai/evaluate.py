"""Evaluation module providing notebook-faithful evaluation and comprehensive metrics."""

import json
import math
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from aircraft_ai import config
from aircraft_ai.errors import ModelNotTrainedError
from aircraft_ai.logging_setup import setup_logger

logger = setup_logger("aircraft_ai.evaluate", log_file=str(config.LOGS_DIR / "evaluation.log"))


def notebook_faithful_evaluation(model, test_generator) -> Dict[str, Any]:
    """Execute evaluation using the exact formula from the academic notebook:
    
    steps = test_generator.samples // test_generator.batch_size
    test_loss, test_accuracy = model.evaluate(test_generator, steps=steps)
    """
    test_generator.reset()
    samples = test_generator.samples
    batch_size = test_generator.batch_size
    truncated_steps = samples // batch_size

    logger.info(
        "Notebook-faithful evaluation: samples=%d, batch_size=%d, integer steps=%d",
        samples,
        batch_size,
        truncated_steps,
    )

    if truncated_steps == 0:
        warning_msg = (
            f"Truncation Warning: test_generator.samples ({samples}) < batch_size ({batch_size}), "
            "resulting in steps = 0. Model evaluation evaluates 0 steps!"
        )
        logger.warning(warning_msg)
        return {
            "formula": "model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)",
            "samples": samples,
            "batch_size": batch_size,
            "steps": 0,
            "test_loss": None,
            "test_accuracy": None,
            "warning": warning_msg,
        }

    eval_result = model.evaluate(test_generator, steps=truncated_steps, verbose=0)
    test_loss = float(eval_result[0])
    test_accuracy = float(eval_result[1])

    logger.info("Notebook eval output: Test Loss: %.4f, Test Accuracy: %.4f", test_loss, test_accuracy)
    return {
        "formula": "model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)",
        "samples": samples,
        "batch_size": batch_size,
        "steps": truncated_steps,
        "evaluated_samples": truncated_steps * batch_size,
        "untested_samples_omitted": samples - (truncated_steps * batch_size),
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_accuracy, 4),
    }


def full_supplementary_evaluation(model, test_generator) -> Dict[str, Any]:
    """Execute complete, non-truncated evaluation across 100% of the test split."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    test_generator.reset()
    samples = test_generator.samples
    batch_size = test_generator.batch_size
    full_steps = math.ceil(samples / batch_size)

    # 1. Un-truncated loss and accuracy
    eval_result = model.evaluate(test_generator, steps=full_steps, verbose=0)
    full_test_loss = float(eval_result[0])
    full_test_accuracy = float(eval_result[1])

    # 2. Extract true labels and compute raw continuous predictions
    test_generator.reset()
    predictions_raw = model.predict(test_generator, steps=full_steps, verbose=0)
    # Truncate to exact samples in case of batch padding
    predictions_raw = predictions_raw[:samples]
    probabilities_dent = predictions_raw.flatten()
    predicted_classes = (probabilities_dent > config.CLASSIFICATION_THRESHOLD).astype(int)
    true_classes = test_generator.classes[:samples]

    # Confusion Matrix: axes [crack=0, dent=1]
    cm = confusion_matrix(true_classes, predicted_classes, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Per-class counts
    crack_count = int(np.sum(true_classes == 0))
    dent_count = int(np.sum(true_classes == 1))

    # Metrics
    prec_dent = float(precision_score(true_classes, predicted_classes, pos_label=1, zero_division=0))
    rec_dent = float(recall_score(true_classes, predicted_classes, pos_label=1, zero_division=0))
    f1_dent = float(f1_score(true_classes, predicted_classes, pos_label=1, zero_division=0))

    prec_crack = float(precision_score(true_classes, predicted_classes, pos_label=0, zero_division=0))
    rec_crack = float(recall_score(true_classes, predicted_classes, pos_label=0, zero_division=0))
    f1_crack = float(f1_score(true_classes, predicted_classes, pos_label=0, zero_division=0))

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Split)")
    plt.colorbar()
    tick_marks = np.arange(len(config.CLASSES))
    plt.xticks(tick_marks, ["crack (0)", "dent (1)"])
    plt.yticks(tick_marks, ["crack (0)", "dent (1)"])

    # Annotate counts inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(config.CONFUSION_MATRIX_PLOT_PATH, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved confusion matrix plot to %s", config.CONFUSION_MATRIX_PLOT_PATH)

    detailed_report = {
        "full_samples_count": samples,
        "test_loss": round(full_test_loss, 4),
        "test_accuracy": round(full_test_accuracy, 4),
        "class_distribution": {
            "crack": crack_count,
            "dent": dent_count,
            "total": samples,
        },
        "confusion_matrix": {
            "labels": ["crack", "dent"],
            "matrix": cm.tolist(),
            "true_negatives_crack": int(tn),
            "false_positives_dent": int(fp),
            "false_negatives_crack": int(fn),
            "true_positives_dent": int(tp),
        },
        "positive_class_dent_metrics": {
            "precision": round(prec_dent, 4),
            "recall": round(rec_dent, 4),
            "f1_score": round(f1_dent, 4),
        },
        "negative_class_crack_metrics": {
            "precision": round(prec_crack, 4),
            "recall": round(rec_crack, 4),
            "f1_score": round(f1_crack, 4),
        },
        "discussion": (
            "Because the dataset has unequal class distribution and small sample size, "
            "overall accuracy can be misleading if one class dominates. Evaluating precision "
            "and recall for both crack and dent is critical for aviation damage detection."
        ),
    }

    return detailed_report


def task7_visualize_prediction(test_generator, model, index_to_plot: int = 1) -> None:
    """Implement exact Task 7 visualization from the academic notebook."""
    test_generator.reset()
    test_images, test_labels = next(test_generator)
    predictions = model.predict(test_images, verbose=0)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    class_indices = test_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}

    image_to_plot = test_images[index_to_plot]
    true_label = int(test_labels[index_to_plot])
    predicted_label = int(predicted_classes[index_to_plot])

    true_name = class_names[true_label]
    pred_name = class_names[predicted_label]

    plt.figure(figsize=(6, 6))
    plt.imshow(image_to_plot)
    plt.title(f"True: {true_name}\nPred: {pred_name}", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.savefig(config.TASK7_PREDICTION_PLOT_PATH, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved Task 7 prediction figure to %s", config.TASK7_PREDICTION_PLOT_PATH)


def plot_deterministic_test_grid(test_generator, model, num_images: int = 6) -> None:
    """Deterministic prediction grid displaying test images with true vs predicted labels."""
    test_generator.reset()
    test_images, test_labels = next(test_generator)
    predictions = model.predict(test_images, verbose=0)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    class_names = {v: k for k, v in test_generator.class_indices.items()}
    count = min(num_images, len(test_images))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(count):
        img = test_images[i]
        true_lbl = class_names[int(test_labels[i])]
        pred_lbl = class_names[int(predicted_classes[i])]
        prob = float(predictions[i][0])
        is_correct = true_lbl == pred_lbl

        color = "#16a34a" if is_correct else "#dc2626"
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {true_lbl}\nPred: {pred_lbl} (P(dent)={prob:.2f})",
            color=color,
            fontsize=11,
            fontweight="bold",
        )
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(config.TEST_PREDICTIONS_GRID_PATH, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved deterministic test predictions grid to %s", config.TEST_PREDICTIONS_GRID_PATH)


def run_evaluation() -> Dict[str, Any]:
    """Execute complete evaluation workflow, assemble metrics, and save metrics.json."""
    from aircraft_ai.inference import get_model
    from aircraft_ai.train import create_generators

    logger.info("Starting model evaluation...")
    model = get_model()
    _, _, test_gen = create_generators()

    # Academic notebook evaluation
    nb_eval = notebook_faithful_evaluation(model, test_gen)

    # Supplementary full test evaluation
    full_eval = full_supplementary_evaluation(model, test_gen)

    # Task 7 visualization
    task7_visualize_prediction(test_gen, model, index_to_plot=1)

    # Deterministic grid
    plot_deterministic_test_grid(test_gen, model, num_images=6)

    combined_metrics = {
        "notebook_faithful_evaluation": nb_eval,
        "full_supplementary_evaluation": full_eval,
    }

    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, indent=2)

    logger.info("Saved comprehensive metrics to %s", config.METRICS_PATH)
    return combined_metrics
