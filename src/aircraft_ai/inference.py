"""Inference module for aircraft damage classification with strict preprocessing parity."""

import io
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image

from aircraft_ai import config
from aircraft_ai.errors import ModelNotTrainedError, PreprocessingMismatchError
from aircraft_ai.logging_setup import setup_logger

logger = setup_logger("aircraft_ai.inference")

# Global singleton cache for loaded Keras model
_LOADED_MODEL: Any = None


def get_model(model_path: Path = config.MODEL_SAVE_PATH) -> Any:
    """Retrieve or lazily load the trained Keras model singleton."""
    global _LOADED_MODEL
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL

    if not model_path.exists():
        raise ModelNotTrainedError(
            f"Trained model artifact not found at {model_path}. "
            "Please train the model first by running scripts/run_train.py."
        )

    import keras
    logger.info("Loading trained VGG16 damage classification model from %s...", model_path)
    _LOADED_MODEL = keras.models.load_model(str(model_path))
    return _LOADED_MODEL


def set_model(model) -> None:
    """Explicitly set or reset the singleton model (used in testing or post-training)."""
    global _LOADED_MODEL
    _LOADED_MODEL = model


def preprocess_image(image_input: Union[str, Path, bytes, Image.Image]) -> np.ndarray:
    """Preprocess an input image to match training ImageDataGenerator(rescale=1./255) identically.
    
    Operations:
    1. Open image if path or bytes.
    2. Convert to RGB mode (3 channels).
    3. Resize to (224, 224) using bilinear interpolation.
    4. Convert to float32 NumPy array.
    5. Scale by 1.0 / 255.0.
    6. Expand dims to batch shape (1, 224, 224, 3).
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input))
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Strict RGB conversion
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize to target size (224, 224)
    img = img.resize((config.IMG_ROWS, config.IMG_COLS), resample=Image.Resampling.BILINEAR)

    # Convert to array and rescale identically to ImageDataGenerator(rescale=1./255)
    img_array = np.asarray(img, dtype=np.float32) * (1.0 / 255.0)

    # Validation check for shape and value range
    if img_array.shape != (config.IMG_ROWS, config.IMG_COLS, config.CHANNELS):
        raise PreprocessingMismatchError(
            f"Preprocessed shape {img_array.shape} does not match expected {config.INPUT_SHAPE}"
        )

    # Add batch dimension: (1, 224, 224, 3)
    batch_array = np.expand_dims(img_array, axis=0)
    return batch_array


def predict_damage(
    image_input: Union[str, Path, bytes, Image.Image],
    model: Any = None,
    threshold: float = config.CLASSIFICATION_THRESHOLD,
) -> Dict[str, Any]:
    """Perform real inference on a single aircraft damage image."""
    active_model: Any = model if model is not None else get_model()

    processed_batch = preprocess_image(image_input)
    prediction = active_model.predict(processed_batch, verbose=0)
    prob_dent = float(prediction[0][0])
    prob_crack = float(1.0 - prob_dent)

    predicted_label = "dent" if prob_dent > threshold else "crack"
    predicted_class_id = 1 if predicted_label == "dent" else 0

    return {
        "predicted_class": predicted_label,
        "predicted_class_id": predicted_class_id,
        "probability_dent": round(prob_dent, 6),
        "probability_crack": round(prob_crack, 6),
        "threshold": threshold,
        "class_indices": config.CLASS_INDICES,
        "confidence": round(max(prob_dent, prob_crack), 6),
        "disclaimer": config.SAFETY_DISCLAIMER,
    }


def predict_batch(
    images: List[Union[str, Path, bytes, Image.Image]],
    model: Any = None,
    threshold: float = config.CLASSIFICATION_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Perform batched inference on multiple aircraft damage images."""
    if not images:
        return []

    active_model: Any = model if model is not None else get_model()

    processed_tensors = [preprocess_image(img)[0] for img in images]
    batch = np.stack(processed_tensors, axis=0)

    predictions = active_model.predict(batch, verbose=0)
    results = []
    for pred in predictions:
        prob_dent = float(pred[0])
        prob_crack = float(1.0 - prob_dent)
        predicted_label = "dent" if prob_dent > threshold else "crack"
        predicted_class_id = 1 if predicted_label == "dent" else 0

        results.append({
            "predicted_class": predicted_label,
            "predicted_class_id": predicted_class_id,
            "probability_dent": round(prob_dent, 6),
            "probability_crack": round(prob_crack, 6),
            "threshold": threshold,
            "class_indices": config.CLASS_INDICES,
            "confidence": round(max(prob_dent, prob_crack), 6),
            "disclaimer": config.SAFETY_DISCLAIMER,
        })

    return results


def analyze_image(
    image_input: Union[str, Path, bytes, Image.Image],
    model: Any = None,
    classification_threshold: float = config.CLASSIFICATION_THRESHOLD,
    domain_threshold: float = config.DOMAIN_ACCEPTANCE_THRESHOLD,
    uncertainty_margin: float = config.UNCERTAINTY_MARGIN,
) -> Dict[str, Any]:
    """Execute the multi-stage damage inspection pipeline:
    
    Stage 1: File & image validation (format, size, mode, decode).
    Stage 2: Semantic aircraft-image relevance screening (CLIP zero-shot).
    Stage 3: Damage classification (VGG16 baseline, executed only if domain-accepted).
    Stage 4: Confidence scoring and uncertainty evaluation.
    """
    from aircraft_ai.domain_screening import screen_image_domain

    # Stage 2: Domain Relevance Screening
    domain_res = screen_image_domain(image_input, threshold=domain_threshold)

    if not domain_res["is_aircraft_domain"]:
        logger.warning(
            "Rejected out-of-domain image: matched '%s' (in_domain_score=%.4f < %.2f)",
            domain_res["matched_category"],
            domain_res["in_domain_score"],
            domain_threshold,
        )
        return {
            "status": config.STATUS_UNSUPPORTED,
            "is_supported": False,
            "predicted_class": None,
            "predicted_class_id": None,
            "decision_message": (
                f"Unsupported image — this image could not be verified as an aircraft inspection image "
                f"({domain_res['matched_category']}). No damage classification was performed. "
                "Please upload a relevant aircraft surface photograph."
            ),
            "raw_dent_score": None,
            "raw_crack_score": None,
            "confidence": domain_res["out_of_domain_score"],
            "confidence_label": "Domain Rejection Confidence",
            "threshold": classification_threshold,
            "uncertainty_margin": uncertainty_margin,
            "domain_screening": domain_res,
            "model_version": "VGG16-AeroDamage-v1.0 (ImageNet frozen)",
            "preprocessing": "Domain validation executed prior to VGG16 input",
            "disclaimer": config.SAFETY_DISCLAIMER,
        }

    # Stage 3: Damage Classification on Supported Aircraft Image
    active_model = model if model is not None else get_model()
    class_res = predict_damage(image_input, model=active_model, threshold=classification_threshold)

    dent_score = class_res["probability_dent"]
    crack_score = class_res["probability_crack"]
    margin_from_decision = abs(dent_score - classification_threshold)

    # Stage 4: Uncertainty Evaluation
    if margin_from_decision < uncertainty_margin:
        status = config.STATUS_UNCERTAIN
        predicted_class = "uncertain"
        predicted_class_id = -1
        decision_message = (
            "Uncertain result — the model score is near the 0.50 decision threshold "
            f"(P(dent)={dent_score:.4f}, margin={margin_from_decision:.4f} < {uncertainty_margin:.2f}). "
            "The model cannot confidently distinguish dent from crack. Manual inspection is recommended."
        )
    else:
        status = config.STATUS_COMPLETED
        predicted_class = class_res["predicted_class"]
        predicted_class_id = class_res["predicted_class_id"]
        decision_message = f"Defect classified as {predicted_class.upper()} based on structural visual features."

    return {
        "status": status,
        "is_supported": True,
        "predicted_class": predicted_class,
        "predicted_class_id": predicted_class_id,
        "decision_message": decision_message,
        "raw_dent_score": dent_score,
        "raw_crack_score": crack_score,
        "confidence": class_res["confidence"],
        "confidence_label": "Model Score (Uncalibrated Sigmoid Output)",
        "threshold": classification_threshold,
        "uncertainty_margin": uncertainty_margin,
        "margin_from_decision": round(margin_from_decision, 4),
        "domain_screening": domain_res,
        "class_indices": config.CLASS_INDICES,
        "model_version": "VGG16-AeroDamage-v1.0 (ImageNet frozen)",
        "preprocessing": f"{config.IMG_ROWS}x{config.IMG_COLS} RGB, bilinear resize, scaled 1./255",
        "disclaimer": config.SAFETY_DISCLAIMER,
    }

