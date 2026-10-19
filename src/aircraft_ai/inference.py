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
