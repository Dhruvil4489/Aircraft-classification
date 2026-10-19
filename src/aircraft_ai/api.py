"""FastAPI web service for aircraft damage classification, BLIP captioning, and metrics reporting."""

import io
import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from aircraft_ai import config
from aircraft_ai.captioning import describe_image
from aircraft_ai.errors import (
    CorruptImageError,
    DatasetNotFoundError,
    ModelNotTrainedError,
)
from aircraft_ai.inference import analyze_image, get_model, predict_damage
from aircraft_ai.logging_setup import setup_logger

logger = setup_logger("aircraft_ai.api", log_file=str(config.LOGS_DIR / "api.log"))

# Frontend directory
FRONTEND_DIR = config.WORKSPACE_ROOT / "frontend"

app = FastAPI(
    title="AeroDamage AI Service",
    description="Automated aircraft structural damage classification (VGG16) and multimodal captioning (BLIP)",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_uploaded_image(file_bytes: bytes, filename: str | None = None) -> Image.Image:
    """Validate uploaded file bytes: size limit, PIL verification, and convert to RGB."""
    max_bytes = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds {config.MAX_UPLOAD_SIZE_MB} MB limit.",
        )

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        # Re-open after verify to ensure pixel data is decoded
        img = Image.open(io.BytesIO(file_bytes))
        img.load()
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            alpha_composite = Image.alpha_composite(bg, img)
            img = alpha_composite.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.warning("Rejected invalid image upload '%s': %s", filename, e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image format. Please upload a valid JPEG, PNG, or WebP file: {e}",
        )


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint indicating service status and whether model artifacts exist."""
    model_trained = config.MODEL_SAVE_PATH.exists()
    dataset_prepared = (config.DATASET_DIR / "train").exists()
    return {
        "status": "healthy",
        "model_trained": model_trained,
        "dataset_prepared": dataset_prepared,
        "blip_device": config.BLIP_DEVICE,
        "domain_screening_model": config.DOMAIN_SCREENING_MODEL_ID,
        "class_mapping": config.CLASS_INDICES,
    }


@app.get("/api/model/status")
def get_model_status() -> Dict[str, Any]:
    """Return model availability, artifact paths, parameter details, and device status."""
    model_exists = config.MODEL_SAVE_PATH.exists()
    return {
        "status": "ready" if model_exists else "not_trained",
        "model_trained": model_exists,
        "model_path": str(config.MODEL_SAVE_PATH),
        "backbone": "ImageNet Pretrained VGG16 (Frozen)",
        "head": "Flatten + Dense(512, ReLU) + Dropout(0.3) + Dense(512, ReLU) + Dropout(0.3) + Dense(1, Sigmoid)",
        "weights_size_bytes": config.MODEL_SAVE_PATH.stat().st_size if model_exists else 0,
        "domain_screening_model": config.DOMAIN_SCREENING_MODEL_ID,
        "domain_threshold": config.DOMAIN_ACCEPTANCE_THRESHOLD,
        "uncertainty_margin": config.UNCERTAINTY_MARGIN,
        "blip_model": config.BLIP_MODEL_ID,
        "blip_device": config.BLIP_DEVICE,
    }


@app.get("/api/model/info")
def get_model_info() -> Dict[str, Any]:
    """Return comprehensive academic and production model specifications."""
    return {
        "project": "AeroDamage AI",
        "version": "1.0.0",
        "backbone": {
            "name": "VGG16",
            "weights": "ImageNet",
            "include_top": False,
            "trainable": False,
            "input_shape": list(config.INPUT_SHAPE),
        },
        "head": {
            "layers": [
                {"type": "Flatten", "output_features": 25088},
                {"type": "Dense", "units": 512, "activation": "relu"},
                {"type": "Dropout", "rate": 0.3},
                {"type": "Dense", "units": 512, "activation": "relu"},
                {"type": "Dropout", "rate": 0.3},
                {"type": "Dense", "units": 1, "activation": "sigmoid"},
            ],
            "optimizer": "Adam(lr=0.0001)",
            "loss": "binary_crossentropy",
            "epochs": config.N_EPOCHS,
            "batch_size": config.BATCH_SIZE,
        },
        "class_mapping": config.CLASS_INDICES,
        "threshold": config.CLASSIFICATION_THRESHOLD,
        "uncertainty_margin": config.UNCERTAINTY_MARGIN,
        "domain_screening": {
            "model": config.DOMAIN_SCREENING_MODEL_ID,
            "threshold": config.DOMAIN_ACCEPTANCE_THRESHOLD,
            "strategy": "Zero-shot CLIP contrastive prompt ensembling",
        },
        "captioning": {
            "model": config.BLIP_MODEL_ID,
            "prompts": {
                "caption": config.CAPTION_PROMPT,
                "summary": config.SUMMARY_PROMPT,
            },
        },
        "safety_disclaimer": config.SAFETY_DISCLAIMER,
    }


@app.get("/api/class-mapping")
def get_class_mapping() -> Dict[str, Any]:
    """Return the exact class indices and decision threshold."""
    return {
        "class_indices": config.CLASS_INDICES,
        "inverted_indices": config.INVERTED_CLASS_INDICES,
        "positive_class": "dent (index 1)",
        "negative_class": "crack (index 0)",
        "threshold": config.CLASSIFICATION_THRESHOLD,
        "sigmoid_interpretation": "Sigmoid output represents P(dent). P(crack) = 1.0 - P(dent).",
    }


@app.get("/api/dataset/info")
def get_dataset_info() -> Dict[str, Any]:
    """Return dataset structure report and leakage audit."""
    if not config.DATASET_REPORT_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dataset report not found. Please run scripts/prepare_data.py first.",
        )
    with open(config.DATASET_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.post("/api/analyze")
async def analyze_damage_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Execute complete 4-stage damage inspection:
    Stage 1: Image validation and safe RGB decoding.
    Stage 2: Semantic aircraft-image relevance screening (CLIP OOD rejection).
    Stage 3: VGG16 binary classification (only if domain accepted).
    Stage 4: Confidence scoring and uncertainty evaluation.
    """
    if not config.MODEL_SAVE_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not trained yet. Please run training first (scripts/run_train.py).",
        )

    file_bytes = await file.read()
    image = validate_uploaded_image(file_bytes, filename=file.filename)

    try:
        result = analyze_image(image)
        logger.info(
            "Analysis complete for '%s': status=%s, class=%s",
            file.filename,
            result["status"],
            result.get("predicted_class"),
        )
        return result
    except Exception as e:
        logger.error("Analysis pipeline failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline error: {str(e)}",
        )


@app.post("/api/classify")
async def classify_damage_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Classify damage using the multi-stage guarded analysis pipeline."""
    return await analyze_damage_endpoint(file=file)



@app.post("/api/caption")
async def generate_caption_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Generate a descriptive caption for an aircraft damage image using BLIP."""
    file_bytes = await file.read()
    image = validate_uploaded_image(file_bytes, filename=file.filename)

    try:
        result = describe_image(image, task="caption")
        logger.info("BLIP Caption generated for '%s': %s", file.filename, result["raw_generated_text"])
        return result
    except Exception as e:
        logger.error("Captioning failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"BLIP captioning error: {str(e)}",
        )


@app.post("/api/summary")
async def generate_summary_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Generate a detailed summary for an aircraft damage image using BLIP."""
    file_bytes = await file.read()
    image = validate_uploaded_image(file_bytes, filename=file.filename)

    try:
        result = describe_image(image, task="summary")
        logger.info("BLIP Summary generated for '%s': %s", file.filename, result["raw_generated_text"])
        return result
    except Exception as e:
        logger.error("Summarization failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"BLIP summarization error: {str(e)}",
        )


@app.get("/api/evaluation")
def get_evaluation_metrics() -> Dict[str, Any]:
    """Return training history and test evaluation metrics."""
    if not config.METRICS_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evaluation metrics not found. Please run model evaluation first (scripts/run_eval.py).",
        )

    with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    history = None
    if config.HISTORY_PATH.exists():
        with open(config.HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)

    return {
        "metrics": metrics,
        "history": history,
        "notebook_faithful_evaluation": metrics.get("notebook_faithful_evaluation"),
        "full_supplementary_evaluation": metrics.get("full_supplementary_evaluation"),
    }


@app.get("/api/results/figures/{name}")
def get_result_figure(name: str):
    """Serve allow-listed evaluation PNG plots."""
    if name not in config.ALLOWED_FIGURE_NAMES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Figure '{name}' not found or not allow-listed.",
        )

    figure_path = config.RESULTS_DIR / name
    if not figure_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Figure artifact '{name}' has not been generated yet.",
        )

    return FileResponse(str(figure_path), media_type="image/png")


# Benchmark Example files mapping
EXAMPLE_IMAGES = {
    "dent_example_1.jpg": config.TEST_DIR / "dent" / "144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg",
    "dent_example_2.jpg": config.TEST_DIR / "dent" / "149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg",
    "crack_example_1.jpg": config.TEST_DIR / "crack" / "18_2_JPG_jpg.rf.7f3408a0d0cbfa78c80adfe6635c91a6.jpg",
    "task9_task10_image.jpg": config.TEST_DIR / "dent" / "149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg",
    "144_10_image.jpg": config.TEST_DIR / "dent" / "144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg",
}


@app.get("/api/examples")
def list_examples() -> Dict[str, Any]:
    """List available test benchmark images."""
    return {"examples": list(EXAMPLE_IMAGES.keys())}


@app.get("/api/examples/{name}")
def get_example_image(name: str):
    """Serve allow-listed benchmark image file."""
    if name not in EXAMPLE_IMAGES:
        # Fallback search inside test directory
        test_file_candidates = list(config.TEST_DIR.glob(f"**/{name}"))
        if test_file_candidates:
            return FileResponse(str(test_file_candidates[0]), media_type="image/jpeg")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Example image not found.")

    target_path = EXAMPLE_IMAGES[name]
    if not target_path.exists():
        # Search dynamically if filename hash slightly differs
        stem = target_path.stem.split(".")[0]
        matches = list(config.TEST_DIR.glob(f"**/*{stem}*"))
        if matches:
            return FileResponse(str(matches[0]), media_type="image/jpeg")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset example image file not extracted.")

    return FileResponse(str(target_path), media_type="image/jpeg")


# Mount static frontend directory
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
