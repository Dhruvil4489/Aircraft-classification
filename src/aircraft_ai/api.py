"""FastAPI web service for aircraft damage classification, BLIP captioning, and metrics reporting."""

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
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
async def analyze_damage_endpoint(
    file: UploadFile = File(...),
    context: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Execute complete 4-stage damage inspection:
    Stage 1: Image validation and safe RGB decoding.
    Stage 2: Semantic aircraft-image relevance screening (CLIP OOD rejection).
    Stage 3: VGG16 binary classification (only if domain accepted).
    Stage 4: Confidence scoring and uncertainty evaluation.
    Maintenance Decision Support: Damage-specific repair and inspection guidance.
    """
    if not config.MODEL_SAVE_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not trained yet. Please run training first (scripts/run_train.py).",
        )

    file_bytes = await file.read()
    image = validate_uploaded_image(file_bytes, filename=file.filename)

    parsed_context = None
    if context:
        try:
            parsed_context = json.loads(context)
        except Exception as e:
            logger.warning("Failed to parse context JSON: %s", e)

    try:
        result = analyze_image(image, context=parsed_context)
        logger.info(
            "Analysis complete for '%s': status=%s, class=%s, assessment_status=%s",
            file.filename,
            result["status"],
            result.get("predicted_class"),
            result.get("assessment_status"),
        )
        return result
    except Exception as e:
        logger.error("Analysis pipeline failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline error: {str(e)}",
        )


@app.post("/api/classify")
async def classify_damage_endpoint(
    file: UploadFile = File(...),
    context: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Classify damage using the multi-stage guarded analysis pipeline."""
    return await analyze_damage_endpoint(file=file, context=context)


@app.post("/api/recommendation")
async def evaluate_recommendation_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate damage-specific maintenance guidance based on inspector-supplied context

    without needing to re-run neural network image inference.
    """
    from aircraft_ai.maintenance import get_damage_recommendation

    predicted_class = payload.get("predicted_class")
    model_score = payload.get("model_score")
    uncertainty_status = payload.get("uncertainty_status", "confident")
    context = payload.get("context")
    inspection_id = payload.get("inspection_id")
    uncertainty_reason = payload.get("uncertainty_reason")

    try:
        rec = get_damage_recommendation(
            predicted_class=predicted_class,
            model_score=model_score,
            uncertainty_status=uncertainty_status,
            context=context,
            inspection_id=inspection_id,
            uncertainty_reason=uncertainty_reason,
        )
        return rec.to_dict()
    except Exception as e:
        logger.error("Maintenance recommendation evaluation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation evaluation error: {str(e)}",
        )



@app.post("/api/caption")
async def generate_caption_endpoint(
    file: UploadFile = File(...),
    num_beams: Optional[int] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
    enhanced_prompts: Optional[bool] = Form(True),
) -> Dict[str, Any]:
    """Generate a descriptive caption for an aircraft damage image using BLIP.

    Optional form parameters:
    - num_beams: Beam search width (default: config.BLIP_NUM_BEAMS=5)
    - max_new_tokens: Max generation length (default: config.BLIP_MAX_NEW_TOKENS=75)
    - enhanced_prompts: Use domain-specific aerospace prompts (default: true)
    """
    file_bytes = await file.read()
    image = validate_uploaded_image(file_bytes, filename=file.filename)

    try:
        result = describe_image(
            image, task="caption",
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            enhanced_prompts=enhanced_prompts if enhanced_prompts is not None else True,
        )
        logger.info("BLIP Caption generated for '%s': %s", file.filename, result["generated_text"])
        return result
    except Exception as e:
        logger.error("Captioning failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"BLIP captioning error: {str(e)}",
        )


@app.post("/api/summary")
async def generate_summary_endpoint(
    file: UploadFile = File(...),
    num_beams: Optional[int] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
    enhanced_prompts: Optional[bool] = Form(True),
) -> Dict[str, Any]:
    """Generate a detailed summary for an aircraft damage image using BLIP.

    Optional form parameters:
    - num_beams: Beam search width (default: config.BLIP_NUM_BEAMS=5)
    - max_new_tokens: Max generation length (default: config.BLIP_MAX_NEW_TOKENS=75)
    - enhanced_prompts: Use domain-specific aerospace prompts (default: true)
    """
    file_bytes = await file.read()
    image = validate_uploaded_image(file_bytes, filename=file.filename)

    try:
        result = describe_image(
            image, task="summary",
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            enhanced_prompts=enhanced_prompts if enhanced_prompts is not None else True,
        )
        logger.info("BLIP Summary generated for '%s': %s", file.filename, result["generated_text"])
        return result
    except Exception as e:
        logger.error("Summarization failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"BLIP summarization error: {str(e)}",
        )


@app.post("/api/describe")
async def generate_full_description_endpoint(
    file: UploadFile = File(...),
    num_beams: Optional[int] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
    enhanced_prompts: Optional[bool] = Form(True),
) -> Dict[str, Any]:
    """Generate both caption AND summary in a single request.

    Returns a combined response with caption, summary, generation timing,
    and metadata — avoiding two separate API roundtrips.
    """
    from aircraft_ai.captioning import describe_image_full

    file_bytes = await file.read()
    image = validate_uploaded_image(file_bytes, filename=file.filename)

    try:
        result = describe_image_full(
            image,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            enhanced_prompts=enhanced_prompts if enhanced_prompts is not None else True,
        )
        logger.info(
            "BLIP Full description generated for '%s': caption='%s', summary='%s'",
            file.filename,
            result["caption"]["generated_text"],
            result["summary"]["generated_text"],
        )
        return result
    except Exception as e:
        logger.error("Full description generation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"BLIP description error: {str(e)}",
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
    "crack_example_2.jpg": config.TEST_DIR / "crack" / "101_1_JPG_jpg.rf.695bc0b3ba5ada38ee27d71dbacaf408.jpg",
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


@app.post("/api/inspection/report/html")
async def generate_inspection_report_html(data: Dict[str, Any]):
    """Generate a printable HTML aerospace Non-Destructive Inspection (NDI) report document

    incorporating damage assessment, recommended actions, inspector context, and verified references.
    """
    from fastapi.responses import HTMLResponse

    inspection_id = data.get("inspection_id", "INS-UNKNOWN")
    timestamp = data.get("timestamp", "N/A")
    status_code = data.get("status", "N/A")
    aircraft_status = data.get("aircraft_image_status", "SUPPORTED")
    presence_status = data.get("damage_presence_status", "NOT_ASSESSED")
    damage_type = (data.get("damage_type") or data.get("predicted_class") or "NOT_ASSESSED").upper()
    model_score = data.get("model_score")
    score_display = f"{float(model_score) * 100:.2f}%" if model_score is not None else "N/A (Damage presence unverified)"
    decision_message = data.get("decision_message", "")
    processing_time = data.get("processing_time_ms", 0)

    # Maintenance Recommendation Fields
    assessment_status = data.get("assessment_status", "awaiting_inspection").replace("_", " ").upper()
    qualified_heading = data.get("qualified_heading", f"Defect Status: {damage_type}")
    summary = data.get("summary", decision_message)
    recommended_actions = data.get("recommended_actions", [])
    maintenance_references = data.get("maintenance_references", [])
    missing_info = data.get("missing_information", [])
    limitations = data.get("limitations", [config.SAFETY_DISCLAIMER])
    context_used = data.get("context_used", {})

    loc = data.get("localization") or {}
    overlay_b64 = loc.get("overlay_data_url", "")
    heatmap_b64 = loc.get("heatmap_data_url", "")

    # Build action steps HTML list
    if recommended_actions:
        actions_html = "<ol style='padding-left: 20px; margin: 8px 0;'>"
        for act in recommended_actions:
            step_num = act.get("step", 1)
            action_text = act.get("action", "")
            src_status = act.get("source_status", "general_inspection_guidance").replace("_", " ")
            actions_html += f"""
            <li style='margin-bottom: 8px;'>
              <strong>Step {step_num}:</strong> {action_text}
              <div style='margin-top: 2px;'><span class='source-badge'>{src_status}</span></div>
            </li>"""
        actions_html += "</ol>"
    else:
        fallback_action = data.get("recommended_action", "Conduct manual visual inspection.")
        actions_html = f"<p><strong>Step 1:</strong> {fallback_action}</p>"

    # Build references HTML table
    if maintenance_references:
        refs_html = """<table class='sub-table'>
          <tr><th>Document Title</th><th>ID / Section</th><th>Status</th><th>Applicability Notes</th></tr>"""
        for r in maintenance_references:
            is_v = r.get("verified", False)
            v_badge = "<span style='color: #15803d; font-weight: 600;'>VERIFIED</span>" if is_v else "<span style='color: #b45309; font-weight: 600;'>UNVERIFIED / UNAVAILABLE</span>"
            refs_html += f"""<tr>
              <td><strong>{r.get('document_title', 'N/A')}</strong></td>
              <td>{r.get('document_id', 'N/A')} &mdash; {r.get('applicable_section', 'N/A')}</td>
              <td>{v_badge}</td>
              <td style='font-size: 12px;'>{r.get('applicability_notes', '')}</td>
            </tr>"""
        refs_html += "</table>"
    else:
        refs_html = "<p style='color: #64748b; font-style: italic;'>Applicable approved maintenance data is not available in this system. Obtain current approved documentation for the aircraft before determining repair.</p>"

    # Missing info HTML
    if missing_info:
        missing_html = "<ul style='margin: 4px 0; padding-left: 20px; color: #b45309;'>"
        for item in missing_info:
            missing_html += f"<li>{item}</li>"
        missing_html += "</ul>"
    else:
        missing_html = "<p style='color: #15803d; margin: 4px 0;'>All required physical and context parameters provided.</p>"

    # Limitations HTML
    limits_html = "<ul style='margin: 4px 0; padding-left: 20px; font-size: 12px; color: #92400e;'>"
    for lim in limitations:
        limits_html += f"<li>{lim}</li>"
    limits_html += "</ul>"

    # Measurements extraction
    meas_len = context_used.get("damage_length_mm")
    meas_wid = context_used.get("damage_width_mm")
    meas_dep = context_used.get("damage_depth_mm")
    if meas_len is not None and meas_wid is not None:
        meas_str = f"{meas_len:.1f} mm (L) &times; {meas_wid:.1f} mm (W)"
        if meas_dep is not None:
            meas_str += f" &times; {meas_dep:.2f} mm (Depth)"
    elif meas_dep is not None:
        meas_str = f"Depth: {meas_dep:.2f} mm (Length/Width unmeasured)"
    else:
        meas_str = "Dimensions unmeasured (cannot be established from 2D photograph alone)"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AeroDamage AI — Inspection Record {inspection_id}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      color: #1e293b;
      margin: 40px;
      line-height: 1.5;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      border-bottom: 2px solid #0f172a;
      padding-bottom: 12px;
      margin-bottom: 24px;
    }}
    .title {{ font-size: 20px; font-weight: 700; color: #0f172a; }}
    .subtitle {{ font-size: 13px; color: #64748b; margin-top: 2px; }}
    .ref-block {{ text-align: right; font-size: 13px; font-family: monospace; }}
    .status-badge {{
      display: inline-block;
      padding: 6px 14px;
      font-weight: 700;
      border-radius: 4px;
      font-size: 13px;
      margin-bottom: 16px;
      background: #f1f5f9;
      color: #0f172a;
      border: 1px solid #cbd5e1;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .status-badge.crack {{ background: #fef2f2; color: #b91c1c; border-color: #fca5a5; }}
    .status-badge.dent {{ background: #eff6ff; color: #1d4ed8; border-color: #93c5fd; }}
    .status-badge.uncertain {{ background: #fffbeb; color: #b45309; border-color: #fde68a; }}
    .status-badge.measurements_required {{ background: #fff7ed; color: #c2410c; border-color: #fdba74; }}
    .status-badge.qualified_assessment_required {{ background: #fef2f2; color: #b91c1c; border-color: #fca5a5; }}
    .status-badge.awaiting_authorized_disposition {{ background: #f0fdf4; color: #15803d; border-color: #86efac; }}
    .source-badge {{
      display: inline-block;
      padding: 2px 6px;
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      border-radius: 3px;
      background: #e2e8f0;
      color: #475569;
    }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px; }}
    th, td {{ padding: 8px 12px; border: 1px solid #e2e8f0; text-align: left; }}
    th {{ background: #f8fafc; font-weight: 600; width: 30%; }}
    .sub-table th {{ width: auto; font-size: 12px; }}
    .section-title {{
      font-size: 15px;
      font-weight: 700;
      color: #0f172a;
      margin: 24px 0 8px 0;
      padding-bottom: 4px;
      border-bottom: 1px solid #cbd5e1;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .recommendation-panel {{
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-left: 4px solid #2563eb;
      border-radius: 4px;
      padding: 16px;
      margin: 16px 0;
    }}
    .evidence-grid {{ display: flex; gap: 20px; margin: 20px 0; }}
    .evidence-card {{ flex: 1; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; text-align: center; }}
    .evidence-card img {{ max-width: 100%; height: auto; max-height: 240px; border-radius: 4px; object-fit: contain; }}
    .evidence-card h4 {{ margin: 0 0 8px 0; font-size: 13px; color: #475569; }}
    .disclaimer-box {{
      background: #fffbeb;
      border-left: 4px solid #f59e0b;
      padding: 12px 16px;
      font-size: 12px;
      color: #92400e;
      margin-top: 24px;
    }}
    .signoff-grid {{
      display: flex;
      justify-content: space-between;
      margin-top: 40px;
      padding-top: 20px;
      border-top: 1px dashed #cbd5e1;
      font-size: 13px;
    }}
    .signoff-box {{ width: 45%; }}
    .signoff-line {{ border-bottom: 1px solid #0f172a; margin-top: 40px; }}
    @media print {{
      body {{ margin: 20px; }}
      button {{ display: none; }}
    }}
  </style>
</head>
<body>
  <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
    <button onclick="window.print()" style="padding: 8px 16px; font-weight: 600; cursor: pointer; background: #2563eb; color: #fff; border: none; border-radius: 4px;">Print / Save as PDF</button>
  </div>
  <div class="header">
    <div>
      <div class="title">AERODAMAGE AI — STRUCTURAL INSPECTION RECORD</div>
      <div class="subtitle">AI-Assisted Aerospace Non-Destructive Visual Inspection & Decision Support</div>
    </div>
    <div class="ref-block">
      <div><strong>Record ID:</strong> {inspection_id}</div>
      <div><strong>Date:</strong> {timestamp}</div>
      <div><strong>ATA Spec:</strong> Chapter 51 / 53</div>
    </div>
  </div>

  <div style="margin-bottom: 12px;">
    <span class="status-badge {status_code.lower()}">CLASSIFICATION: {status_code} — {damage_type}</span>
    <span class="status-badge {assessment_status.lower().replace(' ', '_')}">DISPOSITION STATUS: {assessment_status}</span>
  </div>

  <div class="section-title">1. Inspector-Supplied Aircraft & Component Context</div>
  <table>
    <tr><th>Aircraft Manufacturer / Model</th><td>{context_used.get('aircraft_manufacturer') or 'Unknown'} / {context_used.get('aircraft_model') or 'Unknown'}</td></tr>
    <tr><th>Component / Zone ID</th><td>{context_used.get('component_id') or 'Not specified'}</td></tr>
    <tr><th>Structural Location / Proximity</th><td>{context_used.get('structural_location') or 'Not specified'} {f"(Critical Feature Alert)" if context_used.get('is_near_joint_or_fastener') else ""}</td></tr>
    <tr><th>Panel Material</th><td>{context_used.get('material') or 'Unknown / Unspecified'}</td></tr>
    <tr><th>Inspector Physical Measurements</th><td><strong>{meas_str}</strong></td></tr>
    <tr><th>Measurement Instrument / Method</th><td>{context_used.get('measurement_method') or 'Not recorded'}</td></tr>
    <tr><th>Inspector Notes</th><td>{context_used.get('inspection_notes') or 'None recorded'}</td></tr>
  </table>

  <div class="section-title">2. Computer Vision Inference & Diagnostics</div>
  <table>
    <tr><th>Stage A: Aircraft Image Status</th><td><strong>{aircraft_status}</strong></td></tr>
    <tr><th>Stage B: Damage Presence</th><td><strong>{presence_status}</strong></td></tr>
    <tr><th>Stage C: Defect Classification</th><td><strong>{damage_type}</strong></td></tr>
    <tr><th>Model Activation Score</th><td>{score_display}</td></tr>
    <tr><th>Diagnostic Finding</th><td>{decision_message}</td></tr>
    <tr><th>Inference Duration</th><td>{processing_time} ms</td></tr>
    <tr><th>Model Architecture</th><td>VGG16 (ImageNet Backbone Frozen) + Pre-Activation Grad-CAM Head</td></tr>
  </table>

  <h3>Visual Explainability Evidence (Grad-CAM Attention)</h3>
  <div class="evidence-grid">
    <div class="evidence-card">
      <h4>Grad-CAM Attention Heatmap</h4>
      {f'<img src="{heatmap_b64}" alt="Grad-CAM Heatmap">' if heatmap_b64 else '<p>Not generated</p>'}
    </div>
    <div class="evidence-card">
      <h4>Blended Panel Overlay</h4>
      {f'<img src="{overlay_b64}" alt="Blended Overlay">' if overlay_b64 else '<p>Not generated</p>'}
    </div>
  </div>

  <div class="section-title">3. DAMAGE ASSESSMENT & RECOMMENDED ACTION</div>
  <div class="recommendation-panel">
    <h3 style="margin-top: 0; color: #0f172a;">{qualified_heading}</h3>
    <p><strong>Assessment Summary:</strong> {summary}</p>
    
    <div style="margin-top: 16px;">
      <strong>Recommended Next Steps (Ordered Workflow):</strong>
      {actions_html}
    </div>

    <div style="margin-top: 16px;">
      <strong>Applicable Verified Maintenance References:</strong>
      {refs_html}
    </div>

    <div style="margin-top: 16px;">
      <strong>Missing Context & Unverified Parameters:</strong>
      {missing_html}
    </div>
  </div>

  <div class="disclaimer-box">
    <strong>OPERATIONAL LIMITATION & SAFETY NOTICE:</strong><br>
    {limits_html}
    <p style="margin: 6px 0 0 0;">This automated inspection record is an AI-assisted decision support artifact. It does NOT constitute a certified airworthiness approval, return-to-service authorization, or certified NDT sign-off. Defect verification, measurement confirmation, and repair disposition must be conducted by licensed A&P / Part 66 aircraft maintenance personnel.</p>
  </div>

  <div class="signoff-grid">
    <div class="signoff-box">
      <div><strong>NDT Level II/III Technician:</strong></div>
      <div class="signoff-line"></div>
      <div style="margin-top: 4px; font-size: 11px; color: #64748b;">Signature / Stamp / Date</div>
    </div>
    <div class="signoff-box">
      <div><strong>Quality Assurance / Lead Inspector:</strong></div>
      <div class="signoff-line"></div>
      <div style="margin-top: 4px; font-size: 11px; color: #64748b;">Signature / Stamp / Date</div>
    </div>
  </div>
</body>
</html>"""
    return HTMLResponse(content=html_content)



# Mount static frontend directory
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
