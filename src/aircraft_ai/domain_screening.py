"""Domain relevance screening and out-of-distribution (OOD) rejection using CLIP."""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import threading

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from aircraft_ai import config
from aircraft_ai.errors import CorruptImageError
from aircraft_ai.logging_setup import setup_logger

logger = setup_logger("aircraft_ai.domain_screening")

_CLIP_MODEL: Optional[Any] = None
_CLIP_PROCESSOR: Optional[Any] = None
_CLIP_LOCK = threading.Lock()


def get_clip_pipeline() -> Tuple[Any, Any]:
    """Retrieve or lazily load the CLIP processor and model singleton."""
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is not None and _CLIP_PROCESSOR is not None:
        return _CLIP_PROCESSOR, _CLIP_MODEL

    with _CLIP_LOCK:
        if _CLIP_MODEL is None or _CLIP_PROCESSOR is None:
            from transformers import CLIPModel, CLIPProcessor
            logger.info("Loading CLIP model from '%s'...", config.DOMAIN_SCREENING_MODEL_ID)
            _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(config.DOMAIN_SCREENING_MODEL_ID)
            _CLIP_MODEL = CLIPModel.from_pretrained(config.DOMAIN_SCREENING_MODEL_ID)
            _CLIP_MODEL.eval()
            logger.info("CLIP model loaded successfully.")

    return _CLIP_PROCESSOR, _CLIP_MODEL


def load_pil_image_safely(image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    """Safely open, decode, and convert an image to RGB format, handling transparency."""
    try:
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input))
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            # Create a detached copy to ensure valid pixel buffer
            img = image_input.copy()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Handle transparency (RGBA / LA / P with transparency)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            alpha_composite = Image.alpha_composite(background, img)
            img = alpha_composite.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        return img
    except Exception as e:
        raise CorruptImageError(f"Failed to safely decode image: {e}") from e


def screen_image_domain(
    image_input: Union[str, Path, bytes, Image.Image],
    threshold: float = config.DOMAIN_ACCEPTANCE_THRESHOLD,
    in_prompts: Optional[List[str]] = None,
    out_prompts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate whether an input image plausibly depicts aircraft exterior structural damage.
    
    Returns structured domain screening metrics including in-domain score,
    rejection decision, top matched prompt, and category breakdown.
    """
    img = load_pil_image_safely(image_input)
    processor, model = get_clip_pipeline()

    in_domain = in_prompts or config.IN_DOMAIN_PROMPTS
    out_domain = out_prompts or config.OUT_OF_DOMAIN_PROMPTS
    all_prompts = in_domain + out_domain
    n_in = len(in_domain)

    with torch.no_grad():
        inputs = processor(text=all_prompts, images=img, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = F.softmax(logits_per_image, dim=1).squeeze().tolist()

    if isinstance(probs, float):
        probs = [probs]

    in_scores = probs[:n_in]
    out_scores = probs[n_in:]

    total_in_score = float(sum(in_scores))
    total_out_score = float(sum(out_scores))

    # Identify top matched prompt
    top_idx = int(np.argmax(probs))
    top_prompt = all_prompts[top_idx]
    is_top_in_domain = top_idx < n_in

    # Build human-readable breakdown
    breakdown = {prompt: round(float(prob), 4) for prompt, prob in zip(all_prompts, probs)}

    # Map reject categories
    matched_category = "Aircraft Structural Surface"
    rejection_reason = None

    if not is_top_in_domain or total_in_score < threshold:
        is_aircraft = False
        if "signature" in top_prompt or "cursive" in top_prompt:
            matched_category = "Handwritten Signature / Ink Strokes"
            rejection_reason = "Image depicts handwritten text or signature strokes, not an aircraft exterior surface."
        elif "document" in top_prompt or "letter" in top_prompt or "receipt" in top_prompt:
            matched_category = "Document / Printed Paperwork"
            rejection_reason = "Image appears to be a printed document or paperwork."
        elif "person" in top_prompt or "portrait" in top_prompt:
            matched_category = "Human Portrait / Person"
            rejection_reason = "Image depicts a human subject or face."
        elif "vehicle" in top_prompt or "car" in top_prompt:
            matched_category = "Ground Vehicle"
            rejection_reason = "Image depicts an automobile or non-aircraft vehicle."
        elif "landscape" in top_prompt or "nature" in top_prompt:
            matched_category = "Natural Landscape"
            rejection_reason = "Image appears to be a landscape or outdoor nature scenery."
        elif "diagram" in top_prompt or "drawing" in top_prompt:
            matched_category = "Digital Graphic / Diagram"
            rejection_reason = "Image appears to be a digital graphic, screenshot, or diagram."
        else:
            matched_category = "Out-of-Distribution Image"
            rejection_reason = "Image does not exhibit aircraft exterior surface characteristics."
    else:
        is_aircraft = True
        matched_category = "Aircraft Structural Surface"

    logger.info(
        "Domain screening: is_aircraft=%s, in_score=%.4f (threshold=%.2f), category='%s'",
        is_aircraft,
        total_in_score,
        threshold,
        matched_category,
    )

    return {
        "is_aircraft_domain": is_aircraft,
        "in_domain_score": round(total_in_score, 4),
        "out_of_domain_score": round(total_out_score, 4),
        "threshold": threshold,
        "matched_category": matched_category,
        "top_prompt": top_prompt,
        "rejection_reason": rejection_reason,
        "breakdown": breakdown,
    }
