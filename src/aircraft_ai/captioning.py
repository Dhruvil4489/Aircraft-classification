"""BLIP Captioning and Summarization module with concurrency protection and typed diagnostics."""

import io
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from aircraft_ai import config
from aircraft_ai.errors import (
    BlipError,
    BlipLoadError,
    CorruptImageError,
    InvalidTaskError,
)
from aircraft_ai.logging_setup import setup_logger

logger = setup_logger("aircraft_ai.captioning")

# Global inference lock to guarantee concurrency of 1 for memory safety
_BLIP_INFERENCE_LOCK = threading.Lock()
_PROCESSOR: Any = None
_MODEL: Any = None


def get_blip_model() -> Tuple[Any, Any]:
    """Retrieve or lazily initialize the singleton BLIP processor and PyTorch model."""
    global _PROCESSOR, _MODEL
    if _PROCESSOR is not None and _MODEL is not None:
        return _PROCESSOR, _MODEL

    try:
        from transformers.models.blip import BlipForConditionalGeneration, BlipProcessor
        logger.info("Loading pretrained BLIP processor and model from '%s'...", config.BLIP_MODEL_ID)
        proc: Any = BlipProcessor.from_pretrained(config.BLIP_MODEL_ID)
        mdl: Any = BlipForConditionalGeneration.from_pretrained(config.BLIP_MODEL_ID)
        if hasattr(mdl, "eval"):
            mdl.eval()
        _PROCESSOR = proc
        _MODEL = mdl
        logger.info("BLIP model successfully loaded onto device: %s", config.BLIP_DEVICE)
        return _PROCESSOR, _MODEL
    except Exception as e:
        error_msg = (
            f"Failed to load BLIP model '{config.BLIP_MODEL_ID}': {e}. "
            "If operating in an offline environment, please ensure weights are pre-downloaded "
            "to HuggingFace cache or set HF_HUB_OFFLINE=1 after downloading."
        )
        logger.error(error_msg)
        raise BlipLoadError(error_msg) from e


# ─── Post-Processing Pipeline ───────────────────────────────────────────────


def _clean_generated_text(raw_text: str, prompt: str) -> str:
    """Clean and normalize BLIP-generated text for display quality.

    Steps:
      1. Strip the conditioning prompt prefix.
      2. Remove leading punctuation artifacts (colon, comma).
      3. Collapse duplicate whitespace.
      4. Capitalize first letter.
      5. Ensure the text ends with a period.
      6. Remove repeated phrases (BLIP repetition artifact).
    """
    cleaned = raw_text.strip()

    # Strip prompt prefix (case-insensitive)
    if cleaned.lower().startswith(prompt.lower()):
        cleaned = cleaned[len(prompt):].strip()

    # Remove leading punctuation artifacts
    cleaned = re.sub(r"^[,:;\-–—]+\s*", "", cleaned)

    # Collapse multiple spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned)

    # Remove repeated consecutive phrases (common BLIP artifact)
    # e.g., "a plane with a plane with damage" -> "a plane with damage"
    cleaned = re.sub(r"\b(\w+(?:\s+\w+){0,3})\s+\1\b", r"\1", cleaned, flags=re.IGNORECASE)

    # Capitalize first letter
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    # Ensure proper sentence ending
    if cleaned and not cleaned.endswith((".","!","?")):
        cleaned = cleaned.rstrip(",;:") + "."

    return cleaned


def _count_tokens(text: str) -> int:
    """Approximate token count using whitespace splitting."""
    return len(text.split())


# ─── Core Generation ────────────────────────────────────────────────────────


def describe_image(
    image_input: Union[str, Path, bytes, Image.Image],
    task: str = "caption",
    max_new_tokens: int | None = None,
    num_beams: int | None = None,
    enhanced_prompts: bool = True,
) -> Dict[str, Any]:
    """Generate a caption or summary for an aircraft damage image using pretrained BLIP.
    
    Academic framing:
    - Both 'caption' and 'summary' utilize the exact same BLIP model checkpoint.
    - 'caption' is conditioned on: "This is a picture of"
    - 'summary' is conditioned on: "This is a detailed photo showing"
    - Generated descriptions are prompt-conditioned captions and do not constitute
      guaranteed factual or certified structural assessments.

    When `enhanced_prompts=True` (default), domain-specific aerospace prompts are used
    instead of the generic academic prompts, producing more relevant descriptions.
    """
    normalized_task = task.strip().lower()
    if normalized_task not in {"caption", "summary"}:
        raise InvalidTaskError(f"Task must be 'caption' or 'summary', got '{task}'")

    # Load and validate image
    try:
        if isinstance(image_input, (str, Path)):
            path_obj = Path(image_input)
            if not path_obj.exists():
                raise FileNotFoundError(f"Image file does not exist: {path_obj}")
            img = Image.open(str(path_obj))
            img.verify()
            img = Image.open(str(path_obj))
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
            img.verify()
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            img = image_input.copy()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        if img.mode != "RGB":
            img = img.convert("RGB")
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            raise
        raise CorruptImageError(f"Failed to open or decode image: {e}") from e

    # Determine prompt — enhanced aerospace-specific or original academic
    if enhanced_prompts:
        prompt = (
            config.CAPTION_PROMPT_ENHANCED
            if normalized_task == "caption"
            else config.SUMMARY_PROMPT_ENHANCED
        )
    else:
        prompt = config.CAPTION_PROMPT if normalized_task == "caption" else config.SUMMARY_PROMPT

    # Apply generation defaults from config if not explicitly overridden
    effective_max_tokens = max_new_tokens or config.BLIP_MAX_NEW_TOKENS
    effective_num_beams = num_beams or config.BLIP_NUM_BEAMS

    processor: Any
    model: Any
    processor, model = get_blip_model()

    # Inference guarded by lock
    with _BLIP_INFERENCE_LOCK:
        try:
            t_start = time.perf_counter()

            inputs = processor(images=img, text=prompt, return_tensors="pt")
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": effective_max_tokens,
                "num_beams": effective_num_beams,
                "length_penalty": config.BLIP_LENGTH_PENALTY,
                "repetition_penalty": config.BLIP_REPETITION_PENALTY,
            }

            output = model.generate(**inputs, **gen_kwargs)
            raw_text = processor.decode(output[0], skip_special_tokens=True).strip()

            t_end = time.perf_counter()
            generation_time_ms = round((t_end - t_start) * 1000, 1)

            # Post-process for clean, professional output
            cleaned_text = _clean_generated_text(raw_text, prompt)

            return {
                "task": normalized_task,
                "prompt_used": prompt,
                "enhanced_prompts": enhanced_prompts,
                "raw_generated_text": raw_text,
                "generated_text": cleaned_text,
                "generated_text_without_prompt": cleaned_text,  # backward compat
                "model_id": config.BLIP_MODEL_ID,
                "device": config.BLIP_DEVICE,
                "generation_config": {
                    "num_beams": effective_num_beams,
                    "max_new_tokens": effective_max_tokens,
                    "length_penalty": config.BLIP_LENGTH_PENALTY,
                    "repetition_penalty": config.BLIP_REPETITION_PENALTY,
                },
                "metrics": {
                    "generation_time_ms": generation_time_ms,
                    "output_token_count": _count_tokens(cleaned_text),
                    "raw_token_count": _count_tokens(raw_text),
                },
                "note": (
                    "Note: 'summary' is generated by conditioning the BLIP captioning model "
                    "on a detailed prompt prefix ('This is a detailed photo showing'). It is not a "
                    "separate summarization model."
                ),
            }
        except Exception as e:
            logger.error("BLIP generation failed during task '%s': %s", normalized_task, e)
            raise BlipError(f"BLIP generation failed: {e}") from e


def describe_image_full(
    image_input: Union[str, Path, bytes, Image.Image],
    max_new_tokens: int | None = None,
    num_beams: int | None = None,
    enhanced_prompts: bool = True,
) -> Dict[str, Any]:
    """Generate both caption AND summary in a single call, returning a combined result.

    This avoids two separate API roundtrips and provides a unified response
    with side-by-side comparison of the caption vs summary outputs.
    """
    t_start = time.perf_counter()

    caption_result = describe_image(
        image_input, task="caption",
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        enhanced_prompts=enhanced_prompts,
    )
    summary_result = describe_image(
        image_input, task="summary",
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        enhanced_prompts=enhanced_prompts,
    )

    t_end = time.perf_counter()
    total_time_ms = round((t_end - t_start) * 1000, 1)

    return {
        "caption": caption_result,
        "summary": summary_result,
        "model_id": config.BLIP_MODEL_ID,
        "device": config.BLIP_DEVICE,
        "enhanced_prompts": enhanced_prompts,
        "total_generation_time_ms": total_time_ms,
        "note": (
            "Both caption and summary use the same BLIP checkpoint conditioned on different "
            "prompt prefixes. Enhanced prompts use aerospace-specific language for more "
            "relevant descriptions of structural damage."
        ),
    }
