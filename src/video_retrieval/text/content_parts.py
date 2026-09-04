"""Neutral multimodal content parts shared by Gemini / Qwen VL clients."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": str(text)}


def image_part(image: Any) -> dict[str, Any]:
    """``image`` may be a PIL Image, Path, or filesystem path string."""
    return {"type": "image", "image": image}


def normalize_parts(parts: list[Any]) -> list[dict[str, Any]]:
    """Convert mixed provider-specific parts into neutral dicts."""
    out: list[dict[str, Any]] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, dict) and part.get("type") in {"text", "image"}:
            out.append(part)
            continue
        if isinstance(part, str):
            out.append(text_part(part))
            continue
        if isinstance(part, Path):
            out.append(image_part(part))
            continue
        # PIL.Image.Image
        if hasattr(part, "mode") and hasattr(part, "size") and hasattr(part, "convert"):
            out.append(image_part(part))
            continue
        # google.genai types.Part
        text = getattr(part, "text", None)
        if isinstance(text, str) and text:
            out.append(text_part(text))
            continue
        inline = getattr(part, "inline_data", None)
        if inline is not None:
            data = getattr(inline, "data", None)
            mime = getattr(inline, "mime_type", None) or "image/jpeg"
            if data:
                out.append({"type": "image_bytes", "data": data, "mime_type": mime})
                continue
        # Some Gemini Part wrappers store a PIL image on construction via Part(image)
        for attr in ("image", "_image", "pil_image"):
            value = getattr(part, attr, None)
            if value is not None and hasattr(value, "mode"):
                out.append(image_part(value))
                break
        else:
            # Last resort: treat unknown objects with a string form as text
            rendered = str(part).strip()
            if rendered:
                out.append(text_part(rendered))
    return out


def load_pil_image(image: Any):
    from io import BytesIO

    from PIL import Image

    if hasattr(image, "mode") and hasattr(image, "size"):
        pil = image
    elif isinstance(image, (str, Path)):
        pil = Image.open(image)
    elif isinstance(image, (bytes, bytearray)):
        pil = Image.open(BytesIO(image))
    else:
        raise TypeError(f"Unsupported image type: {type(image)!r}")
    if pil.mode not in ("RGB", "L"):
        pil = pil.convert("RGB")
    return pil
