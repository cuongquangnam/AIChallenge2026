from __future__ import annotations

from google.genai import types


def gemini_supports_thinking_minimal(model: str) -> bool:
    """Return True only for Gemini 3 models that accept thinking_level=MINIMAL.

    Preview Pro models (e.g. gemini-3.1-pro-preview) reject MINIMAL with 400.
    """
    name = (model or "").strip().lower()
    if not name.startswith("gemini-3"):
        return False
    if "pro" in name:
        return False
    return True


def gemini_generate_config(model: str, *, json_response: bool = False) -> types.GenerateContentConfig | None:
    """Build a GenerateContentConfig safe for the configured GEMINI_MODEL."""
    config_kwargs: dict = {}
    if json_response:
        config_kwargs["response_mime_type"] = "application/json"
    if gemini_supports_thinking_minimal(model):
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="minimal")
    if not config_kwargs:
        return None
    return types.GenerateContentConfig(**config_kwargs)
