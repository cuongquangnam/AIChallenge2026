from __future__ import annotations

from video_retrieval.config import Settings, get_settings
from video_retrieval.text.llm import LLMClient


def resolve_llm_backend(settings: Settings | None = None) -> str:
    """Return ``gemini``, ``qwen_vl``, or ``none``."""
    cfg = settings or get_settings()
    backend = (cfg.llm_backend or "auto").strip().lower()
    if backend in {"qwen", "qwen2_5_vl", "qwen2.5-vl", "qwen_vl"}:
        return "qwen_vl"
    if backend == "gemini":
        return "gemini" if cfg.gemini_api_key else "none"
    if backend in {"none", "off", "disabled"}:
        return "none"
    # auto: prefer Gemini when keyed, otherwise leave none (callers fall back)
    if cfg.gemini_api_key:
        return "gemini"
    return "none"


def get_llm_client(
    settings: Settings | None = None,
    *,
    force: bool = False,
    backend: str | None = None,
) -> LLMClient | None:
    """Shared text/multimodal LLM used by planner, extractor, QA, OCR."""
    cfg = settings or get_settings()
    chosen = (backend or resolve_llm_backend(cfg)).strip().lower()
    if chosen in {"qwen", "qwen2_5_vl", "qwen2.5-vl"}:
        chosen = "qwen_vl"
    if chosen == "gemini":
        from video_retrieval.text.gemini_client import get_gemini_client

        return get_gemini_client(cfg, force=force)
    if chosen == "qwen_vl":
        from video_retrieval.text.qwen_vl_client import get_qwen_vl_client

        return get_qwen_vl_client(cfg, force=force)
    return None
