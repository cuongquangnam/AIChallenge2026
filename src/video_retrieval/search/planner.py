from __future__ import annotations

import json
import re

from video_retrieval.config import Settings
from video_retrieval.models import QueryPlan

_PLAN_PROMPT = """\
Split a video-search query into three retrieval channels.
Return JSON only, no markdown:
{
  "ocr": "words likely shown on screen (logos, captions, signs, names)",
  "asr": "words likely spoken in the audio",
  "visual": "English visual scene description for image-text embedding search",
  "weights": {"ocr": 0.0-1.0, "asr": 0.0-1.0, "visual": 0.0-1.0}
}
Rules:
- Use empty string if a channel is not relevant.
- Keep ocr/asr in the query language; put visual in concise English.
- Weights should sum to about 1; raise a channel if that is the user's main intent.
Query: {query}
"""


class QueryPlanner:
    """Turn a natural-language query into OCR / ASR / visual search strings."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = _planner_backend(settings)
        self._client = None
        if self.backend == "gemini":
            self._init_gemini()

    def _init_gemini(self) -> None:
        if not self.settings.gemini_api_key:
            self.backend = "heuristic"
            return
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            self.backend = "heuristic"
            return

        self._client = genai.Client(api_key=self.settings.gemini_api_key)
        self._types = types

    def plan(self, query: str) -> QueryPlan:
        query = query.strip()
        if not query:
            return QueryPlan()
        if self.backend != "gemini" or self._client is None:
            return heuristic_plan(query)

        try:
            return self._plan_gemini(query)
        except Exception:
            return heuristic_plan(query)

    def _plan_gemini(self, query: str) -> QueryPlan:
        from google.genai import types

        kwargs: dict = {
            "model": self.settings.gemini_model,
            "contents": _PLAN_PROMPT.format(query=query),
        }
        config_kwargs: dict = {"response_mime_type": "application/json"}
        if self.settings.gemini_model.startswith("gemini-3"):
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="minimal")
        kwargs["config"] = types.GenerateContentConfig(**config_kwargs)

        response = self._client.models.generate_content(**kwargs)
        raw = (response.text or "").strip()
        return parse_plan(raw, fallback_query=query)


def heuristic_plan(query: str) -> QueryPlan:
    """Use the raw query for every channel when no LLM is available."""
    return QueryPlan(ocr=query, asr=query, visual=query)


def parse_plan(raw: str, *, fallback_query: str) -> QueryPlan:
    payload = _load_json(raw)
    if not isinstance(payload, dict):
        return heuristic_plan(fallback_query)

    weights_raw = payload.get("weights") or {}
    weights = {
        "ocr": _as_weight(weights_raw.get("ocr"), 1.0),
        "asr": _as_weight(weights_raw.get("asr"), 1.0),
        "visual": _as_weight(weights_raw.get("visual"), 1.0),
    }
    plan = QueryPlan(
        ocr=str(payload.get("ocr") or "").strip(),
        asr=str(payload.get("asr") or "").strip(),
        visual=str(payload.get("visual") or "").strip(),
        weights=weights,
    )
    if not (plan.ocr or plan.asr or plan.visual):
        return heuristic_plan(fallback_query)
    return plan


def _planner_backend(settings: Settings) -> str:
    backend = (settings.query_planner or "auto").strip().lower()
    if backend == "heuristic":
        return "heuristic"
    if backend in {"gemini", "auto"} and settings.gemini_api_key:
        return "gemini"
    return "heuristic"


def _load_json(raw: str) -> object:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _as_weight(value: object, default: float) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if number < 0:
        return 0.0
    return number
