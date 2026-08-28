from __future__ import annotations

import json
import logging
import re

from video_retrieval.config import Settings
from video_retrieval.models import QueryPlan
from video_retrieval.text.gemini_client import get_gemini_client
from video_retrieval.text.gemini_logging import log_gemini_failure
from video_retrieval.text.llm import LLMClient

logger = logging.getLogger(__name__)

_PLAN_PROMPT = """\
Split a video-search query into three retrieval channels.
Return JSON only, no markdown:
{{
  "ocr": "words likely shown on screen (logos, captions, signs, names)",
  "asr": "words likely spoken in the audio",
  "visual": "English visual scene description for image-text embedding search",
  "weights": {{"ocr": 0.0-1.0, "asr": 0.0-1.0, "visual": 0.0-1.0}}
}}
Rules:
- Use empty string if a channel is not relevant.
- Keep ocr/asr in the query language; put visual in concise English.
- Weights should sum to about 1; raise a channel if that is the user's main intent.
Query: {query}
"""


class QueryPlanner:
    """Turn a natural-language query into OCR / ASR / visual search strings."""

    def __init__(
        self,
        settings: Settings,
        *,
        llm: LLMClient | None = None,
    ):
        self.settings = settings
        self.backend = _planner_backend(settings)
        self._llm = llm
        if self.backend == "gemini" and self._llm is None:
            self._llm = get_gemini_client(settings)
        if self.backend == "gemini" and self._llm is None:
            logger.warning(
                "Gemini query planner disabled: GEMINI_API_KEY is empty; using heuristic plans"
            )
            self.backend = "heuristic"
        elif self.backend == "gemini":
            logger.info(
                "Gemini query planner ready (model=%s)",
                self.settings.gemini_model,
            )

    def plan(self, query: str) -> QueryPlan:
        query = query.strip()
        if not query:
            return QueryPlan()
        if self.backend != "gemini" or self._llm is None:
            return heuristic_plan(query)

        try:
            return self._plan_gemini(query)
        except Exception as exc:
            log_gemini_failure(
                component="query planner",
                model=self.settings.gemini_model,
                exc=exc,
                query=query,
                fallback="heuristic plan (copy query to ocr/asr/visual)",
            )
            return heuristic_plan(query)

    def _plan_gemini(self, query: str) -> QueryPlan:
        assert self._llm is not None
        raw = self._llm.generate_text(
            _PLAN_PROMPT.format(query=query),
            json_response=True,
            component="query planner",
        )
        plan = parse_plan(raw, fallback_query=query)
        if not (plan.ocr or plan.asr or plan.visual):
            logger.warning(
                "Gemini query planner returned empty channels (model=%s); "
                "falling back to heuristic. raw_response=%r",
                self.settings.gemini_model,
                raw[:800],
            )
            return heuristic_plan(query)
        return plan


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
