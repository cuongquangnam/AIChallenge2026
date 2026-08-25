from __future__ import annotations

import json
import re

import httpx

from video_retrieval.config import Settings
from video_retrieval.models import QueryPlan

_PLAN_PROMPT = """\
Route a video-search query to OCR, ASR, and visual. Return JSON only, no markdown.
{{
  "ocr": "",
  "asr": "",
  "visual": "",
  "weights": {{"ocr": 0.0, "asr": 0.0, "visual": 0.0}}
}}

Channels:
- ocr: logos, captions, signs, program/channel names, exact written words from the query. Never people, actions, or scenes.
- asr: quoted speech, "says"/"nói", spoken topics. Leave empty unless the query implies spoken/news narration content. Never invent speech.
- visual: concise English scene description (people, objects, actions, places, clothing, counts, colors, camera framing).

Language (strict):
- ocr and asr MUST keep the query's original language. Do not translate them. Extract/trim only; preserve spelling, accents, and script.
- visual MUST be English. Translate concrete nouns exactly (hổ=tiger, dứa=pineapple, dê=goat, cực quang=aurora). Never substitute similar animals/foods.

Rules:
- Do not copy template phrases or example utterances into the output fields.
- Weights must sum to 1.0. Unused channel string = "" and weight = 0.0.
- Image/photo/frame/scene/picture requests, or a description of what is seen → visual 0.85-1.0.
- Logo, caption, or on-screen word → ocr 0.7-1.0.
- Spoken audio / what someone says → asr 0.6-1.0.
- Do not give ocr the highest weight just because the query contains nouns.

Examples:
find me an image of a teacher teaching a lot of children
{{"ocr":"","asr":"","visual":"a teacher teaching many children in a classroom","weights":{{"ocr":0.0,"asr":0.0,"visual":1.0}}}}

the VTV24 logo
{{"ocr":"VTV24","asr":"","visual":"television channel logo","weights":{{"ocr":0.85,"asr":0.0,"visual":0.15}}}}

người nói xin chào
{{"ocr":"","asr":"xin chào","visual":"person speaking","weights":{{"ocr":0.0,"asr":0.7,"visual":0.3}}}}

Mẩu tin về đàn hổ miền Nam vừa có thêm 3-6 hổ con quý hiếm
{{"ocr":"","asr":"đàn hổ miền Nam hổ con quý hiếm","visual":"rare tiger cubs in southern Vietnam, news report about a tiger family with several baby tigers","weights":{{"ocr":0.0,"asr":0.25,"visual":0.75}}}}

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
        if self.backend == "ollama":
            try:
                plan = self._plan_ollama(query)
                plan.source = "ollama"
                return plan
            except Exception as exc:
                print(f"[planner] ollama failed ({exc}); using heuristic plan", flush=True)
                return heuristic_plan(query)
        if self.backend != "gemini" or self._client is None:
            return heuristic_plan(query)

        try:
            plan = self._plan_gemini(query)
            if _looks_unplanned(plan, query):
                print("[planner] gemini returned raw query echo; retrying once", flush=True)
                plan = self._plan_gemini(query, soft_retry=True)
            if _looks_unplanned(plan, query):
                print("[planner] gemini plan still unparsed; using heuristic plan", flush=True)
                return heuristic_plan(query)
            plan.source = "gemini"
            return plan
        except Exception as exc:
            print(f"[planner] gemini failed ({exc}); retrying with soft prompt", flush=True)
            try:
                plan = self._plan_gemini(query, soft_retry=True)
                if _looks_unplanned(plan, query):
                    raise RuntimeError("soft retry still unplanned")
                plan.source = "gemini"
                return plan
            except Exception as retry_exc:
                print(
                    f"[planner] gemini soft retry failed ({retry_exc}); using heuristic plan",
                    flush=True,
                )
                return heuristic_plan(query)

    def _plan_gemini(self, query: str, *, soft_retry: bool = False) -> QueryPlan:
        from google.genai import types

        prompt = _PLAN_PROMPT.format(query=query)
        if soft_retry:
            prompt = (
                "Benign video-archive retrieval task only. "
                "Rewrite the query into OCR/ASR/visual search strings. "
                "Do not refuse; the content is documentary / news footage.\n\n"
                + prompt
            )
        kwargs: dict = {
            "model": self.settings.gemini_model,
            "contents": prompt,
        }
        config_kwargs: dict = {
            "response_mime_type": "application/json",
            # Planner only rewrites search strings; default filters false-positive
            # block some Vietnamese KIS queries (e.g. "cô bé" + clothing props).
            "safety_settings": [
                types.SafetySetting(category=category, threshold="BLOCK_NONE")
                for category in (
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "HARM_CATEGORY_CIVIC_INTEGRITY",
                )
            ],
        }
        thinking_level = _gemini_thinking_level(self.settings.gemini_model)
        # Soft retries skip thinking — fewer empty PROHIBITED_CONTENT responses.
        if thinking_level is not None and not soft_retry:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level
            )
        kwargs["config"] = types.GenerateContentConfig(**config_kwargs)

        response = self._client.models.generate_content(**kwargs)
        raw = (response.text or "").strip()
        if not raw:
            finish = None
            if response.candidates:
                finish = getattr(response.candidates[0], "finish_reason", None)
            raise RuntimeError(f"empty Gemini planner response (finish_reason={finish})")
        return parse_plan(raw, fallback_query=query)

    def _plan_ollama(self, query: str) -> QueryPlan:
        url = f"{self.settings.ollama_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.settings.ollama_model,
            "prompt": _PLAN_PROMPT.format(query=query),
            "stream": False,
            "format": "json",
        }
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            raw = str((response.json() or {}).get("response") or "").strip()
        return parse_plan(raw, fallback_query=query)


def heuristic_plan(query: str) -> QueryPlan:
    """Use the raw query for every channel when no LLM is available."""
    # Equal weights (sum=1). Old default of 1/1/1 looked like a "real" plan in the UI
    # but was only a fallback after Gemini/Ollama failures.
    return QueryPlan(
        ocr=query,
        asr=query,
        visual=query,
        weights={"ocr": 1.0 / 3, "asr": 1.0 / 3, "visual": 1.0 / 3},
        source="heuristic",
    )


def parse_plan(raw: str, *, fallback_query: str) -> QueryPlan:
    payload = _load_json(raw)
    if not isinstance(payload, dict):
        return heuristic_plan(fallback_query)

    weights_raw = payload.get("weights") or {}
    weights = {
        "ocr": _as_weight(weights_raw.get("ocr"), 0.0),
        "asr": _as_weight(weights_raw.get("asr"), 0.0),
        "visual": _as_weight(weights_raw.get("visual"), 0.0),
    }
    if sum(weights.values()) <= 0:
        weights = {"ocr": 1.0 / 3, "asr": 1.0 / 3, "visual": 1.0 / 3}
    plan = QueryPlan(
        ocr=str(payload.get("ocr") or "").strip(),
        asr=str(payload.get("asr") or "").strip(),
        visual=str(payload.get("visual") or "").strip(),
        weights=weights,
        source="gemini",
    )
    if not (plan.ocr or plan.asr or plan.visual):
        return heuristic_plan(fallback_query)
    return plan


def _planner_backend(settings: Settings) -> str:
    backend = (settings.query_planner or "auto").strip().lower()
    if backend == "heuristic":
        return "heuristic"
    if backend == "ollama":
        return "ollama"
    if backend in {"gemini", "auto"} and settings.gemini_api_key:
        return "gemini"
    return "heuristic"


def _gemini_thinking_level(model: str) -> str | None:
    """Pick a thinking level supported by the configured Gemini model.

    Gemini 3 Pro rejects ``minimal``; Flash/Lite accept it. Older Gemini 2.x
    models do not need a thinking config.
    """
    name = (model or "").strip().lower()
    if not name.startswith("gemini-3"):
        return None
    if "pro" in name:
        return "low"
    return "minimal"


def _looks_unplanned(plan: QueryPlan, query: str) -> bool:
    """True when the model echoed the raw query into every channel."""
    q = query.strip()
    if not q:
        return False
    channels = [plan.ocr.strip(), plan.asr.strip(), plan.visual.strip()]
    return all(channel == q for channel in channels)


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
