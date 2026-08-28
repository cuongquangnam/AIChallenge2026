from __future__ import annotations

import json
import logging
import re
from typing import Literal

from video_retrieval.config import Settings, get_settings
from video_retrieval.models import EventChainPlan, EventSpec
from video_retrieval.text.gemini_logging import log_gemini_failure

logger = logging.getLogger(__name__)

_EVENT_LINE_RE = re.compile(
    r"^\s*(E\d+)\s*[:.\-]?\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)

_EXTRACT_PROMPT = """\
You parse a video retrieval query into an ordered chain of events for search.
Return JSON only, no markdown:
{{
  "context": "short English summary of the overall scene or topic",
  "question_event_id": null,
  "events": [
    {{
      "event_id": "E1",
      "description": "human-readable event text from the query",
      "ocr": "on-screen text likely visible during this event",
      "asr": "spoken words likely heard during this event",
      "visual": "concise English visual description for image-text search",
      "is_question_target": false
    }}
  ]
}}
Rules:
- Preserve strict temporal order as E1, E2, E3, ...
- Use empty string when a channel is irrelevant.
- Keep ocr/asr in the query language; put visual in concise English.
- Do not invent events not described in the query.
{task_rules}

Query:
{query}
"""

_TASK_RULES = {
    "kis": (
        "Task: Known Item Search (KIS). Split the narrative into 1 to 5 ordered events "
        "that happen in sequence. question_event_id must be null; is_question_target false for all."
    ),
    "qa": (
        "Task: Question Answering (QA). Split into ordered events that describe the video context. "
        "Mark exactly one event with is_question_target=true — the moment whose frame holds the answer "
        "(e.g. scale reading, sign number, map legend). Set question_event_id to that event's id."
    ),
    "trake": (
        "Task: TRAKE temporal alignment. Events are usually labeled E1, E2, ... in the query. "
        "Preserve their order and descriptions. question_event_id must be null."
    ),
}


class EventChainExtractor:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = None
        self._types = None
        if self.settings.gemini_api_key and self.settings.query_planner != "heuristic":
            self._init_gemini()

    def _init_gemini(self) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return
        self._client = genai.Client(api_key=self.settings.gemini_api_key)
        self._types = types

    def extract(
        self,
        query: str,
        *,
        task: Literal["kis", "qa", "trake"] = "kis",
    ) -> EventChainPlan:
        query = query.strip()
        if not query:
            return EventChainPlan(task=task, events=[])

        if self._client is not None:
            try:
                return self._extract_gemini(query, task=task)
            except Exception as exc:
                log_gemini_failure(
                    component=f"{task} event extractor",
                    model=self.settings.gemini_model,
                    exc=exc,
                    query=query,
                    fallback="heuristic event split",
                )
        return heuristic_event_plan(query, task=task)

    def _extract_gemini(
        self,
        query: str,
        *,
        task: Literal["kis", "qa", "trake"],
    ) -> EventChainPlan:
        assert self._client is not None
        from video_retrieval.text.gemini_config import gemini_generate_config

        prompt = _EXTRACT_PROMPT.format(
            task_rules=_TASK_RULES[task],
            query=query,
        )
        kwargs: dict = {
            "model": self.settings.gemini_model,
            "contents": prompt,
        }
        config = gemini_generate_config(self.settings.gemini_model, json_response=True)
        if config is not None:
            kwargs["config"] = config
        response = self._client.models.generate_content(**kwargs)
        raw = (response.text or "").strip()
        return parse_event_plan(raw, fallback_query=query, task=task)


def heuristic_event_plan(
    query: str,
    *,
    task: Literal["kis", "qa", "trake"] = "kis",
) -> EventChainPlan:
    if task == "trake":
        return _heuristic_trake(query)
    if task == "qa":
        return _heuristic_qa(query)
    return _heuristic_kis(query)


def _heuristic_trake(query: str) -> EventChainPlan:
    matches = list(_EVENT_LINE_RE.finditer(query))
    if not matches:
        return EventChainPlan(
            task="trake",
            events=[
                EventSpec(
                    event_id="E1",
                    description=query.strip(),
                    visual=query.strip()[:160],
                )
            ],
        )
    context = query[: matches[0].start()].strip()
    events: list[EventSpec] = []
    for match in matches:
        event_id = match.group(1).upper()
        text = match.group(2).strip()
        events.append(
            EventSpec(
                event_id=event_id,
                description=text,
                visual=text[:160],
            )
        )
        return EventChainPlan(task="trake", context=context, events=events)


def _heuristic_kis(query: str) -> EventChainPlan:
    max_events = max(1, int(get_settings().kis_max_events))
    parts = _split_narrative(query)
    if not parts:
        parts = [query.strip()]
    parts = parts[:max_events]
    events = [
        EventSpec(
            event_id=f"E{index}",
            description=part,
            visual=part[:160],
            ocr=part[:120],
            asr=part[:120],
        )
        for index, part in enumerate(parts, start=1)
    ]
    return EventChainPlan(task="kis", context=parts[0][:80] if parts else "", events=events)


def _heuristic_qa(query: str) -> EventChainPlan:
    question_mark = query.rfind("?")
    if question_mark > 0:
        context_text = query[:question_mark].strip()
        question_part = query[question_mark:].strip()
    else:
        context_text = query
        question_part = query

    context_parts = _split_narrative(context_text) or [context_text]
    max_context = max(1, int(get_settings().kis_max_events) - 1)
    context_parts = context_parts[:max_context]

    events: list[EventSpec] = []
    for index, part in enumerate(context_parts, start=1):
        events.append(
            EventSpec(
                event_id=f"E{index}",
                description=part,
                visual=part[:160],
                ocr=part[:120],
                asr=part[:120],
            )
        )
    q_id = f"E{len(events) + 1}"
    events.append(
        EventSpec(
            event_id=q_id,
            description=question_part,
            visual=question_part[:160],
            ocr=question_part[:120],
            is_question_target=True,
        )
    )
    return EventChainPlan(
        task="qa",
        context=context_text[:200],
        events=events,
        question_event_id=q_id,
    )


def _split_narrative(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = re.split(r"\n+|(?<=[.!?])\s+", text)
    cleaned = [chunk.strip() for chunk in chunks if chunk.strip()]
    return cleaned


def parse_event_plan(
    raw: str,
    *,
    fallback_query: str,
    task: Literal["kis", "qa", "trake"],
) -> EventChainPlan:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return heuristic_event_plan(fallback_query, task=task)
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return heuristic_event_plan(fallback_query, task=task)
    if not isinstance(payload, dict):
        return heuristic_event_plan(fallback_query, task=task)

    events_raw = payload.get("events") or []
    events: list[EventSpec] = []
    if isinstance(events_raw, list):
        for index, item in enumerate(events_raw, start=1):
            if not isinstance(item, dict):
                continue
            event_id = str(item.get("event_id") or f"E{index}").strip().upper()
            description = str(
                item.get("description")
                or item.get("visual")
                or item.get("ocr")
                or item.get("asr")
                or ""
            ).strip()
            events.append(
                EventSpec(
                    event_id=event_id,
                    description=description,
                    ocr=str(item.get("ocr") or "").strip(),
                    asr=str(item.get("asr") or "").strip(),
                    visual=str(item.get("visual") or description).strip(),
                    is_question_target=bool(item.get("is_question_target")),
                )
            )

    if not events:
        return heuristic_event_plan(fallback_query, task=task)

    max_events = max(1, int(get_settings().kis_max_events))
    if task == "kis":
        events = events[:max_events]

    question_event_id = payload.get("question_event_id")
    if question_event_id is not None:
        question_event_id = str(question_event_id).strip().upper() or None
    if task == "qa" and not question_event_id:
        for event in events:
            if event.is_question_target:
                question_event_id = event.event_id
                break
        if not question_event_id and events:
            question_event_id = events[-1].event_id
            events[-1] = events[-1].model_copy(update={"is_question_target": True})

    return EventChainPlan(
        task=task,
        context=str(payload.get("context") or "").strip(),
        events=events,
        question_event_id=question_event_id,
    )
