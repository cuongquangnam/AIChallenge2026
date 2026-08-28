from __future__ import annotations

import json
import logging
import math
import re
from typing import Literal

from video_retrieval.config import Settings, get_settings
from video_retrieval.models import EventChainPlan, EventSpec
from video_retrieval.text.gemini_client import get_gemini_client
from video_retrieval.text.gemini_logging import log_gemini_failure
from video_retrieval.text.llm import LLMClient

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
      "is_question_target": false,
      "gap_from_prev_sec": null,
      "gap_min_sec": null,
      "gap_max_sec": null
    }}
  ]
}}
Rules:
- Preserve strict temporal order as E1, E2, E3, ...
- Use empty string when a channel is irrelevant.
- Keep ocr/asr in the query language; put visual in concise English.
- Do not invent events not described in the query.
- For E1, gap_from_prev_sec / gap_min_sec / gap_max_sec must be null.
- For E2+, estimate wall-clock seconds between this event and the previous one
  in a typical 1–4 minute news/cooking/report clip:
  * same continuous action or next cooking step: expected 1–8, max ≤ 15
  * "then / sau đó / tiếp theo / tiếp đến": expected 3–20, max ≤ 40
  * "right after / ngay sau / lập tức": expected 1–4, max ≤ 10
  * start-of-clip vs end-of-clip, or a clear scene change: expected 10–45, max ≤ 90
  * "finally / cuối cùng / later": expected 15–60, max ≤ 120
- Always give a number even if uncertain; widen gap_max_sec when unsure.
- These are short videos, never hours.
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
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
    ):
        self.settings = settings or get_settings()
        self._llm = llm
        if self._llm is None and self.settings.gemini_api_key:
            if self.settings.query_planner != "heuristic":
                self._llm = get_gemini_client(self.settings)

    def extract(
        self,
        query: str,
        *,
        task: Literal["kis", "qa", "trake"] = "kis",
    ) -> EventChainPlan:
        query = query.strip()
        if not query:
            return EventChainPlan(task=task, events=[])

        if self._llm is not None:
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
        assert self._llm is not None
        prompt = _EXTRACT_PROMPT.format(
            task_rules=_TASK_RULES[task],
            query=query,
        )
        raw = self._llm.generate_text(
            prompt,
            json_response=True,
            component=f"{task} event extractor",
        )
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
            events=_apply_event_gaps(
                [
                    EventSpec(
                        event_id="E1",
                        description=query.strip(),
                        visual=query.strip()[:160],
                    )
                ],
                query=query,
            ),
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
    return EventChainPlan(
        task="trake",
        context=context,
        events=_apply_event_gaps(events, query=query),
    )


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
    events = _apply_event_gaps(events, query=query)
    return EventChainPlan(
        task="kis",
        context=parts[0][:80] if parts else "",
        events=events,
    )


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
        events=_apply_event_gaps(events, query=query),
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
                    gap_from_prev_sec=_as_optional_float(item.get("gap_from_prev_sec")),
                    gap_min_sec=_as_optional_float(item.get("gap_min_sec")),
                    gap_max_sec=_as_optional_float(item.get("gap_max_sec")),
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
        events=_apply_event_gaps(events, query=fallback_query),
        question_event_id=question_event_id,
    )


def _as_optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _complete_gap_range(
    expected: float,
    lo: float | None,
    hi: float | None,
) -> tuple[float, float]:
    if lo is None:
        lo = max(0.0, expected * 0.25)
    if hi is None:
        hi = max(expected * 3.0, expected + 5.0)
    lo = max(0.0, lo)
    hi = max(lo, hi)
    return lo, hi


def _heuristic_gap(prev_text: str, curr_text: str, query: str) -> tuple[float, float, float]:
    prev = prev_text.lower()
    curr = curr_text.lower()
    pair = f"{prev} {curr}"
    blob = f"{query} {pair}".lower()
    if re.search(r"ngay sau|lập tức|immediately|right after|cùng lúc", pair):
        return 2.0, 0.3, 8.0
    if re.search(r"sau vài giây|after a few seconds", blob):
        return 4.0, 1.0, 12.0
    if re.search(r"bắt đầu|starts?\s+with", prev) and re.search(
        r"kết thúc|ends?\s+with", curr
    ):
        return 25.0, 5.0, 90.0
    if re.search(r"cuối cùng|finally|\blater\b", curr):
        return 20.0, 4.0, 80.0
    if re.search(
        r"sau đó|tiếp theo|tiếp đến|\bthen\b|after that|followed by",
        pair,
    ):
        return 8.0, 1.5, 30.0
    return 6.0, 0.5, 25.0


def _apply_event_gaps(events: list[EventSpec], *, query: str) -> list[EventSpec]:
    if not events:
        return events
    filled: list[EventSpec] = []
    for index, event in enumerate(events):
        if index == 0:
            filled.append(
                event.model_copy(
                    update={
                        "gap_from_prev_sec": None,
                        "gap_min_sec": None,
                        "gap_max_sec": None,
                    }
                )
            )
            continue
        expected = event.gap_from_prev_sec
        lo = event.gap_min_sec
        hi = event.gap_max_sec
        if expected is None:
            prev = events[index - 1]
            expected, lo, hi = _heuristic_gap(
                prev.description or prev.visual,
                event.description or event.visual,
                query,
            )
        else:
            expected = max(0.0, min(expected, 180.0))
            lo, hi = _complete_gap_range(expected, lo, hi)
        filled.append(
            event.model_copy(
                update={
                    "gap_from_prev_sec": expected,
                    "gap_min_sec": lo,
                    "gap_max_sec": hi,
                }
            )
        )
    return filled
