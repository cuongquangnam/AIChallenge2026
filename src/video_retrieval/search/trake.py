from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

from video_retrieval.config import Settings, get_settings
from video_retrieval.models import (
    SearchHit,
    TrakeChain,
    TrakeEventHit,
    TrakeEventPlan,
    TrakePlan,
    TrakeResult,
)
from video_retrieval.search.service import SearchService
from video_retrieval.text.gemini_logging import log_gemini_failure

logger = logging.getLogger(__name__)

_EVENT_LINE_RE = re.compile(
    r"^\s*(E\d+)\s*[:.\-]?\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)

_TRAKE_PLAN_PROMPT = """\
You parse a Temporal Retrieval and Alignment of Key Events (TRAKE) query.
Return JSON only, no markdown:
{{
  "context": "short English summary of the overall scene",
  "events": [
    {{
      "event_id": "E1",
      "ocr": "on-screen text if any",
      "asr": "spoken words if any",
      "visual": "concise English visual description for image-text search"
    }}
  ]
}}
Rules:
- Preserve event order exactly as E1, E2, E3, ...
- Use empty string when a channel is irrelevant.
- Keep ocr/asr in the query language; put visual in concise English.
- Do not invent events that are not described.

Query:
{query}
"""


class TrakeService:
    """Parse TRAKE events, retrieve per-event frames, align ordered chains."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        search: SearchService | None = None,
    ):
        self.settings = settings or get_settings()
        self.search = search or SearchService(self.settings)
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

    def run(self, query: str, *, top_chains: int | None = None) -> TrakeResult:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")

        plan = self.parse_events(query)
        if not plan.events:
            raise ValueError("No TRAKE events (E1, E2, …) found in the query")

        event_hits = self._retrieve_events(plan)
        chain_limit = top_chains or self.settings.trake_top_chains
        chains = self._align_chains(
            event_hits,
            plan=plan,
            top_videos=max(self.settings.trake_top_videos, min(chain_limit, 24)),
            top_chains=chain_limit,
            per_event=self.settings.trake_candidates_per_event,
        )
        csv_row = ""
        if chains:
            best = chains[0]
            frames = ",".join(str(ev.frame_index) for ev in best.events)
            csv_row = f"{best.video_id},{frames}"
        return TrakeResult(query=query, plan=plan, chains=chains, csv_row=csv_row)

    def parse_events(self, query: str) -> TrakePlan:
        if self._client is not None:
            try:
                return self._parse_gemini(query)
            except Exception as exc:
                log_gemini_failure(
                    component="TRAKE planner",
                    model=self.settings.gemini_model,
                    exc=exc,
                    query=query,
                    fallback="heuristic E1/E2/… line split",
                )
        return heuristic_trake_plan(query)

    def _parse_gemini(self, query: str) -> TrakePlan:
        assert self._client is not None and self._types is not None
        from video_retrieval.text.gemini_config import gemini_generate_config

        kwargs: dict = {
            "model": self.settings.gemini_model,
            "contents": _TRAKE_PLAN_PROMPT.format(query=query),
        }
        config = gemini_generate_config(self.settings.gemini_model, json_response=True)
        if config is not None:
            kwargs["config"] = config
        response = self._client.models.generate_content(**kwargs)
        raw = (response.text or "").strip()
        return parse_trake_plan(raw, fallback_query=query)

    def _retrieve_events(self, plan: TrakePlan) -> dict[str, list[SearchHit]]:
        limit = self.settings.trake_event_limit
        by_event: dict[str, list[SearchHit]] = {}
        for event in plan.events:
            query = _event_search_query(event, context=plan.context)
            # Prefer visual channel with a short string to stay under SigLIP limits.
            try:
                response = self.search.search_visual(query, limit=limit)
            except Exception:
                response = self.search.search_mixed(query, limit=min(limit, 20))
            by_event[event.event_id] = list(response.hits)
        return by_event

    def _retrieve_event_in_video(
        self,
        event: TrakeEventPlan,
        *,
        video_id: str,
        context: str,
        limit: int,
    ) -> list[SearchHit]:
        """Visual search restricted to one video for TRAKE refinement."""
        query = _event_search_query(event, context="")
        hits: list[SearchHit] = []
        try:
            vector = self.search.visual.encode_text(query)
            hits.extend(
                self.search.qdrant.search(
                    vector,
                    vector_name="siglip",
                    limit=limit,
                    video_id=video_id,
                )
            )
        except Exception:
            pass
        return hits

    def _align_chains(
        self,
        event_hits: dict[str, list[SearchHit]],
        *,
        plan: TrakePlan,
        top_videos: int,
        top_chains: int,
        per_event: int,
    ) -> list[TrakeChain]:
        event_ids = list(event_hits.keys())
        if not event_ids:
            return []
        events_by_id = {event.event_id: event for event in plan.events}

        video_scores = _score_videos(event_hits)
        if not video_scores:
            return []
        ranked_videos = sorted(video_scores.items(), key=lambda item: item[1], reverse=True)
        selected = [video_id for video_id, _ in ranked_videos[: max(top_videos, 1)]]

        chains: list[TrakeChain] = []
        for video_id in selected:
            per_event_cands: list[list[tuple[int, float, SearchHit]]] = []
            for event_id in event_ids:
                cands: list[tuple[int, float, SearchHit]] = []
                seen_frames: set[int] = set()
                pool = [
                    hit
                    for hit in event_hits[event_id]
                    if hit.video_id == video_id and hit.frame_index is not None
                ]
                event = events_by_id.get(event_id)
                if event is not None:
                    pool.extend(
                        self._retrieve_event_in_video(
                            event,
                            video_id=video_id,
                            context=plan.context,
                            limit=per_event,
                        )
                    )
                for rank, hit in enumerate(pool):
                    if hit.frame_index is None:
                        continue
                    frame = int(hit.frame_index)
                    if frame in seen_frames:
                        continue
                    seen_frames.add(frame)
                    score = float(hit.score) + 1.0 / (60 + rank + 1)
                    cands.append((frame, score, hit))
                    if len(cands) >= per_event:
                        break
                cands.sort(key=lambda item: item[0])
                per_event_cands.append(cands)

            if any(not cands for cands in per_event_cands):
                continue

            best_path = _best_monotonic_path(per_event_cands)
            if best_path is None:
                continue
            events: list[TrakeEventHit] = []
            total = 0.0
            for event_id, (frame, score, hit) in zip(event_ids, best_path):
                total += score
                events.append(
                    TrakeEventHit(
                        event_id=event_id,
                        frame_index=frame,
                        score=score,
                        timestamp_sec=hit.timestamp_sec,
                        keyframe_path=hit.keyframe_path,
                        text=hit.text,
                        source=hit.source or "trake",
                    )
                )
            chains.append(
                TrakeChain(
                    video_id=video_id,
                    score=total + video_scores.get(video_id, 0.0),
                    events=events,
                )
            )

        chains.sort(key=lambda chain: chain.score, reverse=True)
        return chains[: max(top_chains, 1)]


def heuristic_trake_plan(query: str) -> TrakePlan:
    """Split on E1/E2/... lines; leftover text becomes context."""
    matches = list(_EVENT_LINE_RE.finditer(query))
    if not matches:
        # Whole query as a single visual event.
        return TrakePlan(
            context="",
            events=[
                TrakeEventPlan(event_id="E1", visual=query.strip(), ocr="", asr=""),
            ],
        )

    context = query[: matches[0].start()].strip()
    events: list[TrakeEventPlan] = []
    for match in matches:
        event_id = match.group(1).upper()
        text = match.group(2).strip()
        events.append(
            TrakeEventPlan(
                event_id=event_id,
                visual=text[:160],
                ocr="",
                asr="",
            )
        )
    return TrakePlan(context=context, events=events)


def parse_trake_plan(raw: str, *, fallback_query: str) -> TrakePlan:
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
        return heuristic_trake_plan(fallback_query)
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return heuristic_trake_plan(fallback_query)
    if not isinstance(payload, dict):
        return heuristic_trake_plan(fallback_query)

    events_raw = payload.get("events") or []
    events: list[TrakeEventPlan] = []
    if isinstance(events_raw, list):
        for index, item in enumerate(events_raw, start=1):
            if not isinstance(item, dict):
                continue
            event_id = str(item.get("event_id") or f"E{index}").strip().upper()
            events.append(
                TrakeEventPlan(
                    event_id=event_id,
                    ocr=str(item.get("ocr") or "").strip(),
                    asr=str(item.get("asr") or "").strip(),
                    visual=str(item.get("visual") or "").strip(),
                )
            )
    if not events:
        return heuristic_trake_plan(fallback_query)
    return TrakePlan(
        context=str(payload.get("context") or "").strip(),
        events=events,
    )


def _event_search_query(event: TrakeEventPlan, *, context: str = "") -> str:
    """Build a short retrieval string (SigLIP max ~64 tokens)."""
    base = (event.visual or event.ocr or event.asr or event.event_id).strip()
    # Prefer the event itself; optionally prepend a tiny context cue.
    if context and len(base) < 80:
        cue = " ".join(context.split()[:8])
        if cue:
            base = f"{cue}. {base}"
    if len(base) > 160:
        base = base[:160].rsplit(" ", 1)[0] or base[:160]
    return base


def _score_videos(event_hits: dict[str, list[SearchHit]]) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    coverage: dict[str, set[str]] = defaultdict(set)
    for event_id, hits in event_hits.items():
        best: dict[str, float] = {}
        for rank, hit in enumerate(hits):
            if not hit.video_id:
                continue
            contribution = 1.0 / (60 + rank + 1)
            best[hit.video_id] = max(best.get(hit.video_id, 0.0), contribution)
            coverage[hit.video_id].add(event_id)
        for video_id, contribution in best.items():
            scores[video_id] += contribution
    # Prefer videos that cover more events.
    for video_id, events in coverage.items():
        scores[video_id] += 0.5 * len(events)
    return dict(scores)


def _best_monotonic_path(
    per_event_cands: list[list[tuple[int, float, SearchHit]]],
) -> list[tuple[int, float, SearchHit]] | None:
    """DP: maximize sum of scores with strictly increasing frame indices."""
    n = len(per_event_cands)
    # dp[i][j] = best score using first i+1 events ending at candidate j of event i
    dp: list[list[float]] = []
    prev: list[list[int | None]] = []

    first = per_event_cands[0]
    dp.append([score for _, score, _ in first])
    prev.append([None] * len(first))

    for i in range(1, n):
        cur = per_event_cands[i]
        dp_row: list[float] = []
        prev_row: list[int | None] = []
        for j, (frame_j, score_j, _) in enumerate(cur):
            best_score = float("-inf")
            best_k: int | None = None
            for k, (frame_k, _, _) in enumerate(per_event_cands[i - 1]):
                if frame_k >= frame_j:
                    continue
                candidate = dp[i - 1][k] + score_j
                if candidate > best_score:
                    best_score = candidate
                    best_k = k
            dp_row.append(best_score)
            prev_row.append(best_k)
        dp.append(dp_row)
        prev.append(prev_row)

    last_scores = dp[-1]
    if not last_scores or all(score == float("-inf") for score in last_scores):
        return _greedy_monotonic_path(per_event_cands)

    j = max(range(len(last_scores)), key=lambda idx: last_scores[idx])
    path_rev: list[tuple[int, float, SearchHit]] = []
    for i in range(n - 1, -1, -1):
        path_rev.append(per_event_cands[i][j])
        parent = prev[i][j]
        if i == 0:
            break
        if parent is None:
            return _greedy_monotonic_path(per_event_cands)
        j = parent
    path_rev.reverse()
    return path_rev


def _greedy_monotonic_path(
    per_event_cands: list[list[tuple[int, float, SearchHit]]],
) -> list[tuple[int, float, SearchHit]] | None:
    """Fallback: walk events left-to-right taking the best frame after the previous."""
    if not per_event_cands or any(not cands for cands in per_event_cands):
        return None
    path: list[tuple[int, float, SearchHit]] = []
    prev_frame = -1
    for cands in per_event_cands:
        options = [item for item in cands if item[0] > prev_frame]
        if not options:
            return None
        chosen = max(options, key=lambda item: item[1])
        path.append(chosen)
        prev_frame = chosen[0]
    return path
