from __future__ import annotations

from pathlib import Path
from typing import Any

from video_retrieval.config import Settings, get_settings
from video_retrieval.encoders.visual import VisualEncoder
from video_retrieval.models import QueryPlan, SearchHit, SearchResponse
from video_retrieval.search.planner import QueryPlanner
from video_retrieval.storage.elasticsearch_store import ElasticsearchStore
from video_retrieval.storage.qdrant_store import QdrantStore

_CHANNEL_LIMIT_FACTOR = 5
_ASR_TIME_PAD_SEC = 1.5


class SearchService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        visual: VisualEncoder | None = None,
        qdrant: QdrantStore | None = None,
        es: ElasticsearchStore | None = None,
        planner: QueryPlanner | None = None,
    ):
        self.settings = settings or get_settings()
        self.visual = visual or VisualEncoder(self.settings)
        self.qdrant = qdrant or QdrantStore(self.settings)
        self.es = es or ElasticsearchStore(self.settings)
        self.planner = planner or QueryPlanner(self.settings)

    def search_text(self, query: str, *, limit: int = 10) -> SearchResponse:
        return self.search_mixed(query, limit=limit)

    def search_ocr(self, query: str, *, limit: int = 10) -> SearchResponse:
        plan = self.planner.plan(query)
        text = plan.ocr or query
        hits = self.es.search(text, limit=limit, source="ocr") if text else []
        return SearchResponse(query=query, mode="ocr", hits=hits, plan=plan)

    def search_asr(self, query: str, *, limit: int = 10) -> SearchResponse:
        plan = self.planner.plan(query)
        text = plan.asr or query
        hits = self.es.search(text, limit=limit, source="asr") if text else []
        return SearchResponse(query=query, mode="asr", hits=hits, plan=plan)

    def search_keyword(self, query: str, *, limit: int = 10) -> SearchResponse:
        hits = self.es.search(query, limit=limit)
        return SearchResponse(query=query, mode="keyword", hits=hits)

    def search_visual_text(
        self,
        query: str,
        *,
        limit: int = 10,
        vector_name: str = "siglip",
    ) -> SearchResponse:
        return self.search_visual(query, limit=limit, vector_name=vector_name)

    def search_visual(
        self,
        query: str,
        *,
        limit: int = 10,
        vector_name: str = "siglip",
    ) -> SearchResponse:
        plan = self.planner.plan(query)
        text = plan.visual or query
        hits = self.qdrant.search(
            self.visual.encode_text(text),
            vector_name=vector_name,
            limit=limit,
        )
        return SearchResponse(query=query, mode="visual", hits=hits, plan=plan)

    def search_image(
        self,
        image_path: Path,
        *,
        limit: int = 10,
        vector_name: str = "siglip",
    ) -> SearchResponse:
        siglip, beit3 = self.visual.encode_image(Path(image_path))
        vector = siglip if vector_name == "siglip" else beit3
        hits = self.qdrant.search(vector, vector_name=vector_name, limit=limit)
        return SearchResponse(query=str(image_path), mode=f"visual_image:{vector_name}", hits=hits)

    def search_hybrid(self, query: str, *, limit: int = 10) -> SearchResponse:
        return self.search_mixed(query, limit=limit)

    def search_planned(
        self,
        query: str,
        *,
        limit: int = 10,
        vector_name: str = "siglip",
    ) -> SearchResponse:
        return self.search_mixed(query, limit=limit, vector_name=vector_name)

    def search_mixed(
        self,
        query: str,
        *,
        limit: int = 10,
        vector_name: str = "siglip",
    ) -> SearchResponse:
        plan = self.planner.plan(query)
        channel_limit = max(limit * _CHANNEL_LIMIT_FACTOR, limit)

        ocr_hits: list[SearchHit] = []
        asr_hits: list[SearchHit] = []
        visual_hits: list[SearchHit] = []
        if plan.ocr:
            ocr_hits = self.es.search(plan.ocr, limit=channel_limit, source="ocr")
        if plan.asr:
            asr_hits = self.es.search(plan.asr, limit=channel_limit, source="asr")
        if plan.visual:
            visual_hits = self.qdrant.search(
                self.visual.encode_text(plan.visual),
                vector_name=vector_name,
                limit=channel_limit,
            )

        fused = fuse_frame_scores(
            ocr_hits=ocr_hits,
            asr_hits=asr_hits,
            visual_hits=visual_hits,
            weights=plan.weights,
            limit=limit,
        )
        return SearchResponse(query=query, mode="mixed", hits=fused, plan=plan)

    def search_event_spec(
        self,
        event,
        *,
        limit: int = 50,
        vector_name: str = "siglip",
    ) -> list[SearchHit]:
        """Mixed OCR/ASR/visual retrieval for one event spec (no LLM planner)."""
        from video_retrieval.models import EventSpec

        if not isinstance(event, EventSpec):
            raise TypeError("event must be EventSpec")

        channel_limit = max(limit * _CHANNEL_LIMIT_FACTOR, limit)
        ocr_hits: list[SearchHit] = []
        asr_hits: list[SearchHit] = []
        visual_hits: list[SearchHit] = []
        if event.ocr:
            ocr_hits = self.es.search(event.ocr, limit=channel_limit, source="ocr")
        if event.asr:
            asr_hits = self.es.search(event.asr, limit=channel_limit, source="asr")
        visual_text = (event.visual or event.description or "").strip()
        if visual_text:
            visual_hits = self.qdrant.search(
                self.visual.encode_text(visual_text),
                vector_name=vector_name,
                limit=channel_limit,
            )
        weights = _event_weights(event)
        return fuse_frame_scores(
            ocr_hits=ocr_hits,
            asr_hits=asr_hits,
            visual_hits=visual_hits,
            weights=weights,
            limit=limit,
        )


def fuse_frame_scores(
    *,
    ocr_hits: list[SearchHit],
    asr_hits: list[SearchHit],
    visual_hits: list[SearchHit],
    weights: dict[str, float],
    limit: int,
) -> list[SearchHit]:
    """Combine OCR, ASR, and visual channel scores onto keyframes."""
    weights = _normalized_weights(weights)
    ocr_norm = _minmax_scores(ocr_hits)
    asr_norm = _minmax_scores(asr_hits)
    visual_norm = _minmax_scores(visual_hits)

    buckets: dict[tuple, dict[str, Any]] = {}
    for hit, score in zip(visual_hits, visual_norm, strict=True):
        _merge_bucket(buckets, hit, visual=score)
    for hit, score in zip(ocr_hits, ocr_norm, strict=True):
        _merge_bucket(buckets, hit, ocr=score)
    for hit, score in zip(asr_hits, asr_norm, strict=True):
        matched = _apply_asr_to_frames(buckets, hit, score)
        if not matched:
            _merge_bucket(buckets, hit, asr=score)

    fused: list[SearchHit] = []
    for bucket in buckets.values():
        channels = {
            "ocr": float(bucket["ocr"]),
            "asr": float(bucket["asr"]),
            "visual": float(bucket["visual"]),
        }
        combined = (
            weights["ocr"] * channels["ocr"]
            + weights["asr"] * channels["asr"]
            + weights["visual"] * channels["visual"]
        )
        hit: SearchHit = bucket["hit"]
        fused.append(
            hit.model_copy(
                update={
                    "score": combined,
                    "source": "mixed",
                    "channel_scores": channels,
                    "payload": {
                        **hit.payload,
                        "channel_scores": channels,
                        "weights": weights,
                    },
                }
            )
        )

    fused.sort(key=lambda item: item.score, reverse=True)
    return fused[:limit]


def _event_weights(event) -> dict[str, float]:
    channels = {
        "ocr": bool(event.ocr),
        "asr": bool(event.asr),
        "visual": bool((event.visual or event.description or "").strip()),
    }
    active = sum(1 for value in channels.values() if value)
    if active == 0:
        return {"ocr": 1 / 3, "asr": 1 / 3, "visual": 1 / 3}
    weight = 1.0 / active
    return {key: weight if channels[key] else 0.0 for key in ("ocr", "asr", "visual")}


def _rrf_fuse(rankings: list[list[SearchHit]], *, limit: int, k: int = 60) -> list[SearchHit]:
    scores: dict[str, float] = {}
    best: dict[str, SearchHit] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            key = (
                f"{hit.video_id}|{hit.shot_index}|{hit.frame_index}|{hit.source}|{hit.text or ''}"
            )
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in best or hit.score > best[key].score:
                best[key] = hit

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    fused: list[SearchHit] = []
    for key, score in ordered:
        hit = best[key].model_copy()
        hit.score = score
        fused.append(hit)
    return fused


def _frame_key(hit: SearchHit) -> tuple:
    return (hit.video_id, hit.shot_index, hit.frame_index)


def _merge_bucket(
    buckets: dict[tuple, dict[str, Any]],
    hit: SearchHit,
    *,
    ocr: float = 0.0,
    asr: float = 0.0,
    visual: float = 0.0,
) -> None:
    key = _frame_key(hit)
    bucket = buckets.get(key)
    if bucket is None:
        buckets[key] = {
            "hit": hit,
            "ocr": ocr,
            "asr": asr,
            "visual": visual,
        }
        return
    bucket["ocr"] = max(bucket["ocr"], ocr)
    bucket["asr"] = max(bucket["asr"], asr)
    bucket["visual"] = max(bucket["visual"], visual)
    current: SearchHit = bucket["hit"]
    if not current.keyframe_path and hit.keyframe_path:
        bucket["hit"] = hit
    if not current.text and hit.text:
        bucket["hit"] = current.model_copy(update={"text": hit.text})


def _apply_asr_to_frames(
    buckets: dict[tuple, dict[str, Any]],
    asr_hit: SearchHit,
    score: float,
) -> bool:
    start = asr_hit.timestamp_sec
    end = _hit_end_sec(asr_hit)
    matched = False
    for bucket in buckets.values():
        frame: SearchHit = bucket["hit"]
        if frame.video_id != asr_hit.video_id:
            continue
        if asr_hit.shot_index is not None and frame.shot_index == asr_hit.shot_index:
            bucket["asr"] = max(bucket["asr"], score)
            if not frame.text and asr_hit.text:
                bucket["hit"] = frame.model_copy(update={"text": asr_hit.text})
            matched = True
            continue
        ts = frame.timestamp_sec
        if ts is None or start is None:
            continue
        lo = start - _ASR_TIME_PAD_SEC
        hi = (end if end is not None else start) + _ASR_TIME_PAD_SEC
        if lo <= ts <= hi:
            bucket["asr"] = max(bucket["asr"], score)
            if not frame.text and asr_hit.text:
                bucket["hit"] = frame.model_copy(update={"text": asr_hit.text})
            matched = True
    return matched


def _hit_end_sec(hit: SearchHit) -> float | None:
    value = hit.payload.get("end_sec") if hit.payload else None
    if value is None:
        return hit.timestamp_sec
    try:
        return float(value)
    except (TypeError, ValueError):
        return hit.timestamp_sec


def _minmax_scores(hits: list[SearchHit]) -> list[float]:
    if not hits:
        return []
    values = [hit.score for hit in hits]
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [1.0] * len(hits)
    return [(score - lo) / (hi - lo) for score in values]


def _normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    out = {
        "ocr": max(float(weights.get("ocr", 1.0)), 0.0),
        "asr": max(float(weights.get("asr", 1.0)), 0.0),
        "visual": max(float(weights.get("visual", 1.0)), 0.0),
    }
    total = sum(out.values())
    if total <= 0:
        return {"ocr": 1 / 3, "asr": 1 / 3, "visual": 1 / 3}
    return {key: value / total for key, value in out.items()}
