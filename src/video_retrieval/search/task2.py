"""Retrieval and time grouping for the music-award acceptance question.

This module deliberately stops before visual reasoning.  Its output is a small
set of evidence groups (keyframes plus time ranges) that can be passed to a
VLM to verify the major award and count people walking on stage.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Protocol

from video_retrieval.models import SearchHit, Task2RetrievalResponse, TemporalGroup

TASK2_QUESTION = (
    "Trong video về lễ trao giải thưởng âm nhạc, có bao nhiêu người lên sân khấu "
    "để nhận giải thưởng lớn nhất?"
)

# Keep these separate because the useful words differ by retrieval modality.
TASK2_QUERIES: dict[str, list[str]] = {
    "visual": [
        "music award ceremony stage",
        "winners walking on stage to receive an award",
    ],
    "ocr": ["grand prize", "award of the year", "daesang"],
    "asr": ["the winner is", "grand prize", "award of the year"],
}


class _Task2SearchService(Protocol):
    def search_visual_text(
        self, query: str, *, limit: int = 10, video_id: str | None = None
    ): ...

    def search_text_filtered(
        self,
        query: str,
        *,
        limit: int = 10,
        source: str | None = None,
        video_id: str | None = None,
    ): ...


def retrieve_task2_candidates(
    service: _Task2SearchService,
    *,
    video_id: str | None = None,
    candidates_per_query: int = 20,
    group_limit: int = 10,
    max_gap_sec: float = 10.0,
    max_gap_frames: int = 10,
    context_radius_frames: int = 5,
    manifests_dir: Path | None = None,
) -> Task2RetrievalResponse:
    """Retrieve Task 2 evidence from visual, OCR, and ASR channels."""
    if candidates_per_query < 1:
        raise ValueError("candidates_per_query must be at least 1")
    if group_limit < 1:
        raise ValueError("group_limit must be at least 1")
    if max_gap_sec < 0:
        raise ValueError("max_gap_sec cannot be negative")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames cannot be negative")
    if context_radius_frames < 0:
        raise ValueError("context_radius_frames cannot be negative")

    ranked_hits: list[SearchHit] = []
    for channel, queries in TASK2_QUERIES.items():
        for query in queries:
            if channel == "visual":
                response = service.search_visual_text(
                    query, limit=candidates_per_query, video_id=video_id
                )
            else:
                response = service.search_text_filtered(
                    query,
                    limit=candidates_per_query,
                    source=channel,
                    video_id=video_id,
                )
            ranked_hits.extend(_annotate_hits(response.hits, channel=channel, query=query))

    groups = group_hits_by_time(
        ranked_hits,
        max_gap_sec=max_gap_sec,
        max_gap_frames=max_gap_frames,
        context_radius_frames=context_radius_frames,
        limit=group_limit,
    )
    if manifests_dir is not None:
        _attach_context_keyframe_paths(groups, manifests_dir=manifests_dir)
    return Task2RetrievalResponse(
        question=TASK2_QUESTION,
        queries={channel: list(queries) for channel, queries in TASK2_QUERIES.items()},
        groups=groups,
    )


def group_hits_by_time(
    hits: list[SearchHit],
    *,
    max_gap_sec: float = 10.0,
    max_gap_frames: int = 10,
    context_radius_frames: int = 5,
    limit: int = 10,
) -> list[TemporalGroup]:
    """Merge hits into event windows, falling back to frame IDs when needed.

    Pre-extracted AIC keyframes have frame numbers but no trustworthy timestamps,
    so each video uses frame distance when all of its timestamps are identical.
    """
    if max_gap_sec < 0:
        raise ValueError("max_gap_sec cannot be negative")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames cannot be negative")
    if context_radius_frames < 0:
        raise ValueError("context_radius_frames cannot be negative")

    by_video: dict[str, list[SearchHit]] = defaultdict(list)
    for hit in hits:
        if hit.timestamp_sec is not None or hit.frame_index is not None:
            by_video[hit.video_id].append(hit)

    groups: list[TemporalGroup] = []
    for video_id, video_hits in by_video.items():
        use_frames = _should_group_by_frame(video_hits)
        ordered = sorted(video_hits, key=lambda hit: _temporal_position(hit, use_frames=use_frames))
        current: list[SearchHit] = []
        last_position: float | None = None
        max_gap = float(max_gap_frames if use_frames else max_gap_sec)
        for hit in ordered:
            position = _temporal_position(hit, use_frames=use_frames)
            if current and last_position is not None and position - last_position > max_gap:
                groups.append(_make_group(video_id, current, context_radius_frames))
                current = []
            current.append(hit)
            last_position = position
        if current:
            groups.append(_make_group(video_id, current, context_radius_frames))

    return sorted(groups, key=lambda group: group.score, reverse=True)[:limit]


def _annotate_hits(hits: list[SearchHit], *, channel: str, query: str) -> list[SearchHit]:
    """Use reciprocal rank so scores from Qdrant and Elasticsearch are comparable."""
    annotated: list[SearchHit] = []
    for rank, hit in enumerate(hits):
        copy = hit.model_copy(deep=True)
        copy.score = 1.0 / (60 + rank + 1)
        copy.payload.update(
            {
                "task2_channel": channel,
                "task2_query": query,
                "task2_rank": rank + 1,
            }
        )
        annotated.append(copy)
    return annotated


def _should_group_by_frame(hits: list[SearchHit]) -> bool:
    timestamps = {hit.timestamp_sec for hit in hits if hit.timestamp_sec is not None}
    return len(timestamps) <= 1 and all(hit.frame_index is not None for hit in hits)


def _temporal_position(hit: SearchHit, *, use_frames: bool) -> float:
    if use_frames:
        return float(hit.frame_index or 0)
    return float(hit.timestamp_sec or 0.0)


def _make_group(
    video_id: str, hits: list[SearchHit], context_radius_frames: int
) -> TemporalGroup:
    timestamps = [float(hit.timestamp_sec) for hit in hits if hit.timestamp_sec is not None] or [0.0]
    frame_indices = [hit.frame_index for hit in hits if hit.frame_index is not None]
    sources = sorted({str(hit.payload.get("task2_channel", hit.source)) for hit in hits})
    # Independent modalities agreeing on an event are stronger evidence than
    # repeated results from one modality.
    score = sum(hit.score for hit in hits) + 0.01 * max(0, len(sources) - 1)
    center_frame_index = round(sum(frame_indices) / len(frame_indices)) if frame_indices else None
    context_frame_indices = (
        list(
            range(
                max(0, center_frame_index - context_radius_frames),
                center_frame_index + context_radius_frames + 1,
            )
        )
        if center_frame_index is not None
        else []
    )
    return TemporalGroup(
        video_id=video_id,
        start_sec=min(timestamps),
        end_sec=max(timestamps),
        center_sec=sum(timestamps) / len(timestamps),
        start_frame_index=min(frame_indices) if frame_indices else None,
        end_frame_index=max(frame_indices) if frame_indices else None,
        center_frame_index=center_frame_index,
        context_frame_indices=context_frame_indices,
        score=score,
        sources=sources,
        hits=hits,
    )


def _attach_context_keyframe_paths(groups: list[TemporalGroup], *, manifests_dir: Path) -> None:
    """Resolve a group's frame IDs to nearby images recorded by the indexer."""
    for group in groups:
        keyframes = _read_manifest_keyframes(manifests_dir / f"{group.video_id}.json")
        if not keyframes:
            group.context_keyframe_paths = _paths_from_evidence(group.hits)
            continue

        center = group.center_frame_index
        if center is None:
            group.context_keyframe_paths = _paths_from_evidence(group.hits)
            continue

        nearby = sorted(
            keyframes,
            key=lambda item: abs(int(item.get("frame_index", 0)) - center),
        )[: len(group.context_frame_indices)]
        group.context_keyframe_paths = [
            str(item["path"]) for item in sorted(nearby, key=lambda item: int(item["frame_index"]))
        ]


def _read_manifest_keyframes(manifest_path: Path) -> list[dict[str, object]]:
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    keyframes = manifest.get("keyframes")
    if isinstance(keyframes, list):
        return [item for item in keyframes if isinstance(item, dict) and "path" in item]

    shots = manifest.get("shots")
    if not isinstance(shots, list):
        return []
    return [
        keyframe
        for shot in shots
        if isinstance(shot, dict)
        for keyframe in shot.get("keyframes", [])
        if isinstance(keyframe, dict) and "path" in keyframe
    ]


def _paths_from_evidence(hits: list[SearchHit]) -> list[str]:
    return list(dict.fromkeys(hit.keyframe_path for hit in hits if hit.keyframe_path))
