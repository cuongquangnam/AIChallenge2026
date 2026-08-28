from __future__ import annotations

from video_retrieval.models import EventChain, SearchHit
from video_retrieval.search.kis import hits_to_submission_rows


def chains_to_submission_rows(
    chains: list[EventChain],
    *,
    limit: int = 100,
) -> list[tuple[str, int]]:
    """Flatten chain event frames to unique ``(video_id, frame_index)`` rows."""
    rows: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for chain in chains:
        for event in chain.events:
            key = (chain.video_id, int(event.frame_index))
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
            if len(rows) >= limit:
                return rows[:limit]

    if len(rows) < limit and rows:
        seed_hits = [
            SearchHit(
                video_id=video_id,
                score=1.0,
                source="kis_chain",
                frame_index=frame_idx,
            )
            for video_id, frame_idx in rows
        ]
        try:
            return hits_to_submission_rows(seed_hits, limit=limit)
        except ValueError:
            return rows[:limit]

    if len(rows) < limit:
        raise ValueError(f"Need at least one chain hit to build {limit} submission rows")
    return rows[:limit]


def chains_to_search_hits(chains: list[EventChain]) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for chain_index, chain in enumerate(chains):
        for event in chain.events:
            hits.append(
                SearchHit(
                    video_id=chain.video_id,
                    score=chain.score - chain_index * 0.01 + event.score * 0.001,
                    source=f"kis:{event.event_id}",
                    frame_index=event.frame_index,
                    timestamp_sec=event.timestamp_sec,
                    text=event.text,
                    keyframe_path=event.keyframe_path,
                )
            )
    hits.sort(key=lambda item: item.score, reverse=True)
    return hits


def chain_to_csv_line(chain: EventChain) -> str | None:
    frames = [str(event.frame_index) for event in chain.events]
    if not chain.video_id or not frames:
        return None
    return f"{chain.video_id},{','.join(frames)}"


def chains_to_csv_lines(chains: list[EventChain]) -> list[str]:
    lines: list[str] = []
    for chain in chains:
        line = chain_to_csv_line(chain)
        if line:
            lines.append(line)
    return lines
