from __future__ import annotations

from video_retrieval.models import EventChain, EventHit, SearchHit


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
                return rows

    if not rows:
        raise ValueError("Need at least one chain hit to build submission rows")
    return rows


def chains_to_search_hits(
    chains: list[EventChain],
    *,
    limit: int = 100,
) -> list[SearchHit]:
    """Flatten chains to submission rows in chain / event order (E1, E2, …)."""
    rows = chains_to_submission_rows(chains, limit=limit)
    lookup: dict[tuple[str, int], tuple[EventChain, EventHit]] = {}
    for chain in chains:
        for event in chain.events:
            lookup[(chain.video_id, int(event.frame_index))] = (chain, event)

    hits: list[SearchHit] = []
    for chain_index, (video_id, frame_idx) in enumerate(rows):
        pair = lookup.get((video_id, frame_idx))
        if pair is None:
            continue
        chain, event = pair
        hits.append(
            SearchHit(
                video_id=video_id,
                score=chain.score - chain_index * 0.01 + event.score * 0.001,
                source=f"kis:{event.event_id}",
                frame_index=event.frame_index,
                timestamp_sec=event.timestamp_sec,
                text=event.text,
                keyframe_path=event.keyframe_path,
            )
        )
    return hits


def chains_to_flat_event_hits(
    chains: list[EventChain],
    *,
    source_prefix: str = "trake",
) -> list[SearchHit]:
    """One hit per chain event in chain rank order (no padding)."""
    hits: list[SearchHit] = []
    for chain_index, chain in enumerate(chains):
        for event in chain.events:
            hits.append(
                SearchHit(
                    video_id=chain.video_id,
                    score=chain.score - chain_index * 0.01 + event.score * 0.001,
                    source=f"{source_prefix}:{event.event_id}",
                    frame_index=event.frame_index,
                    timestamp_sec=event.timestamp_sec,
                    text=event.text,
                    keyframe_path=event.keyframe_path,
                )
            )
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
