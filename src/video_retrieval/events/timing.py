from __future__ import annotations

import math
from dataclasses import dataclass

from video_retrieval.models import EventSpec, SearchHit

DEFAULT_FPS = 25.0


@dataclass(frozen=True)
class EventGap:
    """Expected time from the previous event to this one."""

    expected_sec: float
    min_sec: float
    max_sec: float


def gap_from_spec(spec: EventSpec | None) -> EventGap | None:
    if spec is None or spec.gap_from_prev_sec is None:
        return None
    expected = float(spec.gap_from_prev_sec)
    if expected < 0:
        return None
    lo = spec.gap_min_sec
    hi = spec.gap_max_sec
    if lo is None:
        lo = max(0.0, expected * 0.25)
    if hi is None:
        hi = max(expected * 3.0, expected + 5.0)
    lo = max(0.0, float(lo))
    hi = max(lo, float(hi))
    return EventGap(expected_sec=expected, min_sec=lo, max_sec=hi)


def gaps_from_events(events: list[EventSpec]) -> list[EventGap | None]:
    out: list[EventGap | None] = []
    for index, spec in enumerate(events):
        out.append(None if index == 0 else gap_from_spec(spec))
    return out


def hit_time_sec(hit: SearchHit, frame: int, *, fps: float = DEFAULT_FPS) -> float:
    if hit.timestamp_sec is not None:
        return float(hit.timestamp_sec)
    return frame / max(fps, 1.0)


def gap_sigma(gap: EventGap) -> float:
    span = gap.max_sec - gap.min_sec
    return max(span / 2.5, gap.expected_sec * 0.35, 1.5)


def transition_allowed(dt: float, gap: EventGap | None, *, hard_factor: float) -> bool:
    if dt < 0:
        return False
    if gap is None:
        return True
    limit = gap.max_sec * max(hard_factor, 1.0)
    return dt <= limit


def transition_score(dt: float, gap: EventGap | None, *, weight: float) -> float:
    """Additive bonus/penalty for one event-to-event gap."""
    if gap is None or weight <= 0:
        return 0.0
    sigma = gap_sigma(gap)
    z = (dt - gap.expected_sec) / sigma
    bonus = weight * math.exp(-0.5 * z * z)
    if gap.min_sec <= dt <= gap.max_sec:
        return bonus
    if dt < gap.min_sec:
        overshoot = (gap.min_sec - dt) / sigma
    else:
        overshoot = (dt - gap.max_sec) / sigma
    return bonus - weight * 0.5 * min(overshoot, 2.0)


def chain_gap_score(
    path: list[tuple[int, float, SearchHit]],
    gaps: list[EventGap | None],
    *,
    weight: float,
    fps: float = DEFAULT_FPS,
) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for index in range(1, len(path)):
        gap = gaps[index] if index < len(gaps) else None
        prev_frame, _, prev_hit = path[index - 1]
        frame, _, hit = path[index]
        dt = hit_time_sec(hit, frame, fps=fps) - hit_time_sec(
            prev_hit, prev_frame, fps=fps
        )
        total += transition_score(dt, gap, weight=weight)
    return total


def format_gaps_log(events: list[EventSpec]) -> str:
    parts: list[str] = []
    for spec in events:
        if spec.gap_from_prev_sec is None:
            continue
        lo = spec.gap_min_sec
        hi = spec.gap_max_sec
        if lo is not None and hi is not None:
            parts.append(
                f"{spec.event_id}:{spec.gap_from_prev_sec:g}s[{lo:g}-{hi:g}]"
            )
        else:
            parts.append(f"{spec.event_id}:{spec.gap_from_prev_sec:g}s")
    return " ".join(parts)
