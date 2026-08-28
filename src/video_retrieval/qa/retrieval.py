from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QACandidate:
    video_id: str
    score: float
    frame_index: int | None = None
    timestamp_sec: float | None = None
    sources: list[str] = field(default_factory=list)
