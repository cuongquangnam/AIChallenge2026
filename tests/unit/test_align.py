"""Unit tests for monotonic chain alignment."""

import pytest

from video_retrieval.events.align import best_monotonic_path, top_monotonic_paths
from video_retrieval.models import SearchHit


def _hit(video_id: str, frame: int, score: float) -> SearchHit:
    return SearchHit(video_id=video_id, score=score, source="test", frame_index=frame)


@pytest.mark.unit
def test_top_monotonic_paths_returns_distinct_paths() -> None:
    per_event = [
        [(10, 1.0, _hit("v1", 10, 1.0)), (20, 0.9, _hit("v1", 20, 0.9))],
        [(30, 1.0, _hit("v1", 30, 1.0)), (40, 0.8, _hit("v1", 40, 0.8))],
        [(50, 1.0, _hit("v1", 50, 1.0)), (60, 0.7, _hit("v1", 60, 0.7))],
    ]
    paths = top_monotonic_paths(per_event, limit=3)
    assert len(paths) >= 2
    frame_sets = {tuple(frame for frame, _, _ in path) for path in paths}
    assert len(frame_sets) == len(paths)


@pytest.mark.unit
def test_best_monotonic_path_matches_top_one() -> None:
    per_event = [
        [(10, 1.0, _hit("v1", 10, 1.0)), (20, 0.9, _hit("v1", 20, 0.9))],
        [(30, 1.0, _hit("v1", 30, 1.0))],
    ]
    best = best_monotonic_path(per_event)
    top = top_monotonic_paths(per_event, limit=1)[0]
    assert best is not None
    assert [frame for frame, _, _ in best] == [frame for frame, _, _ in top]
