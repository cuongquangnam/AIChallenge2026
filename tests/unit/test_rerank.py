"""Unit tests for cross-encoder reranking at chain alignment."""

import pytest

from video_retrieval.config import Settings
from video_retrieval.events.rerank import CrossEncoderReranker, _normalize_scores
from video_retrieval.models import EventSpec, SearchHit


def _hit(video_id: str, frame: int, score: float, *, path: str | None = None) -> SearchHit:
    return SearchHit(
        video_id=video_id,
        score=score,
        source="test",
        frame_index=frame,
        keyframe_path=path,
    )


@pytest.mark.unit
def test_normalize_scores_spreads_range() -> None:
    assert _normalize_scores([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]


@pytest.mark.unit
def test_rerank_mock_changes_candidate_scores() -> None:
    settings = Settings(chain_rerank_enabled=True, chain_rerank_backend="mock", chain_rerank_blend=1.0)
    reranker = CrossEncoderReranker(settings)
    cands = [
        [(10, 0.1, _hit("v1", 10, 0.1)), (20, 0.9, _hit("v1", 20, 0.9))],
    ]
    spec = EventSpec(event_id="e1", visual="person walking")
    out = reranker.rerank_per_event(
        cands,
        event_ids=["e1"],
        events_by_id={"e1": spec},
        video_id="v1",
    )
    scores = [score for _, score, _ in out[0]]
    assert scores != [0.1, 0.9]
    assert [frame for frame, _, _ in out[0]] == [10, 20]


@pytest.mark.unit
def test_rerank_preserves_frame_order() -> None:
    settings = Settings(chain_rerank_enabled=True, chain_rerank_backend="mock", chain_rerank_blend=0.5)
    reranker = CrossEncoderReranker(settings)
    cands = [
        [(5, 1.0, _hit("v1", 5, 1.0)), (15, 0.5, _hit("v1", 15, 0.5))],
        [(25, 0.8, _hit("v1", 25, 0.8))],
    ]
    specs = {
        "e1": EventSpec(event_id="e1", visual="start"),
        "e2": EventSpec(event_id="e2", visual="end"),
    }
    out = reranker.rerank_per_event(
        cands,
        event_ids=["e1", "e2"],
        events_by_id=specs,
        video_id="v1",
        context="demo clip",
    )
    assert [frame for frame, _, _ in out[0]] == [5, 15]
    assert [frame for frame, _, _ in out[1]] == [25]
