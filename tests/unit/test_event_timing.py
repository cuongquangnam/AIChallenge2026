"""Unit tests for event-gap inference and timed chain alignment."""

import pytest

from video_retrieval.events.align import top_monotonic_paths
from video_retrieval.events.extractor import heuristic_event_plan, parse_event_plan
from video_retrieval.events.timing import EventGap, chain_gap_score
from video_retrieval.models import SearchHit


def _hit(video_id: str, frame: int, timestamp_sec: float, score: float = 1.0) -> SearchHit:
    return SearchHit(
        video_id=video_id,
        score=score,
        source="test",
        frame_index=frame,
        timestamp_sec=timestamp_sec,
    )


@pytest.mark.unit
def test_parse_event_plan_keeps_model_gaps() -> None:
    raw = """
    {
      "context": "cooking",
      "events": [
        {"event_id": "E1", "description": "carrots in the pot", "visual": "carrots"},
        {
          "event_id": "E2",
          "description": "plated dish",
          "visual": "plate",
          "gap_from_prev_sec": 12,
          "gap_min_sec": 4,
          "gap_max_sec": 25
        }
      ]
    }
    """
    plan = parse_event_plan(raw, fallback_query="carrots then plated dish", task="kis")
    assert plan.events[0].gap_from_prev_sec is None
    assert plan.events[1].gap_from_prev_sec == 12
    assert plan.events[1].gap_min_sec == 4
    assert plan.events[1].gap_max_sec == 25


@pytest.mark.unit
def test_parse_event_plan_keeps_only_valid_coco_object_requirements() -> None:
    raw = """
    {
      "events": [{
        "event_id": "E1",
        "visual": "two people beside a car",
        "required_objects": [
          {"label": "person", "min_count": 2},
          {"label": "car", "min_count": 1},
          {"label": "dragon", "min_count": 1}
        ]
      }]
    }
    """
    plan = parse_event_plan(raw, fallback_query="two people beside a car", task="kis")
    assert [(item.label, item.min_count) for item in plan.events[0].required_objects] == [
        ("person", 2),
        ("car", 1),
    ]


@pytest.mark.unit
def test_heuristic_gap_uses_vietnamese_then_cue() -> None:
    query = "Cảnh A. Sau đó cảnh B."
    plan = heuristic_event_plan(query, task="kis")
    assert len(plan.events) >= 2
    assert plan.events[0].gap_from_prev_sec is None
    assert plan.events[1].gap_from_prev_sec == 8.0
    assert plan.events[1].gap_max_sec == 30.0


@pytest.mark.unit
def test_alignment_prefers_path_matching_expected_gap() -> None:
    per_event = [
        [(10, 1.0, _hit("v1", 10, 1.0))],
        [
            (20, 0.9, _hit("v1", 20, 5.0)),
            (200, 1.0, _hit("v1", 200, 21.0)),
        ],
    ]
    gap = EventGap(expected_sec=4.0, min_sec=1.0, max_sec=12.0)
    paths = top_monotonic_paths(
        per_event,
        limit=1,
        gaps=[None, gap],
        gap_weight=1.0,
        hard_factor=3.0,
        fps=25.0,
    )
    assert paths
    frames = [frame for frame, _, _ in paths[0]]
    assert frames == [10, 20]


@pytest.mark.unit
def test_alignment_rejects_gap_beyond_hard_limit() -> None:
    per_event = [
        [(10, 1.0, _hit("v1", 10, 1.0))],
        [(500, 1.0, _hit("v1", 500, 200.0))],
    ]
    gap = EventGap(expected_sec=5.0, min_sec=1.0, max_sec=15.0)
    paths = top_monotonic_paths(
        per_event,
        limit=1,
        gaps=[None, gap],
        gap_weight=1.0,
        hard_factor=3.0,
        fps=25.0,
    )
    assert paths == []


@pytest.mark.unit
def test_chain_gap_score_is_higher_for_expected_dt() -> None:
    close = [
        (10, 1.0, _hit("v1", 10, 1.0)),
        (20, 1.0, _hit("v1", 20, 6.0)),
    ]
    far = [
        (10, 1.0, _hit("v1", 10, 1.0)),
        (80, 1.0, _hit("v1", 80, 40.0)),
    ]
    gaps = [None, EventGap(expected_sec=5.0, min_sec=1.0, max_sec=12.0)]
    close_score = chain_gap_score(close, gaps, weight=1.0)
    far_score = chain_gap_score(far, gaps, weight=1.0)
    assert close_score > far_score
