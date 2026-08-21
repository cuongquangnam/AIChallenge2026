from __future__ import annotations

import json
import unicodedata

import pytest

from video_retrieval.models import SearchHit, SearchResponse
from video_retrieval.search.task2 import group_hits_by_time, retrieve_task2_candidates


def _hit(video_id: str, timestamp: float | None, *, score: float = 1.0, source: str = "text:ocr"):
    return SearchHit(
        video_id=video_id,
        timestamp_sec=timestamp,
        score=score,
        source=source,
        frame_index=int(timestamp * 10) if timestamp is not None else None,
    )


@pytest.mark.unit
def test_group_hits_by_time_keeps_events_and_videos_separate() -> None:
    hits = [
        _hit("awards", 95.0, source="visual:siglip"),
        _hit("awards", 103.0, source="text:asr"),
        _hit("awards", 125.0, source="text:ocr"),
        _hit("other", 96.0, source="visual:siglip"),
        _hit("awards", None),
    ]

    groups = group_hits_by_time(hits, max_gap_sec=10.0, limit=10)

    assert len(groups) == 3
    award_group = next(group for group in groups if group.video_id == "awards" and len(group.hits) == 2)
    assert award_group.start_sec == 95.0
    assert award_group.end_sec == 103.0
    assert award_group.center_sec == 99.0
    assert award_group.context_frame_indices == list(range(985, 996))


@pytest.mark.unit
def test_group_hits_by_time_rewards_multisource_evidence() -> None:
    hits = [
        _hit("awards", 10.0, score=0.02, source="visual:siglip"),
        _hit("awards", 12.0, score=0.02, source="text:asr"),
        _hit("awards", 100.0, score=0.03, source="text:ocr"),
    ]

    groups = group_hits_by_time(hits, max_gap_sec=10.0, limit=10)

    assert groups[0].start_sec == 10.0
    assert groups[0].sources == ["text:asr", "visual:siglip"]
    assert groups[0].score == pytest.approx(0.05)


@pytest.mark.unit
def test_group_hits_by_time_does_not_overcount_duplicate_channel_hits() -> None:
    hits = [
        _hit("awards", 10.0, score=0.02, source="visual:siglip"),
        _hit("awards", 11.0, score=0.03, source="visual:siglip"),
        _hit("awards", 12.0, score=0.01, source="visual:siglip"),
    ]

    groups = group_hits_by_time(hits, max_gap_sec=10.0, limit=10)

    assert groups[0].score == pytest.approx(0.03)


@pytest.mark.unit
def test_group_hits_by_time_falls_back_to_frame_indices() -> None:
    hits = [
        _hit("awards", 0.0, source="visual:siglip"),
        _hit("awards", 0.0, source="text:asr"),
        _hit("awards", 0.0, source="text:ocr"),
    ]
    hits[0].frame_index = 101
    hits[1].frame_index = 108
    hits[2].frame_index = 130

    groups = group_hits_by_time(hits, max_gap_frames=10, limit=10)

    assert len(groups) == 2
    assert groups[0].start_frame_index == 101
    assert groups[0].end_frame_index == 108
    assert groups[0].context_frame_indices == list(range(99, 110))


@pytest.mark.unit
def test_group_hits_by_time_can_sample_a_wider_temporal_context() -> None:
    hits = [
        _hit("awards", 95.0, source="visual:siglip"),
        _hit("awards", 103.0, source="text:asr"),
    ]

    groups = group_hits_by_time(
        hits,
        max_gap_sec=10.0,
        context_radius_frames=2,
        context_stride_frames=10,
        limit=10,
    )

    assert groups[0].context_frame_indices == [970, 980, 990, 1000, 1010]


class _FakeTask2Service:
    def search_visual_text(self, query: str, *, limit: int, video_id: str | None):
        assert limit == 5
        return SearchResponse(query=query, mode="visual", hits=[_hit(video_id or "awards", 20.0)])

    def search_text_filtered(
        self,
        query: str,
        *,
        limit: int,
        source: str | None,
        video_id: str | None,
        strict: bool,
    ):
        assert source in {"ocr", "asr"}
        assert strict is True
        return SearchResponse(
            query=query,
            mode="text",
            hits=[_hit(video_id or "awards", 22.0, source=f"text:{source}")],
        )


class _FixedTask2Planner:
    def plan(self, query: str):
        from video_retrieval.models import QueryPlan

        assert query
        return QueryPlan(ocr="winner name", asr="top award announced", visual="award winners on stage")


@pytest.mark.unit
def test_retrieve_task2_candidates_runs_all_channels() -> None:
    response = retrieve_task2_candidates(
        _FakeTask2Service(), video_id="awards", candidates_per_query=5
    )

    assert len(response.queries["visual"]) == 2
    assert response.groups[0].video_id == "awards"
    assert response.video_id == "awards"
    assert response.groups[0].sources == ["asr", "ocr", "visual"]


@pytest.mark.unit
def test_retrieve_task2_candidates_skips_rejected_video() -> None:
    class _TwoVideoService(_FakeTask2Service):
        def search_visual_text(self, query: str, *, limit: int, video_id: str | None):
            return SearchResponse(
                query=query,
                mode="visual",
                hits=[_hit("rejected", 20.0), _hit("next", 20.0)],
            )

        def search_text_filtered(self, query: str, *, limit: int, source: str | None, video_id: str | None, strict: bool):
            return SearchResponse(
                query=query,
                mode="text",
                hits=[_hit("rejected", 22.0, source=f"text:{source}"), _hit("next", 22.0, source=f"text:{source}")],
            )

    response = retrieve_task2_candidates(
        _TwoVideoService(), candidates_per_query=5, excluded_video_ids={"rejected"}
    )

    assert response.video_id == "next"
    assert response.groups[0].video_id == "next"


@pytest.mark.unit
def test_retrieve_task2_candidates_can_limit_to_available_videos() -> None:
    class _TwoVideoService(_FakeTask2Service):
        def search_visual_text(self, query: str, *, limit: int, video_id: str | None):
            return SearchResponse(query=query, mode="visual", hits=[_hit("remote", 20.0), _hit("local", 20.0)])

        def search_text_filtered(self, query: str, *, limit: int, source: str | None, video_id: str | None, strict: bool):
            return SearchResponse(
                query=query,
                mode="text",
                hits=[_hit("remote", 22.0, source=f"text:{source}"), _hit("local", 22.0, source=f"text:{source}")],
            )

    response = retrieve_task2_candidates(
        _TwoVideoService(), candidates_per_query=5, allowed_video_ids={"local"}
    )

    assert response.video_id == "local"


@pytest.mark.unit
def test_retrieve_task2_candidates_adds_planned_channel_queries() -> None:
    service = _FakeTask2Service()
    service.planner = _FixedTask2Planner()

    response = retrieve_task2_candidates(service, video_id="awards", candidates_per_query=5)

    assert response.queries["ocr"][-1] == "winner name"
    assert response.queries["asr"][-1] == "top award announced"
    assert response.queries["visual"][-1] == "award winners on stage"


@pytest.mark.unit
def test_retrieve_task2_candidates_ignores_generic_planner_fallback() -> None:
    class _FallbackPlanner:
        def plan(self, query: str):
            from video_retrieval.models import QueryPlan

            # Gemini may echo the query using decomposed Unicode characters.
            echoed = unicodedata.normalize("NFD", query)
            return QueryPlan(ocr=echoed, asr=echoed, visual=echoed)

    service = _FakeTask2Service()
    service.planner = _FallbackPlanner()

    response = retrieve_task2_candidates(service, video_id="awards", candidates_per_query=5)

    assert response.queries == {
        "visual": [
            "music awards ceremony winners on stage",
            "award winners walking on stage to receive trophy",
        ],
        "ocr": ["music awards ceremony", "award of the year", "daesang"],
        "asr": ["the winner is", "the grand award goes to", "award of the year"],
    }


@pytest.mark.unit
def test_retrieve_task2_candidates_resolves_context_paths(tmp_path) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "awards.json").write_text(
        json.dumps(
            {
                "keyframes": [
                    {"frame_index": frame_index, "path": f"/frames/{frame_index:03d}.jpg"}
                    for frame_index in range(195, 226)
                ]
            }
        ),
        encoding="utf-8",
    )

    response = retrieve_task2_candidates(
        _FakeTask2Service(),
        video_id="awards",
        candidates_per_query=5,
        manifests_dir=manifest_dir,
    )

    assert response.groups[0].context_keyframe_paths[0] == "/frames/210.jpg"
    assert response.groups[0].context_keyframe_paths[-1] == "/frames/220.jpg"
