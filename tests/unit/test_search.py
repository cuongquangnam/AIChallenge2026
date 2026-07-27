import pytest

from video_retrieval.models import SearchHit
from video_retrieval.search.service import SearchService, _rrf_fuse
from video_retrieval.storage.qdrant_store import QdrantStore
from tests.fakes import FakeElasticsearchStore


@pytest.mark.unit
def test_rrf_fuse_prefers_items_present_in_both_lists() -> None:
    shared = SearchHit(video_id="v1", score=0.5, source="text:ocr", shot_index=0, frame_index=1)
    only_text = SearchHit(video_id="v2", score=0.9, source="text:asr", shot_index=1, frame_index=2)
    only_visual = SearchHit(
        video_id="v3", score=0.8, source="visual:siglip", shot_index=2, frame_index=3
    )

    fused = _rrf_fuse([[shared, only_text], [shared, only_visual]], limit=3)
    assert fused[0].video_id == "v1"
    assert len(fused) == 3


@pytest.mark.unit
def test_rrf_fuse_merges_evidence_by_shot() -> None:
    text = SearchHit(
        video_id="v1",
        score=2.0,
        source="text:ocr",
        shot_index=0,
        frame_index=10,
        text="sign text",
    )
    visual = SearchHit(
        video_id="v1",
        score=0.8,
        source="visual:siglip",
        shot_index=0,
        frame_index=11,
        keyframe_path="/tmp/shot.jpg",
    )

    fused = _rrf_fuse([[text], [visual]], limit=5)

    assert len(fused) == 1
    assert fused[0].source == "hybrid"
    assert fused[0].text == "sign text"
    assert fused[0].keyframe_path == "/tmp/shot.jpg"
    assert len(fused[0].payload["evidence"]) == 2


@pytest.mark.unit
def test_search_service_modes(settings, qdrant_store: QdrantStore) -> None:
    fake_es = FakeElasticsearchStore()
    service = SearchService(settings, qdrant=qdrant_store, es=fake_es)  # type: ignore[arg-type]

    text = service.search_text("hello world")
    assert text.mode == "text"
    assert text.hits == []

    visual = service.search_visual_text("a cat", limit=5)
    assert visual.mode == "visual_text:siglip"
    assert visual.hits == []

    with pytest.raises(ValueError, match="SigLIP"):
        service.search_visual_text("a cat", limit=5, vector_name="beit3")

    hybrid = service.search_hybrid("hello", limit=5)
    assert hybrid.mode == "hybrid"
