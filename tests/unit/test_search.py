import pytest

from video_retrieval.models import SearchHit
from video_retrieval.search.planner import QueryPlan, heuristic_plan, parse_plan
from video_retrieval.search.service import SearchService, fuse_frame_scores
from video_retrieval.storage.qdrant_store import QdrantStore
from tests.fakes import FakeElasticsearchStore


@pytest.mark.unit
def test_fuse_frame_scores_combines_channels_on_the_same_frame() -> None:
    ocr = SearchHit(
        video_id="news",
        score=10.0,
        source="text:ocr",
        shot_index=0,
        frame_index=12,
        timestamp_sec=4.0,
        text="VTV",
    )
    visual = SearchHit(
        video_id="news",
        score=0.9,
        source="visual:siglip",
        shot_index=0,
        frame_index=12,
        timestamp_sec=4.0,
        keyframe_path="/tmp/mid.jpg",
    )
    asr = SearchHit(
        video_id="news",
        score=5.0,
        source="text:asr",
        timestamp_sec=3.5,
        payload={"end_sec": 5.0},
        text="good evening",
    )
    fused = fuse_frame_scores(
        ocr_hits=[ocr],
        asr_hits=[asr],
        visual_hits=[visual],
        weights={"ocr": 1.0, "asr": 1.0, "visual": 1.0},
        limit=5,
    )
    assert len(fused) == 1
    hit = fused[0]
    assert hit.video_id == "news"
    assert hit.frame_index == 12
    assert hit.channel_scores["ocr"] == pytest.approx(1.0)
    assert hit.channel_scores["asr"] == pytest.approx(1.0)
    assert hit.channel_scores["visual"] == pytest.approx(1.0)
    assert hit.score == pytest.approx(1.0)


@pytest.mark.unit
def test_parse_plan_reads_channel_queries() -> None:
    raw = """
    {"ocr": "VTV24", "asr": "xin chao", "visual": "news anchor on television",
     "weights": {"ocr": 0.4, "asr": 0.2, "visual": 0.4}}
    """
    plan = parse_plan(raw, fallback_query="hello")
    assert plan.ocr == "VTV24"
    assert plan.asr == "xin chao"
    assert plan.visual == "news anchor on television"
    assert plan.weights["ocr"] == 0.4


@pytest.mark.unit
def test_search_service_modes(settings, qdrant_store: QdrantStore) -> None:
    settings.query_planner = "heuristic"
    fake_es = FakeElasticsearchStore()
    service = SearchService(settings, qdrant=qdrant_store, es=fake_es)  # type: ignore[arg-type]

    text = service.search_text("hello world")
    assert text.mode == "mixed"
    assert text.plan is not None
    assert text.plan.ocr == "hello world"
    assert text.hits == []

    visual = service.search_visual("a cat", limit=5)
    assert visual.mode == "visual"
    assert visual.hits == []

    ocr = service.search_ocr("hello", limit=5)
    assert ocr.mode == "ocr"

    asr = service.search_asr("hello", limit=5)
    assert asr.mode == "asr"

    mixed = service.search_mixed("hello", limit=5)
    assert mixed.mode == "mixed"


@pytest.mark.unit
def test_search_planned_uses_channel_queries(settings, qdrant_store: QdrantStore) -> None:
    settings.query_planner = "heuristic"

    class _FixedPlanner:
        def plan(self, query: str) -> QueryPlan:
            return QueryPlan(ocr="VTV", asr="hello", visual="news studio")

    fake_es = FakeElasticsearchStore()
    from video_retrieval.models import FrameRole, TextDocument

    fake_es.index_documents(
        [
            TextDocument(
                doc_id="v1:ocr:0:middle",
                video_id="v1",
                source="ocr",
                text="VTV tonight",
                shot_index=0,
                frame_index=8,
                role=FrameRole.MIDDLE,
                start_sec=2.0,
            ),
            TextDocument(
                doc_id="v1:asr:0",
                video_id="v1",
                source="asr",
                text="hello from the studio",
                start_sec=1.5,
                end_sec=3.0,
            ),
        ]
    )
    service = SearchService(
        settings,
        qdrant=qdrant_store,
        es=fake_es,  # type: ignore[arg-type]
        planner=_FixedPlanner(),  # type: ignore[arg-type]
    )
    response = service.search_mixed("find the VTV greeting", limit=5)
    assert response.mode == "mixed"
    assert response.plan is not None
    assert response.plan.ocr == "VTV"
    assert any(hit.video_id == "v1" for hit in response.hits)


@pytest.mark.unit
def test_heuristic_plan_copies_query() -> None:
    plan = heuristic_plan("nhạc rock")
    assert plan.ocr == plan.asr == plan.visual == "nhạc rock"
