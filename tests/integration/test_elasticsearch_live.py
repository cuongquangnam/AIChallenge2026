import pytest

from video_retrieval.models import FrameRole, TextDocument


@pytest.mark.integration
def test_elasticsearch_index_and_search(es_store) -> None:
    docs = [
        TextDocument(
            doc_id="live:ocr:0:middle",
            video_id="live",
            source="ocr",
            text="emergency weather alert scrolling",
            shot_index=0,
            frame_index=12,
            role=FrameRole.MIDDLE,
            start_sec=1.2,
            end_sec=1.2,
            metadata={"keyframe_path": "/tmp/kf.jpg"},
        ),
        TextDocument(
            doc_id="live:asr:0",
            video_id="live",
            source="asr",
            text="the reporter discusses traffic delays",
            start_sec=0.0,
            end_sec=4.0,
        ),
    ]
    assert es_store.index_documents(docs) == 2

    ocr_hits = es_store.search("weather alert", source="ocr")
    assert len(ocr_hits) == 1
    assert ocr_hits[0].video_id == "live"
    assert ocr_hits[0].text is not None
    assert "weather" in ocr_hits[0].text

    asr_hits = es_store.search("traffic delays", source="asr")
    assert len(asr_hits) == 1
    assert asr_hits[0].source == "text:asr"
