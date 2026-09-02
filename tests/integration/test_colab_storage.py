import pytest

from video_retrieval.config import get_settings
from video_retrieval.pipeline.indexer import VideoIndexer
from video_retrieval.search.service import SearchService
from video_retrieval.storage.factory import create_qdrant_store, create_text_store
from tests.helpers import write_dummy_video


@pytest.mark.integration
def test_colab_storage_index_and_search_roundtrip(settings, tmp_path) -> None:
    settings = get_settings(data_dir=tmp_path / "data", colab=True).model_copy(
        update={
            "qdrant_collection": "colab_test_keyframes",
            "es_index": "colab_test_text",
            "query_planner": "heuristic",
        }
    )
    video = write_dummy_video(tmp_path / "clip.mp4", frames=40, fps=10)
    qdrant = create_qdrant_store(settings)
    text_store = create_text_store(settings)

    indexer = VideoIndexer(settings, qdrant=qdrant, es=text_store)
    result = indexer.index_video(video, video_id="clip")
    assert result.num_visual_points > 0
    assert result.num_text_docs > 0

    # Simulate a fresh Colab session: close the first client, then reopen.
    qdrant.client.close()
    qdrant_again = create_qdrant_store(settings)
    text_store_again = create_text_store(settings)
    service = SearchService(settings, qdrant=qdrant_again, es=text_store_again)

    text_hits = service.search_text("mock transcription for clip")
    assert any(hit.video_id == "clip" for hit in text_hits.hits)

    visual_hits = service.search_visual_text("stage ceremony", limit=5)
    assert visual_hits.hits

    mixed = service.search_mixed("mock", limit=5)
    assert mixed.hits
