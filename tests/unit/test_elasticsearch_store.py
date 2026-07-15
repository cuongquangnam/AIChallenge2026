from unittest.mock import MagicMock

import pytest

from video_retrieval.models import FrameRole, TextDocument
from video_retrieval.storage.elasticsearch_store import ElasticsearchStore


@pytest.mark.unit
def test_index_documents_builds_bulk_ops(settings) -> None:
    client = MagicMock()
    client.indices.exists.return_value = False
    store = ElasticsearchStore(settings, client=client)

    docs = [
        TextDocument(
            doc_id="clip:ocr:0:middle",
            video_id="clip",
            source="ocr",
            text="hello board",
            shot_index=0,
            frame_index=5,
            role=FrameRole.MIDDLE,
            start_sec=0.5,
            end_sec=0.5,
            metadata={"keyframe_path": "/tmp/f.jpg"},
        )
    ]
    count = store.index_documents(docs)
    assert count == 1
    client.indices.create.assert_called_once()
    client.bulk.assert_called_once()
    ops = client.bulk.call_args.kwargs["operations"]
    assert ops[0] == {"index": {"_index": settings.es_index, "_id": "clip:ocr:0:middle"}}
    assert ops[1]["text"] == "hello board"


@pytest.mark.unit
def test_search_maps_hits(settings) -> None:
    client = MagicMock()
    client.indices.exists.return_value = True
    client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_score": 2.5,
                    "_source": {
                        "video_id": "clip",
                        "source": "asr",
                        "text": "spoken words",
                        "shot_index": None,
                        "frame_index": None,
                        "role": None,
                        "start_sec": 1.2,
                        "keyframe_path": None,
                    },
                }
            ]
        }
    }
    store = ElasticsearchStore(settings, client=client)
    hits = store.search("spoken words", limit=5, source="asr")
    assert len(hits) == 1
    assert hits[0].video_id == "clip"
    assert hits[0].score == 2.5
    assert hits[0].source == "text:asr"
    assert hits[0].text == "spoken words"


@pytest.mark.unit
def test_index_documents_empty_short_circuits(settings) -> None:
    client = MagicMock()
    store = ElasticsearchStore(settings, client=client)
    assert store.index_documents([]) == 0
    client.bulk.assert_not_called()
