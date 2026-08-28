from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from video_retrieval.models import IndexResult, SearchHit, SearchResponse


from video_retrieval.config import Settings
from video_retrieval.runtime import get_runtime, init_runtime, reset_runtime


@pytest.fixture
def client() -> TestClient:
    from video_retrieval import api

    reset_runtime()
    init_runtime(
        Settings(visual_backend="mock", chain_rerank_enabled=False),
        force=True,
    )
    with TestClient(api.app) as test_client:
        yield test_client
    reset_runtime()


@pytest.mark.unit
def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.unit
def test_index_not_found(client: TestClient) -> None:
    with patch("video_retrieval.api.VideoIndexer") as mock_cls:
        mock_cls.return_value.index_video.side_effect = FileNotFoundError("/missing.mp4")
        resp = client.post("/index", json={"path": "/missing.mp4"})
    assert resp.status_code == 404


@pytest.mark.unit
def test_index_success(client: TestClient, tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    result = IndexResult(
        video_id="clip",
        video_path=video,
        num_shots=2,
        num_keyframes=6,
        num_visual_points=6,
        num_text_docs=3,
        audio_path=tmp_path / "clip.wav",
    )
    with patch("video_retrieval.api.VideoIndexer") as mock_cls:
        mock_cls.return_value.index_video.return_value = result
        resp = client.post("/index", json={"path": str(video), "video_id": "clip"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["video_id"] == "clip"
    assert body["num_keyframes"] == 6


@pytest.mark.unit
def test_search_modes(client: TestClient) -> None:
    hit = SearchHit(video_id="clip", score=1.0, source="text:ocr", text="hello")
    response = SearchResponse(query="hello", mode="ocr", hits=[hit])

    with patch("video_retrieval.api.get_runtime") as mock_get:
        mock_get.return_value.search.search_ocr.return_value = response
        resp = client.post("/search", json={"query": "hello", "mode": "ocr", "limit": 5})
    assert resp.status_code == 200
    assert resp.json()["hits"][0]["video_id"] == "clip"

    with patch("video_retrieval.api.get_runtime") as mock_get:
        mock_get.return_value.search.search_asr.return_value = SearchResponse(
            query="hello", mode="asr", hits=[]
        )
        resp = client.post("/search", json={"query": "hello", "mode": "asr"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "asr"

    with patch("video_retrieval.api.get_runtime") as mock_get:
        mock_get.return_value.search.search_visual.return_value = SearchResponse(
            query="cat", mode="visual", hits=[]
        )
        resp = client.post("/search", json={"query": "cat", "mode": "visual"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "visual"

    with patch("video_retrieval.api.get_runtime") as mock_get:
        mock_get.return_value.search.search_mixed.return_value = SearchResponse(
            query="mix", mode="mixed", hits=[]
        )
        resp = client.post("/search", json={"query": "mix", "mode": "mixed"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "mixed"
