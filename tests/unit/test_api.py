from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from video_retrieval.models import IndexResult, SearchHit, SearchResponse


@pytest.fixture
def client() -> TestClient:
    from video_retrieval import api

    return TestClient(api.app)


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
    response = SearchResponse(query="hello", mode="text", hits=[hit])

    with patch("video_retrieval.api.SearchService") as mock_cls:
        mock_cls.return_value.search_text.return_value = response
        resp = client.post("/search", json={"query": "hello", "mode": "text", "limit": 5})
    assert resp.status_code == 200
    assert resp.json()["hits"][0]["video_id"] == "clip"

    with patch("video_retrieval.api.SearchService") as mock_cls:
        mock_cls.return_value.search_visual_text.return_value = SearchResponse(
            query="cat", mode="visual_text:siglip", hits=[]
        )
        resp = client.post("/search", json={"query": "cat", "mode": "visual"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "visual_text:siglip"

    with patch("video_retrieval.api.SearchService") as mock_cls:
        mock_cls.return_value.search_hybrid.return_value = SearchResponse(
            query="mix", mode="hybrid", hits=[]
        )
        resp = client.post("/search", json={"query": "mix", "mode": "hybrid"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "hybrid"
