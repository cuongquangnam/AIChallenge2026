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
    hit = SearchHit(
        video_id="clip",
        score=1.0,
        source="text:ocr",
        text="hello",
        keyframe_path="/tmp/data/keyframes/clip/shot_0001_middle.jpg",
    )
    response = SearchResponse(query="hello", mode="ocr", hits=[hit])

    with patch("video_retrieval.api.SearchService") as mock_cls:
        mock_cls.return_value.search_ocr.return_value = response
        resp = client.post("/search", json={"query": "hello", "mode": "ocr", "limit": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["hits"][0]["video_id"] == "clip"
    assert body["hits"][0]["image_url"] == "/media/keyframes/clip/shot_0001_middle.jpg"

    with patch("video_retrieval.api.SearchService") as mock_cls:
        mock_cls.return_value.search_asr.return_value = SearchResponse(
            query="hello", mode="asr", hits=[]
        )
        resp = client.post("/search", json={"query": "hello", "mode": "asr"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "asr"

    with patch("video_retrieval.api.SearchService") as mock_cls:
        mock_cls.return_value.search_visual.return_value = SearchResponse(
            query="cat", mode="visual", hits=[]
        )
        resp = client.post("/search", json={"query": "cat", "mode": "visual"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "visual"

    with patch("video_retrieval.api.SearchService") as mock_cls:
        mock_cls.return_value.search_mixed.return_value = SearchResponse(
            query="mix", mode="mixed", hits=[]
        )
        resp = client.post("/search", json={"query": "mix", "mode": "mixed"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "mixed"


@pytest.mark.unit
def test_ui_index(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Framefind" in resp.text


@pytest.mark.unit
def test_keyframe_media(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_retrieval import api

    data_dir = tmp_path / "data"
    video_dir = data_dir / "keyframes" / "L21_V015"
    video_dir.mkdir(parents=True)
    frame = video_dir / "shot_0187_start.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xd9")
    monkeypatch.setattr(api.settings, "data_dir", data_dir)

    resp = client.get("/media/keyframes/L21_V015/shot_0187_start.jpg")
    assert resp.status_code == 200
    assert resp.content.startswith(b"\xff\xd8")

    bad = client.get("/media/keyframes/L21_V015/..secret.jpg")
    assert bad.status_code == 400
    missing = client.get("/media/keyframes/L21_V015/does_not_exist.jpg")
    assert missing.status_code == 404


@pytest.mark.unit
def test_video_media(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_retrieval import api

    data_dir = tmp_path / "data"
    videos = data_dir / "videos"
    videos.mkdir(parents=True)
    clip = videos / "L21_V015.mp4"
    clip.write_bytes(b"fake-mp4")
    monkeypatch.setattr(api.settings, "data_dir", data_dir)

    resp = client.get("/media/videos/L21_V015")
    assert resp.status_code == 200
    assert resp.content == b"fake-mp4"

    missing = client.get("/media/videos/NOPE_V000")
    assert missing.status_code == 404


@pytest.mark.unit
def test_capture_video_frame(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2
    import numpy as np

    from video_retrieval import api

    data_dir = tmp_path / "data"
    videos = data_dir / "videos"
    videos.mkdir(parents=True)
    clip = videos / "L21_V015.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    for i in range(50):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :] = (i * 3) % 255
        writer.write(frame)
    writer.release()
    monkeypatch.setattr(api.settings, "data_dir", data_dir)

    resp = client.post("/media/videos/L21_V015/capture", json={"t": 1.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["video_id"] == "L21_V015"
    assert body["frame_index"] == 25
    assert body["image_data_url"].startswith("data:image/jpeg;base64,")
