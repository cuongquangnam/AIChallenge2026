from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from video_retrieval.models import SearchHit, SearchResponse


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
def test_index_endpoints_removed(client: TestClient) -> None:
    assert client.post("/index", json={"path": "/tmp/x.mp4"}).status_code == 404
    assert client.post("/index/upload").status_code == 404


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
    assert 'href="/videos"' in resp.text
    assert 'id="import-csv"' in resp.text
    assert "Import CSV" in resp.text


@pytest.mark.unit
def test_ui_videos(client: TestClient) -> None:
    resp = client.get("/videos")
    assert resp.status_code == 200
    assert "Video library" in resp.text
    assert "/static/videos.js" in resp.text


@pytest.mark.unit
def test_list_videos_api(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_retrieval import api

    data_dir = tmp_path / "data"
    videos = data_dir / "videos"
    keyframes = data_dir / "keyframes" / "L21_V015"
    videos.mkdir(parents=True)
    keyframes.mkdir(parents=True)
    (videos / "L21_V015.mp4").write_bytes(b"fake-mp4")
    (videos / "L22_V001.mp4").write_bytes(b"fake-mp4")
    (videos / "notes.txt").write_text("ignore")
    (keyframes / "shot_0000_middle.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    monkeypatch.setattr(api.settings, "data_dir", data_dir)

    resp = client.get("/api/videos")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert len(body["videos"]) == 2
    by_id = {item["video_id"]: item for item in body["videos"]}
    assert by_id["L21_V015"]["video_url"] == "/media/videos/L21_V015"
    assert by_id["L21_V015"]["poster_url"] == "/media/keyframes/L21_V015/shot_0000_middle.jpg"
    assert by_id["L21_V015"]["series"] == "L21"
    assert by_id["L22_V001"]["poster_url"] is None

    filtered = client.get("/api/videos", params={"q": "L22"})
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1
    assert filtered.json()["videos"][0]["video_id"] == "L22_V001"

    series = client.get("/api/videos", params={"series": "L21"})
    assert series.status_code == 200
    assert series.json()["total"] == 1
    assert series.json()["videos"][0]["video_id"] == "L21_V015"

    groups = client.get("/api/videos/groups")
    assert groups.status_code == 200
    group_body = groups.json()
    assert group_body["total_videos"] == 2
    assert group_body["total_groups"] == 2
    by_series = {item["series"]: item for item in group_body["groups"]}
    assert by_series["L21"]["count"] == 1
    assert by_series["L21"]["poster_url"] == "/media/keyframes/L21_V015/shot_0000_middle.jpg"
    assert by_series["L22"]["count"] == 1

    page = client.get("/api/videos", params={"offset": 1, "limit": 1})
    assert page.status_code == 200
    assert page.json()["total"] == 2
    assert len(page.json()["videos"]) == 1


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


@pytest.mark.unit
def test_consecutive_frames(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2
    import numpy as np

    from video_retrieval import api

    data_dir = tmp_path / "data"
    videos = data_dir / "videos"
    videos.mkdir(parents=True)
    clip = videos / "L21_V015.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    for i in range(40):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :] = (i * 5) % 255
        writer.write(frame)
    writer.release()
    monkeypatch.setattr(api.settings, "data_dir", data_dir)

    resp = client.get(
        "/media/videos/L21_V015/frames",
        params={"start_frame": 10, "count": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["video_id"] == "L21_V015"
    assert body["start_frame"] == 10
    assert body["requested_count"] == 5
    assert len(body["frames"]) == 5
    assert [f["frame_index"] for f in body["frames"]] == [10, 11, 12, 13, 14]
    assert body["frames"][0]["image_data_url"].startswith("data:image/jpeg;base64,")
    assert body["frames"][0]["timestamp_sec"] == pytest.approx(10 / body["fps"])

    post = client.post(
        "/media/videos/L21_V015/frames",
        json={"start_frame": 35, "count": 10},
    )
    assert post.status_code == 200
    post_body = post.json()
    assert [f["frame_index"] for f in post_body["frames"]] == [35, 36, 37, 38, 39]

    too_many = client.get(
        "/media/videos/L21_V015/frames",
        params={"start_frame": 0, "count": 101},
    )
    assert too_many.status_code == 400

    past_end = client.get(
        "/media/videos/L21_V015/frames",
        params={"start_frame": 1000, "count": 5},
    )
    assert past_end.status_code == 200
    assert past_end.json()["frames"] == []


@pytest.mark.unit
def test_resolve_submission_frames(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cv2
    import numpy as np

    from video_retrieval import api

    data_dir = tmp_path / "data"
    videos = data_dir / "videos"
    videos.mkdir(parents=True)
    clip = videos / "L21_V015.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    for i in range(30):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :] = (i * 7) % 255
        writer.write(frame)
    writer.release()
    monkeypatch.setattr(api.settings, "data_dir", data_dir)

    csv_text = "L21_V015,5\nL21_V015,10\nmissing_video,0\n"
    resp = client.post(
        "/api/submission/frames",
        json={"csv_text": csv_text, "query_id": "query-p1-1-kis"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "csv"
    assert body["query_id"] == "query-p1-1-kis"
    assert body["total_rows"] == 3
    assert body["resolved"] == 2
    assert len(body["hits"]) == 2
    assert [hit["frame_index"] for hit in body["hits"]] == [5, 10]
    assert body["hits"][0]["image_data_url"].startswith("data:image/jpeg;base64,")
    assert body["hits"][0]["source"] == "csv_import"
    assert any(err["video_id"] == "missing_video" for err in body["errors"])

    rows_resp = client.post(
        "/api/submission/frames",
        json={
            "rows": [
                {"video_id": "L21_V015", "frame_index": 1},
                {"video_id": "L21_V015", "frame_index": 2},
            ]
        },
    )
    assert rows_resp.status_code == 200
    assert rows_resp.json()["resolved"] == 2

    too_many = client.post(
        "/api/submission/frames",
        json={
            "rows": [{"video_id": "L21_V015", "frame_index": i} for i in range(101)]
        },
    )
    assert too_many.status_code == 400
