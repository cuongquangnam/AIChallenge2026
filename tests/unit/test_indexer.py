from pathlib import Path
import json

import pytest

from video_retrieval.pipeline.indexer import VideoIndexer
from video_retrieval.storage.qdrant_store import QdrantStore
from tests.fakes import FakeElasticsearchStore
from tests.helpers import write_dummy_video


@pytest.mark.unit
def test_index_video_missing_raises(settings) -> None:
    indexer = VideoIndexer(
        settings,
        qdrant=QdrantStore(settings),
        es=FakeElasticsearchStore(),  # type: ignore[arg-type]
    )
    with pytest.raises(FileNotFoundError):
        indexer.index_video(Path("/no/such/video.mp4"))


@pytest.mark.unit
def test_index_video_writes_manifest_and_counts(settings, tmp_path: Path) -> None:
    video = write_dummy_video(tmp_path / "sample.mp4")
    fake_es = FakeElasticsearchStore()
    indexer = VideoIndexer(
        settings,
        qdrant=QdrantStore(settings),
        es=fake_es,  # type: ignore[arg-type]
    )

    result = indexer.index_video(video, video_id="sample")
    assert result.video_id == "sample"
    assert result.num_shots >= 1
    assert result.num_keyframes >= 3
    assert result.num_visual_points == result.num_keyframes
    assert result.num_text_docs >= 1
    assert result.audio_path is not None and result.audio_path.exists()

    manifest = settings.data_dir / "manifests" / "sample.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    asr_docs = [doc for doc in payload["text_docs"] if doc["source"] == "asr"]
    assert asr_docs
    assert all(doc["frame_index"] is not None for doc in asr_docs)
    assert all(doc["shot_index"] is not None for doc in asr_docs)


@pytest.mark.unit
def test_write_manifest_keeps_vietnamese_readable(settings) -> None:
    indexer = VideoIndexer(
        settings,
        qdrant=QdrantStore(settings),
        es=FakeElasticsearchStore(),  # type: ignore[arg-type]
    )
    indexer._write_manifest(
        video_id="clip",
        stored_video=settings.videos_dir / "clip.mp4",
        audio=None,
        shots=[],
        keyframes=[],
        text_docs=[
            {
                "doc_id": "clip:ocr:0:middle",
                "video_id": "clip",
                "source": "ocr",
                "text": "ĐÃ QUAY TRỞ LẠI!",
            }
        ],
    )
    raw = (settings.manifests_dir / "clip.json").read_text(encoding="utf-8")
    assert "ĐÃ QUAY TRỞ LẠI!" in raw
    assert "\\u" not in raw


@pytest.mark.unit
def test_store_video_does_not_copy_existing_file(settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.helpers import write_dummy_video

    existing = write_dummy_video(settings.videos_dir / "clip.mp4")
    copies: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "video_retrieval.pipeline.indexer.shutil.copy2",
        lambda src, dst: copies.append((str(src), str(dst))),
    )
    indexer = VideoIndexer(
        settings,
        qdrant=QdrantStore(settings),
        es=FakeElasticsearchStore(),  # type: ignore[arg-type]
    )
    stored = indexer._store_video(existing, "clip")
    assert stored == existing.resolve()
    assert copies == []


@pytest.mark.unit
def test_index_directory_filters_extensions(settings, tmp_path: Path) -> None:
    write_dummy_video(tmp_path / "a.mp4")
    (tmp_path / "notes.txt").write_text("skip", encoding="utf-8")
    indexer = VideoIndexer(
        settings,
        qdrant=QdrantStore(settings),
        es=FakeElasticsearchStore(),  # type: ignore[arg-type]
    )
    results = indexer.index_directory(tmp_path)
    assert len(results) == 1
    assert results[0].video_id == "a"


@pytest.mark.unit
def test_index_video_visual_only_skips_text(settings, tmp_path: Path) -> None:
    video = write_dummy_video(tmp_path / "clip.mp4")
    fake_es = FakeElasticsearchStore()
    indexer = VideoIndexer(
        settings,
        qdrant=QdrantStore(settings),
        es=fake_es,  # type: ignore[arg-type]
    )
    result = indexer.index_video(video, video_id="clip", stages=["visual"])
    assert result.stages == ["visual"]
    assert result.num_visual_points >= 3
    assert result.num_text_docs == 0
    assert fake_es.docs == {}


@pytest.mark.unit
def test_index_video_ocr_only_then_asr(settings, tmp_path: Path) -> None:
    video = write_dummy_video(tmp_path / "clip.mp4")
    fake_es = FakeElasticsearchStore()
    indexer = VideoIndexer(
        settings,
        qdrant=QdrantStore(settings),
        es=fake_es,  # type: ignore[arg-type]
    )
    ocr_result = indexer.index_video(video, video_id="clip", stages=["ocr"])
    assert ocr_result.num_visual_points == 0
    assert ocr_result.num_text_docs >= 1
    assert all(doc.source == "ocr" for doc in fake_es.docs.values())

    asr_result = indexer.index_video(video, video_id="clip", stages=["asr"], reuse_extract=True)
    assert asr_result.num_text_docs >= 1
    sources = {doc.source for doc in fake_es.docs.values()}
    assert sources == {"ocr", "asr"}
    manifest = json.loads((settings.manifests_dir / "clip.json").read_text(encoding="utf-8"))
    assert {doc["source"] for doc in manifest["text_docs"]} == {"ocr", "asr"}


@pytest.mark.unit
def test_index_ocr_asr_reuses_existing_keyframes(
    settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.helpers import write_dummy_image

    video = write_dummy_video(settings.videos_dir / "clip.mp4")
    for role in ("start", "middle", "end"):
        write_dummy_image(settings.keyframes_dir / "clip" / f"shot_0000_{role}.jpg")
    _write_silent_wav(settings.audio_dir / "clip.wav")

    monkeypatch.setattr(
        "video_retrieval.pipeline.indexer.extract_keyframes",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should reuse keyframes")),
    )
    monkeypatch.setattr(
        "video_retrieval.pipeline.indexer.extract_audio",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should reuse audio")),
    )

    fake_es = FakeElasticsearchStore()
    indexer = VideoIndexer(
        settings,
        qdrant=QdrantStore(settings),
        es=fake_es,  # type: ignore[arg-type]
    )
    result = indexer.index_video(video, video_id="clip", stages=["ocr", "asr"])
    assert result.stages == ["asr", "ocr"]
    assert result.num_visual_points == 0
    assert result.num_keyframes == 3
    assert result.num_text_docs >= 1
    assert {doc.source for doc in fake_es.docs.values()} == {"ocr", "asr"}


@pytest.mark.unit
def test_index_directory_from_keyframe_folders(settings) -> None:
    from tests.helpers import write_dummy_image

    write_dummy_video(settings.videos_dir / "clip.mp4")
    keyframe_root = settings.data_dir / "incoming-kfs"
    for role in ("start", "middle", "end"):
        write_dummy_image(keyframe_root / "clip" / f"shot_0000_{role}.jpg")
    write_dummy_image(settings.keyframes_dir / "clip" / "shot_0000_start.jpg")
    write_dummy_image(settings.keyframes_dir / "clip" / "shot_0000_middle.jpg")
    write_dummy_image(settings.keyframes_dir / "clip" / "shot_0000_end.jpg")
    _write_silent_wav(settings.audio_dir / "clip.wav")

    indexer = VideoIndexer(
        settings,
        qdrant=QdrantStore(settings),
        es=FakeElasticsearchStore(),  # type: ignore[arg-type]
    )
    results = indexer.index_directory(keyframe_root, stages=["ocr"])
    assert len(results) == 1
    assert results[0].video_id == "clip"
    assert results[0].num_text_docs >= 1


@pytest.mark.unit
def test_index_directory_skips_completed_videos(settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.helpers import write_dummy_image

    video = write_dummy_video(settings.videos_dir / "clip.mp4")
    for role in ("start", "middle", "end"):
        write_dummy_image(settings.keyframes_dir / "clip" / f"shot_0000_{role}.jpg")
    _write_silent_wav(settings.audio_dir / "clip.wav")

    fake_es = FakeElasticsearchStore()
    indexer = VideoIndexer(
        settings,
        qdrant=QdrantStore(settings),
        es=fake_es,  # type: ignore[arg-type]
    )
    first = indexer.index_video(video, video_id="clip", stages=["ocr"])
    assert first.num_text_docs >= 1

    def fail_ocr(*args, **kwargs):
        raise AssertionError("OCR should be skipped on resume")

    monkeypatch.setattr(indexer.ocr, "extract_from_keyframes", fail_ocr)
    skipped = indexer.index_video(video, video_id="clip", stages=["ocr"], resume=True)
    assert skipped.video_id == "clip"

    mixed = indexer.index_video(video, video_id="clip", stages=["ocr", "asr"], resume=True)
    assert mixed.stages == ["asr", "ocr"]
    sources = {doc.source for doc in fake_es.docs.values()}
    assert sources == {"ocr", "asr"}


def _write_silent_wav(path: Path, duration_sec: float = 0.1) -> None:
    import struct
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = max(1, int(duration_sec * 16000))
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(struct.pack("<" + "h" * n_frames, *([0] * n_frames)))