import zipfile
from pathlib import Path

import pytest

from video_retrieval.storage.drive_sync import _copy_keyframes
from video_retrieval.storage.post_pull import extract_keyframes_zip
from video_retrieval.storage.qdrant_bootstrap import find_qdrant_snapshot, resolve_colab_qdrant_url


@pytest.mark.unit
def test_find_qdrant_snapshot_prefers_collection_name(tmp_path: Path) -> None:
    qdir = tmp_path / "qdrant"
    qdir.mkdir()
    other = qdir / "other_collection.snapshot"
    match = qdir / "video_keyframes_transnet.snapshot"
    other.write_bytes(b"x")
    match.write_bytes(b"y")

    found = find_qdrant_snapshot(qdir, "video_keyframes_transnet")
    assert found == match


@pytest.mark.unit
def test_resolve_colab_qdrant_url_uses_server_for_snapshot(settings, tmp_path: Path) -> None:
    settings = settings.model_copy(
        update={
            "data_dir": tmp_path / "data",
            "qdrant_collection": "video_keyframes_transnet",
            "colab_runtime": True,
        }
    )
    qdir = settings.qdrant_dir
    qdir.mkdir(parents=True)
    (qdir / "video_keyframes_transnet.snapshot").write_bytes(b"snap")

    assert resolve_colab_qdrant_url(settings) == "http://127.0.0.1:6333"


@pytest.mark.unit
def test_extract_keyframes_zip_flattens_nested_dir(tmp_path: Path) -> None:
    keyframes_dir = tmp_path / "keyframes"
    keyframes_dir.mkdir()
    zip_path = keyframes_dir / "keyframes.zip"

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("keyframes/L21_V001/shot_0001_middle.jpg", b"jpg")

    result = extract_keyframes_zip(keyframes_dir, progress=False)
    assert result["action"] == "extracted"
    assert (keyframes_dir / "L21_V001" / "shot_0001_middle.jpg").is_file()


@pytest.mark.unit
def test_extract_keyframes_zip_skips_when_already_present(tmp_path: Path) -> None:
    keyframes_dir = tmp_path / "keyframes"
    video_dir = keyframes_dir / "L21_V001"
    video_dir.mkdir(parents=True)
    frame = video_dir / "shot_0001_middle.jpg"
    frame.write_bytes(b"jpg")
    zip_path = keyframes_dir / "keyframes.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("L21_V001/shot_0001_middle.jpg", b"jpg")

    result = extract_keyframes_zip(keyframes_dir, progress=False)
    assert result["action"] == "skip"
    assert result["reason"] == "already_extracted"


@pytest.mark.unit
def test_copy_keyframes_prefers_zip_over_loose_files(tmp_path: Path) -> None:
    source = tmp_path / "drive" / "keyframes"
    loose = source / "L21_V001"
    loose.mkdir(parents=True)
    (loose / "shot_0001_middle.jpg").write_bytes(b"loose")
    (source / "keyframes.zip").write_bytes(b"zip")
    dest = tmp_path / "data" / "keyframes"

    copied = _copy_keyframes(source, dest, progress=False)

    assert copied == 1
    assert (dest / "keyframes.zip").read_bytes() == b"zip"
    assert not (dest / "L21_V001").exists()
