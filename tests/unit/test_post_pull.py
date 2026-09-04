import zipfile
from pathlib import Path

import pytest

from video_retrieval.storage.drive_sync import _copy_keyframes
from video_retrieval.storage.post_pull import extract_keyframes_zip
from video_retrieval.storage.qdrant_bootstrap import (
    find_qdrant_snapshot,
    resolve_colab_qdrant_url,
    stage_snapshot_for_recovery,
)


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
def test_stage_snapshot_for_recovery_hardlinks(tmp_path: Path) -> None:
    src = tmp_path / "data" / "qdrant" / "video_keyframes_transnet.snapshot"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"snapshot-bytes")
    snaps = tmp_path / "qdrant" / "snapshots"

    staged = stage_snapshot_for_recovery(src, snapshots_dir=snaps, progress=False)
    assert staged == snaps / "video_keyframes_transnet.snapshot"
    assert staged.read_bytes() == b"snapshot-bytes"
    assert staged.samefile(src)


@pytest.mark.unit
def test_extract_keyframes_zip_flattens_nested_dir(tmp_path: Path) -> None:
    keyframes_dir = tmp_path / "keyframes"
    keyframes_dir.mkdir()
    zip_path = keyframes_dir / "keyframes.zip"

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("keyframes/L21_V001/shot_0001_middle.jpg", b"jpg")

    result = extract_keyframes_zip(keyframes_dir, progress=False)
    assert result["action"] == "extracted"
    assert result["zips"] == ["keyframes.zip"]
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


@pytest.mark.unit
def test_extract_keyframes_multiple_zip_shards(tmp_path: Path) -> None:
    keyframes_dir = tmp_path / "keyframes"
    keyframes_dir.mkdir()
    with zipfile.ZipFile(keyframes_dir / "keyframes_000.zip", "w") as archive:
        archive.writestr("L21_V001/shot_0001_middle.jpg", b"jpg")
    with zipfile.ZipFile(keyframes_dir / "keyframes_001.zip", "w") as archive:
        archive.writestr("L21_V002/shot_0001_middle.jpg", b"jpg")

    result = extract_keyframes_zip(keyframes_dir, progress=False)

    assert result["action"] == "extracted"
    assert result["zips"] == ["keyframes_000.zip", "keyframes_001.zip"]
    assert (keyframes_dir / "L21_V001" / "shot_0001_middle.jpg").is_file()
    assert (keyframes_dir / "L21_V002" / "shot_0001_middle.jpg").is_file()


@pytest.mark.unit
def test_copy_keyframes_copies_all_zip_shards(tmp_path: Path) -> None:
    source = tmp_path / "drive" / "keyframes"
    source.mkdir(parents=True)
    (source / "keyframes_000.zip").write_bytes(b"zip0")
    (source / "keyframes_001.zip").write_bytes(b"zip1")
    dest = tmp_path / "data" / "keyframes"

    copied = _copy_keyframes(source, dest, progress=False)

    assert copied == 2
    assert (dest / "keyframes_000.zip").read_bytes() == b"zip0"
    assert (dest / "keyframes_001.zip").read_bytes() == b"zip1"
