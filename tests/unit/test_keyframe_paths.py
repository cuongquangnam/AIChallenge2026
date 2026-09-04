"""Tests for resolving indexed keyframe paths onto local DATA_DIR."""

from pathlib import Path

import pytest

from video_retrieval.config import Settings
from video_retrieval.storage.keyframe_paths import resolve_keyframe_path


@pytest.mark.unit
def test_resolve_keyframe_path_uses_video_dir_and_basename(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    frame = settings.keyframes_dir / "L01_V001" / "shot_0001_middle.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"jpg")

    found = resolve_keyframe_path(
        settings,
        video_id="L01_V001",
        keyframe_path="/old/host/data/keyframes/L01_V001/shot_0001_middle.jpg",
    )
    assert found == frame.resolve()


@pytest.mark.unit
def test_resolve_keyframe_path_from_relative_keyframes_suffix(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    frame = settings.keyframes_dir / "L01_V002" / "shot_0002_middle.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"jpg")

    found = resolve_keyframe_path(
        settings,
        video_id="L01_V002",
        keyframe_path="ignored/prefix/keyframes/L01_V002/shot_0002_middle.jpg",
    )
    assert found == frame.resolve()


@pytest.mark.unit
def test_resolve_keyframe_path_missing_returns_none(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    assert (
        resolve_keyframe_path(
            settings,
            video_id="L01_V001",
            keyframe_path="shot_9999_middle.jpg",
        )
        is None
    )
