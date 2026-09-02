from pathlib import Path
from unittest.mock import patch

import pytest

from video_retrieval.config import Settings
from video_retrieval.storage.qa_video_sync import (
    ensure_qa_videos_from_drive,
    local_video_path,
    should_lazy_pull_qa_videos,
)


@pytest.mark.unit
def test_should_lazy_pull_qa_videos_only_on_colab() -> None:
    assert should_lazy_pull_qa_videos(Settings(colab_runtime=True)) is True
    assert should_lazy_pull_qa_videos(Settings(colab_runtime=False)) is False


@pytest.mark.unit
def test_local_video_path_from_videos_dir(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", colab_runtime=True)
    settings.ensure_dirs()
    video = settings.videos_dir / "clip.mp4"
    video.write_bytes(b"fake")

    assert local_video_path(settings, "clip") == video


@pytest.mark.unit
def test_ensure_qa_videos_from_drive_pulls_missing_file(tmp_path: Path) -> None:
    from video_retrieval.storage.drive_sync import DriveDataSync

    settings = Settings(
        data_dir=tmp_path / "data",
        colab_runtime=True,
        drive_data_path="MyDrive/video-retrieval",
    )
    settings.ensure_dirs()

    remote_root = tmp_path / "remote"
    remote_video = remote_root / "videos" / "clip.mp4"
    remote_video.parent.mkdir(parents=True)
    remote_video.write_bytes(b"video-bytes")

    sync = DriveDataSync(
        local_dir=settings.data_dir,
        data_path="unused",
        local_mirror=str(remote_root),
        mount_on_access=False,
    )

    with patch(
        "video_retrieval.storage.qa_video_sync.create_data_sync",
        return_value=sync,
    ):
        pulled = ensure_qa_videos_from_drive(settings, {"clip", "missing"})

    assert pulled == {"clip": "clip.mp4"}
    assert (settings.videos_dir / "clip.mp4").read_bytes() == b"video-bytes"


@pytest.mark.unit
def test_ensure_qa_videos_skips_when_not_colab(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", colab_runtime=False)
    assert ensure_qa_videos_from_drive(settings, {"clip"}) == {}
