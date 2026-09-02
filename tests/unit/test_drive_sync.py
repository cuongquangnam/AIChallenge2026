from pathlib import Path

import pytest

from video_retrieval.storage.drive_sync import DriveDataSync


@pytest.mark.unit
def test_drive_sync_pull_and_push(tmp_path: Path) -> None:
    remote_root = tmp_path / "drive" / "MyDrive" / "video-retrieval"
    manifests = remote_root / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "clip.json").write_text('{"video_id": "clip"}', encoding="utf-8")

    local_dir = tmp_path / "data"
    sync = DriveDataSync(
        data_path="unused",
        local_dir=local_dir,
        local_mirror=remote_root,
        mount_on_access=False,
    )

    pulled = sync.pull(paths=["manifests"])
    assert pulled == 1
    assert (local_dir / "manifests" / "clip.json").is_file()

    (local_dir / "manifests" / "clip.json").write_text('{"video_id": "updated"}', encoding="utf-8")
    pushed = sync.push(paths=["manifests"])
    assert pushed == 1
    assert (remote_root / "manifests" / "clip.json").read_text(encoding="utf-8") == '{"video_id": "updated"}'


@pytest.mark.unit
def test_drive_sync_download_paths(tmp_path: Path) -> None:
    remote_root = tmp_path / "drive-root"
    keyframe = remote_root / "keyframes" / "clip" / "shot_0001_middle.jpg"
    keyframe.parent.mkdir(parents=True)
    keyframe.write_bytes(b"jpeg")

    local_dir = tmp_path / "data"
    sync = DriveDataSync(
        local_dir=local_dir,
        local_mirror=remote_root,
        mount_on_access=False,
    )
    assert sync.download_paths(["keyframes/clip/shot_0001_middle.jpg"]) == 1
    assert (local_dir / "keyframes" / "clip" / "shot_0001_middle.jpg").is_file()
