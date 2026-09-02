import pytest

from video_retrieval.config import Settings
from video_retrieval.storage.data_sync import create_data_sync, validate_remote_storage


@pytest.mark.unit
def test_validate_drive_storage_requires_path() -> None:
    settings = Settings(
        drive_data_path="",
        drive_local_path="",
    )
    with pytest.raises(ValueError, match="DRIVE_DATA_PATH"):
        validate_remote_storage(settings)


@pytest.mark.unit
def test_create_data_sync_drive_uses_local_mirror(tmp_path) -> None:
    mirror = tmp_path / "drive-mirror"
    mirror.mkdir()
    settings = Settings(
        drive_data_path="MyDrive/video-retrieval",
        drive_local_path=str(mirror),
        data_dir=tmp_path / "data",
    )
    sync = create_data_sync(settings, mount_drive=False)
    assert sync.remote_root() == mirror.resolve()
