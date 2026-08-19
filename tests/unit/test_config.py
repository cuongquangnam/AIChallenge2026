from pathlib import Path

import pytest

from video_retrieval.config import Settings


@pytest.mark.unit
def test_ensure_dirs_creates_layout(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    assert settings.videos_dir.is_dir()
    assert settings.keyframes_dir.is_dir()
    assert settings.audio_dir.is_dir()
    assert settings.manifests_dir.is_dir()


@pytest.mark.unit
def test_get_settings_overrides_data_dir(tmp_path: Path) -> None:
    from video_retrieval.config import get_settings

    override = tmp_path / "outputs"
    settings = get_settings(data_dir=override)
    assert settings.data_dir == override.resolve()
    assert settings.keyframes_dir == override.resolve() / "keyframes"


@pytest.mark.unit
def test_settings_with_data_dir_resolves_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(data_dir=Path("./data"))
    updated = settings.with_data_dir("custom-out")
    assert updated.data_dir == (tmp_path / "custom-out").resolve()
    assert updated.videos_dir == updated.data_dir / "videos"


@pytest.mark.unit
def test_settings_accepts_backend_overrides(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        visual_backend="mock",
        ocr_backend="mock",
        asr_backend="mock",
        shot_backend="opencv",
    )
    assert settings.visual_backend == "mock"
    assert settings.ocr_backend == "mock"
    assert settings.asr_backend == "mock"
    assert settings.shot_backend == "opencv"
