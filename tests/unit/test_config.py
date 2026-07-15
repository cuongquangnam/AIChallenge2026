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
