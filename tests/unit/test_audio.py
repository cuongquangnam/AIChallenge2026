from pathlib import Path
from unittest.mock import patch

import pytest

from video_retrieval.extraction.audio import extract_audio
from tests.helpers import write_dummy_video


@pytest.mark.unit
def test_extract_audio_silent_fallback_without_ffmpeg(tmp_path: Path) -> None:
    video = write_dummy_video(tmp_path / "clip.mp4", frames=20, fps=10, with_audio=False)
    with patch("video_retrieval.extraction.audio.shutil.which", return_value=None):
        track = extract_audio(video, tmp_path / "audio", video_id="clip")
    assert track.path.exists()
    assert track.sample_rate == 16000
    assert track.video_id == "clip"
    assert track.duration_sec is not None
    assert track.duration_sec == pytest.approx(2.0, rel=0.2)


@pytest.mark.unit
def test_extract_audio_falls_back_when_ffmpeg_fails(tmp_path: Path) -> None:
    """Video-only clips (no audio stream) should get a silent WAV, not crash."""
    video = write_dummy_video(tmp_path / "clip.mp4", frames=20, fps=10, with_audio=False)

    class _Result:
        returncode = 1
        stderr = "Output file does not contain any stream"

    with (
        patch("video_retrieval.extraction.audio.shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("video_retrieval.extraction.audio.subprocess.run", return_value=_Result()),
    ):
        track = extract_audio(video, tmp_path / "audio", video_id="clip")

    assert track.path.exists()
    assert track.path.stat().st_size > 0
    assert track.sample_rate == 16000


@pytest.mark.unit
def test_extract_audio_from_video_with_audio_track(tmp_path: Path) -> None:
    video = write_dummy_video(tmp_path / "clip.mp4", frames=20, fps=10, with_audio=True)
    track = extract_audio(video, tmp_path / "audio", video_id="clip")
    assert track.path.exists()
    assert track.path.stat().st_size > 44  # more than empty WAV header
    assert track.sample_rate == 16000
