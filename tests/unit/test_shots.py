from pathlib import Path

import pytest

from video_retrieval.extraction.shots import detect_shots_opencv, detect_shots_transnetv2
from tests.helpers import write_dummy_video


@pytest.mark.unit
def test_detect_shots_opencv_finds_boundary(tmp_path: Path) -> None:
    video = write_dummy_video(tmp_path / "two_shots.mp4", frames=40, fps=10)
    shots = detect_shots_opencv(str(video), threshold=0.3, min_shot_len=5)
    assert len(shots) >= 2
    assert shots[0].start_frame == 0
    assert shots[0].end_frame < shots[1].start_frame


@pytest.mark.unit
def test_detect_shots_opencv_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Cannot open video"):
        detect_shots_opencv(str(tmp_path / "missing.mp4"))


@pytest.mark.unit
def test_detect_shots_transnetv2_falls_back(tmp_path: Path) -> None:
    video = write_dummy_video(tmp_path / "clip.mp4")
    shots = detect_shots_transnetv2(str(video))
    assert len(shots) >= 1
