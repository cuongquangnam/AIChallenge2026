from pathlib import Path

import pytest

from video_retrieval.extraction.keyframes import extract_keyframes, load_existing_shots
from video_retrieval.models import FrameRole
from tests.helpers import write_dummy_image, write_dummy_video


@pytest.mark.unit
def test_extract_keyframes_produces_start_middle_end(tmp_path: Path) -> None:
    video = write_dummy_video(tmp_path / "clip.mp4")
    out = tmp_path / "keyframes"

    shots = extract_keyframes(video, out, video_id="clip", shot_backend="opencv")
    assert len(shots) >= 1
    for shot in shots:
        roles = {kf.role.value for kf in shot.keyframes}
        assert roles == {"start", "middle", "end"}
        for kf in shot.keyframes:
            assert kf.path.exists()
            assert kf.video_id == "clip"
            assert kf.timestamp_sec >= 0


@pytest.mark.unit
def test_load_existing_shots_from_keyframe_folder(tmp_path: Path) -> None:
    folder = tmp_path / "clip"
    for shot in (0, 3):
        for role in ("start", "middle", "end"):
            write_dummy_image(folder / f"shot_{shot:04d}_{role}.jpg")

    shots = load_existing_shots(tmp_path, "clip", fps=25.0, duration_sec=10.0)
    assert [shot.shot_index for shot in shots] == [0, 3]
    assert shots[0].start_sec == 0.0
    assert shots[0].end_sec == pytest.approx(5.0)
    assert {kf.role for kf in shots[0].keyframes} == {
        FrameRole.START,
        FrameRole.MIDDLE,
        FrameRole.END,
    }
    assert all(kf.path.exists() for shot in shots for kf in shot.keyframes)


@pytest.mark.unit
def test_extract_keyframes_produces_start_middle_end(tmp_path: Path) -> None:
    video = write_dummy_video(tmp_path / "clip.mp4")
    out = tmp_path / "keyframes"

    shots = extract_keyframes(video, out, video_id="clip", shot_backend="opencv")
    assert len(shots) >= 1
    for shot in shots:
        roles = {kf.role.value for kf in shot.keyframes}
        assert roles == {"start", "middle", "end"}
        for kf in shot.keyframes:
            assert kf.path.exists()
            assert kf.video_id == "clip"
            assert kf.timestamp_sec >= 0
