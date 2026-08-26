from pathlib import Path

import pytest

from video_retrieval.extraction.keyframes import extract_keyframes, load_existing_shots
from video_retrieval.extraction.shots import ShotSpan, subdivide_long_shots
from video_retrieval.models import FrameRole
from tests.helpers import write_dummy_image, write_dummy_video


@pytest.mark.unit
def test_subdivide_long_shots_splits_by_max_sec() -> None:
    spans = [ShotSpan(start_frame=0, end_frame=299)]  # 10s @ 30fps
    out = subdivide_long_shots(spans, fps=30.0, max_shot_sec=2.0)
    assert len(out) == 5
    assert out[0] == ShotSpan(start_frame=0, end_frame=59)
    assert out[-1] == ShotSpan(start_frame=240, end_frame=299)


@pytest.mark.unit
def test_subdivide_long_shots_keeps_short_spans() -> None:
    spans = [ShotSpan(start_frame=10, end_frame=40)]
    out = subdivide_long_shots(spans, fps=25.0, max_shot_sec=10.0)
    assert out == spans


@pytest.mark.unit
def test_extract_keyframes_produces_start_middle_and_final_end(tmp_path: Path) -> None:
    video = write_dummy_video(tmp_path / "clip.mp4", frames=60, fps=10)
    out = tmp_path / "keyframes"

    shots = extract_keyframes(
        video,
        out,
        video_id="clip",
        shot_backend="opencv",
        max_shot_sec=1.0,
        opencv_threshold=1.0,  # one long shot → several subdivided chunks
        opencv_min_shot_len=1,
    )
    assert len(shots) >= 2
    for shot in shots[:-1]:
        roles = {kf.role.value for kf in shot.keyframes}
        assert roles == {"start", "middle"}
        assert not (out / "clip" / f"shot_{shot.shot_index:04d}_end.jpg").exists()
    last = shots[-1]
    last_roles = {kf.role.value for kf in last.keyframes}
    assert "start" in last_roles and "middle" in last_roles
    # End is kept only when it is a distinct frame from start/middle.
    if last.end_frame not in {last.start_frame, (last.start_frame + last.end_frame) // 2}:
        assert "end" in last_roles
    for kf in (kf for shot in shots for kf in shot.keyframes):
        assert kf.path.exists()
        assert kf.video_id == "clip"
        assert kf.timestamp_sec >= 0


@pytest.mark.unit
def test_extract_keyframes_respects_max_shot_sec(tmp_path: Path) -> None:
    # Dummy helper writes a short clip; force tiny max_shot_sec so subdivision runs.
    video = write_dummy_video(tmp_path / "clip.mp4", frames=60, fps=10)
    out = tmp_path / "keyframes"
    shots = extract_keyframes(
        video,
        out,
        video_id="clip",
        shot_backend="opencv",
        max_shot_sec=1.0,
        opencv_threshold=1.0,  # no histogram cuts → one long shot, then subdivided
        opencv_min_shot_len=1,
    )
    assert len(shots) >= 2
    for shot in shots:
        length = shot.end_frame - shot.start_frame + 1
        assert length <= 10  # 1.0s * 10fps


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
