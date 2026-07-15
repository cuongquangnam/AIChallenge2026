from pathlib import Path

import pytest

from video_retrieval.extraction.keyframes import extract_keyframes
from tests.helpers import write_dummy_video


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
