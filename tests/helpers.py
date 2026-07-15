from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def write_dummy_video(
    path: Path,
    frames: int = 40,
    fps: int = 10,
    *,
    with_audio: bool = True,
) -> Path:
    """Write a short two-shot clip (red → green).

    When ``with_audio`` is True and ffmpeg is on PATH, muxes a sine tone so the
    file has a real audio stream (needed to exercise ffmpeg extraction).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    video_only = path if not with_audio else path.with_suffix(".silent.mp4")
    _write_opencv_video(video_only, frames=frames, fps=fps)

    if with_audio and shutil.which("ffmpeg"):
        duration = frames / float(fps)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_only),
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=16000:duration={duration}",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        video_only.unlink(missing_ok=True)
        if result.returncode != 0:
            # Fall back to video-only so tests still run without encoders.
            _write_opencv_video(path, frames=frames, fps=fps)
        return path

    if with_audio and video_only != path:
        video_only.replace(path)
    return path


def _write_opencv_video(path: Path, frames: int, fps: int) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 64),
    )
    for i in range(frames):
        color = (0, 0, 255) if i < frames // 2 else (0, 255, 0)
        frame = np.full((64, 64, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def write_dummy_image(path: Path, color: tuple[int, int, int] = (120, 40, 200)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)
    return path
