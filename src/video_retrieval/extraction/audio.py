from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2

from video_retrieval.models import AudioTrack


def extract_audio(video_path: Path, output_dir: Path, video_id: str) -> AudioTrack:
    """Extract mono 16 kHz WAV via ffmpeg when available.

    Falls back to a silent WAV when ffmpeg is missing or the input has no
    audio stream (common for OpenCV-generated / muted clips).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}.wav"
    duration = _probe_duration(video_path)

    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return AudioTrack(
                video_id=video_id,
                path=out_path,
                sample_rate=16000,
                duration_sec=duration,
            )

    _write_silent_wav(out_path, duration_sec=duration or 1.0)
    return AudioTrack(
        video_id=video_id,
        path=out_path,
        sample_rate=16000,
        duration_sec=duration,
    )


def _probe_duration(video_path: Path) -> float | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0:
        return None
    return float(frame_count / fps)


def _write_silent_wav(path: Path, duration_sec: float, sample_rate: int = 16000) -> None:
    import struct
    import wave

    n_frames = max(1, int(duration_sec * sample_rate))
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * n_frames, *([0] * n_frames)))
