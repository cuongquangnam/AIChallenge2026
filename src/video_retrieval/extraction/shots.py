from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ShotSpan:
    start_frame: int
    end_frame: int


def detect_shots_opencv(
    video_path: str,
    threshold: float = 0.45,
    min_shot_len: int = 8,
) -> list[ShotSpan]:
    """Histogram-diff shot boundary detector (lightweight TransNetV2 stand-in)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    prev_hist: np.ndarray | None = None
    boundaries: list[int] = [0]
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            if diff >= threshold and (frame_idx - boundaries[-1]) >= min_shot_len:
                boundaries.append(frame_idx)
        prev_hist = hist
        frame_idx += 1

    total = frame_idx
    cap.release()

    if total == 0:
        return []

    if boundaries[-1] != total:
        boundaries.append(total)

    shots: list[ShotSpan] = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1] - 1
        if end < start:
            continue
        shots.append(ShotSpan(start_frame=start, end_frame=end))
    return shots


def detect_shots_transnetv2(
    video_path: str,
    *,
    threshold: float = 0.5,
    device: str = "auto",
) -> list[ShotSpan]:
    """TransNetV2 backend. Falls back to OpenCV if the package is missing."""
    try:
        from video_retrieval.extraction.transnet_v2 import predict_shots

        return predict_shots(video_path, threshold=threshold, device=device)
    except Exception as exc:
        print(f"[transnet] failed ({exc}); falling back to OpenCV", flush=True)
        return detect_shots_opencv(video_path)


def subdivide_long_shots(
    spans: list[ShotSpan],
    *,
    fps: float,
    max_shot_sec: float,
) -> list[ShotSpan]:
    """Split spans longer than ``max_shot_sec`` into equal-ish chunks.

    Keeps start/middle keyframe spacing bounded so long mis-detected
    (or truly long) scenes don't leave multi-minute blind spots.
    """
    if max_shot_sec <= 0 or fps <= 0:
        return list(spans)

    max_frames = max(1, int(round(max_shot_sec * fps)))
    out: list[ShotSpan] = []
    for span in spans:
        length = span.end_frame - span.start_frame + 1
        if length <= max_frames:
            out.append(span)
            continue
        start = span.start_frame
        while start <= span.end_frame:
            end = min(span.end_frame, start + max_frames - 1)
            out.append(ShotSpan(start_frame=start, end_frame=end))
            start = end + 1
    return out
