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


def detect_shots_transnetv2(video_path: str) -> list[ShotSpan]:
    """Optional TransNetV2 backend. Falls back to OpenCV if package is missing."""
    try:
        from video_retrieval.extraction.transnet_v2 import predict_shots

        return predict_shots(video_path)
    except Exception:
        return detect_shots_opencv(video_path)
