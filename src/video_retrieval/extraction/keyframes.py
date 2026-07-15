from __future__ import annotations

from pathlib import Path

import cv2

from video_retrieval.extraction.shots import detect_shots_opencv, detect_shots_transnetv2
from video_retrieval.models import FrameRole, KeyFrame, Shot


def extract_keyframes(
    video_path: Path,
    output_dir: Path,
    video_id: str,
    shot_backend: str = "opencv",
) -> list[Shot]:
    """Detect shots and save start / middle / end keyframes per shot."""
    video_path = Path(video_path)
    out_root = Path(output_dir) / video_id
    out_root.mkdir(parents=True, exist_ok=True)

    if shot_backend == "transnetv2":
        spans = detect_shots_transnetv2(str(video_path))
    else:
        spans = detect_shots_opencv(str(video_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    shots: list[Shot] = []

    for shot_index, span in enumerate(spans):
        mid = (span.start_frame + span.end_frame) // 2
        role_to_frame = {
            FrameRole.START: span.start_frame,
            FrameRole.MIDDLE: mid,
            FrameRole.END: span.end_frame,
        }
        keyframes: list[KeyFrame] = []
        for role, frame_index in role_to_frame.items():
            frame = _read_frame(cap, frame_index)
            if frame is None:
                continue
            rel = f"shot_{shot_index:04d}_{role.value}.jpg"
            frame_path = out_root / rel
            cv2.imwrite(str(frame_path), frame)
            keyframes.append(
                KeyFrame(
                    video_id=video_id,
                    shot_index=shot_index,
                    role=role,
                    frame_index=frame_index,
                    timestamp_sec=frame_index / fps,
                    path=frame_path,
                )
            )

        shots.append(
            Shot(
                video_id=video_id,
                shot_index=shot_index,
                start_frame=span.start_frame,
                end_frame=span.end_frame,
                start_sec=span.start_frame / fps,
                end_sec=span.end_frame / fps,
                keyframes=keyframes,
            )
        )

    cap.release()
    return shots


def _read_frame(cap: cv2.VideoCapture, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    return frame if ok else None
