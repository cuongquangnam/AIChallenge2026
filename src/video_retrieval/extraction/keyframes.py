from __future__ import annotations

import re
from pathlib import Path

import cv2

from video_retrieval.extraction.shots import detect_shots_opencv, detect_shots_transnetv2
from video_retrieval.models import FrameRole, KeyFrame, Shot

_KEYFRAME_NAME = re.compile(
    r"^shot_(\d+)_(start|middle|end)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)
_ROLE_ORDER = {FrameRole.START: 0, FrameRole.MIDDLE: 1, FrameRole.END: 2}


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


def load_existing_shots(
    output_dir: Path,
    video_id: str,
    *,
    fps: float = 25.0,
    duration_sec: float | None = None,
) -> list[Shot]:
    """Rebuild shot metadata from already-extracted keyframe files."""
    folder = Path(output_dir) / video_id
    if not folder.is_dir():
        return []

    grouped: dict[int, dict[FrameRole, Path]] = {}
    for path in folder.iterdir():
        if not path.is_file():
            continue
        match = _KEYFRAME_NAME.match(path.name)
        if not match:
            continue
        shot_index = int(match.group(1))
        role = FrameRole(match.group(2).lower())
        grouped.setdefault(shot_index, {})[role] = path
    if not grouped:
        return []

    fps = fps if fps and fps > 0 else 25.0
    n_shots = len(grouped)
    duration = duration_sec if duration_sec and duration_sec > 0 else float(n_shots)
    shots: list[Shot] = []
    for offset, shot_index in enumerate(sorted(grouped)):
        start_sec = (offset / n_shots) * duration
        end_sec = ((offset + 1) / n_shots) * duration
        start_frame = int(round(start_sec * fps))
        end_frame = max(start_frame, int(round(end_sec * fps)) - 1)
        timestamps = {
            FrameRole.START: start_sec,
            FrameRole.MIDDLE: (start_sec + end_sec) / 2.0,
            FrameRole.END: end_sec,
        }
        keyframes = [
            KeyFrame(
                video_id=video_id,
                shot_index=shot_index,
                role=role,
                frame_index=int(round(timestamps[role] * fps)),
                timestamp_sec=timestamps[role],
                path=path,
            )
            for role, path in grouped[shot_index].items()
        ]
        keyframes.sort(key=lambda kf: _ROLE_ORDER.get(kf.role, 9))
        shots.append(
            Shot(
                video_id=video_id,
                shot_index=shot_index,
                start_frame=start_frame,
                end_frame=end_frame,
                start_sec=start_sec,
                end_sec=end_sec,
                keyframes=keyframes,
            )
        )
    return shots


def video_timing(video_path: Path) -> tuple[float, float | None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 25.0, None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()
    duration = frame_count / fps if fps > 0 and frame_count > 0 else None
    return fps, duration


def _read_frame(cap: cv2.VideoCapture, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    return frame if ok else None
