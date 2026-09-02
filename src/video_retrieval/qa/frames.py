from __future__ import annotations

import uuid
from pathlib import Path

import cv2

from video_retrieval.config import Settings
from video_retrieval.models import QAFrame, QAFrameGroup
from video_retrieval.storage.qa_video_sync import ensure_qa_videos_from_drive, local_video_path
from video_retrieval.qa.retrieval import QACandidate


class VideoNotFoundForQAError(FileNotFoundError):
    pass


class QAFrameExtractionError(RuntimeError):
    pass


class VideoLocator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def find(self, video_id: str) -> Path:
        path = local_video_path(self.settings, video_id)
        if path is not None:
            return path

        ensure_qa_videos_from_drive(self.settings, {video_id})
        path = local_video_path(self.settings, video_id)
        if path is not None:
            return path

        raise VideoNotFoundForQAError(f"Indexed video file was not found for {video_id}")


class QAFrameSampler:
    def __init__(self, settings: Settings, locator: VideoLocator | None = None):
        self.settings = settings
        self.locator = locator or VideoLocator(settings)

    def sample(
        self,
        *,
        video_id: str,
        candidates: list[QACandidate],
        group_count: int,
        radius: int,
        stride: int,
        min_center_gap: int,
    ) -> list[QAFrameGroup]:
        video_path = self.locator.find(video_id)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise QAFrameExtractionError(f"Cannot open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            cap.release()
            raise QAFrameExtractionError(f"Video contains no readable frames: {video_path}")

        centers = _select_centers(
            candidates,
            fps=fps,
            frame_count=frame_count,
            limit=max(group_count, 1),
            min_gap=max(min_center_gap, 0),
        )
        request_id = uuid.uuid4().hex
        root = self.settings.data_dir / "qa_frames" / request_id / video_id
        groups: list[QAFrameGroup] = []
        try:
            for group_index, (center, candidate) in enumerate(centers, start=1):
                group_dir = root / f"group_{group_index:02d}_center_{center:08d}"
                group_dir.mkdir(parents=True, exist_ok=True)
                frame_ids = _neighbor_frame_ids(
                    center,
                    frame_count=frame_count,
                    radius=max(radius, 0),
                    stride=max(stride, 1),
                )
                frames: list[QAFrame] = []
                for frame_id in frame_ids:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ok, image = cap.read()
                    if not ok:
                        continue
                    path = group_dir / f"frame_{frame_id:08d}.jpg"
                    if not cv2.imwrite(str(path), image):
                        continue
                    frames.append(
                        QAFrame(
                            frame_id=frame_id,
                            timestamp_sec=frame_id / fps,
                            path=path,
                        )
                    )
                if frames:
                    groups.append(
                        QAFrameGroup(
                            video_id=video_id,
                            center_frame_id=center,
                            retrieval_score=candidate.score,
                            frames=frames,
                            sources=candidate.sources,
                        )
                    )
        finally:
            cap.release()

        if not groups:
            raise QAFrameExtractionError(f"Could not extract Q&A evidence from {video_path}")
        return groups


def _select_centers(
    candidates: list[QACandidate],
    *,
    fps: float,
    frame_count: int,
    limit: int,
    min_gap: int,
) -> list[tuple[int, QACandidate]]:
    selected: list[tuple[int, QACandidate]] = []
    for candidate in candidates:
        if candidate.frame_index is not None:
            center = candidate.frame_index
        else:
            center = round((candidate.timestamp_sec or 0.0) * fps)
        center = min(max(center, 0), frame_count - 1)
        if any(abs(center - existing) < min_gap for existing, _ in selected):
            continue
        selected.append((center, candidate))
        if len(selected) >= limit:
            break
    return selected


def _neighbor_frame_ids(center: int, *, frame_count: int, radius: int, stride: int) -> list[int]:
    frame_ids = {
        min(max(center + offset, 0), frame_count - 1)
        for offset in range(-radius, radius + 1, stride)
    }
    frame_ids.add(center)
    return sorted(frame_ids)
