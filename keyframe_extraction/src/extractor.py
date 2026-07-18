"""Frame-accurate OpenCV extraction and atomic result metadata writing."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from .detector import InvalidShotBoundaryError, ShotBoundary

LOGGER = logging.getLogger(__name__)


class VideoOpenError(RuntimeError):
    """Raised when OpenCV cannot open or inspect an input video."""


class FrameExtractionError(RuntimeError):
    """Raised when an exact requested frame cannot be decoded."""


class OutputWriteError(RuntimeError):
    """Raised when a keyframe or metadata file cannot be written."""


class OutputConflictError(RuntimeError):
    """Raised when generated output paths already exist without overwrite mode."""


@dataclass(frozen=True)
class VideoMetadata:
    """Validated video stream properties reported by OpenCV."""

    fps: float
    frame_count: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate all values needed for frame bounds and timestamps."""
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise VideoOpenError(
                f"Video FPS must be finite and greater than zero; received {self.fps}."
            )
        integer_values = (self.frame_count, self.width, self.height)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_values
        ):
            raise VideoOpenError("Frame count, width, and height must be integers.")
        if self.frame_count < 1:
            raise VideoOpenError(
                f"Video must contain at least one frame; received {self.frame_count}."
            )
        if self.width < 1 or self.height < 1:
            raise VideoOpenError(
                "Video dimensions must be positive; received "
                f"{self.width}x{self.height}."
            )

    @property
    def duration_seconds(self) -> float:
        """Return nominal stream duration as frame count divided by FPS."""
        return self.frame_count / self.fps


@dataclass(frozen=True)
class ExtractionResult:
    """Paths and counts produced by one successful extraction operation."""

    jpeg_paths: tuple[Path, ...]
    metadata_path: Path

    @property
    def jpeg_count(self) -> int:
        """Return the number of generated JPEG files."""
        return len(self.jpeg_paths)


def probe_video(video_path: Path | str) -> VideoMetadata:
    """Open a video with OpenCV and return validated stream metadata."""
    path = Path(video_path).expanduser().resolve()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoOpenError(f"OpenCV could not open input video: {path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        raw_frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not math.isfinite(raw_frame_count) or raw_frame_count < 1.0:
            raise VideoOpenError(
                f"OpenCV reported an invalid or empty frame count "
                f"({raw_frame_count}) for '{path}'."
            )
        if not math.isfinite(raw_width) or not math.isfinite(raw_height):
            raise VideoOpenError(
                f"OpenCV reported invalid dimensions for '{path}'."
            )

        return VideoMetadata(
            fps=fps,
            frame_count=int(raw_frame_count),
            width=int(raw_width),
            height=int(raw_height),
        )
    finally:
        capture.release()


class KeyframeExtractor:
    """Extract start, floor-middle, and inclusive-end JPEGs for every shot."""

    metadata_filename = "shots.json"

    def __init__(self, jpeg_quality: int = 95, *, overwrite: bool = False) -> None:
        """Configure JPEG encoding and output conflict behavior."""
        if (
            isinstance(jpeg_quality, bool)
            or not isinstance(jpeg_quality, int)
            or not 1 <= jpeg_quality <= 100
        ):
            raise ValueError(
                "JPEG quality must be an integer from 1 to 100; "
                f"received {jpeg_quality}."
            )
        self.jpeg_quality = jpeg_quality
        self.overwrite = overwrite

    def extract(
        self,
        video_path: Path | str,
        shots: Sequence[ShotBoundary],
        output_dir: Path | str,
        *,
        video_metadata: VideoMetadata,
        detector_threshold: float,
    ) -> ExtractionResult:
        """Extract all keyframes and atomically publish their JSON metadata.

        Frames and metadata are first written to temporary paths. The final
        ``shots.json`` is replaced only after every requested JPEG has been
        decoded, encoded, and moved into place successfully.
        """
        source = Path(video_path).expanduser().resolve()
        output = Path(output_dir).expanduser().resolve()
        if not source.exists() or not source.is_file():
            raise VideoOpenError(
                f"Input video does not exist or is not a regular file: {source}"
            )
        self._validate_shots(shots, video_metadata.frame_count)
        output.mkdir(parents=True, exist_ok=True)

        output_names = self._build_output_names(shots)
        final_jpegs = tuple(output / name for name in output_names.values())
        metadata_path = output / self.metadata_filename
        self._check_conflicts((*final_jpegs, metadata_path))

        targets = self._group_targets(shots)
        frames = self._decode_target_frames(source, tuple(sorted(targets)))
        metadata_temp: Path | None = None

        try:
            with tempfile.TemporaryDirectory(
                prefix=".keyframes-", dir=output
            ) as staging_name:
                staging_dir = Path(staging_name)
                for frame_index, references in targets.items():
                    frame = frames[frame_index]
                    for shot_id, position in references:
                        filename = output_names[(shot_id, position)]
                        self._write_jpeg(staging_dir / filename, frame, frame_index)

                payload = self._build_metadata(
                    source=source,
                    shots=shots,
                    output_names=output_names,
                    video_metadata=video_metadata,
                    detector_threshold=detector_threshold,
                )
                metadata_temp = self._write_metadata_temp(output, payload)

                for final_path in final_jpegs:
                    staged_path = staging_dir / final_path.name
                    try:
                        os.replace(staged_path, final_path)
                    except OSError as exc:
                        raise OutputWriteError(
                            f"Could not publish keyframe '{final_path}'."
                        ) from exc

                try:
                    os.replace(metadata_temp, metadata_path)
                except OSError as exc:
                    raise OutputWriteError(
                        f"Could not atomically publish metadata '{metadata_path}'."
                    ) from exc
                metadata_temp = None
        finally:
            if metadata_temp is not None:
                try:
                    metadata_temp.unlink(missing_ok=True)
                except OSError:
                    LOGGER.warning(
                        "Could not remove temporary metadata file '%s'.",
                        metadata_temp,
                    )

        return ExtractionResult(
            jpeg_paths=final_jpegs,
            metadata_path=metadata_path,
        )

    @staticmethod
    def _validate_shots(shots: Sequence[ShotBoundary], frame_count: int) -> None:
        if not shots:
            raise InvalidShotBoundaryError(
                "At least one shot is required for extraction."
            )
        previous_end = -1
        for expected_id, shot in enumerate(shots, start=1):
            if shot.shot_id != expected_id:
                raise InvalidShotBoundaryError(
                    f"Expected sequential shot ID {expected_id}, received "
                    f"{shot.shot_id}."
                )
            if shot.start_frame <= previous_end:
                raise InvalidShotBoundaryError(
                    f"Shot {shot.shot_id} overlaps or is unordered."
                )
            if shot.end_frame >= frame_count:
                raise InvalidShotBoundaryError(
                    f"Shot {shot.shot_id} ends at frame {shot.end_frame}, outside the "
                    f"valid range 0..{frame_count - 1}."
                )
            previous_end = shot.end_frame

    @staticmethod
    def _build_output_names(
        shots: Sequence[ShotBoundary],
    ) -> dict[tuple[int, str], str]:
        names: dict[tuple[int, str], str] = {}
        for shot in shots:
            for position in ("start", "middle", "end"):
                names[(shot.shot_id, position)] = (
                    f"shot_{shot.shot_id:03d}_{position}.jpg"
                )
        return names

    @staticmethod
    def _group_targets(
        shots: Sequence[ShotBoundary],
    ) -> dict[int, list[tuple[int, str]]]:
        targets: dict[int, list[tuple[int, str]]] = {}
        for shot in shots:
            positions = (
                ("start", shot.start_frame),
                ("middle", shot.middle_frame),
                ("end", shot.end_frame),
            )
            for position, frame_index in positions:
                targets.setdefault(frame_index, []).append((shot.shot_id, position))
        return targets

    def _check_conflicts(self, final_paths: Sequence[Path]) -> None:
        if self.overwrite:
            return
        conflicts = [path for path in final_paths if path.exists()]
        if conflicts:
            preview = ", ".join(str(path) for path in conflicts[:5])
            remainder = len(conflicts) - 5
            suffix = f" (and {remainder} more)" if remainder > 0 else ""
            raise OutputConflictError(
                "Refusing to overwrite existing generated output: "
                f"{preview}{suffix}. Use --overwrite to allow replacement."
            )

    def _decode_target_frames(
        self, video_path: Path, target_indices: Sequence[int]
    ) -> dict[int, NDArray[np.uint8]]:
        cache: dict[int, NDArray[np.uint8]] = {}
        fallback_targets: list[int] = []
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise VideoOpenError(
                    f"OpenCV could not open input video for extraction: {video_path}"
                )
            for frame_index in target_indices:
                frame = self._read_with_verified_seek(capture, frame_index)
                if frame is None:
                    fallback_targets.append(frame_index)
                else:
                    cache[frame_index] = frame.copy()
        finally:
            capture.release()

        if fallback_targets:
            LOGGER.debug(
                "Random access was unreliable for %d target frame(s); decoding "
                "sequentially from frame zero.",
                len(fallback_targets),
            )
            cache.update(
                self._decode_sequentially(video_path, tuple(fallback_targets))
            )

        missing = [index for index in target_indices if index not in cache]
        if missing:
            raise FrameExtractionError(
                f"Could not decode requested frame(s) {missing} from '{video_path}'."
            )
        return cache

    @staticmethod
    def _read_with_verified_seek(
        capture: cv2.VideoCapture, frame_index: int
    ) -> NDArray[np.uint8] | None:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index)):
            return None
        position_before = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
        success, frame = capture.read()
        position_after = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if not success or frame is None or frame.size == 0:
            return None

        before_matches = math.isfinite(position_before) and math.isclose(
            position_before, frame_index, abs_tol=0.25
        )
        after_matches = math.isfinite(position_after) and math.isclose(
            position_after, frame_index + 1, abs_tol=0.25
        )
        if not before_matches or not after_matches:
            return None
        return frame

    @staticmethod
    def _decode_sequentially(
        video_path: Path, target_indices: Sequence[int]
    ) -> dict[int, NDArray[np.uint8]]:
        requested = set(target_indices)
        decoded: dict[int, NDArray[np.uint8]] = {}
        maximum = max(requested)
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise VideoOpenError(
                    f"OpenCV could not reopen input video for sequential decoding: "
                    f"{video_path}"
                )
            frame_index = 0
            while frame_index <= maximum:
                success, frame = capture.read()
                if not success or frame is None or frame.size == 0:
                    unresolved = sorted(requested.difference(decoded))
                    raise FrameExtractionError(
                        f"Sequential decoding of '{video_path}' stopped at frame "
                        f"{frame_index}; requested frame(s) {unresolved} remain "
                        "unavailable."
                    )
                if frame_index in requested:
                    decoded[frame_index] = frame.copy()
                    if len(decoded) == len(requested):
                        break
                frame_index += 1
        finally:
            capture.release()
        return decoded

    def _write_jpeg(
        self, output_path: Path, frame: NDArray[np.uint8], frame_index: int
    ) -> None:
        try:
            written = cv2.imwrite(
                str(output_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
        except cv2.error as exc:
            raise OutputWriteError(
                f"OpenCV failed to encode requested frame {frame_index} as "
                f"'{output_path}'."
            ) from exc
        if not written:
            raise OutputWriteError(
                f"cv2.imwrite returned false for requested frame {frame_index} at "
                f"'{output_path}'."
            )

    @staticmethod
    def _build_metadata(
        *,
        source: Path,
        shots: Sequence[ShotBoundary],
        output_names: Mapping[tuple[int, str], str],
        video_metadata: VideoMetadata,
        detector_threshold: float,
    ) -> dict[str, object]:
        fps = video_metadata.fps
        if not math.isfinite(fps) or fps <= 0.0:
            raise OutputWriteError(
                f"Cannot calculate frame timestamps from invalid FPS value {fps}."
            )

        shot_records: list[dict[str, object]] = []
        for shot in shots:
            middle = shot.middle_frame
            shot_records.append(
                {
                    "shot_id": shot.shot_id,
                    "start_frame": shot.start_frame,
                    "middle_frame": middle,
                    "end_frame": shot.end_frame,
                    "start_time_seconds": shot.start_frame / fps,
                    "middle_time_seconds": middle / fps,
                    "end_time_seconds": shot.end_frame / fps,
                    "keyframes": {
                        "start": output_names[(shot.shot_id, "start")],
                        "middle": output_names[(shot.shot_id, "middle")],
                        "end": output_names[(shot.shot_id, "end")],
                    },
                }
            )

        return {
            "schema_version": "1.0",
            "source_video": str(source),
            "video": {
                "fps": video_metadata.fps,
                "frame_count": video_metadata.frame_count,
                "width": video_metadata.width,
                "height": video_metadata.height,
                "duration_seconds": video_metadata.duration_seconds,
            },
            "detector": {
                "name": "TransNetV2",
                "threshold": detector_threshold,
            },
            "shot_count": len(shots),
            "shots": shot_records,
        }

    @staticmethod
    def _write_metadata_temp(
        output_dir: Path, payload: Mapping[str, object]
    ) -> Path:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_dir,
                prefix=".shots-",
                suffix=".json.tmp",
                delete=False,
            ) as temporary_file:
                temp_path = Path(temporary_file.name)
                json.dump(
                    payload,
                    temporary_file,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
        except (OSError, TypeError, ValueError) as exc:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    LOGGER.warning(
                        "Could not remove failed metadata temporary file '%s'.",
                        temp_path,
                    )
            raise OutputWriteError(
                f"Could not serialize metadata in '{output_dir}'."
            ) from exc
        if temp_path is None:
            raise OutputWriteError(
                f"Metadata temporary file was not created in '{output_dir}'."
            )
        return temp_path
