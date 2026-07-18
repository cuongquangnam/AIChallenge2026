"""Verify keyframe extraction metadata against a video and output directory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import cv2


class VerificationError(RuntimeError):
    """Raised when generated metadata or keyframe output is inconsistent."""


def build_parser() -> argparse.ArgumentParser:
    """Create the output verification command-line parser."""
    parser = argparse.ArgumentParser(
        description="Validate shots.json and all generated keyframe JPEGs."
    )
    parser.add_argument("--video", required=True, type=Path, help="Source video path.")
    parser.add_argument(
        "--output", required=True, type=Path, help="Extraction output directory."
    )
    return parser


def require_mapping(value: object, context: str) -> Mapping[str, object]:
    """Return a JSON object or raise a contextual validation error."""
    if not isinstance(value, dict):
        raise VerificationError(f"{context} must be a JSON object.")
    if not all(isinstance(key, str) for key in value):
        raise VerificationError(f"{context} contains a non-string key.")
    return value


def require_list(value: object, context: str) -> list[object]:
    """Return a JSON array or raise a contextual validation error."""
    if not isinstance(value, list):
        raise VerificationError(f"{context} must be a JSON array.")
    return value


def require_int(value: object, context: str, *, minimum: int = 0) -> int:
    """Return a bounded JSON integer, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerificationError(f"{context} must be an integer.")
    if value < minimum:
        raise VerificationError(f"{context} must be at least {minimum}; got {value}.")
    return value


def require_number(value: object, context: str, *, positive: bool = False) -> float:
    """Return a finite JSON number with optional positivity validation."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationError(f"{context} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise VerificationError(f"{context} must be finite.")
    if positive and numeric <= 0.0:
        raise VerificationError(f"{context} must be greater than zero.")
    return numeric


def probe_video(video_path: Path) -> tuple[float, int, int, int]:
    """Read video properties and guarantee release of the VideoCapture."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VerificationError(f"OpenCV cannot open source video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if capture.isOpened():
        raise VerificationError(f"VideoCapture did not release cleanly: {video_path}")
    if not math.isfinite(fps) or fps <= 0.0:
        raise VerificationError(f"Video reports invalid FPS {fps}: {video_path}")
    if frame_count < 1 or width < 1 or height < 1:
        raise VerificationError(
            "Video reports invalid frame count or dimensions: "
            f"frames={frame_count}, size={width}x{height}."
        )
    return fps, frame_count, width, height


def load_metadata(metadata_path: Path) -> Mapping[str, object]:
    """Load and validate the top-level metadata JSON object."""
    if not metadata_path.is_file():
        raise VerificationError(f"Metadata file does not exist: {metadata_path}")
    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            payload: object = json.load(metadata_file)
    except json.JSONDecodeError as exc:
        raise VerificationError(
            f"Metadata is not valid JSON: {metadata_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise VerificationError(f"Cannot read metadata: {metadata_path}") from exc
    return require_mapping(payload, "metadata root")


def validate_video_metadata(
    payload: Mapping[str, object],
    source_video: Path,
    actual_video: tuple[float, int, int, int],
) -> tuple[float, int]:
    """Validate recorded source identity and stream properties."""
    recorded_source = payload.get("source_video")
    if not isinstance(recorded_source, str):
        raise VerificationError("source_video must be a string.")
    if Path(recorded_source).expanduser().resolve() != source_video:
        raise VerificationError(
            f"source_video does not match --video: {recorded_source}"
        )

    video = require_mapping(payload.get("video"), "video")
    metadata_fps = require_number(video.get("fps"), "video.fps", positive=True)
    metadata_count = require_int(
        video.get("frame_count"), "video.frame_count", minimum=1
    )
    metadata_width = require_int(video.get("width"), "video.width", minimum=1)
    metadata_height = require_int(video.get("height"), "video.height", minimum=1)
    duration = require_number(
        video.get("duration_seconds"), "video.duration_seconds", positive=True
    )

    actual_fps, actual_count, actual_width, actual_height = actual_video
    if not math.isclose(metadata_fps, actual_fps, rel_tol=1e-6, abs_tol=1e-6):
        raise VerificationError(
            f"Metadata FPS {metadata_fps} does not match video FPS {actual_fps}."
        )
    if metadata_count != actual_count:
        raise VerificationError(
            f"Metadata frame count {metadata_count} does not match video frame "
            f"count {actual_count}."
        )
    if (metadata_width, metadata_height) != (actual_width, actual_height):
        raise VerificationError(
            f"Metadata dimensions {metadata_width}x{metadata_height} do not match "
            f"video dimensions {actual_width}x{actual_height}."
        )
    expected_duration = metadata_count / metadata_fps
    if not math.isclose(duration, expected_duration, rel_tol=1e-9, abs_tol=1e-9):
        raise VerificationError(
            f"Metadata duration {duration} does not match {expected_duration}."
        )
    return metadata_fps, metadata_count


def validate_shots(
    payload: Mapping[str, object],
    output_dir: Path,
    fps: float,
    frame_count: int,
) -> tuple[int, set[Path]]:
    """Validate every shot, timestamp, filename, and readable JPEG."""
    shot_count = require_int(payload.get("shot_count"), "shot_count", minimum=1)
    shots = require_list(payload.get("shots"), "shots")
    if len(shots) != shot_count:
        raise VerificationError(
            f"shot_count is {shot_count}, but shots contains {len(shots)} records."
        )

    expected_positions = ("start", "middle", "end")
    recorded_jpegs: set[Path] = set()
    previous_end = -1
    for expected_id, raw_shot in enumerate(shots, start=1):
        context = f"shots[{expected_id - 1}]"
        shot = require_mapping(raw_shot, context)
        shot_id = require_int(shot.get("shot_id"), f"{context}.shot_id", minimum=1)
        if shot_id != expected_id:
            raise VerificationError(
                f"{context}.shot_id must be sequential and one-based; expected "
                f"{expected_id}, got {shot_id}."
            )

        start = require_int(shot.get("start_frame"), f"{context}.start_frame")
        middle = require_int(shot.get("middle_frame"), f"{context}.middle_frame")
        end = require_int(shot.get("end_frame"), f"{context}.end_frame")
        if expected_id == 1 and start != 0:
            raise VerificationError("The first zero-based shot must start at frame 0.")
        if not start <= middle <= end:
            raise VerificationError(
                f"{context} violates start <= middle <= inclusive end: "
                f"{start}, {middle}, {end}."
            )
        if middle != (start + end) // 2:
            raise VerificationError(
                f"{context}.middle_frame is {middle}; expected {(start + end) // 2}."
            )
        if start <= previous_end:
            raise VerificationError(f"{context} overlaps or is not ordered.")
        if end >= frame_count:
            raise VerificationError(
                f"{context}.end_frame {end} is outside 0..{frame_count - 1}."
            )
        previous_end = end

        for position, frame_index in zip(
            expected_positions, (start, middle, end)
        ):
            timestamp_key = f"{position}_time_seconds"
            timestamp = require_number(
                shot.get(timestamp_key), f"{context}.{timestamp_key}"
            )
            expected_timestamp = frame_index / fps
            if not math.isclose(
                timestamp, expected_timestamp, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise VerificationError(
                    f"{context}.{timestamp_key} is {timestamp}; expected "
                    f"{expected_timestamp}."
                )

        keyframes = require_mapping(shot.get("keyframes"), f"{context}.keyframes")
        if set(keyframes) != set(expected_positions):
            raise VerificationError(
                f"{context}.keyframes must contain exactly start, middle, and end."
            )
        for position in expected_positions:
            filename = keyframes[position]
            if not isinstance(filename, str) or not filename:
                raise VerificationError(
                    f"{context}.keyframes.{position} must be a filename."
                )
            filename_path = Path(filename)
            if filename_path.is_absolute() or len(filename_path.parts) != 1:
                raise VerificationError(
                    f"{context}.keyframes.{position} must be a local basename."
                )
            expected_name = f"shot_{shot_id:03d}_{position}.jpg"
            if filename != expected_name:
                raise VerificationError(
                    f"{context}.keyframes.{position} is '{filename}'; expected "
                    f"'{expected_name}'."
                )
            jpeg_path = output_dir / filename
            if jpeg_path in recorded_jpegs:
                raise VerificationError(f"Duplicate keyframe filename: {filename}")
            if not jpeg_path.is_file():
                raise VerificationError(
                    f"Recorded keyframe does not exist: {jpeg_path}"
                )
            image = cv2.imread(str(jpeg_path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise VerificationError(f"OpenCV cannot read keyframe: {jpeg_path}")
            recorded_jpegs.add(jpeg_path)

    return shot_count, recorded_jpegs


def validate_output(video_path: Path, output_dir: Path) -> tuple[int, int]:
    """Validate a complete extraction output and return shot/JPEG counts."""
    source_video = video_path.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    if not source_video.is_file():
        raise VerificationError(f"Source video does not exist: {source_video}")
    if not output.is_dir():
        raise VerificationError(f"Output directory does not exist: {output}")

    metadata_path = output / "shots.json"
    payload = load_metadata(metadata_path)
    actual_video = probe_video(source_video)
    fps, frame_count = validate_video_metadata(payload, source_video, actual_video)
    shot_count, recorded_jpegs = validate_shots(
        payload, output, fps, frame_count
    )

    actual_jpegs = set(output.glob("*.jpg"))
    expected_jpeg_count = shot_count * 3
    if len(recorded_jpegs) != expected_jpeg_count:
        raise VerificationError(
            f"Metadata records {len(recorded_jpegs)} JPEGs; expected "
            f"{expected_jpeg_count}."
        )
    if actual_jpegs != recorded_jpegs:
        extra = sorted(str(path.name) for path in actual_jpegs - recorded_jpegs)
        missing = sorted(str(path.name) for path in recorded_jpegs - actual_jpegs)
        raise VerificationError(
            f"JPEG file set differs from metadata; extra={extra}, missing={missing}."
        )

    temporary_metadata = list(output.glob(".shots-*.json.tmp"))
    staging_paths = list(output.glob(".keyframes-*"))
    if temporary_metadata or staging_paths:
        leftovers = [
            path.name for path in (*temporary_metadata, *staging_paths)
        ]
        raise VerificationError(f"Temporary output remains: {leftovers}")
    return shot_count, len(actual_jpegs)


def main(argv: Sequence[str] | None = None) -> int:
    """Run verification and return zero only for a consistent output."""
    args = build_parser().parse_args(argv)
    try:
        shot_count, jpeg_count = validate_output(args.video, args.output)
    except VerificationError as exc:
        print(f"Verification failed: {exc}")
        return 1
    print(
        f"Verification passed: {shot_count} shot(s), {jpeg_count} readable JPEG(s), "
        "valid metadata, timestamps, and released video resources."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
