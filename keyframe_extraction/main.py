"""Command-line entry point for TransNet V2 keyframe extraction."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Sequence

LOGGER = logging.getLogger("keyframe_extraction")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def threshold_value(raw_value: str) -> float:
    """Parse and validate a probability threshold for argparse."""
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "threshold must be a number between 0.0 and 1.0"
        ) from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(
            "threshold must be between 0.0 and 1.0 inclusive"
        )
    return value


def jpeg_quality_value(raw_value: str) -> int:
    """Parse and validate a JPEG quality value for argparse."""
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "JPEG quality must be an integer from 1 to 100"
        ) from exc
    if not 1 <= value <= 100:
        raise argparse.ArgumentTypeError(
            "JPEG quality must be an integer from 1 to 100"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without initializing the model."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect shots with TransNet V2 and extract start, middle, and end "
            "keyframes."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory in which JPEG files and shots.json will be written.",
    )
    parser.add_argument(
        "--threshold",
        type=threshold_value,
        default=0.5,
        help="TransNet V2 transition threshold (default: 0.5).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=jpeg_quality_value,
        default=95,
        help="JPEG encoding quality from 1 to 100 (default: 95).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing generated files.",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    return parser


def configure_logging(level_name: str) -> None:
    """Configure concise process-wide logging."""
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(levelname)s: %(message)s",
    )


def validate_paths(input_path: Path, output_path: Path) -> tuple[Path, Path]:
    """Resolve and validate CLI filesystem paths."""
    source = input_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if not source.exists():
        raise ValueError(f"Input video does not exist: {source}")
    if not source.is_file():
        raise ValueError(f"Input path is not a regular file: {source}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {output}")
    if source == output:
        raise ValueError("Input video and output directory cannot be the same path.")
    return source, output


def run(args: argparse.Namespace) -> int:
    """Execute one complete detection and extraction operation."""
    try:
        from src import KeyframeExtractor, TransNetV2ShotDetector
        from src.extractor import probe_video
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise RuntimeError(
            f"Missing Python dependency '{missing}'. Install requirements with "
            "`python -m pip install -r requirements.txt`. A system-level "
            "FFmpeg executable is also required for video inference."
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the keyframe extraction dependencies. Reinstall "
            "them with `python -m pip install -r requirements.txt`; a "
            "system-level FFmpeg executable is also required."
        ) from exc

    source, output = validate_paths(args.input, args.output)
    LOGGER.info("Inspecting video: %s", source)
    video_metadata = probe_video(source)

    LOGGER.info("Initializing TransNet V2.")
    detector = TransNetV2ShotDetector(threshold=args.threshold)
    LOGGER.info("Detecting shot boundaries.")
    shots = detector.detect(source, frame_count=video_metadata.frame_count)

    LOGGER.info("Extracting three keyframes for each of %d shots.", len(shots))
    extractor = KeyframeExtractor(
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
    )
    result = extractor.extract(
        source,
        shots,
        output,
        video_metadata=video_metadata,
        detector_threshold=detector.threshold,
    )

    print(
        "Keyframe extraction completed successfully.\n"
        f"Source video: {source}\n"
        f"Shots: {len(shots)}\n"
        f"JPEG files: {result.jpeg_count}\n"
        f"Output directory: {output}\n"
        f"Metadata: {result.metadata_path}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, execute the pipeline, and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    try:
        return run(args)
    except (RuntimeError, ValueError) as exc:
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            LOGGER.exception("Keyframe extraction failed: %s", exc)
        else:
            LOGGER.error("Keyframe extraction failed: %s", exc)
        return 1
    except Exception as exc:
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            LOGGER.exception("Unexpected keyframe extraction failure: %s", exc)
        else:
            LOGGER.error(
                "Unexpected keyframe extraction failure: %s. Re-run with "
                "--log-level DEBUG for a stack trace.",
                exc,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
