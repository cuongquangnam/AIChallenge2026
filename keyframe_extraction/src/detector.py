"""TransNet V2 model integration and inclusive shot-boundary validation."""

from __future__ import annotations

import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence, cast

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


class TransNetV2DependencyError(RuntimeError):
    """Raised when TransNet V2 or one of its runtime dependencies is absent."""


class ModelInitializationError(RuntimeError):
    """Raised when the pretrained TransNet V2 model cannot be initialized."""


class ShotDetectionError(RuntimeError):
    """Raised when video inference or prediction validation fails."""


class InvalidShotBoundaryError(ValueError):
    """Raised when a shot boundary violates the inclusive boundary contract."""


@dataclass(frozen=True)
class ShotBoundary:
    """An immutable inclusive shot range.

    Shot identifiers are one-based; frame indices are zero-based.
    """

    shot_id: int
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        """Validate identifier and inclusive frame indices."""
        values = (self.shot_id, self.start_frame, self.end_frame)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise InvalidShotBoundaryError(
                "Shot ID and frame boundaries must all be integers."
            )
        if self.shot_id < 1:
            raise InvalidShotBoundaryError(
                f"Shot ID must be one-based and positive; received {self.shot_id}."
            )
        if self.start_frame < 0 or self.end_frame < 0:
            raise InvalidShotBoundaryError(
                "Frame indices must be non-negative; received "
                f"[{self.start_frame}, {self.end_frame}] for shot {self.shot_id}."
            )
        if self.end_frame < self.start_frame:
            raise InvalidShotBoundaryError(
                "Inclusive end frame cannot precede start frame; received "
                f"[{self.start_frame}, {self.end_frame}] for shot {self.shot_id}."
            )

    @property
    def middle_frame(self) -> int:
        """Return the floor-based middle frame index."""
        return (self.start_frame + self.end_frame) // 2

    @property
    def frame_count(self) -> int:
        """Return the number of frames in this inclusive range."""
        return self.end_frame - self.start_frame + 1


class _TransNetV2Model(Protocol):
    """Structural type for the official external inference object."""

    predict_video: Callable[[str], Sequence[object]]
    predictions_to_scenes: Callable[..., object]


class TransNetV2ShotDetector:
    """Detect inclusive shot boundaries with the official TransNet V2 model."""

    name = "TransNetV2"

    def __init__(
        self,
        threshold: float = 0.5,
        *,
        model: _TransNetV2Model | None = None,
    ) -> None:
        """Initialize one reusable model instance.

        Args:
            threshold: Transition probability threshold in the closed interval
                ``[0.0, 1.0]``.
            model: Optional compatible model object, primarily for controlled
                integration testing. When omitted, the official pretrained model
                is loaded immediately and exactly once for this detector.
        """
        self.threshold = self._validate_threshold(threshold)
        self._model = model if model is not None else self._load_model()

    @staticmethod
    def _validate_threshold(threshold: float) -> float:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("Prediction threshold must be a numeric value.")
        validated = float(threshold)
        if not math.isfinite(validated) or not 0.0 <= validated <= 1.0:
            raise ValueError(
                "Prediction threshold must be between 0.0 and 1.0; "
                f"received {threshold}."
            )
        return validated

    @staticmethod
    def _dependency_guidance(dependency: str) -> str:
        return (
            f"Missing or unusable dependency '{dependency}'. Install this project's "
            "dependencies with `python -m pip install -r requirements.txt`. The "
            "system-level FFmpeg executable must also be installed and available "
            "on PATH for TransNet V2 video inference."
        )

    def _load_model(self) -> _TransNetV2Model:
        try:
            from transnetv2 import TransNetV2
        except ModuleNotFoundError as exc:
            missing = exc.name or "transnetv2"
            raise TransNetV2DependencyError(
                self._dependency_guidance(missing)
            ) from exc
        except ImportError as exc:
            raise TransNetV2DependencyError(
                self._dependency_guidance("transnetv2/TensorFlow")
            ) from exc

        try:
            model = TransNetV2()
        except Exception as exc:
            raise ModelInitializationError(
                "TransNet V2 was imported but its pretrained model could not be "
                "initialized. Reinstall with `python -m pip install -r "
                "requirements.txt`, ensure the packaged Git LFS model weights are "
                "present, and ensure the system-level FFmpeg executable is "
                "installed."
            ) from exc
        return cast(_TransNetV2Model, model)

    def detect(
        self, video_path: Path | str, *, frame_count: int | None = None
    ) -> list[ShotBoundary]:
        """Run inference and return validated inclusive shot boundaries.

        Args:
            video_path: Path to a readable input video.
            frame_count: Optional independently probed upper bound. When supplied,
                boundaries are also constrained to this many OpenCV-visible frames.

        Raises:
            TransNetV2DependencyError: If FFmpeg is not installed.
            ShotDetectionError: If inference output is empty or malformed.
            InvalidShotBoundaryError: If scene conversion returns invalid ranges.
        """
        path = Path(video_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise ShotDetectionError(
                f"Input video does not exist or is not a regular file: {path}"
            )
        if frame_count is not None and (
            isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or frame_count < 1
        ):
            raise ShotDetectionError(
                f"Frame count must be a positive integer; received {frame_count}."
            )
        if shutil.which("ffmpeg") is None:
            raise TransNetV2DependencyError(self._dependency_guidance("ffmpeg"))

        try:
            raw_result = self._model.predict_video(str(path))
        except ModuleNotFoundError as exc:
            missing = exc.name or "ffmpeg-python"
            raise TransNetV2DependencyError(
                self._dependency_guidance(missing)
            ) from exc
        except Exception as exc:
            raise ShotDetectionError(
                f"TransNet V2 inference failed for '{path}'. Confirm that the video "
                "is readable, the model weights are intact, and FFmpeg can decode "
                "the file."
            ) from exc

        video_frames, single_predictions, all_predictions = self._validate_result(
            raw_result, path
        )
        decoded_frame_count = int(video_frames.shape[0])
        valid_frame_count = (
            min(decoded_frame_count, frame_count)
            if frame_count is not None
            else decoded_frame_count
        )
        if frame_count is not None and frame_count != decoded_frame_count:
            LOGGER.warning(
                "OpenCV reported %d frames while TransNet V2/FFmpeg decoded %d; "
                "shot boundaries will be limited to %d frames.",
                frame_count,
                decoded_frame_count,
                valid_frame_count,
            )

        try:
            raw_scenes = self._model.predictions_to_scenes(
                single_predictions, threshold=self.threshold
            )
        except Exception as exc:
            raise ShotDetectionError(
                f"TransNet V2 could not convert predictions into scenes for '{path}'."
            ) from exc

        scenes = self._validate_scenes(raw_scenes, valid_frame_count, path)
        shots = [
            ShotBoundary(
                shot_id=index,
                start_frame=int(scene[0]),
                end_frame=int(scene[1]),
            )
            for index, scene in enumerate(scenes, start=1)
        ]
        self._validate_shot_sequence(shots, valid_frame_count, path)
        return shots

    @staticmethod
    def _validate_result(
        raw_result: object, video_path: Path
    ) -> tuple[
        NDArray[np.uint8], NDArray[np.floating], NDArray[np.floating]
    ]:
        if not isinstance(raw_result, (tuple, list)) or len(raw_result) != 3:
            raise ShotDetectionError(
                "TransNet V2 returned an unexpected result for "
                f"'{video_path}'; expected (frames, single_predictions, "
                "all_predictions)."
            )

        frames = np.asarray(raw_result[0])
        if (
            frames.ndim != 4
            or frames.shape[0] < 1
            or frames.shape[1:] != (27, 48, 3)
            or frames.dtype != np.uint8
        ):
            raise ShotDetectionError(
                f"TransNet V2 decoded no usable frames from '{video_path}'. The "
                f"returned frame tensor had shape {frames.shape}."
            )

        single = TransNetV2ShotDetector._validate_prediction_vector(
            raw_result[1], "single-frame", frames.shape[0], video_path
        )
        all_frames = TransNetV2ShotDetector._validate_prediction_vector(
            raw_result[2], "all-frame", frames.shape[0], video_path
        )
        return frames, single, all_frames

    @staticmethod
    def _validate_prediction_vector(
        raw_predictions: object,
        label: str,
        expected_length: int,
        video_path: Path,
    ) -> NDArray[np.floating]:
        predictions = np.asarray(raw_predictions)
        if predictions.ndim != 1 or predictions.shape[0] != expected_length:
            raise ShotDetectionError(
                f"Invalid {label} TransNet V2 predictions for '{video_path}': "
                f"expected a vector of length {expected_length}, received shape "
                f"{predictions.shape}."
            )
        if not np.issubdtype(predictions.dtype, np.number):
            raise ShotDetectionError(
                f"Invalid {label} TransNet V2 predictions for '{video_path}': "
                "values are not numeric."
            )
        numeric = predictions.astype(np.float32, copy=False)
        if not np.all(np.isfinite(numeric)):
            raise ShotDetectionError(
                f"Invalid {label} TransNet V2 predictions for '{video_path}': "
                "values contain NaN or infinity."
            )
        if np.any(numeric < 0.0) or np.any(numeric > 1.0):
            raise ShotDetectionError(
                f"Invalid {label} TransNet V2 predictions for '{video_path}': "
                "probabilities must be within [0.0, 1.0]."
            )
        return numeric

    @staticmethod
    def _validate_scenes(
        raw_scenes: object, frame_count: int, video_path: Path
    ) -> NDArray[np.int64]:
        scenes = np.asarray(raw_scenes)
        if scenes.size == 0:
            return np.array([[0, frame_count - 1]], dtype=np.int64)
        if scenes.ndim != 2 or scenes.shape[1] != 2:
            raise ShotDetectionError(
                f"TransNet V2 returned malformed scenes for '{video_path}': "
                f"expected shape (n, 2), received {scenes.shape}."
            )
        if not np.issubdtype(scenes.dtype, np.number):
            raise ShotDetectionError(
                f"TransNet V2 returned non-numeric scene boundaries for '{video_path}'."
            )
        numeric_scenes = scenes.astype(np.float64, copy=False)
        if not np.all(np.isfinite(numeric_scenes)) or not np.all(
            numeric_scenes == np.floor(numeric_scenes)
        ):
            raise ShotDetectionError(
                f"TransNet V2 returned non-integer scene boundaries for '{video_path}'."
            )

        inclusive = numeric_scenes.astype(np.int64)
        if np.any(inclusive[:, 0] < 0) or np.any(inclusive[:, 1] < 0):
            raise InvalidShotBoundaryError(
                f"TransNet V2 returned a negative scene boundary for '{video_path}'."
            )
        if np.any(inclusive[:, 1] < inclusive[:, 0]):
            raise InvalidShotBoundaryError(
                "TransNet V2 returned an end frame before its start for "
                f"'{video_path}'."
            )
        if np.any(inclusive[1:, 0] <= inclusive[:-1, 0]):
            raise InvalidShotBoundaryError(
                f"TransNet V2 returned unordered scene starts for '{video_path}'."
            )
        if np.any(inclusive[1:, 0] <= inclusive[:-1, 1]):
            raise InvalidShotBoundaryError(
                f"TransNet V2 returned overlapping scenes for '{video_path}'."
            )
        if np.any(inclusive[:, 0] >= frame_count):
            raise InvalidShotBoundaryError(
                f"TransNet V2 returned a scene starting beyond the {frame_count} "
                f"decoded frames in '{video_path}'."
            )

        inclusive[:, 1] = np.minimum(inclusive[:, 1], frame_count - 1)
        return inclusive

    @staticmethod
    def _validate_shot_sequence(
        shots: Sequence[ShotBoundary], frame_count: int, video_path: Path
    ) -> None:
        if not shots:
            raise ShotDetectionError(f"No shots were produced for '{video_path}'.")
        previous_end = -1
        for expected_id, shot in enumerate(shots, start=1):
            if shot.shot_id != expected_id:
                raise InvalidShotBoundaryError(
                    f"Shot IDs are not sequential in '{video_path}'."
                )
            if shot.start_frame <= previous_end:
                raise InvalidShotBoundaryError(
                    f"Shot {shot.shot_id} overlaps its predecessor in '{video_path}'."
                )
            if shot.end_frame >= frame_count:
                raise InvalidShotBoundaryError(
                    f"Shot {shot.shot_id} ends beyond frame {frame_count - 1} in "
                    f"'{video_path}'."
                )
            previous_end = shot.end_frame
