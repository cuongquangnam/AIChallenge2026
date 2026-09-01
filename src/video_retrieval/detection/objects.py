from __future__ import annotations

from typing import Any

from video_retrieval.config import Settings
from video_retrieval.models import FrameObjectDetections, KeyFrame, ObjectDetection


class ObjectDetector:
    """Detect COCO objects in keyframes using an optional Ultralytics YOLO backend."""

    def __init__(self, settings: Settings, *, model: Any | None = None):
        self.settings = settings
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "YOLO backend requires the 'ml' extra: pip install -e .[ml]"
                ) from exc
            self._model = YOLO(self.settings.object_model_id)
        return self._model

    def detect_keyframes(self, keyframes: list[KeyFrame]) -> list[FrameObjectDetections]:
        if not keyframes:
            return []
        backend = self.settings.object_backend.strip().lower()
        if backend == "mock":
            return [FrameObjectDetections(keyframe=keyframe) for keyframe in keyframes]
        if backend != "yolo":
            raise ValueError(f"Unsupported object backend: {self.settings.object_backend}")

        output: list[FrameObjectDetections] = []
        batch_size = max(1, int(self.settings.object_batch_size))
        for offset in range(0, len(keyframes), batch_size):
            batch = keyframes[offset : offset + batch_size]
            kwargs: dict[str, Any] = {
                "source": [str(keyframe.path) for keyframe in batch],
                "conf": self.settings.object_confidence,
                "iou": self.settings.object_iou,
                "verbose": False,
            }
            if self.settings.object_device.strip():
                kwargs["device"] = self.settings.object_device.strip()
            results = list(self.model.predict(**kwargs))
            if len(results) != len(batch):
                raise RuntimeError("YOLO returned a different number of results than input frames")
            output.extend(
                _parse_result(keyframe, result) for keyframe, result in zip(batch, results, strict=True)
            )
        return output


def _parse_result(keyframe: KeyFrame, result: Any) -> FrameObjectDetections:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return FrameObjectDetections(keyframe=keyframe)
    xyxy = _to_list(getattr(boxes, "xyxy", []))
    confidences = _to_list(getattr(boxes, "conf", []))
    class_ids = _to_list(getattr(boxes, "cls", []))
    names = getattr(result, "names", {})
    detections: list[ObjectDetection] = []
    for coords, confidence, class_id in zip(xyxy, confidences, class_ids, strict=True):
        class_index = int(class_id)
        label = names.get(class_index, str(class_index)) if isinstance(names, dict) else names[class_index]
        detections.append(
            ObjectDetection(
                label=str(label).strip().lower(),
                confidence=float(confidence),
                bbox_xyxy=tuple(float(value) for value in coords),
            )
        )
    return FrameObjectDetections(keyframe=keyframe, detections=detections)


def _to_list(value: Any) -> list:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)
