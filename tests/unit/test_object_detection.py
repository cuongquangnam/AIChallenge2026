from pathlib import Path
from types import SimpleNamespace

import pytest

from video_retrieval.detection.objects import ObjectDetector
from video_retrieval.models import (
    FrameObjectDetections,
    FrameRole,
    KeyFrame,
    ObjectDetection,
    ObjectRequirement,
    SearchHit,
)
from video_retrieval.search.object_filter import rerank_hits_by_objects
from tests.helpers import write_dummy_image


class _FakeYolo:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [
            SimpleNamespace(
                names={0: "person", 56: "chair"},
                boxes=SimpleNamespace(
                    xyxy=[[1.0, 2.0, 10.0, 20.0], [4.0, 5.0, 14.0, 25.0]],
                    conf=[0.9, 0.8],
                    cls=[0.0, 56.0],
                ),
            )
            for _ in kwargs["source"]
        ]


@pytest.mark.unit
def test_yolo_detector_parses_batch_results(settings, tmp_path: Path) -> None:
    settings.object_backend = "yolo"
    settings.object_batch_size = 2
    model = _FakeYolo()
    detector = ObjectDetector(settings, model=model)
    keyframes = [
        KeyFrame(
            video_id="clip",
            shot_index=index,
            role=FrameRole.MIDDLE,
            frame_index=index * 10,
            timestamp_sec=float(index),
            path=write_dummy_image(tmp_path / f"frame-{index}.jpg"),
        )
        for index in range(3)
    ]

    frames = detector.detect_keyframes(keyframes)

    assert len(model.calls) == 2
    assert [frame.counts for frame in frames] == [
        {"person": 1, "chair": 1},
        {"person": 1, "chair": 1},
        {"person": 1, "chair": 1},
    ]


@pytest.mark.unit
def test_object_rerank_boosts_matching_indexed_frame_but_keeps_unknown() -> None:
    requirements = [ObjectRequirement(label="person", min_count=2)]
    hits = [
        SearchHit(
            video_id="missing",
            score=1.0,
            source="mixed",
            payload={"objects_indexed": True, "object_counts": {"person": 0}},
        ),
        SearchHit(
            video_id="match",
            score=0.95,
            source="mixed",
            payload={"objects_indexed": True, "object_counts": {"person": 2}},
        ),
        SearchHit(video_id="legacy", score=0.9, source="mixed"),
    ]

    reranked = rerank_hits_by_objects(hits, requirements, boost=0.2, penalty=0.2)

    assert [hit.video_id for hit in reranked] == ["match", "legacy", "missing"]
    assert reranked[0].payload["object_match_ratio"] == 1.0
    assert "object_match_ratio" not in reranked[1].payload


def fake_person_detections(keyframes: list[KeyFrame]) -> list[FrameObjectDetections]:
    return [
        FrameObjectDetections(
            keyframe=keyframe,
            detections=[
                ObjectDetection(
                    label="person",
                    confidence=0.9,
                    bbox_xyxy=(1.0, 2.0, 10.0, 20.0),
                )
            ],
        )
        for keyframe in keyframes
    ]
