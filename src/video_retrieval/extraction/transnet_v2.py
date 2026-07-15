"""Optional TransNetV2 shot detector.

Install separately if you want the real model, e.g.:
  pip install git+https://github.com/soCzech/TransNetV2.git
"""

from __future__ import annotations

from video_retrieval.extraction.shots import ShotSpan


def predict_shots(video_path: str) -> list[ShotSpan]:
    try:
        from transnetv2 import TransNetV2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TransNetV2 is not installed. Use SHOT_BACKEND=opencv or install TransNetV2."
        ) from exc

    model = TransNetV2()
    video_frames, single_frame_predictions, _ = model.predict_video(video_path)
    scenes = model.predictions_to_scenes(single_frame_predictions)

    shots: list[ShotSpan] = []
    for start, end in scenes:
        shots.append(ShotSpan(start_frame=int(start), end_frame=int(end)))
    if not shots and len(video_frames) > 0:
        shots.append(ShotSpan(start_frame=0, end_frame=len(video_frames) - 1))
    return shots
