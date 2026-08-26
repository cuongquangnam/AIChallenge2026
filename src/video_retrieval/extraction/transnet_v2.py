"""TransNetV2 shot detector (PyTorch).

Prefer ``transnetv2-pytorch`` on Mac (MPS/CPU). Falls back to OpenCV when the
package is missing so ``SHOT_BACKEND=transnetv2`` still runs.
"""

from __future__ import annotations

from video_retrieval.extraction.shots import ShotSpan, detect_shots_opencv

_MODEL = None


def _pick_device(requested: str = "auto") -> str:
    import torch

    choice = (requested or "auto").strip().lower()
    if choice == "auto":
        # Prefer MPS on Apple Silicon for throughput; CUDA when available.
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if choice in {"mps", "cpu", "cuda"}:
        return choice
    raise ValueError(f"Unknown TRANSNET_DEVICE={requested!r}; expected auto|mps|cpu|cuda")


def _get_model(device: str = "auto"):
    global _MODEL
    try:
        from transnetv2_pytorch import TransNetV2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Install TransNetV2 PyTorch: pip install '.[ml]' "
            "(adds transnetv2-pytorch)."
        ) from exc

    resolved = _pick_device(device)
    if _MODEL is None or getattr(_MODEL, "_vr_device", None) != resolved:
        model = TransNetV2(device=resolved)
        model._vr_device = resolved  # type: ignore[attr-defined]
        _MODEL = model
        print(f"TransNetV2 ready on device={resolved}", flush=True)
    return _MODEL


def predict_shots(
    video_path: str,
    *,
    threshold: float = 0.5,
    device: str = "auto",
) -> list[ShotSpan]:
    """Detect shot spans with TransNetV2; OpenCV fallback if package missing."""
    try:
        model = _get_model(device)
    except ImportError as exc:
        print(f"[transnet] {exc}; falling back to OpenCV shot detect", flush=True)
        return detect_shots_opencv(video_path)

    with __import__("torch").no_grad():
        scenes = model.detect_scenes(video_path, threshold=threshold)

    shots: list[ShotSpan] = []
    for scene in scenes:
        # Package may return dicts or (start, end) tuples.
        if isinstance(scene, dict):
            start = scene.get("start_frame", scene.get("start"))
            end = scene.get("end_frame", scene.get("end"))
            if start is None and "start_time" in scene:
                # Some builds only expose times; skip incomplete rows.
                continue
        else:
            start, end = scene[0], scene[1]
        start_i = int(start)
        end_i = int(end)
        if end_i < start_i:
            continue
        shots.append(ShotSpan(start_frame=start_i, end_frame=end_i))

    if not shots:
        print("[transnet] no scenes returned; falling back to OpenCV", flush=True)
        return detect_shots_opencv(video_path)
    return shots
