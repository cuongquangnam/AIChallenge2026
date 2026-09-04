"""Resolve on-disk keyframe JPGs for rerank / QA / UI."""

from __future__ import annotations

from pathlib import Path

from video_retrieval.config import Settings


def resolve_keyframe_path(
    settings: Settings,
    *,
    video_id: str,
    keyframe_path: str | Path | None,
) -> Path | None:
    """Map an indexed ``keyframe_path`` to a file under ``DATA_DIR/keyframes``.

    Index payloads often store absolute paths from the machine that indexed.
    On Colab those paths are missing; we fall back to
    ``keyframes/{video_id}/{filename}``.
    """
    if not keyframe_path:
        return None
    raw = Path(str(keyframe_path))
    if raw.is_file():
        return raw

    name = raw.name
    candidates: list[Path] = []
    if video_id and name:
        candidates.append(settings.keyframes_dir / video_id / name)
    if name:
        candidates.append(settings.keyframes_dir / name)
    candidates.append(settings.data_dir / raw)
    if not raw.is_absolute():
        candidates.append(settings.keyframes_dir / raw)

    # Paths like .../keyframes/L01_V001/shot_0001_middle.jpg from another host.
    parts = raw.parts
    if "keyframes" in parts:
        idx = parts.index("keyframes")
        rel = Path(*parts[idx + 1 :])
        if rel.parts:
            candidates.append(settings.keyframes_dir / rel)

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None
