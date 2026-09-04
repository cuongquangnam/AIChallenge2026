"""Resolve on-disk keyframe JPGs for rerank / QA / UI."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from video_retrieval.config import Settings

logger = logging.getLogger(__name__)

_MIN_LOG_INTERVAL_SEC = 5.0
_last_log_monotonic = 0.0
_ok_since_log = 0
_miss_since_log = 0
_last_detail = ""


def log_keyframe_access(
    *,
    source: str,
    video_id: str,
    path: Path | None,
    keyframe_path: str | Path | None = None,
    force: bool = False,
) -> None:
    """Throttle keyframe load logs to at least ``_MIN_LOG_INTERVAL_SEC`` apart."""
    global _last_log_monotonic, _ok_since_log, _miss_since_log, _last_detail

    if path is not None:
        _ok_since_log += 1
        _last_detail = f"{source} ok video_id={video_id} path={path}"
    else:
        _miss_since_log += 1
        _last_detail = (
            f"{source} miss video_id={video_id} keyframe_path={keyframe_path!s}"
        )

    now = time.monotonic()
    if not force and (now - _last_log_monotonic) < _MIN_LOG_INTERVAL_SEC:
        return

    message = (
        f"[keyframes] load ok={_ok_since_log} miss={_miss_since_log} "
        f"since last log; {_last_detail}"
    )
    logger.info(message)
    print(message, flush=True)
    _ok_since_log = 0
    _miss_since_log = 0
    _last_log_monotonic = now


def resolve_keyframe_path(
    settings: Settings,
    *,
    video_id: str,
    keyframe_path: str | Path | None,
    source: str = "resolve",
    log: bool = True,
) -> Path | None:
    """Map an indexed ``keyframe_path`` to a file under ``DATA_DIR/keyframes``.

    Index payloads often store absolute paths from the machine that indexed.
    On Colab those paths are missing; we fall back to
    ``keyframes/{video_id}/{filename}``.
    """
    if not keyframe_path:
        if log:
            log_keyframe_access(
                source=source,
                video_id=video_id,
                path=None,
                keyframe_path=keyframe_path,
            )
        return None
    raw = Path(str(keyframe_path))
    if raw.is_file():
        if log:
            log_keyframe_access(source=source, video_id=video_id, path=raw)
        return raw

    name = raw.name
    candidates: list[Path] = []
    if video_id and name:
        candidates.append(settings.keyframes_dir / video_id / name)
        # Zip often contains data_transnet/keyframes/... before normalize.
        candidates.append(
            settings.keyframes_dir / "data_transnet" / "keyframes" / video_id / name
        )
        candidates.append(settings.keyframes_dir / "data" / "keyframes" / video_id / name)
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
            if log:
                log_keyframe_access(source=source, video_id=video_id, path=resolved)
            return resolved

    if log:
        log_keyframe_access(
            source=source,
            video_id=video_id,
            path=None,
            keyframe_path=keyframe_path,
        )
    return None
