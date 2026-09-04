from __future__ import annotations

import time
import zipfile
from pathlib import Path

from video_retrieval.config import Settings
from video_retrieval.storage.qdrant_bootstrap import (
    DEFAULT_QDRANT_URL,
    collection_has_points,
    ensure_qdrant_server,
    find_qdrant_snapshot,
    recover_qdrant_snapshot,
    resolve_colab_qdrant_url,
)


def prepare_colab_data(
    settings: Settings,
    *,
    progress: bool = False,
) -> tuple[Settings, dict[str, object]]:
    """Post-process Drive pulls: unzip keyframes and restore Qdrant snapshots."""
    summary: dict[str, object] = {}

    zip_paths = _find_keyframes_zips(settings.keyframes_dir)
    if not zip_paths:
        root_zip = settings.data_dir / "keyframes.zip"
        if root_zip.is_file():
            zip_paths = [root_zip]
        zip_paths.extend(sorted(settings.data_dir.glob("keyframes_*.zip")))
    keyframes_result = extract_keyframes_zip(
        settings.keyframes_dir,
        zip_paths=zip_paths or None,
        progress=progress,
    )
    # Always normalize — zip may have been extracted earlier with nested
    # data_transnet/keyframes/… layout.
    hoisted = normalize_keyframes_layout(settings.keyframes_dir, progress=progress)
    if hoisted:
        keyframes_result = {
            **keyframes_result,
            "normalized": True,
            "hoisted_dirs": hoisted,
            "video_dirs": len(_video_subdirs(settings.keyframes_dir)),
        }
    summary["keyframes"] = keyframes_result

    qdrant_result, qdrant_url = prepare_colab_qdrant(settings, progress=progress)
    summary["qdrant"] = qdrant_result

    if qdrant_url != settings.qdrant_url:
        settings = settings.model_copy(update={"qdrant_url": qdrant_url})
    return settings, summary


def extract_keyframes_zip(
    keyframes_dir: Path,
    *,
    zip_paths: list[Path] | None = None,
    progress: bool = False,
    force: bool = False,
) -> dict[str, object]:
    """Extract keyframe zip archive(s) into ``keyframes_dir``.

    ``zip_paths`` may point at Drive-mounted archives; members are written
    locally so Colab does not need a second 40GB copy of the zip.

    If video folders already exist under ``keyframes_dir``, extraction is
    skipped unless ``force=True`` (re-extracting tens of GB is expensive).
    """
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    # Fix prior bad extracts (e.g. data_transnet/keyframes/…) before skip check.
    normalize_keyframes_layout(keyframes_dir, progress=progress)

    resolved_zips = list(zip_paths) if zip_paths is not None else _find_keyframes_zips(keyframes_dir)
    video_dirs = _video_subdirs(keyframes_dir)
    if video_dirs and not force:
        return {
            "action": "skip",
            "reason": "already_extracted",
            "video_dirs": len(video_dirs),
            "zips": [path.name for path in resolved_zips],
        }
    if not resolved_zips:
        return {"action": "skip", "reason": "no_zip"}

    total_members = 0
    for idx, zip_path in enumerate(resolved_zips, start=1):
        if progress:
            try:
                size_gb = zip_path.stat().st_size / (1024**3)
            except OSError:
                size_gb = 0.0
            print(
                f"[keyframes] extracting {idx}/{len(resolved_zips)} {zip_path} "
                f"({size_gb:.1f} GiB) -> {keyframes_dir} ...",
                flush=True,
            )
        with zipfile.ZipFile(zip_path) as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            total_members += len(members)
            started = time.monotonic()
            last_log = 0.0
            for member_idx, info in enumerate(members, start=1):
                archive.extract(info, keyframes_dir)
                if not progress:
                    continue
                now = time.monotonic()
                is_first = member_idx == 1
                is_last = member_idx == len(members)
                if is_first or is_last or (now - last_log) >= 5.0:
                    elapsed = now - started
                    rate = member_idx / elapsed if elapsed > 0 else 0.0
                    pct = 100.0 * member_idx / len(members) if members else 100.0
                    print(
                        f"[keyframes]   {zip_path.name}: {member_idx}/{len(members)} "
                        f"({pct:.1f}%) files  elapsed={elapsed:.0f}s  "
                        f"~{rate:.0f} files/s",
                        flush=True,
                    )
                    last_log = now

    hoisted = normalize_keyframes_layout(keyframes_dir, progress=progress)
    video_dirs = _video_subdirs(keyframes_dir)
    if progress:
        print(
            f"[keyframes] extracted {total_members} member(s); "
            f"{len(video_dirs)} video folder(s) under {keyframes_dir}"
            + (f"; hoisted {hoisted} nested dir(s)" if hoisted else ""),
            flush=True,
        )
    return {
        "action": "extracted",
        "zips": [path.name for path in resolved_zips],
        "members": total_members,
        "video_dirs": len(video_dirs),
        "hoisted_dirs": hoisted,
        "dest": str(keyframes_dir),
    }


def prepare_colab_qdrant(
    settings: Settings,
    *,
    progress: bool = False,
) -> tuple[dict[str, object], str]:
    """Start Qdrant server and recover ``.snapshot`` files when needed."""
    qdrant_url = resolve_colab_qdrant_url(settings)
    qdrant_dir = settings.qdrant_dir
    collection = settings.qdrant_collection
    snapshot = find_qdrant_snapshot(qdrant_dir, collection)

    if snapshot is None:
        if qdrant_url == "local":
            return {"action": "skip", "reason": "embedded_storage"}, qdrant_url
        client_url = qdrant_url or DEFAULT_QDRANT_URL
        ensure_qdrant_server(
            client_url,
            install_dir=Path(settings.colab_qdrant_install_dir),
            storage_path=settings.qdrant_dir / "storage",
            startup_timeout_sec=1800.0,
        )
        return {"action": "skip", "reason": "no_snapshot", "url": client_url}, client_url

    if collection not in snapshot.stem and progress:
        print(
            f"[qdrant] warning: snapshot {snapshot.name!r} does not mention "
            f"collection {collection!r}; check QDRANT_COLLECTION in .env",
            flush=True,
        )

    storage_path = settings.qdrant_dir / "storage"
    ensure_qdrant_server(
        DEFAULT_QDRANT_URL,
        install_dir=Path(settings.colab_qdrant_install_dir),
        storage_path=storage_path,
        startup_timeout_sec=1800.0,
    )

    from qdrant_client import QdrantClient

    client = QdrantClient(url=DEFAULT_QDRANT_URL, check_compatibility=False)
    if collection_has_points(client, collection):
        count = int(client.get_collection(collection).points_count or 0)
        return {
            "action": "skip",
            "reason": "collection_ready",
            "collection": collection,
            "points": count,
            "url": DEFAULT_QDRANT_URL,
        }, DEFAULT_QDRANT_URL

    snapshots_dir = Path(settings.colab_qdrant_install_dir) / "snapshots"
    install_dir = Path(settings.colab_qdrant_install_dir)
    points = recover_qdrant_snapshot(
        DEFAULT_QDRANT_URL,
        collection=collection,
        snapshot_path=snapshot,
        snapshots_dir=snapshots_dir,
        storage_path=storage_path,
        install_dir=install_dir,
        progress=progress,
    )
    return {
        "action": "recovered",
        "snapshot": snapshot.name,
        "collection": collection,
        "points": points,
        "url": DEFAULT_QDRANT_URL,
    }, DEFAULT_QDRANT_URL


def _find_keyframes_zips(keyframes_dir: Path) -> list[Path]:
    preferred = keyframes_dir / "keyframes.zip"
    if preferred.is_file():
        others = sorted(path for path in keyframes_dir.glob("*.zip") if path != preferred)
        return [preferred, *others]
    zips = sorted(keyframes_dir.glob("*.zip"))
    return zips


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _looks_like_video_keyframe_dir(path: Path) -> bool:
    """True if ``path`` is a per-video folder of JPG keyframes (not a nesting wrapper)."""
    if not path.is_dir():
        return False
    try:
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() in _IMAGE_SUFFIXES:
                return True
    except OSError:
        return False
    return False


def _video_subdirs(keyframes_dir: Path) -> list[Path]:
    ignored = {"__MACOSX"}
    if not keyframes_dir.is_dir():
        return []
    return [
        path
        for path in keyframes_dir.iterdir()
        if path.is_dir()
        and path.name not in ignored
        and not path.name.startswith(".")
        and _looks_like_video_keyframe_dir(path)
    ]


def normalize_keyframes_layout(keyframes_dir: Path, *, progress: bool = False) -> int:
    """Hoist nested video folders up to ``keyframes_dir``.

    Handles zip layouts like:
    - ``keyframes/L01_V001/...``
    - ``data_transnet/keyframes/L01_V001/...``
    - ``data/keyframes/L01_V001/...``
    """
    if not keyframes_dir.is_dir():
        return 0
    root = keyframes_dir.resolve()
    hoisted = 0

    for _ in range(6):
        nested_markers = sorted(
            (
                path
                for path in keyframes_dir.rglob("keyframes")
                if path.is_dir() and path.resolve() != root
            ),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        changed = False
        for nested in nested_markers:
            try:
                children = list(nested.iterdir())
            except OSError:
                continue
            for child in children:
                if not child.is_dir():
                    continue
                if child.name.startswith(".") or child.name == "__MACOSX":
                    continue
                if not _looks_like_video_keyframe_dir(child):
                    continue
                target = keyframes_dir / child.name
                if target.exists():
                    continue
                if progress:
                    print(
                        f"[keyframes] hoist {child.relative_to(keyframes_dir)} "
                        f"-> {child.name}",
                        flush=True,
                    )
                child.rename(target)
                hoisted += 1
                changed = True
            _remove_empty_parents(nested, stop=keyframes_dir)
        if not changed:
            break

    # One more pass: direct child wrapper named keyframes/
    _flatten_nested_keyframes_dir(keyframes_dir)
    return hoisted


def _remove_empty_parents(path: Path, *, stop: Path) -> None:
    current = path
    stop_resolved = stop.resolve()
    for _ in range(8):
        try:
            resolved = current.resolve()
        except OSError:
            break
        if resolved == stop_resolved or not current.is_dir():
            break
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _flatten_nested_keyframes_dir(keyframes_dir: Path) -> None:
    nested = keyframes_dir / "keyframes"
    if not nested.is_dir() or nested.resolve() == keyframes_dir.resolve():
        return
    for child in list(nested.iterdir()):
        target = keyframes_dir / child.name
        if target.exists():
            continue
        child.rename(target)
    try:
        nested.rmdir()
    except OSError:
        pass
