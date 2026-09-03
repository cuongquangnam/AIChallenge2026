from __future__ import annotations

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

    keyframes_result = extract_keyframes_zip(settings.keyframes_dir, progress=progress)
    summary["keyframes"] = keyframes_result

    qdrant_result, qdrant_url = prepare_colab_qdrant(settings, progress=progress)
    summary["qdrant"] = qdrant_result

    if qdrant_url != settings.qdrant_url:
        settings = settings.model_copy(update={"qdrant_url": qdrant_url})
    return settings, summary


def extract_keyframes_zip(keyframes_dir: Path, *, progress: bool = False) -> dict[str, object]:
    """Extract keyframe zip archive(s) into ``keyframes_dir``."""
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    zip_paths = _find_keyframes_zips(keyframes_dir)
    if not zip_paths:
        video_dirs = _video_subdirs(keyframes_dir)
        if video_dirs:
            return {
                "action": "skip",
                "reason": "already_extracted",
                "video_dirs": len(video_dirs),
            }
        return {"action": "skip", "reason": "no_zip"}

    video_dirs = _video_subdirs(keyframes_dir)
    if video_dirs and not any(_zip_is_newer(zip_path, keyframes_dir) for zip_path in zip_paths):
        return {
            "action": "skip",
            "reason": "already_extracted",
            "video_dirs": len(video_dirs),
            "zips": [path.name for path in zip_paths],
        }

    total_members = 0
    for idx, zip_path in enumerate(zip_paths, start=1):
        if progress:
            print(
                f"[keyframes] extracting {idx}/{len(zip_paths)} {zip_path.name} "
                f"-> {keyframes_dir} ...",
                flush=True,
            )
        with zipfile.ZipFile(zip_path) as archive:
            members = archive.namelist()
            total_members += len(members)
            archive.extractall(keyframes_dir)

    _flatten_nested_keyframes_dir(keyframes_dir)
    video_dirs = _video_subdirs(keyframes_dir)
    if progress:
        print(
            f"[keyframes] extracted {total_members} member(s); "
            f"{len(video_dirs)} video folder(s)",
            flush=True,
        )
    return {
        "action": "extracted",
        "zips": [path.name for path in zip_paths],
        "members": total_members,
        "video_dirs": len(video_dirs),
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
    )

    from qdrant_client import QdrantClient

    client = QdrantClient(url=DEFAULT_QDRANT_URL)
    if collection_has_points(client, collection):
        count = int(client.get_collection(collection).points_count or 0)
        return {
            "action": "skip",
            "reason": "collection_ready",
            "collection": collection,
            "points": count,
            "url": DEFAULT_QDRANT_URL,
        }, DEFAULT_QDRANT_URL

    points = recover_qdrant_snapshot(
        DEFAULT_QDRANT_URL,
        collection=collection,
        snapshot_path=snapshot,
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


def _video_subdirs(keyframes_dir: Path) -> list[Path]:
    ignored = {"__MACOSX"}
    return [
        path
        for path in keyframes_dir.iterdir()
        if path.is_dir() and path.name not in ignored and not path.name.startswith(".")
    ]


def _zip_is_newer(zip_path: Path, keyframes_dir: Path) -> bool:
    try:
        zip_mtime = zip_path.stat().st_mtime
    except OSError:
        return True
    newest = zip_mtime
    for path in _video_subdirs(keyframes_dir):
        try:
            newest = max(newest, path.stat().st_mtime)
            for child in path.rglob("*"):
                if child.is_file():
                    newest = max(newest, child.stat().st_mtime)
        except OSError:
            continue
    return zip_mtime > newest


def _flatten_nested_keyframes_dir(keyframes_dir: Path) -> None:
    nested = keyframes_dir / "keyframes"
    if not nested.is_dir() or nested.resolve() == keyframes_dir.resolve():
        return
    for child in nested.iterdir():
        target = keyframes_dir / child.name
        if target.exists():
            continue
        child.rename(target)
    try:
        nested.rmdir()
    except OSError:
        pass
