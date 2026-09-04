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
            for member_idx, info in enumerate(members, start=1):
                archive.extract(info, keyframes_dir)
                if progress and (
                    member_idx == 1
                    or member_idx == len(members)
                    or member_idx % 2000 == 0
                ):
                    print(
                        f"[keyframes]   {zip_path.name}: {member_idx}/{len(members)} files",
                        flush=True,
                    )

    _flatten_nested_keyframes_dir(keyframes_dir)
    video_dirs = _video_subdirs(keyframes_dir)
    if progress:
        print(
            f"[keyframes] extracted {total_members} member(s); "
            f"{len(video_dirs)} video folder(s) under {keyframes_dir}",
            flush=True,
        )
    return {
        "action": "extracted",
        "zips": [path.name for path in resolved_zips],
        "members": total_members,
        "video_dirs": len(video_dirs),
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


def _video_subdirs(keyframes_dir: Path) -> list[Path]:
    ignored = {"__MACOSX"}
    if not keyframes_dir.is_dir():
        return []
    return [
        path
        for path in keyframes_dir.iterdir()
        if path.is_dir() and path.name not in ignored and not path.name.startswith(".")
    ]


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
