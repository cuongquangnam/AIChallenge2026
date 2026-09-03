from __future__ import annotations

import logging
import platform
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

QDRANT_VERSION = "1.12.5"
QDRANT_ARTIFACT = "qdrant-x86_64-unknown-linux-gnu.tar.gz"
QDRANT_DOWNLOAD_URL = (
    f"https://github.com/qdrant/qdrant/releases/download/v{QDRANT_VERSION}/{QDRANT_ARTIFACT}"
)
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"


def find_qdrant_snapshot(qdrant_dir: Path, collection: str) -> Path | None:
    """Return a ``.snapshot`` file under ``qdrant_dir`` (prefers name matching ``collection``)."""
    if not qdrant_dir.is_dir():
        return None
    candidates = sorted(qdrant_dir.glob("*.snapshot"))
    if not candidates:
        return None
    for candidate in candidates:
        if collection in candidate.stem:
            return candidate
    return candidates[0]


def collection_has_points(client: QdrantClient, collection: str) -> bool:
    try:
        if not client.collection_exists(collection):
            return False
        info = client.get_collection(collection)
        return int(getattr(info, "points_count", 0) or 0) > 0
    except Exception:  # noqa: BLE001
        return False


def is_qdrant_ready(url: str, *, timeout: float = 1.0) -> bool:
    health_url = f"{url.rstrip('/')}/readyz"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def ensure_qdrant_server(
    url: str,
    *,
    install_dir: Path,
    storage_path: Path,
    startup_timeout_sec: float = 120.0,
) -> None:
    """Download and start a single-node Qdrant HTTP server if ``url`` is not reachable."""
    if is_qdrant_ready(url):
        logger.info("Qdrant already running at %s", url)
        print(f"[qdrant] already running at {url}", flush=True)
        return

    if not _can_run_qdrant_binary():
        raise RuntimeError(
            "Qdrant snapshot recovery requires the Qdrant server binary (Linux x86_64). "
            "Run this step on the Colab VM, or upload the Qdrant storage directory instead "
            "of a .snapshot file."
        )

    install_dir = Path(install_dir)
    storage_path = Path(storage_path)
    install_dir.mkdir(parents=True, exist_ok=True)
    storage_path.mkdir(parents=True, exist_ok=True)

    binary = install_dir / "qdrant"
    if not binary.is_file():
        print(f"[qdrant] downloading Qdrant {QDRANT_VERSION} ...", flush=True)
        _download_qdrant_binary(install_dir)
        print(f"[qdrant] installed to {binary}", flush=True)

    config_path = install_dir / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "storage:",
                f"  storage_path: {storage_path.as_posix()}",
                "service:",
                "  host: 127.0.0.1",
                "  http_port: 6333",
                "  grpc_port: 6334",
                "",
            ]
        ),
        encoding="utf-8",
    )

    log_path = install_dir / "qdrant.log"
    print(f"[qdrant] starting server (timeout={startup_timeout_sec:.0f}s) ...", flush=True)
    print(f"[qdrant] logs: {log_path}", flush=True)
    with log_path.open("ab") as log_handle:
        log_handle.write(f"\n==== start {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n".encode())
        log_handle.flush()
        subprocess.Popen(
            [str(binary), "--config-path", str(config_path)],
            cwd=str(install_dir),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    deadline = time.monotonic() + startup_timeout_sec
    last_print = 0.0
    while time.monotonic() < deadline:
        if is_qdrant_ready(url):
            print(f"[qdrant] ready at {url}", flush=True)
            return
        now = time.monotonic()
        if now - last_print >= 10.0:
            print(
                f"[qdrant] still starting… ({int(deadline - now)}s left) see {log_path}",
                flush=True,
            )
            last_print = now
        time.sleep(1.0)

    tail = _tail_text(log_path)
    raise RuntimeError(
        f"Qdrant failed to start at {url} within {startup_timeout_sec:.0f}s. "
        f"Log tail ({log_path}):\n{tail}"
    )


def recover_qdrant_snapshot(
    url: str,
    *,
    collection: str,
    snapshot_path: Path,
    progress: bool = False,
    timeout_sec: float = 3600.0,
) -> int:
    """Recover a collection from a local ``.snapshot`` file via the Qdrant HTTP API."""
    client = QdrantClient(url=url, timeout=timeout_sec)
    if collection_has_points(client, collection):
        count = int(client.get_collection(collection).points_count or 0)
        if progress:
            print(f"[qdrant] collection {collection!r} already loaded ({count} points)", flush=True)
        return count

    location = snapshot_path.resolve().as_uri()
    if progress:
        print(
            f"[qdrant] recovering {snapshot_path.name} into {collection!r} "
            f"(timeout={timeout_sec:.0f}s) ...",
            flush=True,
        )
    client.recover_snapshot(
        collection_name=collection,
        location=location,
        wait=True,
        timeout=timeout_sec,
    )
    count = _wait_for_collection_points(
        client,
        collection,
        timeout_sec=timeout_sec,
        progress=progress,
    )
    if progress:
        print(f"[qdrant] recovered {count} point(s)", flush=True)
    return count


def _wait_for_collection_points(
    client: QdrantClient,
    collection: str,
    *,
    timeout_sec: float,
    progress: bool,
) -> int:
    deadline = time.monotonic() + timeout_sec
    last_print = 0.0
    while time.monotonic() < deadline:
        try:
            info = client.get_collection(collection)
            count = int(getattr(info, "points_count", 0) or 0)
            if count > 0:
                return count
        except Exception:
            pass
        now = time.monotonic()
        if progress and now - last_print >= 30.0:
            print(f"[qdrant] waiting for collection {collection!r} to become readable ...", flush=True)
            last_print = now
        time.sleep(2.0)
    raise RuntimeError(f"Timed out waiting for Qdrant collection {collection!r} after recovery")


def _can_run_qdrant_binary() -> bool:
    machine = platform.machine().lower()
    return platform.system() == "Linux" and machine in {"x86_64", "amd64"}


def _download_qdrant_binary(install_dir: Path) -> None:
    archive_path = install_dir / QDRANT_ARTIFACT
    if not archive_path.is_file():
        logger.info("Downloading Qdrant %s", QDRANT_VERSION)
        urllib.request.urlretrieve(QDRANT_DOWNLOAD_URL, archive_path)

    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(path=install_dir)

    extracted = install_dir / "qdrant"
    if not extracted.is_file():
        raise RuntimeError(f"Qdrant binary missing after extract: {extracted}")

    extracted.chmod(extracted.stat().st_mode | 0o111)
    archive_path.unlink(missing_ok=True)


def _tail_text(path: Path, *, max_chars: int = 4000) -> str:
    if not path.is_file():
        return "(no log file)"
    try:
        data = path.read_bytes()
    except OSError as exc:
        return f"(could not read log: {exc})"
    text = data.decode("utf-8", errors="replace")
    return text[-max_chars:] if len(text) > max_chars else text or "(empty log)"


def embedded_storage_has_collection(storage_path: Path, collection: str) -> bool:
    """True when local embedded Qdrant already contains ``collection`` with points."""
    if not storage_path.is_dir():
        return False
    try:
        client = QdrantClient(path=str(storage_path))
        return collection_has_points(client, collection)
    except Exception:  # noqa: BLE001
        return False


def looks_like_qdrant_storage(path: Path) -> bool:
    if not path.is_dir():
        return False
    markers = ("meta.json", "collections", "collection")
    if any((path / marker).exists() for marker in markers):
        return True
    return any(child.is_dir() and child.name.startswith("collection") for child in path.iterdir())


def resolve_colab_qdrant_url(settings) -> str:
    """Pick embedded local storage or HTTP server (required for ``.snapshot`` recovery)."""
    from video_retrieval.storage.backends import qdrant_storage_path

    qdrant_dir = settings.qdrant_dir
    collection = settings.qdrant_collection
    storage_path = Path(qdrant_storage_path(settings))

    snapshot = find_qdrant_snapshot(qdrant_dir, collection)
    if snapshot is not None:
        return DEFAULT_QDRANT_URL

    if embedded_storage_has_collection(storage_path, collection):
        return "local"
    if looks_like_qdrant_storage(storage_path) and embedded_storage_has_collection(qdrant_dir, collection):
        return "local"
    if looks_like_qdrant_storage(qdrant_dir) and embedded_storage_has_collection(qdrant_dir, collection):
        return "local"

    configured = settings.qdrant_url.strip()
    if configured and configured not in {"local", "file", ":memory:", "memory"}:
        return configured
    return "local"
