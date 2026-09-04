from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

# Colab needs the musl (static) Linux build: gnu needs glibc ≥ 2.38, Colab is 2.35.
# Snapshots exported from Qdrant ≤1.12 (RocksDB / payload_storage_type=on_disk) cannot be
# recovered by GitHub Linux binaries ≥1.17. Migrate first with Docker 1.16.3 → 1.19:
#   scripts/colab/migrate_qdrant_snapshot_for_colab.sh
# Then recover with this 1.19.0 server. Client may stay 1.19.x (check_compatibility=False).
QDRANT_VERSION = "1.19.0"
QDRANT_ARTIFACT = "qdrant-x86_64-unknown-linux-musl.tar.gz"
QDRANT_DOWNLOAD_URL = (
    f"https://github.com/qdrant/qdrant/releases/download/v{QDRANT_VERSION}/{QDRANT_ARTIFACT}"
)
QDRANT_VERSION_STAMP = f"{QDRANT_VERSION}-musl"
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
    startup_timeout_sec: float = 1800.0,
) -> None:
    """Download and start a single-node Qdrant HTTP server if ``url`` is not reachable."""
    install_dir = Path(install_dir)
    storage_path = Path(storage_path)
    install_dir.mkdir(parents=True, exist_ok=True)
    storage_path.mkdir(parents=True, exist_ok=True)

    binary = install_dir / "qdrant"
    version_stamp = install_dir / f".qdrant-version-{QDRANT_VERSION_STAMP}"
    needs_install = (
        not binary.is_file()
        or not version_stamp.is_file()
        or not _binary_is_runnable(binary)
    )

    if is_qdrant_ready(url) and not needs_install:
        logger.info("Qdrant already running at %s", url)
        print(f"[qdrant] already running at {url}", flush=True)
        return

    if is_qdrant_ready(url) and needs_install:
        print(
            f"[qdrant] outdated server is running; stop it before upgrading to "
            f"{QDRANT_VERSION_STAMP}",
            flush=True,
        )
        _stop_qdrant_on_port(6333)

    if not _can_run_qdrant_binary():
        raise RuntimeError(
            "Qdrant snapshot recovery requires the Qdrant server binary (Linux x86_64). "
            "Run this step on the Colab VM, or upload the Qdrant storage directory instead "
            "of a .snapshot file."
        )

    if needs_install:
        print(
            f"[qdrant] downloading Qdrant {QDRANT_VERSION} ({QDRANT_ARTIFACT}) ...",
            flush=True,
        )
        _download_qdrant_binary(install_dir)
        if not _binary_is_runnable(binary):
            raise RuntimeError(
                f"Downloaded Qdrant binary is not runnable on this host ({binary}). "
                "Colab/Ubuntu 22.04 needs the musl build; re-check QDRANT_ARTIFACT."
            )
        for stale in install_dir.glob(".qdrant-version-*"):
            stale.unlink(missing_ok=True)
        version_stamp.write_text(QDRANT_VERSION_STAMP + "\n", encoding="utf-8")
        print(f"[qdrant] installed to {binary}", flush=True)

    if is_qdrant_ready(url):
        print(f"[qdrant] already running at {url}", flush=True)
        return

    # Qdrant 1.13+ only allows file:// recover paths under snapshots_path.
    snapshots_path = install_dir / "snapshots"
    snapshots_path.mkdir(parents=True, exist_ok=True)

    config_path = install_dir / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "storage:",
                f"  storage_path: {storage_path.as_posix()}",
                f"  snapshots_path: {snapshots_path.as_posix()}",
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
        proc = subprocess.Popen(
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
            print(f"[qdrant] ready at {url} (pid={proc.pid})", flush=True)
            return
        if proc.poll() is not None:
            tail = _tail_text(log_path)
            raise RuntimeError(
                f"Qdrant exited early with code {proc.returncode}. "
                f"Log tail ({log_path}):\n{tail}"
            )
        now = time.monotonic()
        if now - last_print >= 30.0:
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


def _stop_qdrant_on_port(port: int) -> None:
    """Best-effort stop of a local Qdrant process listening on ``port``."""
    try:
        out = subprocess.check_output(["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return
    for raw in out.split():
        try:
            pid = int(raw.strip())
        except ValueError:
            continue
        print(f"[qdrant] stopping pid={pid} on :{port}", flush=True)
        try:
            subprocess.check_call(["kill", str(pid)])
        except subprocess.CalledProcessError:
            continue
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if not is_qdrant_ready(DEFAULT_QDRANT_URL):
            return
        time.sleep(0.5)


def stage_snapshot_for_recovery(
    snapshot_path: Path,
    *,
    snapshots_dir: Path,
    progress: bool = False,
) -> Path:
    """Place ``snapshot_path`` under Qdrant's snapshots dir (required for file:// recover).

    Prefers a hard link (no extra disk), then falls back to a copy.
    """
    src = snapshot_path.resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Snapshot not found: {src}")

    snapshots_dir = Path(snapshots_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    dest = (snapshots_dir / src.name).resolve()

    try:
        if dest.exists() and dest.samefile(src):
            return dest
    except OSError:
        pass

    if dest.exists():
        dest.unlink()

    try:
        os.link(src, dest)
        if progress:
            print(f"[qdrant] hard-linked snapshot into {dest}", flush=True)
        return dest
    except OSError:
        if progress:
            size_gb = src.stat().st_size / (1024**3)
            print(
                f"[qdrant] copying snapshot into {dest} ({size_gb:.2f} GiB) ...",
                flush=True,
            )
        shutil.copy2(src, dest)
        if progress:
            print(f"[qdrant] staged snapshot at {dest}", flush=True)
        return dest


def wipe_qdrant_storage(
    *,
    url: str,
    storage_path: Path,
    install_dir: Path,
    progress: bool = False,
    startup_timeout_sec: float = 1800.0,
) -> None:
    """Stop Qdrant, delete ``storage_path``, and start a clean server.

    Used before snapshot recover when a prior attempt left corrupt segment files.
    """
    storage_path = Path(storage_path)
    install_dir = Path(install_dir)
    if progress:
        print(f"[qdrant] wiping storage at {storage_path} for clean recover ...", flush=True)
    _stop_qdrant_on_port(6333)
    if storage_path.exists():
        shutil.rmtree(storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    ensure_qdrant_server(
        url,
        install_dir=install_dir,
        storage_path=storage_path,
        startup_timeout_sec=startup_timeout_sec,
    )


def _collection_dir_exists(storage_path: Path, collection: str) -> bool:
    return (Path(storage_path) / "collections" / collection).is_dir()


def _delete_collection_best_effort(client: QdrantClient, collection: str, *, progress: bool) -> None:
    try:
        if not client.collection_exists(collection):
            return
        if progress:
            print(f"[qdrant] deleting empty/broken collection {collection!r} ...", flush=True)
        client.delete_collection(collection)
    except Exception as exc:  # noqa: BLE001
        if progress:
            print(f"[qdrant] delete_collection failed ({exc}); will wipe storage", flush=True)


def recover_qdrant_snapshot(
    url: str,
    *,
    collection: str,
    snapshot_path: Path,
    snapshots_dir: Path | None = None,
    storage_path: Path | None = None,
    install_dir: Path | None = None,
    progress: bool = False,
    timeout_sec: float = 3600.0,
) -> int:
    """Recover a collection from a local ``.snapshot`` file via the Qdrant HTTP API."""
    # Long client-side HTTP timeout only. Do not pass timeout= into recover_snapshot:
    # older Qdrant servers reject that query param, and newer clients still work with
    # the constructor timeout + wait=True.
    from qdrant_client.http.models import SnapshotPriority

    client = QdrantClient(url=url, timeout=timeout_sec, check_compatibility=False)
    if collection_has_points(client, collection):
        count = int(client.get_collection(collection).points_count or 0)
        if progress:
            print(f"[qdrant] collection {collection!r} already loaded ({count} points)", flush=True)
        return count

    # Prior failed recovers often leave half-written segment dirs that break the next attempt.
    if storage_path is not None and (
        _collection_dir_exists(storage_path, collection)
        or client.collection_exists(collection)
    ):
        _delete_collection_best_effort(client, collection, progress=progress)
        if install_dir is not None and (
            _collection_dir_exists(storage_path, collection) or client.collection_exists(collection)
        ):
            wipe_qdrant_storage(
                url=url,
                storage_path=storage_path,
                install_dir=install_dir,
                progress=progress,
            )
            client = QdrantClient(url=url, timeout=timeout_sec, check_compatibility=False)

    # Qdrant ≥1.13 rejects file:// URIs outside storage.snapshots_path (default ./snapshots).
    staged = snapshot_path
    if snapshots_dir is not None:
        staged = stage_snapshot_for_recovery(
            snapshot_path,
            snapshots_dir=snapshots_dir,
            progress=progress,
        )
    location = staged.resolve().as_uri()

    def _do_recover() -> None:
        if progress:
            print(
                f"[qdrant] recovering {staged.name} into {collection!r} "
                f"(client_timeout={timeout_sec:.0f}s) ...",
                flush=True,
            )
        client.recover_snapshot(
            collection_name=collection,
            location=location,
            priority=SnapshotPriority.SNAPSHOT,
            wait=True,
        )

    try:
        _do_recover()
    except Exception as exc:
        if storage_path is None or install_dir is None:
            raise
        if progress:
            print(
                f"[qdrant] recover failed ({exc}); wiping storage and retrying once ...",
                flush=True,
            )
        wipe_qdrant_storage(
            url=url,
            storage_path=storage_path,
            install_dir=install_dir,
            progress=progress,
        )
        client = QdrantClient(url=url, timeout=timeout_sec, check_compatibility=False)
        _do_recover()

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


def _binary_is_runnable(binary: Path) -> bool:
    """True when ``binary`` executes (catches glibc mismatches before long waits)."""
    if not binary.is_file():
        return False
    try:
        result = subprocess.run(
            [str(binary), "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    combined = f"{result.stdout}\n{result.stderr}"
    if "GLIBC_" in combined and "not found" in combined:
        return False
    # --help may exit non-zero on some builds; treat spawn success as enough.
    return result.returncode in {0, 1, 2} or "Usage" in combined or "qdrant" in combined.lower()


def _download_qdrant_binary(install_dir: Path) -> None:
    archive_path = install_dir / QDRANT_ARTIFACT
    # Always re-fetch when installing so a stale gnu tarball cannot be reused.
    archive_path.unlink(missing_ok=True)
    logger.info("Downloading Qdrant %s from %s", QDRANT_VERSION, QDRANT_DOWNLOAD_URL)
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
