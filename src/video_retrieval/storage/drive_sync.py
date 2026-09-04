from __future__ import annotations

import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from video_retrieval.storage.sync_paths import QA_PULL_PATHS, SEARCH_PULL_PATHS

logger = logging.getLogger(__name__)


def _progress(message: str) -> None:
    """Emit to logger and stdout (worker.log captures both on Colab)."""
    logger.info(message)
    print(message, flush=True)


def mount_google_drive(mount_point: str | Path = "/content/drive", *, force_remount: bool = False) -> Path:
    """Mount Google Drive on a Colab VM (no-op if already mounted).

    ``google.colab.drive.mount`` only works inside the notebook UI (needs a kernel
    for the auth popup). From laptop scripts / ``colab exec`` / console, mount
    first with::

        colab drivemount -s <session> /content/drive
    """
    mount = Path(mount_point)
    my_drive = mount / "MyDrive"
    if my_drive.is_dir() and not force_remount:
        logger.info("Google Drive already mounted at %s", mount)
        return mount

    has_kernel = False
    try:
        from IPython import get_ipython

        ipy = get_ipython()
        has_kernel = ipy is not None and getattr(ipy, "kernel", None) is not None
    except Exception:  # noqa: BLE001
        has_kernel = False

    if not has_kernel:
        raise RuntimeError(
            f"Google Drive is not mounted at {mount} (missing {my_drive}). "
            "drive.mount() cannot authenticate outside the Colab notebook UI. "
            "From your laptop run:\n"
            f"  colab drivemount -s video-retrieval {mount}\n"
            "or: ./scripts/colab/laptop_drivemount.sh\n"
            "Then re-run bootstrap."
        )

    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError(
            "google.colab.drive is only available on a Colab runtime. "
            "On your laptop, set DRIVE_LOCAL_PATH to the Google Drive desktop sync folder."
        ) from exc

    logger.info("Mounting Google Drive at %s", mount)
    drive.mount(str(mount), force_remount=force_remount)
    if not my_drive.is_dir():
        raise RuntimeError(
            f"drive.mount completed but {my_drive} is missing. "
            "Re-auth in the Colab UI or run: colab drivemount -s <session> "
            f"{mount}"
        )
    return mount


class DriveDataSync:
    """Sync DATA_DIR layout to/from a folder on Google Drive."""

    def __init__(
        self,
        *,
        mount_point: str | Path = "/content/drive",
        data_path: str = "MyDrive/video-retrieval",
        local_dir: Path,
        local_mirror: str | Path = "",
        mount_on_access: bool = True,
    ):
        if not str(data_path).strip() and not str(local_mirror).strip():
            raise ValueError("drive_data_path or drive_local_path must be configured for Drive sync")
        self.mount_point = Path(mount_point)
        self.data_path = data_path.strip("/")
        self.local_dir = Path(local_dir)
        self.local_mirror = Path(local_mirror).expanduser() if str(local_mirror).strip() else None
        self.mount_on_access = mount_on_access

    def remote_root(self) -> Path:
        if self.local_mirror is not None:
            mirror = self.local_mirror.expanduser().resolve()
            logger.info("Drive sync using local mirror %s", mirror)
            if mirror.is_dir():
                return mirror
            raise FileNotFoundError(
                f"DRIVE_LOCAL_PATH does not exist: {mirror}. "
                "Point it at the synced Google Drive folder on your laptop."
            )

        if self.mount_on_access:
            logger.info("Drive sync: ensuring mount at %s ...", self.mount_point)
            mount_google_drive(self.mount_point)
        root = self.mount_point / self.data_path if self.data_path else self.mount_point
        logger.info("Drive sync: resolving data folder %s ...", root)
        if not root.is_dir():
            raise FileNotFoundError(
                f"Drive data folder not found: {root}. "
                f"Create {self.data_path} on Google Drive or adjust DRIVE_DATA_PATH."
            )
        resolved = root.resolve()
        logger.info("Drive sync: remote root ready %s", resolved)
        return resolved

    def pull(self, *, paths: list[str] | tuple[str, ...] | None = None) -> int:
        """Copy selected subfolders from Drive into local_dir."""
        selected = list(paths or SEARCH_PULL_PATHS)
        _progress(f"[pull] starting {len(selected)} path(s) from Drive → {self.local_dir}")
        source_root = self.remote_root()
        copied = 0
        total_paths = len(selected)
        for idx, path_name in enumerate(selected, start=1):
            src = source_root / path_name
            _progress(f"[pull {idx}/{total_paths}] {path_name} from {src} ...")
            if path_name == "keyframes":
                n = _hydrate_keyframes(
                    source_root,
                    self.local_dir / "keyframes",
                    progress=True,
                )
            else:
                n = _copy_tree(
                    src,
                    self.local_dir / path_name,
                    label=path_name,
                    progress=True,
                )
            _progress(f"[pull {idx}/{total_paths}] {path_name}: copied {n} file(s)")
            copied += n
        _progress(f"[pull] finished total_copied={copied}")
        return copied

    def push(self, *, paths: list[str] | tuple[str, ...] | None = None) -> int:
        """Copy selected local_dir subfolders up to Drive."""
        selected = list(paths or QA_PULL_PATHS)
        _progress(f"[push] starting {len(selected)} path(s) → Drive")
        dest_root = self.remote_root()
        copied = 0
        for path_name in selected:
            _progress(f"[push] {path_name} ...")
            n = _copy_tree(
                self.local_dir / path_name,
                dest_root / path_name,
                label=path_name,
                progress=True,
            )
            _progress(f"[push] {path_name}: copied {n} file(s)")
            copied += n
        _progress(f"[push] finished total_copied={copied}")
        return copied

    def upload_file(self, local_path: Path, *, dest_relative: str) -> None:
        dest = self.remote_root() / dest_relative.strip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Drive upload: %s → %s", local_path, dest)
        shutil.copy2(local_path, dest)

    def download_file(self, *, src_relative: str, local_path: Path) -> None:
        src = self.remote_root() / src_relative.strip("/")
        if not src.is_file():
            raise FileNotFoundError(src)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Drive download: %s → %s", src, local_path)
        shutil.copy2(src, local_path)

    def download_paths(self, relative_paths: list[str]) -> int:
        logger.info("Drive download_paths: %s path(s) requested", len(relative_paths))
        copied = 0
        for relative in relative_paths:
            relative = relative.strip("/")
            if not relative:
                continue
            target = self.local_dir / relative
            if target.is_file():
                continue
            src = self.remote_root() / relative
            if not src.is_file():
                logger.debug("Drive download_paths: missing %s", src)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Drive download_paths: copying %s → %s", src, target)
            t0 = time.monotonic()
            shutil.copy2(src, target)
            logger.info("Drive download_paths: copied %s in %.1fs", relative, time.monotonic() - t0)
            copied += 1
        logger.info("Drive download_paths: done copied=%s", copied)
        return copied


def _copy_tree(
    source: Path,
    destination: Path,
    *,
    label: str = "",
    progress: bool = False,
    progress_every: int = 50,
    skip_existing: bool = True,
    file_timeout_sec: float = 60.0,
) -> int:
    if not source.exists():
        if progress:
            _progress(f"  skip (missing): {source}")
        return 0
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_file(source, destination, timeout_sec=file_timeout_sec)
        return 1

    files = [p for p in source.rglob("*") if p.is_file()]
    total = len(files)
    name = label or source.name
    if progress:
        _progress(f"  scanning {name}: {total} file(s)")

    copied = 0
    skipped = 0
    failed = 0
    last_beat = time.monotonic()
    for src in files:
        rel = src.relative_to(source)
        dest = destination / rel
        now = time.monotonic()
        if progress and (now - last_beat) >= 10.0:
            _progress(
                f"  {name}: still working… {copied + skipped + failed}/{total} "
                f"(copied={copied} skipped={skipped} failed={failed}) current={rel}"
            )
            last_beat = now

        if skip_existing and dest.is_file():
            try:
                if dest.stat().st_size == src.stat().st_size:
                    skipped += 1
                    if progress and ((copied + skipped) % progress_every == 0):
                        done = copied + skipped + failed
                        pct = (100.0 * done / total) if total else 100.0
                        _progress(
                            f"  {name}: {done}/{total} ({pct:.1f}%) "
                            f"[copied={copied} skipped={skipped}]"
                        )
                    continue
            except OSError:
                pass

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            _copy_file(src, dest, timeout_sec=file_timeout_sec)
            copied += 1
        except Exception as exc:  # noqa: BLE001 - keep pulling other files
            failed += 1
            if progress:
                _progress(f"  WARN skip {rel}: {exc}")
            continue

        done = copied + skipped + failed
        if progress and (done % progress_every == 0 or done == total):
            pct = (100.0 * done / total) if total else 100.0
            _progress(
                f"  {name}: {done}/{total} ({pct:.1f}%) "
                f"[copied={copied} skipped={skipped} failed={failed}]"
            )
            last_beat = time.monotonic()

    if progress:
        _progress(
            f"  {name}: finished copied={copied} skipped={skipped} failed={failed}"
        )
    return copied


def _discover_keyframe_zips(source_root: Path) -> list[Path]:
    """Find keyframe archives under the Drive data root.

    Supported layouts:
    - ``{root}/keyframes.zip``
    - ``{root}/keyframes_000.zip`` (shards)
    - ``{root}/keyframes/*.zip``
    """
    found: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen or not path.is_file():
            return
        seen.add(key)
        found.append(path)

    preferred = source_root / "keyframes.zip"
    if preferred.is_file():
        _add(preferred)
    for path in sorted(source_root.glob("keyframes_*.zip")):
        _add(path)

    nested = source_root / "keyframes"
    if nested.is_dir():
        nested_preferred = nested / "keyframes.zip"
        if nested_preferred.is_file():
            _add(nested_preferred)
        for path in sorted(nested.glob("*.zip")):
            _add(path)
    return found


def _hydrate_keyframes(source_root: Path, destination: Path, *, progress: bool) -> int:
    """Hydrate local keyframes from Drive zip(s) or a loose JPG tree.

    Prefer extracting zip archive(s) **from Drive into** ``destination`` so a
    large archive is not copied twice. Looks for ``keyframes.zip`` at the Drive
    data root and under ``keyframes/``.
    """
    zip_paths = _discover_keyframe_zips(source_root)
    if zip_paths:
        if progress:
            names = ", ".join(path.name for path in zip_paths[:5])
            suffix = " ..." if len(zip_paths) > 5 else ""
            locations = ", ".join(str(path.parent) for path in zip_paths[:3])
            _progress(
                f"  found {len(zip_paths)} zip archive(s) ({names}{suffix}) "
                f"under {locations}; extracting from Drive into local keyframes "
                "(no zip copy)"
            )
        from video_retrieval.storage.post_pull import extract_keyframes_zip

        result = extract_keyframes_zip(
            destination,
            zip_paths=zip_paths,
            progress=progress,
        )
        if result.get("action") == "extracted":
            return int(result.get("members") or len(zip_paths))
        if result.get("action") == "skip":
            return int(result.get("video_dirs") or 0)
        return len(zip_paths)

    keyframes_dir = source_root / "keyframes"
    return _copy_tree(
        keyframes_dir,
        destination,
        label="keyframes",
        progress=progress,
    )


# Back-compat alias used by older tests / imports.
def _copy_keyframes(source: Path, destination: Path, *, progress: bool) -> int:
    """Hydrate from a ``keyframes/`` folder or treat ``source`` as the Drive root."""
    if source.name == "keyframes" and source.parent.is_dir():
        return _hydrate_keyframes(source.parent, destination, progress=progress)
    return _hydrate_keyframes(source, destination, progress=progress)


def _copy_file(src: Path, dest: Path, *, timeout_sec: float) -> None:
    """Copy file contents (no metadata). Abort if Drive FUSE stalls."""
    logger.debug("Drive copyfile %s → %s (timeout=%.0fs)", src, dest, timeout_sec)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(shutil.copyfile, src, dest)
        future.result(timeout=timeout_sec)
