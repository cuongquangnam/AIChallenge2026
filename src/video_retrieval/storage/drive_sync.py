from __future__ import annotations

import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from video_retrieval.storage.sync_paths import QA_PULL_PATHS, SEARCH_PULL_PATHS

logger = logging.getLogger(__name__)


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
            if mirror.is_dir():
                return mirror
            raise FileNotFoundError(
                f"DRIVE_LOCAL_PATH does not exist: {mirror}. "
                "Point it at the synced Google Drive folder on your laptop."
            )

        if self.mount_on_access:
            mount_google_drive(self.mount_point)
        root = self.mount_point / self.data_path if self.data_path else self.mount_point
        if not root.is_dir():
            raise FileNotFoundError(
                f"Drive data folder not found: {root}. "
                f"Create {self.data_path} on Google Drive or adjust DRIVE_DATA_PATH."
            )
        return root.resolve()

    def pull(self, *, paths: list[str] | tuple[str, ...] | None = None) -> int:
        """Copy selected subfolders from Drive into local_dir."""
        selected = list(paths or SEARCH_PULL_PATHS)
        source_root = self.remote_root()
        copied = 0
        total_paths = len(selected)
        for idx, path_name in enumerate(selected, start=1):
            src = source_root / path_name
            print(
                f"[pull {idx}/{total_paths}] {path_name} from {src} ...",
                flush=True,
            )
            if path_name == "keyframes":
                n = _copy_keyframes(src, self.local_dir / path_name, progress=True)
            else:
                n = _copy_tree(
                    src,
                    self.local_dir / path_name,
                    label=path_name,
                    progress=True,
                )
            print(f"[pull {idx}/{total_paths}] {path_name}: copied {n} file(s)", flush=True)
            copied += n
        return copied

    def push(self, *, paths: list[str] | tuple[str, ...] | None = None) -> int:
        """Copy selected local_dir subfolders up to Drive."""
        selected = list(paths or QA_PULL_PATHS)
        dest_root = self.remote_root()
        copied = 0
        for path_name in selected:
            print(f"[push] {path_name} ...", flush=True)
            n = _copy_tree(
                self.local_dir / path_name,
                dest_root / path_name,
                label=path_name,
                progress=True,
            )
            print(f"[push] {path_name}: copied {n} file(s)", flush=True)
            copied += n
        return copied

    def upload_file(self, local_path: Path, *, dest_relative: str) -> None:
        dest = self.remote_root() / dest_relative.strip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)

    def download_file(self, *, src_relative: str, local_path: Path) -> None:
        src = self.remote_root() / src_relative.strip("/")
        if not src.is_file():
            raise FileNotFoundError(src)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)

    def download_paths(self, relative_paths: list[str]) -> int:
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
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            copied += 1
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
            print(f"  skip (missing): {source}", flush=True)
        return 0
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_file(source, destination, timeout_sec=file_timeout_sec)
        return 1

    files = [p for p in source.rglob("*") if p.is_file()]
    total = len(files)
    name = label or source.name
    if progress:
        print(f"  scanning {name}: {total} file(s)", flush=True)

    copied = 0
    skipped = 0
    failed = 0
    last_beat = time.monotonic()
    for src in files:
        rel = src.relative_to(source)
        dest = destination / rel
        now = time.monotonic()
        if progress and (now - last_beat) >= 10.0:
            print(
                f"  {name}: still working… {copied + skipped + failed}/{total} "
                f"(copied={copied} skipped={skipped} failed={failed}) current={rel}",
                flush=True,
            )
            last_beat = now

        if skip_existing and dest.is_file():
            try:
                if dest.stat().st_size == src.stat().st_size:
                    skipped += 1
                    if progress and ((copied + skipped) % progress_every == 0):
                        done = copied + skipped + failed
                        pct = (100.0 * done / total) if total else 100.0
                        print(
                            f"  {name}: {done}/{total} ({pct:.1f}%) "
                            f"[copied={copied} skipped={skipped}]",
                            flush=True,
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
                print(f"  WARN skip {rel}: {exc}", flush=True)
            continue

        done = copied + skipped + failed
        if progress and (done % progress_every == 0 or done == total):
            pct = (100.0 * done / total) if total else 100.0
            print(
                f"  {name}: {done}/{total} ({pct:.1f}%) "
                f"[copied={copied} skipped={skipped} failed={failed}]",
                flush=True,
            )
            last_beat = time.monotonic()

    if progress:
        print(
            f"  {name}: finished copied={copied} skipped={skipped} failed={failed}",
            flush=True,
        )
    return copied


def _copy_keyframes(source: Path, destination: Path, *, progress: bool) -> int:
    """Copy keyframes from Drive, preferring a zip archive over many loose files."""
    zip_path = source / "keyframes.zip"
    if zip_path.is_file():
        if progress:
            print(f"  found {zip_path.name}; copying archive instead of loose keyframes", flush=True)
        destination.mkdir(parents=True, exist_ok=True)
        _copy_file(zip_path, destination / zip_path.name, timeout_sec=60.0)
        return 1
    return _copy_tree(source, destination, label="keyframes", progress=progress)


def _copy_file(src: Path, dest: Path, *, timeout_sec: float) -> None:
    """Copy file contents (no metadata). Abort if Drive FUSE stalls."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(shutil.copyfile, src, dest)
        future.result(timeout=timeout_sec)
