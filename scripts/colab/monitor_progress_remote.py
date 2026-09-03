#!/usr/bin/env python3
"""Print Colab setup/pull progress. Run on the Colab VM."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_REPO_ROOT = Path("/content/video-retrieval")
DEFAULT_DATA_DIR = Path("/content/data")
DEFAULT_DRIVE_MOUNT = Path("/content/drive")
DEFAULT_DRIVE_DATA_PATH = "MyDrive/video-retrieval"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("COLAB_REMOTE_DATA_DIR", str(DEFAULT_DATA_DIR)),
        help="Remote DATA_DIR to inspect.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(DEFAULT_REPO_ROOT),
        help="Remote repo root to inspect.",
    )
    args, _unknown = parser.parse_known_args()

    repo_root = Path(args.repo_root)
    data_dir = Path(args.data_dir)
    env = _read_dotenv(repo_root / ".env.colab") or _read_dotenv(Path("/content/.env.colab"))
    drive_mount = Path(env.get("DRIVE_MOUNT") or os.environ.get("DRIVE_MOUNT") or DEFAULT_DRIVE_MOUNT)
    drive_data_path = (
        env.get("DRIVE_DATA_PATH")
        or os.environ.get("DRIVE_DATA_PATH")
        or DEFAULT_DRIVE_DATA_PATH
    ).strip("/")
    drive_root = drive_mount / drive_data_path
    qdrant_collection = env.get("QDRANT_COLLECTION") or os.environ.get("QDRANT_COLLECTION") or "video_keyframes"
    es_index = env.get("ES_INDEX") or os.environ.get("ES_INDEX") or "video_text"
    worker_port = int(env.get("COLAB_WORKER_PORT") or os.environ.get("COLAB_WORKER_PORT") or "8765")

    status = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo": _repo_status(repo_root),
        "drive": _drive_status(drive_root),
        "data": _data_status(data_dir, drive_root, qdrant_collection, es_index),
        "services": _service_status(worker_port),
        "processes": _interesting_processes(),
        "logs": _logs(repo_root, data_dir),
    }

    if args.json:
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return

    _print_human(status)


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _repo_status(repo_root: Path) -> dict[str, object]:
    status: dict[str, object] = {
        "path": str(repo_root),
        "exists": repo_root.is_dir(),
        "env_colab": (repo_root / ".env.colab").is_file(),
    }
    if not (repo_root / ".git").is_dir():
        return status
    status["branch"] = _run(["git", "branch", "--show-current"], cwd=repo_root)
    status["commit"] = _run(["git", "log", "-1", "--oneline"], cwd=repo_root)
    return status


def _drive_status(drive_root: Path) -> dict[str, object]:
    return {
        "path": str(drive_root),
        "mounted": (drive_root.parent if drive_root.name else drive_root).exists(),
        "exists": drive_root.is_dir(),
        "subdirs": _existing_children(
            drive_root,
            ("elasticsearch", "qdrant", "keyframes", "manifests", "videos"),
        ),
    }


def _data_status(
    data_dir: Path,
    drive_root: Path,
    qdrant_collection: str,
    es_index: str,
) -> dict[str, object]:
    keyframes_dir = data_dir / "keyframes"
    drive_keyframes = drive_root / "keyframes"
    data_qdrant = data_dir / "qdrant"
    data_es = data_dir / "elasticsearch"
    data_manifests = data_dir / "manifests"

    return {
        "path": str(data_dir),
        "exists": data_dir.is_dir(),
        "elasticsearch": {
            "local_files": _file_summary(data_es, "*.ndjson"),
            "expected_index": es_index,
        },
        "qdrant": {
            "local_snapshots": _file_summary(data_qdrant, "*.snapshot"),
            "collection": qdrant_collection,
            "storage_size": _du(data_qdrant / "storage"),
        },
        "keyframes": {
            "local_zip": _file_summary(keyframes_dir, "*.zip"),
            "drive_zip": _file_summary(drive_keyframes, "*.zip"),
            "local_video_dirs": _count_dirs(keyframes_dir),
            "local_images": _count_files(keyframes_dir, (".jpg", ".jpeg", ".png", ".webp")),
            "drive_files": _count_files(drive_keyframes, None) if drive_keyframes.is_dir() else 0,
            "local_size": _du(keyframes_dir),
        },
        "manifests": {
            "local_json": _count_files(data_manifests, (".json",)),
            "drive_json": _count_files(drive_root / "manifests", (".json",)),
        },
    }


def _service_status(worker_port: int) -> dict[str, object]:
    return {
        "worker": _http_json(f"http://127.0.0.1:{worker_port}/health"),
        "qdrant_ready": _http_ok("http://127.0.0.1:6333/readyz"),
        "elasticsearch_ready": _http_ok("http://127.0.0.1:9200/_cluster/health"),
    }


def _interesting_processes() -> list[dict[str, object]]:
    output = _run(["ps", "-eo", "pid,etimes,cmd"])
    rows: list[dict[str, object]] = []
    needles = (
        "bootstrap_remote.py",
        "pull_data_remote.py",
        "setup_on_vm.sh",
        "elasticsearch",
        "qdrant",
        "uvicorn",
        "worker_server",
    )
    for raw in output.splitlines()[1:]:
        line = raw.strip()
        if not line or not any(needle in line for needle in needles):
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        rows.append(
            {
                "pid": parts[0],
                "elapsed_sec": int(parts[1]) if parts[1].isdigit() else parts[1],
                "cmd": parts[2],
            }
        )
    return rows


def _logs(repo_root: Path, data_dir: Path) -> dict[str, object]:
    return {
        "worker_log": _log_tail(repo_root / "worker.log"),
        "elasticsearch_log": _log_tail(data_dir / "elasticsearch_data" / "logs" / "bootstrap.out"),
        "qdrant_log": _log_tail(Path("/content/qdrant/qdrant.log")),
    }


def _existing_children(root: Path, names: tuple[str, ...]) -> list[str]:
    if not root.is_dir():
        return []
    return [name for name in names if (root / name).exists()]


def _file_summary(root: Path, pattern: str) -> list[dict[str, object]]:
    if not root.is_dir():
        return []
    files = sorted(path for path in root.glob(pattern) if path.is_file())
    return [
        {
            "name": path.name,
            "size": _format_bytes(path.stat().st_size),
            "mtime": time.strftime("%H:%M:%S", time.localtime(path.stat().st_mtime)),
        }
        for path in files
    ]


def _count_dirs(root: Path) -> int:
    if not root.is_dir():
        return 0
    ignored = {"__MACOSX"}
    return sum(
        1
        for path in root.iterdir()
        if path.is_dir() and path.name not in ignored and not path.name.startswith(".")
    )


def _count_files(root: Path, suffixes: tuple[str, ...] | None) -> int:
    if not root.is_dir():
        return 0
    total = 0
    suffixes_lower = tuple(s.lower() for s in suffixes) if suffixes else None
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if suffixes_lower is None or path.suffix.lower() in suffixes_lower:
            total += 1
    return total


def _du(path: Path) -> str:
    if not path.exists():
        return "missing"
    output = _run(["du", "-sh", str(path)])
    return output.split("\t", 1)[0].strip() if output else "unknown"


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _http_json(url: str) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def _log_tail(path: Path, *, max_lines: int = 5) -> dict[str, object]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "tail": []}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"path": str(path), "exists": True, "tail": lines[-max_lines:]}


def _run(cmd: list[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            cmd,
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.1f}{unit}"
        amount /= 1024
    return f"{amount:.1f}TB"


def _print_human(status: dict[str, object]) -> None:
    repo = status["repo"]
    drive = status["drive"]
    data = status["data"]
    services = status["services"]
    processes = status["processes"]
    logs = status["logs"]

    print(f"== Colab progress @ {status['time']} ==")
    print(
        f"repo: {repo['path']} exists={repo['exists']} "
        f"env_colab={repo['env_colab']} branch={repo.get('branch', '-')}"
    )
    if repo.get("commit"):
        print(f"commit: {repo['commit']}")
    print(
        f"drive: {drive['path']} exists={drive['exists']} "
        f"subdirs={','.join(drive['subdirs']) or '-'}"
    )
    print("")

    es = data["elasticsearch"]
    qdrant = data["qdrant"]
    keyframes = data["keyframes"]
    manifests = data["manifests"]
    print(
        "elasticsearch: "
        f"ndjson={_names(es['local_files']) or '-'} expected_index={es['expected_index']} "
        f"ready={services['elasticsearch_ready']}"
    )
    print(
        "qdrant: "
        f"snapshots={_names(qdrant['local_snapshots']) or '-'} "
        f"storage={qdrant['storage_size']} collection={qdrant['collection']} "
        f"ready={services['qdrant_ready']}"
    )
    print(
        "keyframes: "
        f"local_zip={_names(keyframes['local_zip']) or '-'} "
        f"drive_zip={_names(keyframes['drive_zip']) or '-'} "
        f"video_dirs={keyframes['local_video_dirs']} images={keyframes['local_images']} "
        f"drive_files={keyframes['drive_files']} size={keyframes['local_size']}"
    )
    print(
        "manifests: "
        f"local_json={manifests['local_json']} drive_json={manifests['drive_json']}"
    )
    worker = services["worker"]
    print(f"worker: healthy={worker is not None} health={worker or '-'}")
    print("")

    print("processes:")
    if processes:
        for proc in processes:
            print(f"  pid={proc['pid']} elapsed={proc['elapsed_sec']}s cmd={proc['cmd'][:140]}")
    else:
        print("  none")

    print("")
    for name, log in logs.items():
        print(f"{name}: {log['path']} exists={log['exists']}")
        for line in log["tail"]:
            print(f"  {line}")


def _names(files: list[dict[str, object]]) -> str:
    return ", ".join(str(item["name"]) for item in files)


if __name__ == "__main__":
    main()
