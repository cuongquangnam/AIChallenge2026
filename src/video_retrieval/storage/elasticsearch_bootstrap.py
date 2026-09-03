from __future__ import annotations

import logging
import os
import pwd
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

ES_VERSION = "8.15.3"
ES_ARTIFACT = f"elasticsearch-{ES_VERSION}-linux-x86_64.tar.gz"
ES_DOWNLOAD_URL = f"https://artifacts.elastic.co/downloads/elasticsearch/{ES_ARTIFACT}"
ES_RUNTIME_USER = "elasticsearch"


def is_elasticsearch_ready(url: str, *, timeout: float = 1.0) -> bool:
    health_url = f"{url.rstrip('/')}/_cluster/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def ensure_elasticsearch(
    url: str,
    *,
    install_dir: Path,
    data_dir: Path,
    java_opts: str = "-Xms512m -Xmx512m",
    startup_timeout_sec: float = 180.0,
) -> None:
    """Download and start a single-node Elasticsearch if ``url`` is not reachable.

    On Colab the process runs as root; Elasticsearch 8 refuses that, so we create
    a dedicated ``elasticsearch`` user and launch via ``sudo -u``.
    """
    if is_elasticsearch_ready(url):
        logger.info("Elasticsearch already running at %s", url)
        print(f"[es] already running at {url}", flush=True)
        return

    install_dir = Path(install_dir)
    data_dir = Path(data_dir)
    install_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    es_home = install_dir / f"elasticsearch-{ES_VERSION}"
    if not (es_home / "bin" / "elasticsearch").is_file():
        print(f"[es] downloading Elasticsearch {ES_VERSION} ...", flush=True)
        _download_and_extract(install_dir)
        print(f"[es] extracted to {es_home}", flush=True)

    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_log = log_dir / "bootstrap.out"

    run_as_root = os.geteuid() == 0
    if run_as_root:
        _ensure_runtime_user(ES_RUNTIME_USER)
        _chown_tree(es_home, ES_RUNTIME_USER)
        _chown_tree(data_dir, ES_RUNTIME_USER)

    cmd = [
        str(es_home / "bin" / "elasticsearch"),
        "-E",
        "discovery.type=single-node",
        "-E",
        f"path.data={data_dir / 'nodes'}",
        "-E",
        f"path.logs={log_dir}",
        "-E",
        "xpack.security.enabled=false",
        "-E",
        "network.host=127.0.0.1",
        "-E",
        "http.port=9200",
        "-E",
        "ingest.geoip.downloader.enabled=false",
    ]
    env = os.environ.copy()
    env["ES_JAVA_OPTS"] = java_opts
    if run_as_root:
        cmd = ["sudo", "-u", ES_RUNTIME_USER, "env", f"ES_JAVA_OPTS={java_opts}", *cmd]
    print(f"[es] starting Elasticsearch (timeout={startup_timeout_sec:.0f}s) ...", flush=True)
    print(f"[es] logs: {bootstrap_log}", flush=True)
    with bootstrap_log.open("ab") as log_handle:
        log_handle.write(f"\n==== start {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n".encode())
        log_handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(es_home),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    deadline = time.monotonic() + startup_timeout_sec
    last_print = 0.0
    while time.monotonic() < deadline:
        if is_elasticsearch_ready(url):
            logger.info("Elasticsearch started at %s", url)
            print(f"[es] ready at {url} (pid={proc.pid})", flush=True)
            return
        if proc.poll() is not None:
            tail = _tail_text(bootstrap_log, max_chars=4000)
            raise RuntimeError(
                f"Elasticsearch exited early with code {proc.returncode}. "
                f"Log tail ({bootstrap_log}):\n{tail}"
            )
        now = time.monotonic()
        if now - last_print >= 15.0:
            print(
                f"[es] still starting… ({int(deadline - now)}s left) "
                f"see {bootstrap_log}",
                flush=True,
            )
            last_print = now
        time.sleep(1.0)

    tail = _tail_text(bootstrap_log, max_chars=4000)
    raise RuntimeError(
        f"Elasticsearch failed to start at {url} within {startup_timeout_sec:.0f}s. "
        f"Log tail ({bootstrap_log}):\n{tail}"
    )


def _ensure_runtime_user(username: str) -> None:
    try:
        pwd.getpwnam(username)
        return
    except KeyError:
        pass
    subprocess.check_call(
        ["useradd", "--system", "--create-home", "--shell", "/bin/false", username]
    )


def _chown_tree(path: Path, username: str) -> None:
    if not path.exists():
        return
    subprocess.check_call(["chown", "-R", f"{username}:{username}", str(path)])


def _tail_text(path: Path, *, max_chars: int) -> str:
    if not path.is_file():
        return "(no log file)"
    try:
        data = path.read_bytes()
    except OSError as exc:
        return f"(could not read log: {exc})"
    text = data.decode("utf-8", errors="replace")
    return text[-max_chars:] if len(text) > max_chars else text or "(empty log)"


def _download_and_extract(install_dir: Path) -> None:
    archive_path = install_dir / ES_ARTIFACT
    if not archive_path.is_file():
        logger.info("Downloading Elasticsearch %s", ES_VERSION)
        urllib.request.urlretrieve(ES_DOWNLOAD_URL, archive_path)
    logger.info("Extracting Elasticsearch to %s", install_dir)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(path=install_dir)
    archive_path.unlink(missing_ok=True)


def find_es_ndjson_export(data_dir: Path, es_index: str) -> Path | None:
    """Locate a bulk-export ndjson file under ``data_dir/elasticsearch/``."""
    es_dir = data_dir / "elasticsearch"
    if not es_dir.is_dir():
        return None

    exact = es_dir / f"{es_index}.ndjson"
    if exact.is_file():
        return exact

    prefixed = sorted(es_dir.glob(f"{es_index}*.ndjson"))
    if prefixed:
        return prefixed[0]

    any_ndjson = sorted(es_dir.glob("*.ndjson"))
    return any_ndjson[0] if any_ndjson else None
