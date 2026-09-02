from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

ES_VERSION = "8.15.3"
ES_ARTIFACT = f"elasticsearch-{ES_VERSION}-linux-x86_64.tar.gz"
ES_DOWNLOAD_URL = f"https://artifacts.elastic.co/downloads/elasticsearch/{ES_ARTIFACT}"


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
    startup_timeout_sec: float = 120.0,
) -> None:
    """Download and start a single-node Elasticsearch if ``url`` is not reachable."""
    if is_elasticsearch_ready(url):
        logger.info("Elasticsearch already running at %s", url)
        return

    install_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    es_home = install_dir / f"elasticsearch-{ES_VERSION}"
    if not (es_home / "bin" / "elasticsearch").is_file():
        _download_and_extract(install_dir)

    env = os.environ.copy()
    env["ES_JAVA_OPTS"] = java_opts
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(
        [
            str(es_home / "bin" / "elasticsearch"),
            "-E",
            "discovery.type=single-node",
            "-E",
            f"path.data={data_dir}",
            "-E",
            f"path.logs={log_dir}",
            "-E",
            "xpack.security.enabled=false",
            "-E",
            "network.host=_local_",
            "-E",
            "http.port=9200",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    deadline = time.monotonic() + startup_timeout_sec
    while time.monotonic() < deadline:
        if is_elasticsearch_ready(url):
            logger.info("Elasticsearch started at %s", url)
            return
        time.sleep(1.0)
    raise RuntimeError(f"Elasticsearch failed to start at {url} within {startup_timeout_sec:.0f}s")


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
