from __future__ import annotations

import logging
import os
import pwd
import shutil
import subprocess
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

from video_retrieval.storage.progress_log import Heartbeat, print_log_heartbeat, tail_text

logger = logging.getLogger(__name__)

ES_VERSION = "8.15.3"
ES_ARTIFACT = f"elasticsearch-{ES_VERSION}-linux-x86_64.tar.gz"
ES_DOWNLOAD_URL = f"https://artifacts.elastic.co/downloads/elasticsearch/{ES_ARTIFACT}"
ES_RUNTIME_USER = "elasticsearch"

# Colab resolves cgroup cpu.stat under paths like
# /sys/fs/cgroup/../../jupyter-children/cpu.stat which escape the default
# /sys/fs/cgroup/- Java grant and crash OsService at boot.
_COLAB_POLICY_MARKER = "AIChallenge2026-colab-cgroup-grant"
_COLAB_CGROUP_GRANT = f"""
grant {{
  // {_COLAB_POLICY_MARKER}
  permission java.io.FilePermission "/sys/jupyter-children/-", "read";
  permission java.io.FilePermission "/jupyter-children/-", "read";
  permission java.io.FilePermission "/sys/fs/cgroup/../../jupyter-children/-", "read";
  permission java.io.FilePermission "/sys/fs/cgroup/../../jupyter-children/cpu.stat", "read";
}};
"""
_COLAB_CPU_STAT = "usage_usec 0\nuser_usec 0\nsystem_usec 0\nnr_periods 0\nnr_throttled 0\nthrottled_usec 0\n"


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

    if es_home.is_dir() and _es_install_needs_repair(es_home):
        print(
            "[es] previous Colab ES install is corrupted/patched incorrectly; re-extracting...",
            flush=True,
        )
        shutil.rmtree(es_home)

    if not (es_home / "bin" / "elasticsearch").is_file():
        print(f"[es] downloading Elasticsearch {ES_VERSION} ...", flush=True)
        with Heartbeat("[es]", interval_sec=20.0, message="downloading/extracting Elasticsearch…"):
            _download_and_extract(install_dir)
        print(f"[es] extracted to {es_home}", flush=True)

    _prepare_colab_cgroup_files()
    policy_note = _apply_colab_cgroup_policy(es_home)
    print(f"[es] Colab cgroup workaround: {policy_note}", flush=True)

    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_log = log_dir / "bootstrap.out"

    run_as_root = os.geteuid() == 0
    if run_as_root:
        _ensure_runtime_user(ES_RUNTIME_USER)
        _chown_tree(es_home, ES_RUNTIME_USER)
        _chown_tree(data_dir, ES_RUNTIME_USER)

    java_opts_full = java_opts.strip()
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
        # Avoid early ML boot probe; OsService still needs the Java policy above.
        "-E",
        "xpack.ml.enabled=false",
        "-E",
        "network.host=127.0.0.1",
        "-E",
        "http.port=9200",
        "-E",
        "ingest.geoip.downloader.enabled=false",
    ]
    env = os.environ.copy()
    env["ES_JAVA_OPTS"] = java_opts_full
    # Drop notebook/Colab overrides that point ES at a custom conf + broken cgroup probes.
    env.pop("ES_PATH_CONF", None)
    if run_as_root:
        cmd = [
            "sudo",
            "-u",
            ES_RUNTIME_USER,
            "env",
            "-u",
            "ES_PATH_CONF",
            f"ES_JAVA_OPTS={java_opts_full}",
            *cmd,
        ]
    print(f"[es] starting Elasticsearch (timeout={startup_timeout_sec:.0f}s) ...", flush=True)
    print(f"[es] logs: {bootstrap_log}", flush=True)
    # Truncate so heartbeats show only the current boot attempt.
    bootstrap_log.write_text("", encoding="utf-8")
    with bootstrap_log.open("ab") as log_handle:
        log_handle.write(f"\n==== start {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n".encode())
        log_handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(es_home),
            env=None if run_as_root else env,
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
            tail = tail_text(bootstrap_log, max_chars=4000)
            raise RuntimeError(
                f"Elasticsearch exited early with code {proc.returncode}. "
                f"Log tail ({bootstrap_log}):\n{tail}"
            )
        now = time.monotonic()
        if now - last_print >= 15.0:
            print_log_heartbeat(
                "[es]",
                bootstrap_log,
                seconds_left=deadline - now,
                max_chars=1500,
            )
            last_print = now
        time.sleep(1.0)

    tail = tail_text(bootstrap_log, max_chars=4000)
    raise RuntimeError(
        f"Elasticsearch failed to start at {url} within {startup_timeout_sec:.0f}s. "
        f"Log tail ({bootstrap_log}):\n{tail}"
    )


def _es_install_needs_repair(es_home: Path) -> bool:
    """True when an earlier Colab patch broke JDK crypto or module policies."""
    jdk_security = es_home / "jdk" / "conf" / "security"
    if jdk_security.is_dir():
        for policy in jdk_security.rglob("*.policy"):
            try:
                if _COLAB_POLICY_MARKER in policy.read_text(encoding="utf-8", errors="replace"):
                    return True
            except OSError:
                continue

    modules_dir = es_home / "modules"
    if modules_dir.is_dir():
        for policy in modules_dir.rglob("plugin-security.policy"):
            try:
                text = policy.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if _COLAB_POLICY_MARKER in text or "<<ALL FILES>>" in text:
                return True
    return False


def _prepare_colab_cgroup_files() -> None:
    """Create readable stub cpu.stat files where Colab's cgroup path resolves."""
    stubs = (
        Path("/sys/jupyter-children/cpu.stat"),
        Path("/jupyter-children/cpu.stat"),
        Path("/sys/fs/cgroup/../../jupyter-children/cpu.stat"),
    )
    for stub in stubs:
        try:
            stub.parent.mkdir(parents=True, exist_ok=True)
            if not stub.is_file():
                stub.write_text(_COLAB_CPU_STAT, encoding="utf-8")
            stub.chmod(0o644)
        except OSError as exc:
            print(f"[es] warn: could not create cgroup stub {stub}: {exc}", flush=True)


def _apply_colab_cgroup_policy(es_home: Path) -> str:
    """Patch server security.policy inside the ES jar with Colab-specific path grants.

    Do not touch ``jdk/conf/security/**`` (breaks JCE) or ``plugin-security.policy``
    (ES rejects broad FilePermission grants there).
    """
    lib_dir = es_home / "lib"
    jars = sorted(lib_dir.glob("elasticsearch-*.jar")) if lib_dir.is_dir() else []
    if not jars:
        return "skipped (elasticsearch jar not found)"

    target_jar = jars[0]
    member = "org/elasticsearch/bootstrap/security.policy"
    try:
        with zipfile.ZipFile(target_jar, "r") as zin:
            if member not in zin.namelist():
                return f"skipped ({member} missing in {target_jar.name})"
            original = zin.read(member).decode("utf-8")
            if _COLAB_POLICY_MARKER in original:
                return f"already patched {target_jar.name}"

            updated = original.rstrip() + "\n" + _COLAB_CGROUP_GRANT.strip() + "\n"
            tmp_jar = target_jar.with_suffix(".jar.colab-tmp")
            with zipfile.ZipFile(tmp_jar, "w") as zout:
                for info in zin.infolist():
                    data = updated.encode("utf-8") if info.filename == member else zin.read(info.filename)
                    zout.writestr(info, data)
        tmp_jar.replace(target_jar)
    except OSError as exc:
        return f"failed to patch {target_jar.name}: {exc}"
    return f"patched {target_jar.name}"


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


def _download_and_extract(install_dir: Path) -> None:
    archive_path = install_dir / ES_ARTIFACT
    if not archive_path.is_file():
        logger.info("Downloading Elasticsearch %s", ES_VERSION)
        urllib.request.urlretrieve(ES_DOWNLOAD_URL, archive_path)
    logger.info("Extracting Elasticsearch to %s", install_dir)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(path=install_dir)
    # Keep the tarball so a later repair re-extract does not re-download ~600MB.
    # Callers that wiped es_home can extract again from the cached archive.


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
