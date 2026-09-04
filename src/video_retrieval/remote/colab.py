from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from video_retrieval.config import Settings
from video_retrieval.remote.models import RemoteJobRequest, RemoteJobResponse
from video_retrieval.storage.data_sync import validate_remote_storage
from video_retrieval.storage.sync_paths import SESSION_PULL_PATHS

logger = logging.getLogger(__name__)

_WORKER_MODULE = "video_retrieval.remote.worker"


class ColabRunner:
    """Run search tasks on a Colab VM via google-colab-cli from this machine."""

    def __init__(self, settings: Settings):
        validate_remote_storage(settings)
        self.settings = settings
        self._bootstrapped = False
        self._session_data_pulled = False

    def search(
        self,
        query: str,
        *,
        mode: str = "mixed",
        limit: int = 10,
        vector_name: str = "siglip",
    ) -> dict[str, Any]:
        request = self._base_request(
            job="search",
            query=query,
            mode=mode,  # type: ignore[arg-type]
            limit=limit,
            vector_name=vector_name,  # type: ignore[arg-type]
        )
        return self._run(request).result or {}

    def kis(
        self,
        query: str,
        *,
        limit: int = 100,
        top_chains: int | None = None,
    ) -> dict[str, Any]:
        request = self._base_request(
            job="kis",
            query=query,
            limit=limit,
            top_chains=top_chains,
        )
        return self._run(request).result or {}

    def qa(
        self,
        question: str,
        *,
        limit: int = 24,
        frame_radius: int | None = None,
    ) -> dict[str, Any]:
        request = self._base_request(
            job="qa",
            query=question,
            limit=limit,
            frame_radius=frame_radius,
        )
        return self._run(request).result or {}

    def trake(
        self,
        query: str,
        *,
        top_chains: int | None = None,
    ) -> dict[str, Any]:
        request = self._base_request(
            job="trake",
            query=query,
            top_chains=top_chains,
        )
        return self._run(request).result or {}

    def ensure_session(self) -> None:
        """Verify Colab is reachable; optionally auto-bootstrap when colab_auto_manage=true."""
        if self.settings.uses_public_worker:
            self._ensure_public_worker()
            return
        if not self._session_running():
            raise RuntimeError(
                f"Colab session {self.settings.colab_session!r} is not running. "
                "Start it manually and run scripts/colab/MANUAL_SETUP.md steps. "
                "Or set COLAB_WORKER_PUBLIC_URL to your Cloudflare tunnel URL."
            )
        if not self.settings.colab_auto_manage:
            return
        if not self._bootstrapped:
            self._bootstrap_remote()
        if not self._session_data_pulled:
            self._pull_session_data()
            self._session_data_pulled = True

    def _ensure_public_worker(self) -> None:
        url = f"{self.settings.colab_worker_url.rstrip('/')}/health"
        try:
            import urllib.request

            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"health status {resp.status}")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Public Colab worker not reachable at {url}. "
                "On the notebook: start worker, then run the Cloudflare tunnel cell, "
                "and set COLAB_WORKER_PUBLIC_URL on the laptop.\n"
                f"Detail: {exc}"
            ) from exc

    def _run(self, request: RemoteJobRequest) -> RemoteJobResponse:
        self.ensure_session()
        if self.settings.uses_public_worker:
            return self._run_public_http(request)
        return self._exec_request(request)

    def _run_public_http(self, request: RemoteJobRequest) -> RemoteJobResponse:
        import urllib.error
        import urllib.request

        url = f"{self.settings.colab_worker_url.rstrip('/')}/job"
        payload = request.model_dump_json().encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.settings.colab_timeout_sec) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Worker HTTP {exc.code} at {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Worker unreachable at {url}: {exc}") from exc
        response = RemoteJobResponse.model_validate_json(body)
        if not response.ok:
            raise RuntimeError(response.error or "Remote Colab job failed")
        return response

    def _ensure_drive_mounted(self) -> None:
        """Mount Drive via Colab CLI (works without notebook UI auth popup)."""
        mount = self.settings.drive_mount or "/content/drive"
        logger.info("Ensuring Google Drive is mounted at %s via colab drivemount", mount)
        completed = self._colab(
            "drivemount",
            "-s",
            self.settings.colab_session,
            mount,
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                f"colab drivemount failed for {mount}. "
                f"Run: colab drivemount -s {self.settings.colab_session} {mount}\n"
                f"{detail}"
            )

    def _pull_session_data(self) -> None:
        self._ensure_drive_mounted()
        request = self._base_request(
            job="session_pull",
            pull_paths=list(SESSION_PULL_PATHS),
        )
        response = self._exec_request(request)
        if not response.ok:
            raise RuntimeError(response.error or "Session data pull failed")
        pulled = (response.result or {}).get("pulled", 0)
        es_info = (response.result or {}).get("elasticsearch") or {}
        logger.info(
            "Pulled %s file(s) from Drive into %s at session start",
            pulled,
            self.settings.colab_remote_data_dir,
        )
        if es_info:
            logger.info(
                "Elasticsearch loaded %s document(s) from %s",
                es_info.get("imported", 0),
                es_info.get("source", "unknown"),
            )

    def _base_request(self, *, job: str, **kwargs: Any) -> RemoteJobRequest:
        return RemoteJobRequest(
            job=job,  # type: ignore[arg-type]
            drive_mount=self.settings.drive_mount,
            drive_data_path=self.settings.drive_data_path,
            drive_local_path=self.settings.drive_local_path,
            remote_data_dir=self.settings.colab_remote_data_dir,
            settings_overrides=self.settings.remote_settings_overrides(),
            **kwargs,
        )

    def _exec_request(self, request: RemoteJobRequest) -> RemoteJobResponse:
        with tempfile.TemporaryDirectory(prefix="vr-colab-") as tmp_dir:
            request_path = Path(tmp_dir) / "request.json"
            request_path.write_text(
                request.model_dump_json(indent=2),
                encoding="utf-8",
            )
            remote_request = f"/content/{request_path.name}"
            self._colab(
                "upload",
                "-s",
                self.settings.colab_session,
                str(request_path),
                remote_request,
            )
            mode = self.settings.colab_worker_mode.strip().lower() or "auto"
            if mode in {"auto", "persistent", "tunnel"}:
                try:
                    payload = self._colab_exec_worker(
                        remote_request,
                        via_http=True,
                    )
                except Exception as exc:
                    if mode in {"persistent", "tunnel"}:
                        raise RuntimeError(
                            "Persistent Colab worker is required but unreachable. "
                            "Run the notebook worker + Cloudflare tunnel cells, "
                            "or: ./scripts/colab/laptop_start_worker.sh\n"
                            f"Detail: {exc}"
                        ) from exc
                    logger.warning(
                        "Persistent worker unavailable (%s); falling back to oneshot exec",
                        exc,
                    )
                    payload = self._colab_exec_worker(remote_request, via_http=False)
            else:
                payload = self._colab_exec_worker(remote_request, via_http=False)
        response = RemoteJobResponse.model_validate_json(payload)
        if not response.ok:
            raise RuntimeError(response.error or "Remote Colab job failed")
        return response

    def _colab_exec_worker(self, remote_request_path: str, *, via_http: bool) -> str:
        script = (
            self._http_proxy_script(remote_request_path)
            if via_http
            else self._oneshot_worker_script(remote_request_path)
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            handle.write(script)
            script_path = handle.name
        try:
            completed = self._colab(
                "exec",
                "-s",
                self.settings.colab_session,
                "-f",
                script_path,
                timeout=self.settings.colab_timeout_sec,
                capture_output=True,
            )
        finally:
            Path(script_path).unlink(missing_ok=True)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "colab exec failed"
            raise RuntimeError(detail)
        return _extract_json_payload(completed.stdout)

    def _http_proxy_script(self, remote_request_path: str) -> str:
        # Localhost on the VM — public URL is used only by the laptop direct path.
        worker_url = f"http://127.0.0.1:{int(self.settings.colab_worker_port)}"
        timeout = int(self.settings.colab_timeout_sec)
        return textwrap.dedent(
            f"""
            import json
            import urllib.error
            import urllib.request

            worker_url = {worker_url!r} + "/job"
            with open({remote_request_path!r}, encoding="utf-8") as handle:
                request = json.load(handle)
            data = json.dumps(request).encode("utf-8")
            headers = {{"Content-Type": "application/json"}}
            req = urllib.request.Request(
                worker_url,
                data=data,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout={timeout}) as resp:
                    print(resp.read().decode("utf-8"))
            except urllib.error.URLError as exc:
                raise SystemExit(
                    "Persistent worker not reachable at "
                    f"{{worker_url}}. Start it with ./scripts/colab/laptop_start_worker.sh "
                    f"({{exc}})"
                ) from exc
            """
        ).strip() + "\n"

    def _oneshot_worker_script(self, remote_request_path: str) -> str:
        return textwrap.dedent(
            f"""
            import json
            from {_WORKER_MODULE} import run_request

            with open({remote_request_path!r}, encoding="utf-8") as handle:
                request = json.load(handle)
            response = run_request(request)
            print(json.dumps(response.model_dump(mode="json"), ensure_ascii=False))
            if not response.ok:
                raise SystemExit(1)
            """
        ).strip() + "\n"

    # Back-compat alias used by older tests / callers.
    def _worker_script(self, remote_request_path: str) -> str:
        return self._oneshot_worker_script(remote_request_path)

    def _bootstrap_remote(self) -> None:
        repo_root = _project_root()
        remote_root = "/content/video-retrieval"
        self._colab("upload", "-s", self.settings.colab_session, str(repo_root), remote_root)
        bootstrap = textwrap.dedent(
            f"""
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", "{remote_root}[ml]"])
            """
        ).strip()
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            handle.write(bootstrap + "\n")
            bootstrap_path = handle.name
        try:
            self._colab("exec", "-s", self.settings.colab_session, "-f", bootstrap_path)
        finally:
            Path(bootstrap_path).unlink(missing_ok=True)
        self._bootstrapped = True

    def _create_session(self) -> None:
        args = [
            "new",
            "-s",
            self.settings.colab_session,
        ]
        if self.settings.colab_gpu:
            args.extend(["--gpu", self.settings.colab_gpu])
        if self.settings.colab_high_mem:
            args.append("--high-mem")
        self._colab(*args)

    def _session_running(self) -> bool:
        completed = self._colab(
            "status",
            "-s",
            self.settings.colab_session,
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            return False
        output = completed.stdout.lower()
        return "no active session" not in output and "not found" not in output

    def _colab(self, *args: str, timeout: float | None = None, capture_output: bool = False, check: bool = True):
        command = [self.settings.colab_cli, *args]
        return subprocess.run(
            command,
            check=check,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _extract_json_payload(stdout: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            return line
    raise RuntimeError(f"Colab worker did not return JSON. Output was:\n{stdout}")
