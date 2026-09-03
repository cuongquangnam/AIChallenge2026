from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from video_retrieval.config import Settings, get_settings
from video_retrieval.remote.models import RemoteJobRequest, RemoteJobResponse
from video_retrieval.runtime import AppRuntime, build_task_runtime
from video_retrieval.storage.data_sync import create_data_sync
from video_retrieval.storage.elasticsearch_hydrate import hydrate_elasticsearch_index
from video_retrieval.storage.post_pull import prepare_colab_data
from video_retrieval.storage.sync_paths import SESSION_PULL_PATHS

_RUNTIME: AppRuntime | None = None
_RUNTIME_KEY: tuple[str, str] | None = None


def warm_runtime(settings: Settings) -> AppRuntime:
    """Load models once for the persistent Colab worker process."""
    global _RUNTIME, _RUNTIME_KEY
    if settings.colab_runtime:
        from video_retrieval.storage.post_pull import prepare_colab_qdrant

        _, qdrant_url = prepare_colab_qdrant(settings, progress=True)
        if qdrant_url != settings.qdrant_url:
            settings = settings.model_copy(update={"qdrant_url": qdrant_url})
    key = (str(settings.data_dir), settings.qdrant_collection, settings.qdrant_url)
    if _RUNTIME is not None and _RUNTIME_KEY == key:
        return _RUNTIME
    _RUNTIME = build_task_runtime(settings)
    _RUNTIME_KEY = key
    return _RUNTIME


def get_cached_runtime() -> AppRuntime | None:
    return _RUNTIME


def run_request(request: RemoteJobRequest | dict[str, Any]) -> RemoteJobResponse:
    """Execute a remote search job on the Colab VM."""
    if not isinstance(request, RemoteJobRequest):
        request = RemoteJobRequest.model_validate(request)

    try:
        result = _dispatch(request)
        return RemoteJobResponse(ok=True, result=result)
    except Exception as exc:  # noqa: BLE001
        return RemoteJobResponse(ok=False, error=str(exc))


def _dispatch(request: RemoteJobRequest) -> dict[str, Any]:
    settings = _settings_from_request(request)

    if request.job == "session_pull":
        sync = create_data_sync(
            settings,
            local_dir=Path(request.remote_data_dir),
            mount_drive=True,
        )
        pull_paths = request.pull_paths or list(SESSION_PULL_PATHS)
        print(f"[session_pull] paths={pull_paths}", flush=True)
        pulled = sync.pull(paths=pull_paths)
        print(f"[session_pull] drive copy done ({pulled} files)", flush=True)
        settings, post_pull = prepare_colab_data(settings, progress=True)
        print(f"[session_pull] post_pull={post_pull}", flush=True)
        es_result = hydrate_elasticsearch_index(settings, progress=True)
        print(f"[session_pull] elasticsearch={es_result}", flush=True)
        return {
            "pulled": pulled,
            "paths": pull_paths,
            "post_pull": post_pull,
            "qdrant_url": settings.qdrant_url,
            "elasticsearch": es_result,
        }

    runtime = _runtime_for(settings)

    if request.job == "search":
        service = runtime.search
        assert service is not None
        if request.mode == "ocr":
            response = service.search_ocr(request.query or "", limit=request.limit)
        elif request.mode == "asr":
            response = service.search_asr(request.query or "", limit=request.limit)
        elif request.mode == "visual":
            response = service.search_visual(
                request.query or "",
                limit=request.limit,
                vector_name=request.vector_name,
            )
        else:
            response = service.search_mixed(
                request.query or "",
                limit=request.limit,
                vector_name=request.vector_name,
            )
        return response.model_dump(mode="json")

    if request.job == "kis":
        assert runtime.kis is not None
        result = runtime.kis.run(
            request.query or "",
            limit=request.limit,
            top_chains=request.top_chains,
        )
        return result.model_dump(mode="json")

    if request.job == "qa":
        assert runtime.qa is not None
        result = runtime.qa.answer(
            request.query or "",
            limit=request.limit,
            frame_radius=request.frame_radius,
        )
        return result.model_dump(mode="json")

    if request.job == "trake":
        assert runtime.trake is not None
        result = runtime.trake.run(
            request.query or "",
            top_chains=request.top_chains,
        )
        return result.model_dump(mode="json")

    raise ValueError(f"Unsupported remote job: {request.job}")


def _runtime_for(settings: Settings) -> AppRuntime:
    """Reuse a warmed process runtime when present; otherwise build once for this process."""
    cached = get_cached_runtime()
    if cached is not None and str(cached.settings.data_dir) == str(settings.data_dir):
        return cached
    return warm_runtime(settings)


def _settings_from_request(request: RemoteJobRequest):
    settings = get_settings(data_dir=request.remote_data_dir, colab=True)
    settings = settings.model_copy(
        update={
            "drive_mount": request.drive_mount,
            "drive_data_path": request.drive_data_path,
            "drive_local_path": request.drive_local_path,
        }
    )
    if request.settings_overrides:
        settings = settings.model_copy(update=request.settings_overrides)
    settings.ensure_dirs()
    if settings.colab_runtime:
        from video_retrieval.storage.qdrant_bootstrap import resolve_colab_qdrant_url

        qdrant_url = resolve_colab_qdrant_url(settings)
        if qdrant_url != settings.qdrant_url:
            settings = settings.model_copy(update={"qdrant_url": qdrant_url})
    return settings


def main() -> None:
    raw = _read_request_payload()
    response = run_request(raw)
    print(json.dumps(response.model_dump(mode="json"), ensure_ascii=False))
    if not response.ok:
        sys.exit(1)


def _read_request_payload() -> dict[str, Any]:
    if len(sys.argv) >= 2:
        request_path = Path(sys.argv[1])
        if request_path.is_file():
            return json.loads(request_path.read_text(encoding="utf-8"))
    env_payload = __import__("os").environ.get("VIDEO_RETRIEVAL_REQUEST")
    if env_payload:
        return json.loads(env_payload)
    raise ValueError(
        "Remote worker needs a request JSON file path argument or VIDEO_RETRIEVAL_REQUEST"
    )
