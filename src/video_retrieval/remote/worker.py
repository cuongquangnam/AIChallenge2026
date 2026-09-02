from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from video_retrieval.config import get_settings
from video_retrieval.remote.models import RemoteJobRequest, RemoteJobResponse
from video_retrieval.storage.data_sync import create_data_sync
from video_retrieval.storage.elasticsearch_hydrate import hydrate_elasticsearch_index
from video_retrieval.storage.sync_paths import SESSION_PULL_PATHS


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
        pulled = sync.pull(paths=pull_paths)
        es_result = hydrate_elasticsearch_index(settings)
        return {"pulled": pulled, "paths": pull_paths, "elasticsearch": es_result}

    if request.job == "search":
        from video_retrieval.search.service import SearchService

        service = SearchService(settings)
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
        from video_retrieval.runtime import build_task_runtime

        runtime = build_task_runtime(settings)
        result = runtime.kis.run(
            request.query or "",
            limit=request.limit,
            top_chains=request.top_chains,
        )
        return result.model_dump(mode="json")

    if request.job == "qa":
        from video_retrieval.runtime import build_task_runtime

        runtime = build_task_runtime(settings)
        result = runtime.qa.answer(
            request.query or "",
            limit=request.limit,
            frame_radius=request.frame_radius,
        )
        return result.model_dump(mode="json")

    if request.job == "trake":
        from video_retrieval.runtime import build_task_runtime

        runtime = build_task_runtime(settings)
        result = runtime.trake.run(
            request.query or "",
            top_chains=request.top_chains,
        )
        return result.model_dump(mode="json")

    raise ValueError(f"Unsupported remote job: {request.job}")


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
