from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from video_retrieval.remote import worker_server as ws
from video_retrieval.remote.models import RemoteJobResponse


@pytest.fixture(autouse=True)
def _clear_jobs() -> None:
    with ws._jobs_lock:
        ws._jobs.clear()
    yield
    with ws._jobs_lock:
        ws._jobs.clear()


@pytest.mark.unit
def test_async_jobs_submit_poll_and_sync_job_still_works() -> None:
    settings = MagicMock()
    settings.data_dir = Path("/content/data")
    runtime = MagicMock()
    runtime.search = object()
    done = RemoteJobResponse(ok=True, result={"hits": [{"video_id": "v1"}]})

    with (
        patch.object(ws, "get_settings", return_value=settings),
        patch.object(ws, "warm_runtime", return_value=runtime),
        patch.object(ws, "run_request", return_value=done) as run_mock,
        TestClient(ws.app) as client,
    ):
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["runtime_ready"] is True

        submitted = client.post(
            "/jobs",
            json={
                "job": "search",
                "query": "hello",
                "limit": 5,
                "remote_data_dir": "/content/data",
            },
        )
        assert submitted.status_code == 200
        job_id = submitted.json()["job_id"]
        assert job_id

        status_payload = None
        for _ in range(100):
            polled = client.get(f"/jobs/{job_id}")
            assert polled.status_code == 200
            status_payload = polled.json()
            if status_payload["status"] in {"done", "error"}:
                break
            time.sleep(0.02)
        assert status_payload is not None
        assert status_payload["status"] == "done"
        assert status_payload["ok"] is True
        assert status_payload["result"]["hits"][0]["video_id"] == "v1"
        run_mock.assert_called()

        sync = client.post(
            "/job",
            json={
                "job": "kis",
                "query": "hello",
                "limit": 10,
                "remote_data_dir": "/content/data",
            },
        )
        assert sync.status_code == 200
        assert sync.json()["ok"] is True

        missing = client.get("/jobs/does-not-exist")
        assert missing.status_code == 404
