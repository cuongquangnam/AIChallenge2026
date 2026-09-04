import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from video_retrieval.config import Settings
from video_retrieval.remote.colab import ColabRunner, _extract_json_payload
from video_retrieval.remote.gateway import (
    RemoteComputeGateway,
    _collect_keyframe_hits,
    _keyframe_relative_path,
)
from video_retrieval.remote.models import RemoteJobRequest


@pytest.mark.unit
def test_extract_json_payload_reads_last_json_line() -> None:
  stdout = "installing...\n{\"ok\": true, \"result\": {\"hits\": []}}\n"
  payload = _extract_json_payload(stdout)
  assert json.loads(payload)["ok"] is True


@pytest.mark.unit
def test_ensure_session_manual_mode_only_checks_running(tmp_path: Path) -> None:
  settings = Settings(
    data_dir=tmp_path / "data",
    drive_data_path="MyDrive/video-retrieval",
    remote_compute="colab",
    colab_session="video-retrieval",
    colab_auto_manage=False,
  )
  runner = ColabRunner(settings)

  with (
    patch.object(runner, "_session_running", return_value=True),
    patch.object(runner, "_bootstrap_remote") as bootstrap_mock,
    patch.object(runner, "_pull_session_data") as pull_mock,
  ):
    runner.ensure_session()

  bootstrap_mock.assert_not_called()
  pull_mock.assert_not_called()


@pytest.mark.unit
def test_ensure_session_auto_manage_pulls_once(tmp_path: Path) -> None:
  settings = Settings(
    data_dir=tmp_path / "data",
    drive_data_path="MyDrive/video-retrieval",
    remote_compute="colab",
    colab_session="video-retrieval",
    colab_auto_manage=True,
  )
  runner = ColabRunner(settings)
  runner._bootstrapped = True

  with (
    patch.object(runner, "_session_running", return_value=True),
    patch.object(runner, "_ensure_drive_mounted") as mount_mock,
    patch.object(runner, "_pull_session_data") as pull_mock,
  ):
    runner.ensure_session()
    runner.ensure_session()

  mount_mock.assert_not_called()  # only called from _pull_session_data
  pull_mock.assert_called_once()


@pytest.mark.unit
def test_ensure_session_raises_when_not_running(tmp_path: Path) -> None:
  settings = Settings(
    data_dir=tmp_path / "data",
    drive_data_path="MyDrive/video-retrieval",
    remote_compute="colab",
    colab_auto_manage=False,
  )
  runner = ColabRunner(settings)

  with patch.object(runner, "_session_running", return_value=False):
    with pytest.raises(RuntimeError, match="not running"):
      runner.ensure_session()


@pytest.mark.unit
def test_remote_settings_overrides_omit_gemini_api_key(settings) -> None:
  settings = settings.model_copy(update={"gemini_api_key": "secret-key"})
  overrides = settings.remote_settings_overrides()
  assert "gemini_api_key" not in overrides


@pytest.mark.unit
def test_colab_runner_search_invokes_exec(tmp_path: Path) -> None:
  settings = Settings(
    data_dir=tmp_path / "data",
    drive_data_path="MyDrive/video-retrieval",
    remote_compute="colab",
    colab_session="video-retrieval",
    colab_worker_mode="oneshot",
  )
  runner = ColabRunner(settings)
  response_json = json.dumps(
    {
      "ok": True,
      "result": {"query": "hello", "mode": "mixed", "hits": []},
    }
  )

  with (
    patch.object(runner, "ensure_session"),
    patch.object(
      runner,
      "_colab",
      side_effect=[
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout=response_json, stderr=""),
      ],
    ) as colab_mock,
  ):
    result = runner.search("hello", mode="mixed", limit=5)

  assert result["query"] == "hello"
  assert colab_mock.call_count == 2
  assert colab_mock.call_args_list[0].args[0] == "upload"


@pytest.mark.unit
def test_colab_runner_prefers_http_worker_in_auto_mode(tmp_path: Path) -> None:
  settings = Settings(
    data_dir=tmp_path / "data",
    drive_data_path="MyDrive/video-retrieval",
    remote_compute="colab",
    colab_worker_mode="auto",
    colab_worker_port=8765,
  )
  runner = ColabRunner(settings)
  response_json = json.dumps({"ok": True, "result": {"hits": []}})

  with (
    patch.object(runner, "ensure_session"),
    patch.object(
      runner,
      "_colab",
      side_effect=[
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout=response_json, stderr=""),
      ],
    ),
  ):
    result = runner.search("hello")

  assert result == {"hits": []}
  script = runner._http_proxy_script("/content/request.json")
  assert "http://127.0.0.1:8765" in script
  assert '"/job"' in script or "/job" in script


@pytest.mark.unit
def test_run_request_search_uses_elasticsearch(settings, tmp_path: Path) -> None:
  import sys
  from unittest.mock import MagicMock, patch

  google_genai = MagicMock()
  google_genai.types = MagicMock()
  sys.modules.setdefault("google.genai", google_genai)

  from video_retrieval.models import SearchHit, SearchResponse
  from video_retrieval.remote import worker as worker_mod
  from video_retrieval.remote.worker import run_request

  data_dir = tmp_path / "remote-data"
  data_dir.mkdir(parents=True)

  mock_response = SearchResponse(query="winner", mode="asr", hits=[
    SearchHit(
      video_id="clip",
      score=1.0,
      source="text:asr",
      text="music awards ceremony winner",
    )
  ])

  request = RemoteJobRequest(
    job="search",
    drive_data_path="MyDrive/video-retrieval",
    remote_data_dir=str(data_dir),
    query="winner",
    mode="asr",
    limit=5,
    settings_overrides={"query_planner": "heuristic"},
  )

  mock_service = MagicMock()
  mock_service.search_asr.return_value = mock_response
  mock_runtime = MagicMock()
  mock_runtime.settings = Settings(data_dir=data_dir)
  mock_runtime.search = mock_service
  worker_mod._RUNTIME = None
  worker_mod._RUNTIME_KEY = None
  with patch("video_retrieval.remote.worker.warm_runtime", return_value=mock_runtime):
    response = run_request(request)

  assert response.ok is True
  assert response.result is not None
  assert response.result["mode"] == "asr"
  mock_service.search_asr.assert_called_once()
  worker_mod._RUNTIME = None
  worker_mod._RUNTIME_KEY = None

@pytest.mark.unit
def test_run_request_session_pull(tmp_path: Path) -> None:
  from video_retrieval.remote.worker import run_request

  data_dir = tmp_path / "remote-data"

  class _FakeSync:
    def pull(self, *, paths):
      assert paths
      return 3

  request = RemoteJobRequest(
    job="session_pull",
    drive_data_path="MyDrive/video-retrieval",
    remote_data_dir=str(data_dir),
  )

  with (
    patch("video_retrieval.remote.worker.create_data_sync", return_value=_FakeSync()),
    patch(
      "video_retrieval.remote.worker.hydrate_elasticsearch_index",
      return_value={"source": "ndjson", "imported": 42, "path": "video_text.ndjson"},
    ),
  ):
    response = run_request(request)

  assert response.ok is True
  assert response.result is not None
  assert response.result["pulled"] == 3
  assert response.result["elasticsearch"]["imported"] == 42


@pytest.mark.unit
def test_keyframe_relative_path_from_colab_absolute(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    hit = {
        "video_id": "clip",
        "keyframe_path": "/content/data/keyframes/clip/shot_0001_middle.jpg",
    }
    assert _keyframe_relative_path(hit, data_dir=data_dir) == "keyframes/clip/shot_0001_middle.jpg"


@pytest.mark.unit
def test_collect_keyframe_hits_from_chains() -> None:
    payload = {
        "chains": [
            {
                "video_id": "clip",
                "events": [
                    {"keyframe_path": "/content/data/keyframes/clip/shot_0001_middle.jpg"}
                ],
            }
        ],
        "hits": [{"video_id": "clip", "keyframe_path": "/content/data/keyframes/clip/x.jpg"}],
    }
    hits = _collect_keyframe_hits(payload)
    assert len(hits) == 2


@pytest.mark.unit
def test_remote_compute_gateway_requires_colab_mode(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        remote_compute="colab",
        drive_data_path="MyDrive/video-retrieval",
    )
    RemoteComputeGateway(settings)


@pytest.mark.unit
def test_remote_compute_gateway_rejects_without_colab_mode(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        drive_data_path="MyDrive/video-retrieval",
        remote_compute="",
    )
    with pytest.raises(ValueError, match="REMOTE_COMPUTE"):
        RemoteComputeGateway(settings)


@pytest.mark.unit
def test_public_worker_submit_and_poll(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        drive_data_path="MyDrive/video-retrieval",
        remote_compute="colab",
        colab_worker_public_url="https://example.trycloudflare.com",
        colab_job_poll_interval_sec=0.01,
        colab_timeout_sec=5,
    )
    runner = ColabRunner(settings)
    calls: list[str] = []

    class _Resp:
        def __init__(self, payload: dict, status: int = 200):
            self.status = status
            self._payload = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        calls.append(f"{req.get_method()} {url}")
        if url.endswith("/health"):
            return _Resp({"ok": True})
        if url.endswith("/jobs") and req.get_method() == "POST":
            return _Resp({"job_id": "abc123", "status": "queued"})
        if url.endswith("/jobs/abc123"):
            if sum(1 for c in calls if "/jobs/abc123" in c) < 2:
                return _Resp({"job_id": "abc123", "status": "running"})
            return _Resp(
                {
                    "job_id": "abc123",
                    "status": "done",
                    "ok": True,
                    "result": {"query": "q", "hits": []},
                }
            )
        raise AssertionError(f"unexpected url {url}")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = runner.search("q", limit=3)

    assert result["hits"] == []
    assert any(c.startswith("POST ") and c.endswith("/jobs") for c in calls)
    assert sum(1 for c in calls if c.endswith("/jobs/abc123")) >= 2
