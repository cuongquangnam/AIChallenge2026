from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import json

import pytest

from video_retrieval.config import Settings
from video_retrieval.models import AudioTrack, FrameRole, KeyFrame
from video_retrieval.text.asr import ASREngine
from video_retrieval.text.ocr import (
    OCREngine,
    _GeminiRateLimiter,
    _daily_quota_message,
    _gemini_generation_config,
    _is_daily_quota_exhausted,
    _is_retryable_api_error,
    _model_unavailable_message,
    _parse_batch_ocr_response,
    _rapidocr_texts,
    _retry_after_seconds,
)
from tests.helpers import write_dummy_image


@pytest.mark.unit
def test_mock_ocr_only_middle_frames(tmp_path: Path, settings: Settings) -> None:
    ocr = OCREngine(settings)
    middle = KeyFrame(
        video_id="clip",
        shot_index=0,
        role=FrameRole.MIDDLE,
        frame_index=5,
        timestamp_sec=0.5,
        path=write_dummy_image(tmp_path / "shot_0000_middle.jpg"),
    )
    start = KeyFrame(
        video_id="clip",
        shot_index=0,
        role=FrameRole.START,
        frame_index=0,
        timestamp_sec=0.0,
        path=write_dummy_image(tmp_path / "shot_0000_start.jpg"),
    )
    docs = ocr.extract_from_keyframes([start, middle])
    assert len(docs) == 1
    assert docs[0].source == "ocr"
    assert "middle" in docs[0].text


@pytest.mark.unit
def test_extract_from_keyframes_only_processes_middle_frames(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.ocr_backend = "gemini"
    settings.gemini_api_key = "test-key"
    called_batches: list[list[str]] = []

    def fake_extract_gemini_batch(self, keyframes: list[KeyFrame]) -> dict[str, str]:
        called_batches.append([kf.path.name for kf in keyframes])
        return {kf.path.name: "ocr text" for kf in keyframes}

    monkeypatch.setattr(OCREngine, "_extract_gemini_batch", fake_extract_gemini_batch)

    ocr = OCREngine(settings)
    middle = KeyFrame(
        video_id="clip",
        shot_index=0,
        role=FrameRole.MIDDLE,
        frame_index=5,
        timestamp_sec=0.5,
        path=write_dummy_image(tmp_path / "shot_0000_middle.jpg"),
    )
    start = KeyFrame(
        video_id="clip",
        shot_index=0,
        role=FrameRole.START,
        frame_index=0,
        timestamp_sec=0.0,
        path=write_dummy_image(tmp_path / "shot_0000_start.jpg"),
    )
    docs = ocr.extract_from_keyframes([start, middle])
    assert len(docs) == 1
    assert called_batches == [["shot_0000_middle.jpg"]]


@pytest.mark.unit
def test_extract_from_keyframes_batches_gemini_requests(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.ocr_backend = "gemini"
    settings.gemini_api_key = "test-key"
    settings.gemini_batch_size = 2
    called_batches: list[list[str]] = []

    def fake_extract_gemini_batch(self, keyframes: list[KeyFrame]) -> dict[str, str]:
        called_batches.append([kf.path.name for kf in keyframes])
        return {kf.path.name: f"text from {kf.path.name}" for kf in keyframes}

    monkeypatch.setattr(OCREngine, "_extract_gemini_batch", fake_extract_gemini_batch)

    ocr = OCREngine(settings)
    keyframes = [
        KeyFrame(
            video_id="clip",
            shot_index=index,
            role=FrameRole.MIDDLE,
            frame_index=index,
            timestamp_sec=float(index),
            path=write_dummy_image(tmp_path / f"shot_{index:04d}_middle.jpg"),
        )
        for index in range(3)
    ]
    docs = ocr.extract_from_keyframes(keyframes)
    assert len(docs) == 3
    assert called_batches == [
        ["shot_0000_middle.jpg", "shot_0001_middle.jpg"],
        ["shot_0002_middle.jpg"],
    ]


@pytest.mark.unit
def test_parse_batch_ocr_response_maps_image_ids() -> None:
    raw = json.dumps(
        {
            "results": [
                {"image_id": "a.jpg", "text": "hello"},
                {"image_id": "b.jpg", "text": ""},
            ]
        }
    )
    parsed = _parse_batch_ocr_response(raw, ["a.jpg", "b.jpg", "c.jpg"])
    assert parsed == {"a.jpg": "hello", "b.jpg": "", "c.jpg": ""}


@pytest.mark.unit
def test_parse_batch_ocr_response_handles_invalid_json() -> None:
    parsed = _parse_batch_ocr_response("not-json", ["a.jpg"])
    assert parsed == {"a.jpg": ""}


@pytest.mark.unit
def test_is_daily_quota_exhausted_detects_per_day_limit() -> None:
    from google.genai import errors as genai_errors

    exc = genai_errors.ClientError(
        429,
        {
            "error": {
                "message": "quota exceeded",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                            }
                        ],
                    }
                ],
            }
        },
    )
    assert _is_daily_quota_exhausted(exc) is True


@pytest.mark.unit
def test_retry_after_seconds_reads_retry_info() -> None:
    from google.genai import errors as genai_errors

    exc = genai_errors.ClientError(
        429,
        {
            "error": {
                "message": "Please retry in 57.755254636s.",
                "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "57s"}],
            }
        },
    )
    assert _retry_after_seconds(exc) == 57.0


@pytest.mark.unit
def test_daily_quota_message_is_actionable() -> None:
    from google.genai import errors as genai_errors

    exc = genai_errors.ClientError(429, {"error": {"message": "quota exceeded"}})
    message = _daily_quota_message(exc)
    assert "daily request quota" in message.lower()
    assert "GEMINI_MODEL" in message


@pytest.mark.unit
def test_gemini_rate_limiter_waits_between_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    limiter = _GeminiRateLimiter(requests_per_minute=5)
    assert limiter.min_interval == 12.0

    monotonic_values = iter([0.0, 0.0, 5.0, 5.0])
    slept: list[float] = []

    monkeypatch.setattr("video_retrieval.text.ocr.time.monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        "video_retrieval.text.ocr.time.sleep",
        lambda seconds: slept.append(seconds),
    )

    limiter.wait()
    limiter.wait()
    assert slept == [7.0]


@pytest.mark.unit
def test_generate_with_retries_on_rate_limit(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from google.genai import errors as genai_errors

    settings.ocr_backend = "gemini"
    settings.gemini_api_key = "test-key"
    settings.gemini_max_retries = 3

    ocr = OCREngine(settings)
    rate_limiter = MagicMock()
    rate_limiter.min_interval = 12.0
    ocr._rate_limiter = rate_limiter
    ocr._client = MagicMock()

    response_ok = SimpleNamespace(text="hello")
    ocr._client.models.generate_content.side_effect = [
        genai_errors.APIError(429, {"error": {"message": "rate limit"}}),
        response_ok,
    ]

    slept: list[float] = []
    monkeypatch.setattr("video_retrieval.text.ocr.time.sleep", lambda seconds: slept.append(seconds))

    text = ocr._generate_with_retries({"model": "gemini-2.0-flash"})
    assert text == "hello"
    assert ocr._client.models.generate_content.call_count == 2
    assert slept == [12.0]


@pytest.mark.unit
def test_generate_with_retries_on_unavailable(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from google.genai import errors as genai_errors

    settings.ocr_backend = "gemini"
    settings.gemini_api_key = "test-key"
    settings.gemini_max_retries = 3

    ocr = OCREngine(settings)
    rate_limiter = MagicMock()
    rate_limiter.min_interval = 12.0
    ocr._rate_limiter = rate_limiter
    ocr._client = MagicMock()

    response_ok = SimpleNamespace(text="recovered")
    ocr._client.models.generate_content.side_effect = [
        genai_errors.ServerError(
            503,
            {
                "error": {
                    "code": 503,
                    "message": "This model is currently experiencing high demand.",
                    "status": "UNAVAILABLE",
                }
            },
        ),
        response_ok,
    ]

    slept: list[float] = []
    monkeypatch.setattr("video_retrieval.text.ocr.time.sleep", lambda seconds: slept.append(seconds))

    text = ocr._generate_with_retries({"model": "gemini-3.1-flash-lite"})
    assert text == "recovered"
    assert ocr._client.models.generate_content.call_count == 2
    assert slept == [15.0]


@pytest.mark.unit
def test_generate_with_retries_raises_model_unavailable(settings: Settings) -> None:
    from google.genai import errors as genai_errors

    settings.ocr_backend = "gemini"
    settings.gemini_api_key = "test-key"
    settings.gemini_model = "gemini-2.5-flash-lite"

    ocr = OCREngine(settings)
    ocr._rate_limiter = MagicMock()
    ocr._rate_limiter.min_interval = 12.0
    ocr._client = MagicMock()
    ocr._client.models.generate_content.side_effect = genai_errors.ClientError(
        404,
        {
            "error": {
                "code": 404,
                "message": "This model models/gemini-2.5-flash-lite is no longer available to new users.",
                "status": "NOT_FOUND",
            }
        },
    )

    with pytest.raises(RuntimeError, match="GEMINI_MODEL"):
        ocr._generate_with_retries({"model": settings.gemini_model})


@pytest.mark.unit
def test_is_retryable_api_error_for_503() -> None:
    from google.genai import errors as genai_errors

    exc = genai_errors.ServerError(
        503,
        {"error": {"message": "high demand", "status": "UNAVAILABLE"}},
    )
    assert _is_retryable_api_error(exc) is True


@pytest.mark.unit
def test_model_unavailable_message_mentions_setting() -> None:
    from google.genai import errors as genai_errors

    settings = Settings(gemini_model="gemini-2.5-flash-lite")
    exc = genai_errors.ClientError(404, {"error": {"message": "no longer available"}})
    message = _model_unavailable_message(settings, exc)
    assert "gemini-2.5-flash-lite" in message
    assert "GEMINI_MODEL" in message


@pytest.mark.unit
def test_gemini_generation_config_for_v3_models() -> None:
    config = _gemini_generation_config("gemini-3.5-flash", json_response=True)
    assert config is not None
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level is not None


@pytest.mark.unit
def test_gemini_generation_config_skips_older_models() -> None:
    assert _gemini_generation_config("gemini-2.0-flash", json_response=False) is None


@pytest.mark.unit
def test_gemini_ocr_requires_api_key(settings: Settings) -> None:
    settings.ocr_backend = "gemini"
    settings.gemini_api_key = ""
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        OCREngine(settings)


@pytest.mark.unit
def test_mock_asr_returns_transcript(tmp_path: Path, settings: Settings) -> None:
    asr = ASREngine(settings)
    audio = AudioTrack(video_id="clip", path=tmp_path / "clip.wav", duration_sec=1.5)
    docs = asr.transcribe(audio)
    assert len(docs) == 1
    assert docs[0].source == "asr"
    assert "clip" in docs[0].text


@pytest.mark.unit
def test_rapidocr_backend_only_processes_middle_frames(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.ocr_backend = "rapidocr"
    called: list[str] = []

    def fake_extract_rapidocr(self, image_path: Path) -> str:
        called.append(image_path.name)
        return f"local ocr {image_path.name}"

    monkeypatch.setattr(OCREngine, "_extract_rapidocr", fake_extract_rapidocr)
    ocr = OCREngine(settings)
    middle = KeyFrame(
        video_id="clip",
        shot_index=0,
        role=FrameRole.MIDDLE,
        frame_index=5,
        timestamp_sec=0.5,
        path=write_dummy_image(tmp_path / "shot_0000_middle.jpg"),
    )
    start = KeyFrame(
        video_id="clip",
        shot_index=0,
        role=FrameRole.START,
        frame_index=0,
        timestamp_sec=0.0,
        path=write_dummy_image(tmp_path / "shot_0000_start.jpg"),
    )
    docs = ocr.extract_from_keyframes([start, middle])
    assert called == ["shot_0000_middle.jpg"]
    assert len(docs) == 1
    assert docs[0].text == "local ocr shot_0000_middle.jpg"


@pytest.mark.unit
def test_rapidocr_skips_empty_text(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.ocr_backend = "rapidocr"
    monkeypatch.setattr(OCREngine, "_extract_rapidocr", lambda self, path: "")
    ocr = OCREngine(settings)
    middle = KeyFrame(
        video_id="clip",
        shot_index=1,
        role=FrameRole.MIDDLE,
        frame_index=8,
        timestamp_sec=1.0,
        path=write_dummy_image(tmp_path / "shot_0001_middle.jpg"),
    )
    assert ocr.extract_from_keyframes([middle]) == []


@pytest.mark.unit
def test_unknown_ocr_backend_raises(settings: Settings) -> None:
    settings.ocr_backend = "colab"
    with pytest.raises(ValueError, match="OCR_BACKEND"):
        OCREngine(settings)


@pytest.mark.unit
def test_rapidocr_texts_handles_versions() -> None:
    assert _rapidocr_texts(SimpleNamespace(txts=("Hello", "World"))) == "Hello\nWorld"
    assert _rapidocr_texts([["box", "VTV24", 0.9], ["box", "news", 0.8]]) == "VTV24\nnews"
    assert _rapidocr_texts(([["box", "logo", 0.99]], 0.12)) == "logo"
    assert _rapidocr_texts(None) == ""
