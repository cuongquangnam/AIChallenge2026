"""Unit tests for shared Gemini client retry and rate limiting."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from video_retrieval.config import Settings
from video_retrieval.text.gemini_client import (
    GeminiClient,
    _GeminiRateLimiter,
    daily_quota_message,
    is_daily_quota_exhausted,
    is_retryable_api_error,
    model_unavailable_message,
    reset_gemini_client,
    retry_after_seconds,
)


@pytest.fixture(autouse=True)
def _reset_client() -> None:
    reset_gemini_client()
    yield
    reset_gemini_client()


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
    assert is_daily_quota_exhausted(exc) is True


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
    assert retry_after_seconds(exc) == 57.0


@pytest.mark.unit
def test_daily_quota_message_is_actionable() -> None:
    from google.genai import errors as genai_errors

    exc = genai_errors.ClientError(429, {"error": {"message": "quota exceeded"}})
    message = daily_quota_message(exc)
    assert "daily request quota" in message.lower()
    assert "GEMINI_MODEL" in message


@pytest.mark.unit
def test_gemini_rate_limiter_waits_between_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    limiter = _GeminiRateLimiter(requests_per_minute=5)
    assert limiter.min_interval == 12.0

    monotonic_values = iter([0.0, 0.0, 5.0, 5.0])
    slept: list[float] = []

    monkeypatch.setattr(
        "video_retrieval.text.gemini_client.time.monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(
        "video_retrieval.text.gemini_client.time.sleep",
        lambda seconds: slept.append(seconds),
    )

    limiter.wait()
    limiter.wait()
    assert slept == [7.0]


@pytest.mark.unit
def test_generate_with_retries_on_rate_limit(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from google.genai import errors as genai_errors

    settings.gemini_api_key = "test-key"
    settings.gemini_max_retries = 3

    client = GeminiClient(settings)
    rate_limiter = MagicMock()
    rate_limiter.min_interval = 12.0
    client._rate_limiter = rate_limiter
    client._client = MagicMock()

    response_ok = SimpleNamespace(text="hello")
    client._client.models.generate_content.side_effect = [
        genai_errors.APIError(429, {"error": {"message": "rate limit"}}),
        response_ok,
    ]

    slept: list[float] = []
    monkeypatch.setattr(
        "video_retrieval.text.gemini_client.time.sleep",
        lambda seconds: slept.append(seconds),
    )

    text = client._generate_with_retries({"model": "gemini-2.0-flash"}, component="test")
    assert text == "hello"
    assert client._client.models.generate_content.call_count == 2
    assert slept == [12.0]


@pytest.mark.unit
def test_generate_with_retries_on_unavailable(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from google.genai import errors as genai_errors

    settings.gemini_api_key = "test-key"
    settings.gemini_max_retries = 3

    client = GeminiClient(settings)
    rate_limiter = MagicMock()
    rate_limiter.min_interval = 12.0
    client._rate_limiter = rate_limiter
    client._client = MagicMock()

    response_ok = SimpleNamespace(text="recovered")
    client._client.models.generate_content.side_effect = [
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
    monkeypatch.setattr(
        "video_retrieval.text.gemini_client.time.sleep",
        lambda seconds: slept.append(seconds),
    )

    text = client._generate_with_retries({"model": "gemini-3.1-flash-lite"}, component="test")
    assert text == "recovered"
    assert client._client.models.generate_content.call_count == 2
    assert slept == [15.0]


@pytest.mark.unit
def test_generate_with_retries_raises_model_unavailable(settings: Settings) -> None:
    from google.genai import errors as genai_errors

    settings.gemini_api_key = "test-key"
    settings.gemini_model = "gemini-2.5-flash-lite"

    client = GeminiClient(settings)
    client._rate_limiter = MagicMock()
    client._rate_limiter.min_interval = 12.0
    client._client = MagicMock()
    client._client.models.generate_content.side_effect = genai_errors.ClientError(
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
        client._generate_with_retries({"model": settings.gemini_model}, component="test")


@pytest.mark.unit
def test_is_retryable_api_error_for_503() -> None:
    from google.genai import errors as genai_errors

    exc = genai_errors.ServerError(
        503,
        {"error": {"message": "high demand", "status": "UNAVAILABLE"}},
    )
    assert is_retryable_api_error(exc) is True


@pytest.mark.unit
def test_model_unavailable_message_mentions_setting() -> None:
    from google.genai import errors as genai_errors

    settings = Settings(gemini_model="gemini-2.5-flash-lite")
    exc = genai_errors.ClientError(404, {"error": {"message": "no longer available"}})
    message = model_unavailable_message(settings, exc)
    assert "gemini-2.5-flash-lite" in message
    assert "GEMINI_MODEL" in message
