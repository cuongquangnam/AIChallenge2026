from __future__ import annotations

import logging
import re
import time
from typing import Any

from video_retrieval.config import Settings, get_settings
from video_retrieval.text.gemini_config import gemini_generate_config

logger = logging.getLogger(__name__)

RETRYABLE_API_CODES = {429, 500, 502, 503, 504}

_client: GeminiClient | None = None
_client_key: tuple[str, str, int, int] | None = None


class GeminiClient:
    """Shared Gemini wrapper with RPM throttling and rate-limit backoff."""

    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiClient")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError("Install ML extras: pip install '.[ml]'") from exc

        self.settings = settings
        self.model = settings.gemini_model
        self._types = types
        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self._rate_limiter = _GeminiRateLimiter(settings.gemini_rpm)

    def generate_text(
        self,
        prompt: str,
        *,
        json_response: bool = False,
        component: str = "gemini",
    ) -> str:
        return self.generate_parts(
            [self._types.Part.from_text(text=prompt)],
            json_response=json_response,
            component=component,
        )

    def generate_parts(
        self,
        parts: list[Any],
        *,
        json_response: bool = False,
        component: str = "gemini",
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "contents": [self._types.Content(role="user", parts=parts)],
        }
        config = gemini_generate_config(self.model, json_response=json_response)
        if config is not None:
            kwargs["config"] = config
        return self._generate_with_retries(kwargs, component=component)

    def _generate_with_retries(self, kwargs: dict[str, Any], *, component: str) -> str:
        from google.genai import errors as genai_errors

        retry_delay = self._rate_limiter.min_interval
        max_retries = max(1, int(self.settings.gemini_max_retries))

        for attempt in range(max_retries):
            self._rate_limiter.wait()
            try:
                response = self._client.models.generate_content(**kwargs)
                return (response.text or "").strip()
            except genai_errors.APIError as exc:
                if getattr(exc, "code", None) == 404:
                    raise RuntimeError(model_unavailable_message(self.settings, exc)) from exc
                if is_daily_quota_exhausted(exc):
                    raise RuntimeError(daily_quota_message(exc)) from exc
                if not is_retryable_api_error(exc) or attempt >= max_retries - 1:
                    raise
                wait_seconds = retry_backoff_seconds(
                    exc,
                    attempt=attempt,
                    base_delay=retry_delay,
                )
                logger.warning(
                    "Gemini %s failed (model=%s, attempt=%s/%s, code=%s); "
                    "retrying in %.0fs: %r",
                    component,
                    self.model,
                    attempt + 1,
                    max_retries,
                    getattr(exc, "code", "error"),
                    wait_seconds,
                    exc,
                )
                time.sleep(wait_seconds)

        return ""


class _GeminiRateLimiter:
    """Enforce a minimum interval between Gemini API calls."""

    def __init__(self, requests_per_minute: int):
        rpm = max(requests_per_minute, 1)
        self.min_interval = 60.0 / rpm
        self._last_request_at: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self._last_request_at = time.monotonic()


def get_gemini_client(
    settings: Settings | None = None,
    *,
    force: bool = False,
) -> GeminiClient | None:
    """Return a process-wide Gemini client (shared rate limiter), or None."""
    global _client, _client_key
    cfg = settings or get_settings()
    if not cfg.gemini_api_key:
        return None
    key = (
        cfg.gemini_api_key,
        cfg.gemini_model,
        int(cfg.gemini_rpm),
        int(cfg.gemini_max_retries),
    )
    if _client is not None and not force and _client_key == key:
        return _client
    _client = GeminiClient(cfg)
    _client_key = key
    return _client


def reset_gemini_client() -> None:
    """Clear cached client (tests only)."""
    global _client, _client_key
    _client = None
    _client_key = None


def retry_backoff_seconds(
    exc: Exception,
    *,
    attempt: int,
    base_delay: float,
) -> float:
    code = getattr(exc, "code", None)
    return max(
        retry_after_seconds(exc),
        base_delay * (2**attempt),
        15.0 if code in {500, 502, 503, 504} else 0.0,
    )


def parse_error_details(exc: Exception) -> list[dict]:
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return []
    error = details.get("error", details)
    if not isinstance(error, dict):
        return []
    raw_details = error.get("details", [])
    return raw_details if isinstance(raw_details, list) else []


def api_error_status(exc: Exception) -> str:
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        error = details.get("error", details)
        if isinstance(error, dict):
            return str(error.get("status") or "")
    return str(getattr(exc, "status", "") or "")


def is_retryable_api_error(exc: Exception) -> bool:
    if is_daily_quota_exhausted(exc):
        return False
    code = getattr(exc, "code", None)
    if code in RETRYABLE_API_CODES:
        return True
    status = api_error_status(exc).upper()
    message = str(getattr(exc, "message", "")).lower()
    return status in {"UNAVAILABLE", "RESOURCE_EXHAUSTED", "INTERNAL"} or "high demand" in message


def is_daily_quota_exhausted(exc: Exception) -> bool:
    for item in parse_error_details(exc):
        if not isinstance(item, dict):
            continue
        if item.get("@type", "").endswith("QuotaFailure"):
            for violation in item.get("violations", []):
                quota_id = str(violation.get("quotaId", ""))
                if "PerDay" in quota_id or "PerDay" in str(violation.get("quotaMetric", "")):
                    return True
    message = str(getattr(exc, "message", "")) + str(getattr(exc, "details", ""))
    return "PerDay" in message and "quota" in message.lower()


def retry_after_seconds(exc: Exception) -> float:
    for item in parse_error_details(exc):
        if not isinstance(item, dict):
            continue
        if item.get("@type", "").endswith("RetryInfo"):
            delay = item.get("retryDelay")
            if isinstance(delay, str) and delay.endswith("s"):
                try:
                    return float(delay[:-1])
                except ValueError:
                    pass
    message = str(getattr(exc, "message", ""))
    match = re.search(r"retry in ([0-9.]+)s", message, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


def daily_quota_message(exc: Exception) -> str:
    return (
        "Gemini daily request quota exceeded for this model. "
        "Free tier limits are per-model and much lower than RPM (e.g. gemini-3.5-flash: 20/day). "
        "Wait for the quota to reset, switch GEMINI_MODEL (e.g. gemini-3.1-flash-lite), "
        "increase GEMINI_BATCH_SIZE to use fewer requests, or enable billing. "
        f"API message: {getattr(exc, 'message', exc)}"
    )


def model_unavailable_message(settings: Settings, exc: Exception) -> str:
    return (
        f"Gemini model {settings.gemini_model!r} is not available for this API key. "
        "Update GEMINI_MODEL to a current model (e.g. gemini-3.1-flash-lite or gemini-2.0-flash). "
        f"API message: {getattr(exc, 'message', exc)}"
    )
