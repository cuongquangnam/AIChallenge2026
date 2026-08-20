from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from video_retrieval.config import Settings
from video_retrieval.models import FrameRole, KeyFrame, TextDocument

_OCR_BACKENDS = {"mock", "gemini", "rapidocr"}


class OCREngine:
    """OCR via Gemini, on-device RapidOCR, or a mock backend."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = (settings.ocr_backend or "mock").strip().lower()
        if self.backend not in _OCR_BACKENDS:
            raise ValueError(
                f"Unknown OCR_BACKEND={settings.ocr_backend!r}; "
                "expected mock, gemini, or rapidocr"
            )
        self._client = None
        self._rate_limiter: _GeminiRateLimiter | None = None
        self._rapidocr_local = threading.local()
        if self.backend == "gemini":
            self._init_gemini()

    def _init_gemini(self) -> None:
        if not self.settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for OCR_BACKEND=gemini")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError("Install ML extras: pip install '.[ml]'") from exc

        self._client = genai.Client(
            api_key=self.settings.gemini_api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self._rate_limiter = _GeminiRateLimiter(self.settings.gemini_rpm)

    def extract_from_keyframes(self, keyframes: list[KeyFrame]) -> list[TextDocument]:
        ocr_keyframes = [kf for kf in keyframes if kf.role == FrameRole.MIDDLE]
        if self.backend == "gemini":
            return self._extract_gemini_keyframes(ocr_keyframes)
        if self.backend == "rapidocr":
            return self._extract_rapidocr_keyframes(ocr_keyframes)

        docs: list[TextDocument] = []
        for kf in ocr_keyframes:
            text = self.extract_text(kf.path)
            if not text.strip():
                continue
            docs.append(self._to_text_document(kf, text))
        return docs

    def _extract_gemini_keyframes(self, keyframes: list[KeyFrame]) -> list[TextDocument]:
        docs: list[TextDocument] = []
        batch_size = max(self.settings.gemini_batch_size, 1)
        batches = list(_chunked(keyframes, batch_size))
        total_batches = len(batches)

        for batch_index, batch in enumerate(batches, start=1):
            print(
                f"OCR batch {batch_index}/{total_batches}: "
                f"{len(batch)} frame(s), "
                f"shots {batch[0].shot_index}-{batch[-1].shot_index}"
            )
            text_by_image = self._extract_gemini_batch(batch)
            for kf in batch:
                text = text_by_image.get(kf.path.name, "")
                if not text.strip():
                    continue
                docs.append(self._to_text_document(kf, text))
        return docs

    def _extract_rapidocr_keyframes(self, keyframes: list[KeyFrame]) -> list[TextDocument]:
        if not keyframes:
            return []
        workers = max(self.settings.ocr_workers, 1)
        total = len(keyframes)
        print(f"OCR rapidocr: {total} middle frame(s), {workers} worker(s)")

        def _one(kf: KeyFrame) -> tuple[KeyFrame, str]:
            return kf, self._extract_rapidocr(kf.path)

        results: list[tuple[KeyFrame, str]] = []
        if workers == 1 or total == 1:
            for index, kf in enumerate(keyframes, start=1):
                results.append(_one(kf))
                if index == 1 or index % 50 == 0 or index == total:
                    print(f"OCR {index}/{total}")
        else:
            done = 0
            with ThreadPoolExecutor(max_workers=min(workers, total)) as pool:
                futures = [pool.submit(_one, kf) for kf in keyframes]
                for future in as_completed(futures):
                    results.append(future.result())
                    done += 1
                    if done == 1 or done % 50 == 0 or done == total:
                        print(f"OCR {done}/{total}")

        results.sort(key=lambda item: (item[0].shot_index, item[0].frame_index or 0))
        docs: list[TextDocument] = []
        for kf, text in results:
            if text.strip():
                docs.append(self._to_text_document(kf, text))
        return docs

    def extract_text(self, image_path: Path) -> str:
        if self.backend == "gemini":
            return self._extract_gemini(image_path)
        if self.backend == "rapidocr":
            return self._extract_rapidocr(image_path)
        # Mock: tag middle frames so textual search is exerciseable end-to-end.
        name = image_path.stem
        if FrameRole.MIDDLE.value in name:
            return f"mock ocr text from {image_path.name}"
        return ""

    def _extract_gemini_batch(self, keyframes: list[KeyFrame]) -> dict[str, str]:
        from google.genai import types
        from PIL import Image

        image_ids = [kf.path.name for kf in keyframes]
        parts: list[types.Part] = [
            types.Part.from_text(text=_batch_ocr_instructions(image_ids)),
        ]
        for kf in keyframes:
            image = Image.open(kf.path)
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            parts.append(types.Part.from_text(text=f"[IMAGE: {kf.path.name}]"))
            parts.append(types.Part(image))

        kwargs: dict = {
            "model": self.settings.gemini_model,
            "contents": [types.Content(role="user", parts=parts)],
        }
        config = _gemini_ocr_config(self.settings.gemini_model, json_response=True)
        if config is not None:
            kwargs["config"] = config

        raw = self._generate_with_retries(kwargs)
        return _parse_batch_ocr_response(raw, image_ids)

    def _extract_gemini(self, image_path: Path) -> str:
        from google.genai import types
        from PIL import Image

        image = Image.open(image_path)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        prompt = (
            "Extract all visible text from this video frame. "
            "Return plain text only, preserve line breaks for distinct regions. "
            "If there is no visible text, return an empty response."
        )
        kwargs: dict = {
            "model": self.settings.gemini_model,
            "contents": [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part(image),
                    ],
                )
            ],
        }
        config = _gemini_ocr_config(self.settings.gemini_model, json_response=False)
        if config is not None:
            kwargs["config"] = config

        return self._generate_with_retries(kwargs)

    def _extract_rapidocr(self, image_path: Path) -> str:
        engine = getattr(self._rapidocr_local, "engine", None)
        if engine is None:
            engine = _new_rapidocr()
            self._rapidocr_local.engine = engine
        return _rapidocr_texts(engine(str(image_path)))

    def _generate_with_retries(self, kwargs: dict) -> str:
        from google.genai import errors as genai_errors

        assert self._rate_limiter is not None
        retry_delay = self._rate_limiter.min_interval
        max_retries = self.settings.gemini_max_retries

        for attempt in range(max_retries):
            self._rate_limiter.wait()
            try:
                response = self._client.models.generate_content(**kwargs)
                return (response.text or "").strip()
            except genai_errors.APIError as exc:
                if getattr(exc, "code", None) == 404:
                    raise RuntimeError(_model_unavailable_message(self.settings, exc)) from exc
                if _is_daily_quota_exhausted(exc):
                    raise RuntimeError(_daily_quota_message(exc)) from exc
                if not _is_retryable_api_error(exc) or attempt >= max_retries - 1:
                    raise
                wait_seconds = max(
                    _retry_after_seconds(exc),
                    retry_delay * (2**attempt),
                    15.0 if getattr(exc, "code", None) in {500, 502, 503, 504} else 0.0,
                )
                print(
                    f"Gemini {getattr(exc, 'code', 'error')} "
                    f"({_api_error_status(exc) or 'transient'}); "
                    f"retrying in {wait_seconds:.0f}s "
                    f"({attempt + 1}/{max_retries})"
                )
                time.sleep(wait_seconds)

        return ""

    @staticmethod
    def _to_text_document(kf: KeyFrame, text: str) -> TextDocument:
        return TextDocument(
            doc_id=f"{kf.video_id}:ocr:{kf.shot_index}:{kf.role.value}",
            video_id=kf.video_id,
            source="ocr",
            text=text.strip(),
            shot_index=kf.shot_index,
            frame_index=kf.frame_index,
            role=kf.role,
            start_sec=kf.timestamp_sec,
            end_sec=kf.timestamp_sec,
            metadata={"keyframe_path": str(kf.path)},
        )


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


def _new_rapidocr():
    try:
        from rapidocr import RapidOCR
    except ImportError:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise ImportError(
                "Install ML extras: pip install '.[ml]' before OCR_BACKEND=rapidocr"
            ) from exc
    return RapidOCR()


def _rapidocr_texts(result: object) -> str:
    """Normalize RapidOCR 2.x / 3.x return values into a single string."""
    if result is None:
        return ""
    txts = getattr(result, "txts", None)
    if txts:
        return "\n".join(str(text) for text in txts if str(text).strip()).strip()
    if isinstance(result, tuple) and result:
        result = result[0]
    if not result:
        return ""
    if isinstance(result, dict):
        return str(result.get("text") or "").strip()
    lines: list[str] = []
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
            elif isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], str):
                text = item[1].strip()
            elif isinstance(item, str):
                text = item.strip()
            else:
                continue
            if text:
                lines.append(text)
    return "\n".join(lines)


def _chunked(items: list[KeyFrame], size: int) -> list[list[KeyFrame]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _batch_ocr_instructions(image_ids: list[str]) -> str:
    ids_json = json.dumps(image_ids)
    return (
        "Extract all visible text from each labeled video frame below. "
        f"The image_id values are exactly: {ids_json}. "
        'Return JSON only with shape {"results":[{"image_id":"<filename>","text":"..."}]}. '
        "Include one entry per image_id, in the same order. "
        "Use an empty string when a frame has no visible text."
    )


def _parse_batch_ocr_response(raw: str, image_ids: list[str]) -> dict[str, str]:
    text_by_id = {image_id: "" for image_id in image_ids}
    if not raw.strip():
        return text_by_id

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return text_by_id

    results = payload.get("results", [])
    if not isinstance(results, list):
        return text_by_id

    for item in results:
        if not isinstance(item, dict):
            continue
        image_id = item.get("image_id")
        if image_id in text_by_id:
            text_by_id[image_id] = str(item.get("text") or "").strip()
    return text_by_id


def _parse_error_details(exc: Exception) -> list[dict]:
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return []
    error = details.get("error", details)
    if not isinstance(error, dict):
        return []
    raw_details = error.get("details", [])
    return raw_details if isinstance(raw_details, list) else []


RETRYABLE_API_CODES = {429, 500, 502, 503, 504}


def _api_error_status(exc: Exception) -> str:
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        error = details.get("error", details)
        if isinstance(error, dict):
            return str(error.get("status") or "")
    return str(getattr(exc, "status", "") or "")


def _is_retryable_api_error(exc: Exception) -> bool:
    if _is_daily_quota_exhausted(exc):
        return False
    code = getattr(exc, "code", None)
    if code in RETRYABLE_API_CODES:
        return True
    status = _api_error_status(exc).upper()
    message = str(getattr(exc, "message", "")).lower()
    return status in {"UNAVAILABLE", "RESOURCE_EXHAUSTED", "INTERNAL"} or "high demand" in message


def _is_daily_quota_exhausted(exc: Exception) -> bool:
    for item in _parse_error_details(exc):
        if not isinstance(item, dict):
            continue
        if item.get("@type", "").endswith("QuotaFailure"):
            for violation in item.get("violations", []):
                quota_id = str(violation.get("quotaId", ""))
                if "PerDay" in quota_id or "PerDay" in str(violation.get("quotaMetric", "")):
                    return True
    message = str(getattr(exc, "message", "")) + str(getattr(exc, "details", ""))
    return "PerDay" in message and "quota" in message.lower()


def _retry_after_seconds(exc: Exception) -> float:
    for item in _parse_error_details(exc):
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


def _daily_quota_message(exc: Exception) -> str:
    return (
        "Gemini daily request quota exceeded for this model. "
        "Free tier limits are per-model and much lower than RPM (e.g. gemini-3.5-flash: 20/day). "
        "Wait for the quota to reset, switch GEMINI_MODEL (e.g. gemini-3.1-flash-lite), "
        "increase GEMINI_BATCH_SIZE to use fewer requests, or enable billing. "
        f"API message: {getattr(exc, 'message', exc)}"
    )


def _model_unavailable_message(settings: Settings, exc: Exception) -> str:
    return (
        f"Gemini model {settings.gemini_model!r} is not available for this API key. "
        "Update GEMINI_MODEL to a current model (e.g. gemini-3.1-flash-lite or gemini-2.0-flash). "
        f"API message: {getattr(exc, 'message', exc)}"
    )


def _gemini_ocr_config(model: str, *, json_response: bool):
    """Build Gemini generation config for OCR requests."""
    from google.genai import types

    config_kwargs: dict = {}
    if json_response:
        config_kwargs["response_mime_type"] = "application/json"
    thinking_level = _gemini_thinking_level(model)
    if thinking_level is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level
        )
    if not config_kwargs:
        return None
    return types.GenerateContentConfig(**config_kwargs)


def _gemini_thinking_level(model: str) -> str | None:
    name = (model or "").strip().lower()
    if not name.startswith("gemini-3"):
        return None
    if "pro" in name:
        return "low"
    return "minimal"


# Backwards-compatible alias for tests.
_gemini_generation_config = _gemini_ocr_config
