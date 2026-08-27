from __future__ import annotations

import logging
import traceback

logger = logging.getLogger(__name__)


def log_gemini_failure(
    *,
    component: str,
    model: str,
    exc: BaseException,
    query: str | None = None,
    raw_response: str | None = None,
    fallback: str | None = None,
    attempt: int | None = None,
    max_attempts: int | None = None,
) -> None:
    """Emit a multi-line ERROR log when a Gemini call fails."""
    lines = [
        f"Gemini {component} failed",
        f"  model={model!r}",
        f"  error_type={type(exc).__name__}",
        f"  error={exc!r}",
    ]
    if attempt is not None and max_attempts is not None:
        lines.append(f"  attempt={attempt}/{max_attempts}")
    if query:
        preview = query if len(query) <= 400 else query[:400] + "…"
        lines.append(f"  query={preview!r}")
    if raw_response is not None:
        preview = raw_response if len(raw_response) <= 800 else raw_response[:800] + "…"
        lines.append(f"  raw_response={preview!r}")
    if fallback:
        lines.append(f"  fallback={fallback}")
    lines.append("  traceback:")
    lines.extend("    " + row for row in traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error("\n".join(lines))
