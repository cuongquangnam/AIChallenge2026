from __future__ import annotations

import logging
import os
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.logging import RichHandler

_PACKAGE_LOGGER = "video_retrieval"
_PACKAGE_MARKER = "/video_retrieval/"


class _StageHighlighter(RegexHighlighter):
    """Color `[kis]` / `[qa]` / `[trake]` / `[search]` tags in progress lines."""

    highlights = [
        r"(?P<cyan>\[kis\])",
        r"(?P<magenta>\[qa\])",
        r"(?P<green>\[trake\])",
        r"(?P<blue>\[search\])",
    ]


class _FlushingRichHandler(RichHandler):
    """Rich handler that flushes after every record so pipeline progress shows live."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()

    def render(self, *, record, traceback, message_renderable):
        path = _short_location(record.pathname)
        level = self.get_level_text(record)
        time_format = None if self.formatter is None else self.formatter.datefmt
        log_time = datetime.fromtimestamp(record.created)
        return self._log_render(
            self.console,
            [message_renderable] if not traceback else [message_renderable, traceback],
            log_time=log_time,
            time_format=time_format,
            level=level,
            path=path,
            line_no=record.lineno,
            link_path=record.pathname if self.enable_link_path else None,
        )


def _short_location(pathname: str) -> str:
    """Turn an absolute path into `events/searcher.py` when it is in this package."""
    normalized = pathname.replace("\\", "/")
    index = normalized.rfind(_PACKAGE_MARKER)
    if index >= 0:
        return normalized[index + len(_PACKAGE_MARKER) :]
    return os.path.basename(pathname)


def rich_handler(*, level: int | str = logging.INFO) -> logging.Handler:
    """Factory for uvicorn dictConfig and configure_logging()."""
    handler = _FlushingRichHandler(
        console=Console(stderr=True, highlight=False),
        highlighter=_StageHighlighter(),
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=True,
        markup=False,
        log_time_format="[%H:%M:%S]",
        omit_repeated_times=False,
    )
    handler.setLevel(level)
    return handler


def configure_logging(level: int = logging.INFO, *, force: bool = False) -> None:
    """Attach a colored stderr handler to video_retrieval (idempotent)."""
    if not force and os.environ.get("PYTEST_CURRENT_TEST"):
        return
    logger = logging.getLogger(_PACKAGE_LOGGER)
    logger.setLevel(level)
    logger.propagate = False
    for existing in list(logger.handlers):
        if isinstance(existing, RichHandler):
            existing.setLevel(level)
            return
    logger.addHandler(rich_handler(level=level))


def uvicorn_log_config() -> dict[str, Any]:
    """Uvicorn logging dict that keeps access logs and our progress handler."""
    from uvicorn.config import LOGGING_CONFIG

    config = deepcopy(LOGGING_CONFIG)
    config["disable_existing_loggers"] = False
    config["handlers"]["app"] = {
        "()": "video_retrieval.log_setup.rich_handler",
        "level": "INFO",
    }
    config["loggers"][_PACKAGE_LOGGER] = {
        "handlers": ["app"],
        "level": "INFO",
        "propagate": False,
    }
    return config


def format_stage_message(task: str, stage: str, **details: object) -> str:
    """Plain `[task] stage key=value` line; location comes from the log record."""
    if details:
        parts = " ".join(f"{key}={value!r}" for key, value in details.items())
        return f"[{task}] {stage} {parts}"
    return f"[{task}] {stage}"


def reset_logging() -> None:
    """Tests only: drop package handlers."""
    logger = logging.getLogger(_PACKAGE_LOGGER)
    logger.handlers.clear()
    logger.propagate = True
    sys.stderr.flush()
