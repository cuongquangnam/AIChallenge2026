"""Unit tests for query stage logging."""

import logging

import pytest

from video_retrieval.log_setup import (
    _short_location,
    configure_logging,
    format_stage_message,
    reset_logging,
)
from video_retrieval.query_stages import log_query_stage


@pytest.mark.unit
def test_log_query_stage(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="video_retrieval.query_stages")
    log_query_stage("kis", "extract_events", events=3)
    assert "[kis] extract_events events=3" in caplog.text


@pytest.mark.unit
def test_log_query_stage_points_at_caller(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="video_retrieval.query_stages")
    log_query_stage("qa", "vlm_batch", batch="1/2")
    record = caplog.records[-1]
    assert record.filename == "test_query_stages.py"
    assert record.funcName == "test_log_query_stage_points_at_caller"
    assert record.lineno > 0


@pytest.mark.unit
def test_configure_logging_attaches_rich_handler() -> None:
    from rich.logging import RichHandler

    reset_logging()
    try:
        configure_logging(force=True)
        handlers = logging.getLogger("video_retrieval").handlers
        assert any(isinstance(handler, RichHandler) for handler in handlers)
        assert logging.getLogger("video_retrieval").propagate is False
    finally:
        reset_logging()


@pytest.mark.unit
def test_format_stage_message_plain_text() -> None:
    assert format_stage_message("kis", "start", limit=100) == "[kis] start limit=100"
    assert format_stage_message("qa", "done") == "[qa] done"


@pytest.mark.unit
def test_short_location_is_package_relative() -> None:
    path = "/Users/me/AIChallenge2026/src/video_retrieval/events/searcher.py"
    assert _short_location(path) == "events/searcher.py"
    assert _short_location("/tmp/foo.py") == "foo.py"
