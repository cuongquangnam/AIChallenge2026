from __future__ import annotations

import logging

from video_retrieval.log_setup import format_stage_message

logger = logging.getLogger(__name__)


def log_query_stage(task: str, stage: str, **details: object) -> None:
    """Log pipeline progress for KIS / QA / TRAKE / search requests.

    ``stacklevel=2`` so file:line points at the caller, not this helper.
    """
    logger.info(format_stage_message(task, stage, **details), stacklevel=2)
