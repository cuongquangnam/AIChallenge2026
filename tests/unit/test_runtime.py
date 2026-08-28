"""Unit tests for process-wide runtime singleton."""

from unittest.mock import patch

import pytest

from video_retrieval.config import Settings
from video_retrieval.runtime import get_runtime, init_runtime, reset_runtime


@pytest.mark.unit
def test_init_runtime_reuses_search_service() -> None:
    reset_runtime()
    settings = Settings(visual_backend="mock", chain_rerank_enabled=False)
    with patch("video_retrieval.runtime.SearchService") as mock_search_cls:
        mock_search_cls.return_value = object()
        init_runtime(settings)
        first = get_runtime().search
        second = get_runtime().search
    assert first is second
    assert mock_search_cls.call_count == 1
    reset_runtime()
