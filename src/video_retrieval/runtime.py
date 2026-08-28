from __future__ import annotations

import logging
from dataclasses import dataclass

from video_retrieval.config import Settings, get_settings
from video_retrieval.events.searcher import EventChainSearcher
from video_retrieval.qa.service import QAService
from video_retrieval.search.kis_service import KisService
from video_retrieval.search.service import SearchService
from video_retrieval.search.trake import TrakeService

logger = logging.getLogger(__name__)


@dataclass
class AppRuntime:
    """Process-wide search + chain task services (models loaded once)."""

    settings: Settings
    search: SearchService
    chain_search: EventChainSearcher
    kis: KisService
    qa: QAService
    trake: TrakeService


_runtime: AppRuntime | None = None


def init_runtime(settings: Settings | None = None, *, force: bool = False) -> AppRuntime:
    """Build or rebuild shared services for this process."""
    global _runtime
    if _runtime is not None and not force:
        return _runtime
    cfg = settings or get_settings()
    logger.info("Initializing shared search runtime")
    search = SearchService(cfg)
    chain_search = EventChainSearcher(cfg, search=search)
    _runtime = AppRuntime(
        settings=cfg,
        search=search,
        chain_search=chain_search,
        kis=KisService(cfg, search=search, chain_search=chain_search),
        qa=QAService(cfg, search=search, chain_search=chain_search),
        trake=TrakeService(cfg, search=search, chain_search=chain_search),
    )
    logger.info("Shared search runtime ready")
    return _runtime


def get_runtime() -> AppRuntime:
    if _runtime is None:
        return init_runtime()
    return _runtime


def reset_runtime() -> None:
    """Clear cached runtime (tests only)."""
    global _runtime
    _runtime = None
