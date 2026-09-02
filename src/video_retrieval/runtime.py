from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Union

from video_retrieval.config import Settings, get_settings
from video_retrieval.encoders.pooled import PooledVisualEncoder
from video_retrieval.encoders.visual import VisualEncoder
from video_retrieval.events.rerank import CrossEncoderReranker, PooledCrossEncoderReranker
from video_retrieval.events.searcher import EventChainSearcher
from video_retrieval.model_pool import ModelPool
from video_retrieval.qa.service import QAService
from video_retrieval.search.kis_service import KisService
from video_retrieval.search.service import SearchService
from video_retrieval.search.trake import TrakeService

logger = logging.getLogger(__name__)

Reranker = Union[CrossEncoderReranker, PooledCrossEncoderReranker]


@dataclass
class AppRuntime:
    """Process-wide search + chain task services (models loaded once)."""

    settings: Settings
    search: SearchService | None
    chain_search: EventChainSearcher | None
    kis: KisService | None
    qa: QAService | None
    trake: TrakeService | None
    siglip_pool: ModelPool[VisualEncoder] | None = None
    rerank_pool: ModelPool[CrossEncoderReranker] | None = None


_runtime: AppRuntime | None = None


def _build_siglip_pool(cfg: Settings) -> tuple[PooledVisualEncoder, ModelPool[VisualEncoder]]:
    pool_size = max(1, cfg.model_pool_size)
    pool: ModelPool[VisualEncoder] = ModelPool(
        lambda _index: VisualEncoder(cfg, load_beit=False),
        size=pool_size,
        name="SigLIP",
    )
    return PooledVisualEncoder(pool, cfg), pool


def _build_reranker(cfg: Settings) -> tuple[Reranker | None, ModelPool[CrossEncoderReranker] | None]:
    if not cfg.chain_rerank_enabled:
        return None, None
    pool_size = max(1, cfg.model_pool_size)
    logger.info("Loading chain rerank pool (%s)", cfg.chain_rerank_model_id)
    pool: ModelPool[CrossEncoderReranker] = ModelPool(
        lambda _index: CrossEncoderReranker(cfg),
        size=pool_size,
        name="BLIP-ITM",
    )
    return PooledCrossEncoderReranker(pool, settings=cfg), pool


def build_task_runtime(settings: Settings | None = None) -> AppRuntime:
    """Build full search + KIS/QA/TRAKE services (local process or Colab worker)."""
    cfg = settings or get_settings()
    logger.info("Initializing task runtime")
    visual, siglip_pool = _build_siglip_pool(cfg)
    search = SearchService(cfg, visual=visual)
    reranker, rerank_pool = _build_reranker(cfg)
    chain_search = EventChainSearcher(cfg, search=search, reranker=reranker)
    runtime = AppRuntime(
        settings=cfg,
        search=search,
        chain_search=chain_search,
        kis=KisService(cfg, search=search, chain_search=chain_search),
        qa=QAService(cfg, search=search, chain_search=chain_search),
        trake=TrakeService(cfg, search=search, chain_search=chain_search),
        siglip_pool=siglip_pool,
        rerank_pool=rerank_pool,
    )
    logger.info(
        "Task runtime ready (SigLIP pool=%s, BLIP pool=%s)",
        siglip_pool.size if siglip_pool else 0,
        rerank_pool.size if rerank_pool else 0,
    )
    return runtime


def init_runtime(settings: Settings | None = None, *, force: bool = False) -> AppRuntime:
    """Build or rebuild shared services for this process."""
    global _runtime
    if _runtime is not None and not force:
        return _runtime
    cfg = settings or get_settings()
    if cfg.uses_remote_compute:
        logger.info(
            "Remote compute enabled (REMOTE_COMPUTE=colab); skipping local model load"
        )
        _runtime = AppRuntime(
            settings=cfg,
            search=None,  # type: ignore[arg-type]
            chain_search=None,  # type: ignore[arg-type]
            kis=None,  # type: ignore[arg-type]
            qa=None,  # type: ignore[arg-type]
            trake=None,  # type: ignore[arg-type]
        )
        return _runtime
    _runtime = build_task_runtime(cfg)
    return _runtime


def get_runtime() -> AppRuntime:
    if _runtime is None:
        return init_runtime()
    return _runtime


def reset_runtime() -> None:
    """Clear cached runtime (tests only)."""
    global _runtime
    _runtime = None
