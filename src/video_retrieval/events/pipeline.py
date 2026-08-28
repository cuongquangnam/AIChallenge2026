from __future__ import annotations

from video_retrieval.config import Settings, get_settings
from video_retrieval.events.extractor import EventChainExtractor
from video_retrieval.events.searcher import EventChainSearcher
from video_retrieval.events.timing import format_gaps_log
from video_retrieval.models import EventChain, EventChainPlan
from video_retrieval.query_stages import log_query_stage
from video_retrieval.search.service import SearchService


class EventChainTaskBase:
    """Shared extractor + chain searcher wiring for KIS / QA / TRAKE."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        search: SearchService | None = None,
        extractor: EventChainExtractor | None = None,
        chain_search: EventChainSearcher | None = None,
    ):
        self.settings = settings or get_settings()
        self.search_service = search or SearchService(self.settings)
        self.extractor = extractor or EventChainExtractor(self.settings)
        self.chain_search = chain_search or EventChainSearcher(
            self.settings,
            search=self.search_service,
        )

    def extract_events(
        self,
        query: str,
        task: str,
        *,
        empty_message: str,
    ) -> EventChainPlan:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")
        log_query_stage(task, "extract_events")
        plan = self.extractor.extract(query, task=task)
        if not plan.events:
            raise ValueError(empty_message)
        stage_details: dict[str, object] = {
            "events": len(plan.events),
            "question_event": plan.question_event_id,
        }
        gap_log = format_gaps_log(plan.events)
        if gap_log:
            stage_details["gaps"] = gap_log
        log_query_stage(task, "extract_events_done", **stage_details)
        return plan

    def search_event_chains(
        self,
        plan: EventChainPlan,
        *,
        top_chains: int,
        top_videos: int | None = None,
    ) -> list[EventChain]:
        video_limit = (
            top_videos
            if top_videos is not None
            else max(self.settings.trake_top_videos, min(top_chains, 24))
        )
        log_query_stage(
            plan.task,
            "chain_search",
            top_chains=top_chains,
            top_videos=video_limit,
        )
        chains = self.chain_search.search(
            plan,
            top_chains=top_chains,
            top_videos=video_limit,
        )
        log_query_stage(plan.task, "chain_search_done", chains=len(chains))
        return chains
