from __future__ import annotations

from video_retrieval.config import Settings
from video_retrieval.events.export import chains_to_search_hits, chains_to_submission_rows
from video_retrieval.events.pipeline import EventChainTaskBase
from video_retrieval.models import KisResult
from video_retrieval.query_stages import log_query_stage


class KisService(EventChainTaskBase):
    """Known Item Search via ordered event chains, flattened for submission."""

    def run(
        self,
        query: str,
        *,
        limit: int = 100,
        top_chains: int | None = None,
    ) -> KisResult:
        if limit < 1:
            raise ValueError("limit must be >= 1")

        log_query_stage("kis", "start", limit=limit)
        plan = self.extract_events(
            query,
            "kis",
            empty_message="No events extracted from query",
        )
        chain_limit = top_chains or self.settings.kis_top_chains
        chains = self.search_event_chains(plan, top_chains=chain_limit)
        log_query_stage("kis", "export_rows", limit=limit)
        rows = chains_to_submission_rows(chains, limit=limit)
        hits = chains_to_search_hits(chains, limit=limit)
        log_query_stage("kis", "done", rows=len(rows), chains=len(chains))
        return KisResult(
            query=query.strip(),
            plan=plan,
            chains=chains,
            hits=hits,
            submission_rows=rows,
        )
