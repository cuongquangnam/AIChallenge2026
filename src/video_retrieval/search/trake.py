from __future__ import annotations

from video_retrieval.events.export import chain_to_csv_line
from video_retrieval.events.pipeline import EventChainTaskBase
from video_retrieval.models import EventChainPlan, TrakeResult
from video_retrieval.query_stages import log_query_stage


class TrakeService(EventChainTaskBase):
    """Parse events and align temporal chains via the shared event pipeline."""

    def run(self, query: str, *, top_chains: int | None = None) -> TrakeResult:
        log_query_stage("trake", "start", top_chains=top_chains or self.settings.trake_top_chains)
        plan = self.parse_events(query)
        chain_limit = top_chains or self.settings.trake_top_chains
        chains = self.search_event_chains(plan, top_chains=chain_limit)
        csv_row = chain_to_csv_line(chains[0]) or "" if chains else ""
        log_query_stage("trake", "done", chains=len(chains))
        return TrakeResult(
            query=query.strip(),
            plan=plan,
            chains=chains,
            csv_row=csv_row,
        )

    def parse_events(self, query: str) -> EventChainPlan:
        return self.extract_events(
            query,
            "trake",
            empty_message="No TRAKE events (E1, E2, …) found in the query",
        )


# Backward-compatible helpers for tests and imports.
def heuristic_trake_plan(query: str) -> EventChainPlan:
    from video_retrieval.events.extractor import heuristic_event_plan

    return heuristic_event_plan(query, task="trake")


def parse_trake_plan(raw: str, *, fallback_query: str) -> EventChainPlan:
    from video_retrieval.events.extractor import parse_event_plan

    return parse_event_plan(raw, fallback_query=fallback_query, task="trake")
