from __future__ import annotations

from video_retrieval.config import Settings, get_settings
from video_retrieval.events.align import score_videos, top_monotonic_paths
from video_retrieval.models import EventChain, EventChainPlan, EventHit, EventSpec, SearchHit
from video_retrieval.search.service import SearchService


class EventChainSearcher:
    """Retrieve per-event hits and align monotonic chains in-video."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        search: SearchService | None = None,
    ):
        self.settings = settings or get_settings()
        self.search_service = search or SearchService(self.settings)

    def search(
        self,
        plan: EventChainPlan,
        *,
        top_videos: int | None = None,
        top_chains: int | None = None,
        per_event: int | None = None,
        event_limit: int | None = None,
    ) -> list[EventChain]:
        if not plan.events:
            return []

        limit = event_limit or self.settings.trake_event_limit
        chain_limit = top_chains or self.settings.trake_top_chains
        video_limit = top_videos or max(
            self.settings.trake_top_videos,
            min(chain_limit, 24),
        )
        per_event_limit = per_event or self.settings.trake_candidates_per_event

        event_hits = self._retrieve_events(plan, limit=limit)
        return self._align_chains(
            event_hits,
            plan=plan,
            top_videos=video_limit,
            top_chains=chain_limit,
            per_event=per_event_limit,
        )

    def _retrieve_events(
        self,
        plan: EventChainPlan,
        *,
        limit: int,
    ) -> dict[str, list[SearchHit]]:
        by_event: dict[str, list[SearchHit]] = {}
        for event in plan.events:
            hits = self.search_service.search_event_spec(event, limit=limit)
            if not hits:
                query = _event_search_query(event, context=plan.context)
                try:
                    response = self.search_service.search_mixed(query, limit=min(limit, 20))
                    hits = list(response.hits)
                except Exception:
                    hits = []
            by_event[event.event_id] = hits
        return by_event

    def _retrieve_event_in_video(
        self,
        event: EventSpec,
        *,
        video_id: str,
        limit: int,
    ) -> list[SearchHit]:
        query = _event_search_query(event, context="")
        hits: list[SearchHit] = []
        try:
            vector = self.search_service.visual.encode_text(query)
            hits.extend(
                self.search_service.qdrant.search(
                    vector,
                    vector_name="siglip",
                    limit=limit,
                    video_id=video_id,
                )
            )
        except Exception:
            pass
        if event.ocr:
            try:
                hits.extend(
                    self.search_service.es.search(
                        event.ocr,
                        limit=limit,
                        source="ocr",
                        video_id=video_id,
                    )
                )
            except Exception:
                pass
        return hits

    def _align_chains(
        self,
        event_hits: dict[str, list[SearchHit]],
        *,
        plan: EventChainPlan,
        top_videos: int,
        top_chains: int,
        per_event: int,
    ) -> list[EventChain]:
        event_ids = [event.event_id for event in plan.events]
        if not event_ids:
            return []
        events_by_id = {event.event_id: event for event in plan.events}

        video_scores = score_videos(event_hits)
        if not video_scores:
            return []
        ranked_videos = sorted(video_scores.items(), key=lambda item: item[1], reverse=True)
        selected = [video_id for video_id, _ in ranked_videos[: max(top_videos, 1)]]
        paths_per_video = min(per_event, top_chains)

        chains: list[EventChain] = []
        for video_id in selected:
            per_event_cands = self._per_event_candidates(
                event_ids,
                event_hits,
                events_by_id,
                video_id=video_id,
                per_event=per_event,
            )
            if per_event_cands is None:
                continue

            paths = top_monotonic_paths(per_event_cands, limit=paths_per_video)
            for path in paths:
                event_hits_out: list[EventHit] = []
                total = 0.0
                for event_id, (frame, score, hit) in zip(event_ids, path, strict=True):
                    total += score
                    spec = events_by_id.get(event_id)
                    event_hits_out.append(
                        EventHit(
                            event_id=event_id,
                            frame_index=frame,
                            score=score,
                            timestamp_sec=hit.timestamp_sec,
                            keyframe_path=hit.keyframe_path,
                            text=hit.text,
                            source=hit.source or plan.task,
                            description=spec.description if spec else None,
                        )
                    )
                chains.append(
                    EventChain(
                        video_id=video_id,
                        score=total + video_scores.get(video_id, 0.0),
                        events=event_hits_out,
                    )
                )

        chains.sort(key=lambda chain: chain.score, reverse=True)
        return chains[: max(top_chains, 1)]

    def _per_event_candidates(
        self,
        event_ids: list[str],
        event_hits: dict[str, list[SearchHit]],
        events_by_id: dict[str, EventSpec],
        *,
        video_id: str,
        per_event: int,
    ) -> list[list[tuple[int, float, SearchHit]]] | None:
        per_event_cands: list[list[tuple[int, float, SearchHit]]] = []
        for event_id in event_ids:
            cands: list[tuple[int, float, SearchHit]] = []
            seen_frames: set[int] = set()
            pool = [
                hit
                for hit in event_hits[event_id]
                if hit.video_id == video_id and hit.frame_index is not None
            ]
            event = events_by_id.get(event_id)
            if event is not None:
                pool.extend(
                    self._retrieve_event_in_video(
                        event,
                        video_id=video_id,
                        limit=per_event,
                    )
                )
            for rank, hit in enumerate(pool):
                if hit.frame_index is None:
                    continue
                frame = int(hit.frame_index)
                if frame in seen_frames:
                    continue
                seen_frames.add(frame)
                score = float(hit.score) + 1.0 / (60 + rank + 1)
                cands.append((frame, score, hit))
                if len(cands) >= per_event:
                    break
            cands.sort(key=lambda item: item[0])
            per_event_cands.append(cands)

        if any(not cands for cands in per_event_cands):
            return None
        return per_event_cands


def _event_search_query(event: EventSpec, *, context: str = "") -> str:
    base = (
        event.visual
        or event.description
        or event.ocr
        or event.asr
        or event.event_id
    ).strip()
    if context and len(base) < 80:
        cue = " ".join(context.split()[:8])
        if cue:
            base = f"{cue}. {base}"
    if len(base) > 160:
        base = base[:160].rsplit(" ", 1)[0] or base[:160]
    return base


