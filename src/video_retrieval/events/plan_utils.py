from __future__ import annotations

from video_retrieval.models import EventChain, EventChainPlan


def questioned_frame(plan: EventChainPlan, chain: EventChain) -> tuple[str, int]:
    """Return the questioned event id and aligned frame for a chain."""
    q_id = plan.question_event_id
    if q_id:
        for event in chain.events:
            if event.event_id == q_id:
                return q_id, int(event.frame_index)

    for spec in plan.events:
        if spec.is_question_target:
            for hit in chain.events:
                if hit.event_id == spec.event_id:
                    return hit.event_id, int(hit.frame_index)

    if chain.events:
        last = chain.events[-1]
        return last.event_id, int(last.frame_index)
    return "E1", 0


def event_description(plan: EventChainPlan, event_id: str) -> str:
    for event in plan.events:
        if event.event_id == event_id:
            return event.description or event.visual or ""
    return ""
