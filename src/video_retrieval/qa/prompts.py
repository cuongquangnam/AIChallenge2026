from __future__ import annotations

from video_retrieval.models import QAFrameGroup


DECOMPOSITION_SYSTEM_PROMPT = """You decompose video question-answering queries for retrieval.
Return JSON only. Do not answer the question and do not invent details."""


def build_decomposition_prompt(question: str) -> str:
    return f"""Split the question below into 2 to 5 short, independently searchable visual or
spoken descriptions. Preserve important entities, actions, setting, and superlatives.

Question: {question}

Return exactly this JSON shape:
{{"descriptions":["description 1","description 2"]}}"""


ANSWER_SYSTEM_PROMPT = """You answer a question using only the supplied video frame.
Return JSON only. Read on-screen numbers, signs, and text carefully. Prefer a short concrete
answer (e.g. a number or place name) when visible. Only use an empty answer if nothing
relevant is visible at all."""


SINGLE_FRAME_ANSWER_PROMPT = """You answer a question using only the single video frame provided.
Return JSON only with a short concrete answer when visible."""


def build_single_frame_answer_prompt(
    question: str,
    video_id: str,
    frame_id: int,
    event_description: str = "",
) -> str:
    context = f"Event context: {event_description}\n" if event_description else ""
    return f"""Question: {question}
{context}The image is frame_id={frame_id} from video_id={video_id}.

Return exactly this JSON shape:
{{"video_id":"{video_id}","frame_id":{frame_id},"answer":"..."}}"""


def build_answer_prompt(
    question: str,
    descriptions: list[str],
    video_id: str,
    frame_groups: list[QAFrameGroup],
) -> str:
    group_lines = []
    for index, group in enumerate(frame_groups, start=1):
        frame_ids = ", ".join(str(frame.frame_id) for frame in group.frames)
        group_lines.append(
            f"Group {index}: center={group.center_frame_id}; "
            f"available frame_id values=[{frame_ids}]"
        )

    retrieval_context = "\n".join(f"- {item}" for item in descriptions)
    groups = "\n".join(group_lines)
    return f"""Question: {question}
Retrieved descriptions:
{retrieval_context}

All evidence comes from video_id={video_id}.
The images are provided after this text in the same group/frame order shown below:
{groups}

Count carefully across adjacent frames without double-counting the same person. The returned
frame_id must be one of the available frame_id values above and must best support the answer.

Return exactly this JSON shape:
{{"video_id":"{video_id}","frame_id":123,"answer":"..."}}"""
