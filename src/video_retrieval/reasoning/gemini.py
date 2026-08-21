from __future__ import annotations

import json
from pathlib import Path
import time

from video_retrieval.config import Settings
from video_retrieval.models import Task2BatchVerdict, Task2GroupVerdict, TemporalGroup


class GeminiTask2Verifier:
    """Compare retrieved Task 2 evidence groups in one Gemini request."""

    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be configured for Task 2 VLM verification")
        from google import genai

        self.settings = settings
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model

    def verify_groups(
        self, groups: list[TemporalGroup]
    ) -> tuple[TemporalGroup | None, Task2GroupVerdict]:
        from google.genai import types

        readable_groups = [
            group
            for group in groups
            if any(Path(path).exists() for path in group.context_keyframe_paths)
        ]
        if not readable_groups:
            raise ValueError("No readable context frames for Task 2 candidates")

        prompt = f"""You are reviewing {len(readable_groups)} chronological evidence groups
from the same candidate video. Each group contains consecutive frames.

Question: In a music awards ceremony video, how many distinct people walk onto
the stage to receive the ceremony's largest award?

Compare every group and choose the ONE group that clearly shows the target
event. Count only recipients who walk onto the stage for the largest award; do
not count MCs, performers, staff, or people already on stage. Do not infer a
major award merely because a stage is visible. Red-carpet, performances,
certificate handouts, lower-rank prizes, and ordinary award presentations are
not the target.

Return JSON only:
{{
  "selected_group_index": 1,
  "is_major_award": true,
  "winner_count": 0,
  "evidence_frame_ids": [0],
  "confidence": 0.0,
  "reason": "short evidence-based explanation"
}}
Use only frame IDs supplied in the selected group. Group indexes start at 1.
If no group clearly shows the major-award acceptance, set
selected_group_index=null, is_major_award=false, and winner_count=null."""
        parts: list[object] = [prompt]
        for index, group in enumerate(readable_groups, start=1):
            parts.append(
                f"GROUP {index}: video_id={group.video_id}; "
                f"frame IDs in chronological order={group.context_frame_indices}."
            )
            for path_string in group.context_keyframe_paths:
                path = Path(path_string)
                if not path.exists():
                    continue
                with path.open("rb") as image_file:
                    parts.append(
                        types.Part.from_bytes(data=image_file.read(), mime_type="image/jpeg")
                    )

        from google.genai import errors

        response = None
        for attempt in range(self.settings.gemini_max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=parts,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                break
            except errors.ServerError as exc:
                if exc.code != 503 or attempt + 1 == self.settings.gemini_max_retries:
                    raise
                time.sleep(min(2**attempt, 30))
        if response is None:
            raise RuntimeError("Gemini did not return a response")
        try:
            data = json.loads(response.text or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("Gemini returned invalid Task 2 JSON") from exc
        batch_verdict = Task2BatchVerdict.model_validate(data)
        if not batch_verdict.is_major_award or batch_verdict.selected_group_index is None:
            return None, Task2GroupVerdict(
                is_major_award=False,
                winner_count=None,
                evidence_frame_ids=[],
                confidence=batch_verdict.confidence,
                reason=batch_verdict.reason,
            )
        group_index = batch_verdict.selected_group_index - 1
        if group_index < 0 or group_index >= len(readable_groups):
            raise ValueError("Gemini selected an invalid Task 2 group index")
        group = readable_groups[group_index]
        verdict = Task2GroupVerdict(
            is_major_award=True,
            winner_count=batch_verdict.winner_count,
            evidence_frame_ids=batch_verdict.evidence_frame_ids,
            confidence=batch_verdict.confidence,
            reason=batch_verdict.reason,
        )
        allowed = set(group.context_frame_indices)
        verdict.evidence_frame_ids = [frame_id for frame_id in verdict.evidence_frame_ids if frame_id in allowed]
        if verdict.winner_count is None:
            verdict.winner_count = None
        return group, verdict
