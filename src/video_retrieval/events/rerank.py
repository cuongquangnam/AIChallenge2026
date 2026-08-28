from __future__ import annotations

from pathlib import Path

import numpy as np

from video_retrieval.config import Settings
from video_retrieval.models import EventSpec, SearchHit


class CrossEncoderReranker:
    """Image–text cross-encoder rerank for per-event alignment candidates."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = settings.chain_rerank_backend
        if self.backend == "auto":
            self.backend = settings.visual_backend
        self._model = None
        self._processor = None
        if self.backend == "real":
            self._load_real_model()

    def rerank_per_event(
        self,
        per_event_cands: list[list[tuple[int, float, SearchHit]]],
        *,
        event_ids: list[str],
        events_by_id: dict[str, EventSpec],
        video_id: str,
        context: str = "",
    ) -> list[list[tuple[int, float, SearchHit]]]:
        out: list[list[tuple[int, float, SearchHit]]] = []
        blend = self.settings.chain_rerank_blend
        for event_id, cands in zip(event_ids, per_event_cands, strict=True):
            spec = events_by_id.get(event_id)
            query = _event_query(spec, event_id=event_id, context=context)
            ce_scores = self._score_candidates(query, cands, video_id=video_id)
            retrieval = [score for _, score, _ in cands]
            norm_retrieval = _normalize_scores(retrieval)
            reranked: list[tuple[int, float, SearchHit]] = []
            for (frame, score, hit), r_norm, ce in zip(
                cands, norm_retrieval, ce_scores, strict=True
            ):
                blended = (1.0 - blend) * r_norm + blend * ce
                reranked.append((frame, blended, hit))
            reranked.sort(key=lambda item: item[0])
            out.append(reranked)
        return out

    def _score_candidates(
        self,
        query: str,
        cands: list[tuple[int, float, SearchHit]],
        *,
        video_id: str,
    ) -> list[float]:
        if not cands:
            return []
        if self.backend == "real":
            return self._score_candidates_real(query, cands, video_id=video_id)
        return [_mock_ce_score(query, hit, video_id=video_id) for _, _, hit in cands]

    def _score_candidates_real(
        self,
        query: str,
        cands: list[tuple[int, float, SearchHit]],
        *,
        video_id: str,
    ) -> list[float]:
        from PIL import Image

        images: list[Image.Image] = []
        indices: list[int] = []
        fallback = _mock_ce_score
        for idx, (_, _, hit) in enumerate(cands):
            path = resolve_keyframe_path(self.settings, hit, video_id=video_id)
            if path is None:
                continue
            try:
                images.append(Image.open(path).convert("RGB"))
                indices.append(idx)
            except OSError:
                continue

        scores = [fallback(query, hit, video_id=video_id) for _, _, hit in cands]
        if not images:
            return scores

        texts = [query] * len(images)
        inputs = self._processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
            probs = outputs.itm_score.softmax(dim=1)[:, 1]

        for idx, prob in zip(indices, probs.tolist(), strict=True):
            scores[idx] = float(prob)
        return scores

    def _load_real_model(self) -> None:
        try:
            import torch
            from transformers import BlipForImageTextRetrieval, BlipProcessor
        except ImportError as exc:
            raise ImportError(
                "Install ML extras: pip install '.[ml]' before CHAIN_RERANK_BACKEND=real"
            ) from exc

        self._torch = torch
        if torch.cuda.is_available():
            self._device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        model_id = self.settings.chain_rerank_model_id
        self._processor = BlipProcessor.from_pretrained(model_id)
        self._model = BlipForImageTextRetrieval.from_pretrained(model_id).to(self._device)
        self._model.eval()


def resolve_keyframe_path(settings: Settings, hit: SearchHit, *, video_id: str) -> Path | None:
    if not hit.keyframe_path:
        return None
    raw = Path(hit.keyframe_path)
    if raw.is_file():
        return raw
    for candidate in (
        settings.keyframes_dir / video_id / raw.name,
        settings.keyframes_dir / raw.name,
        settings.data_dir / raw,
    ):
        if candidate.is_file():
            return candidate
    return None


def _event_query(spec: EventSpec | None, *, event_id: str, context: str) -> str:
    if spec is None:
        return event_id
    base = (
        spec.visual
        or spec.description
        or spec.ocr
        or spec.asr
        or spec.event_id
    ).strip()
    if context and len(base) < 80:
        cue = " ".join(context.split()[:8])
        if cue:
            base = f"{cue}. {base}"
    if len(base) > 160:
        base = base[:160].rsplit(" ", 1)[0] or base[:160]
    return base


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return scores
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-9:
        return [1.0] * len(scores)
    return [(score - lo) / (hi - lo) for score in scores]


def _mock_ce_score(query: str, hit: SearchHit, *, video_id: str) -> float:
    seed = f"{query}:{video_id}:{hit.frame_index}:{hit.keyframe_path}"
    rng = np.random.default_rng(abs(hash(seed)) % (2**32))
    return float(rng.random())
