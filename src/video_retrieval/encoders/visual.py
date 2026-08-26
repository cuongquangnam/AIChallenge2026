from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from video_retrieval.config import Settings
from video_retrieval.models import KeyFrame, VisualEmbedding

# Leave room for special tokens inside SigLIP's 64-position budget.
_SIGLIP_SPECIAL_TOKEN_RESERVE = 2
_SIGLIP_CHUNK_OVERLAP = 8


def chunk_token_ids(
    token_ids: list[int],
    *,
    chunk_size: int,
    overlap: int = _SIGLIP_CHUNK_OVERLAP,
) -> list[list[int]]:
    """Split token ids into overlapping windows of at most ``chunk_size``."""
    if not token_ids:
        return [[]]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if len(token_ids) <= chunk_size:
        return [list(token_ids)]
    overlap = max(0, min(overlap, chunk_size - 1))
    step = max(1, chunk_size - overlap)
    chunks: list[list[int]] = []
    for start in range(0, len(token_ids), step):
        piece = token_ids[start : start + chunk_size]
        if not piece:
            break
        chunks.append(piece)
        if start + chunk_size >= len(token_ids):
            break
    return chunks


class VisualEncoder:
    """SigLIP image/text encoder for cross-modal keyframe search."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = settings.visual_backend
        self._siglip = None
        self._siglip_processor = None
        if self.backend == "real":
            self._load_real_models()

    def _load_real_models(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Install ML extras: pip install '.[ml]' before VISUAL_BACKEND=real"
            ) from exc

        self._torch = torch
        if torch.cuda.is_available():
            self._device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        self._siglip_processor = AutoProcessor.from_pretrained(self.settings.siglip_model_id)
        self._siglip = AutoModel.from_pretrained(self.settings.siglip_model_id).to(self._device)
        self._siglip.eval()

    def encode_image(self, image_path: Path) -> list[float]:
        if self.backend == "real":
            return self._encode_real(image_path)
        return self._encode_mock(image_path)

    def encode_text(self, text: str) -> list[float]:
        """Text → SigLIP space for cross-modal query.

        Long queries are split into overlapping token windows (SigLIP max 64),
        embedded separately, then mean-pooled and re-normalized so the tail of
        the query is not discarded.
        """
        if self.backend == "real":
            return self._encode_text_real(text)
        return self._hash_embed(text, self.settings.siglip_dim)

    def encode_keyframes(self, keyframes: list[KeyFrame]) -> list[VisualEmbedding]:
        if not keyframes:
            return []
        if self.backend != "real":
            return [
                VisualEmbedding(keyframe=kf, siglip=self.encode_image(kf.path))
                for kf in keyframes
            ]
        return self._encode_keyframes_batched(keyframes)

    def _encode_keyframes_batched(self, keyframes: list[KeyFrame]) -> list[VisualEmbedding]:
        batch_size = max(int(self.settings.siglip_batch_size or 1), 1)
        total = len(keyframes)
        out: list[VisualEmbedding] = []
        torch = self._torch
        for start in range(0, total, batch_size):
            batch = keyframes[start : start + batch_size]
            images = [Image.open(kf.path).convert("RGB") for kf in batch]
            with torch.no_grad():
                inputs = self._siglip_processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                feats = self._as_embedding_tensor(
                    self._siglip.get_image_features(**inputs)
                )
                if feats.ndim == 1:
                    feats = feats.unsqueeze(0)
                feats = torch.nn.functional.normalize(feats, dim=-1)
                vectors = feats.detach().cpu().tolist()
            for kf, vec in zip(batch, vectors, strict=True):
                out.append(VisualEmbedding(keyframe=kf, siglip=list(vec)))
            done = min(start + len(batch), total)
            if done == len(batch) or done % max(batch_size * 4, 64) == 0 or done == total:
                print(
                    f"SigLIP {done}/{total} "
                    f"(batch={batch_size}, device={self._device})",
                    flush=True,
                )
        return out

    def _encode_mock(self, image_path: Path) -> list[float]:
        seed = f"{image_path.resolve()}:{image_path.stat().st_size}:siglip"
        return self._hash_embed(seed, self.settings.siglip_dim)

    def _encode_real(self, image_path: Path) -> list[float]:
        image = Image.open(image_path).convert("RGB")
        torch = self._torch
        with torch.no_grad():
            inputs = self._siglip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            feats = self._siglip.get_image_features(**inputs)
            return self._to_unit_vector(feats)

    def _siglip_max_length(self) -> int:
        return int(getattr(self._siglip.config.text_config, "max_position_embeddings", 64))

    def _text_chunks(self, text: str, max_length: int) -> list[str]:
        tokenizer = self._siglip_processor.tokenizer
        budget = max(8, max_length - _SIGLIP_SPECIAL_TOKEN_RESERVE)
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        chunks = chunk_token_ids(
            list(token_ids),
            chunk_size=budget,
            overlap=_SIGLIP_CHUNK_OVERLAP,
        )
        decoded: list[str] = []
        for piece in chunks:
            if not piece:
                continue
            chunk = tokenizer.decode(piece, skip_special_tokens=True).strip()
            if chunk:
                decoded.append(chunk)
        return decoded or [text]

    def _encode_text_chunk(self, text: str, max_length: int) -> list[float]:
        torch = self._torch
        with torch.no_grad():
            inputs = self._siglip_processor(
                text=[text],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            feats = self._siglip.get_text_features(**inputs)
            return self._to_unit_vector(feats)

    def _encode_text_real(self, text: str) -> list[float]:
        max_length = self._siglip_max_length()
        chunks = self._text_chunks(text, max_length)
        if len(chunks) == 1:
            return self._encode_text_chunk(chunks[0], max_length)

        vectors = [self._encode_text_chunk(chunk, max_length) for chunk in chunks]
        pooled = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
        pooled /= float(np.linalg.norm(pooled) + 1e-8)
        return pooled.tolist()

    def _to_unit_vector(self, features: Any) -> list[float]:
        torch = self._torch
        tensor = self._as_embedding_tensor(features)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        vec = torch.nn.functional.normalize(tensor, dim=-1)[0]
        return vec.detach().cpu().tolist()

    def _as_embedding_tensor(self, features: Any):
        """Normalize SigLIP feature outputs across transformers versions.

        Older builds return a bare tensor from ``get_*_features``. Newer ones
        may return ``BaseModelOutputWithPooling`` (or similar) instead.
        """
        torch = self._torch
        if torch.is_tensor(features):
            return features
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return features.pooler_output
        if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
            return features.last_hidden_state[:, 0]
        if hasattr(features, "embeddings") and features.embeddings is not None:
            return features.embeddings
        raise TypeError(f"Unsupported feature type: {type(features)!r}")

    @staticmethod
    def _hash_embed(seed: str, dim: int) -> list[float]:
        rng = np.random.default_rng(abs(hash(seed)) % (2**32))
        vec = rng.normal(size=dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        return vec.tolist()
