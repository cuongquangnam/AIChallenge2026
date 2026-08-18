from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from video_retrieval.config import Settings
from video_retrieval.models import KeyFrame, VisualEmbedding


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
        """Text → SigLIP space for cross-modal query."""
        if self.backend == "real":
            return self._encode_text_real(text)
        return self._hash_embed(text, self.settings.siglip_dim)

    def encode_keyframes(self, keyframes: list[KeyFrame]) -> list[VisualEmbedding]:
        return [
            VisualEmbedding(keyframe=kf, siglip=self.encode_image(kf.path))
            for kf in keyframes
        ]

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

    def _encode_text_real(self, text: str) -> list[float]:
        torch = self._torch
        with torch.no_grad():
            inputs = self._siglip_processor(
                text=[text],
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            feats = self._siglip.get_text_features(**inputs)
            return self._to_unit_vector(feats)

    def _to_unit_vector(self, features: Any) -> list[float]:
        torch = self._torch
        tensor = features if torch.is_tensor(features) else self._as_embedding_tensor(features)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        vec = torch.nn.functional.normalize(tensor, dim=-1)[0]
        return vec.detach().cpu().tolist()

    def _as_embedding_tensor(self, features: Any):
        torch = self._torch
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return features.pooler_output
        if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
            return features.last_hidden_state[:, 0]
        raise TypeError(f"Unsupported feature type: {type(features)!r}")

    @staticmethod
    def _hash_embed(seed: str, dim: int) -> list[float]:
        rng = np.random.default_rng(abs(hash(seed)) % (2**32))
        vec = rng.normal(size=dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        return vec.tolist()
