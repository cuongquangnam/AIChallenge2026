from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import numpy as np
from PIL import Image

from video_retrieval.config import Settings
from video_retrieval.models import KeyFrame, VisualEmbedding


class VisualEncoder:
    """SigLIP + BEiT3 visual encoders with a deterministic mock backend."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = settings.visual_backend
        self._siglip = None
        self._siglip_processor = None
        self._beit = None
        self._beit_processor = None

    def _ensure_real_runtime(self) -> None:
        if hasattr(self, "_torch"):
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Install ML extras: pip install '.[ml]' before VISUAL_BACKEND=real"
            ) from exc

        self._torch = torch
        self._auto_image_processor = AutoImageProcessor
        self._auto_model = AutoModel
        self._auto_processor = AutoProcessor
        if torch.cuda.is_available():
            self._device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

    def _load_siglip(self) -> None:
        if self._siglip is not None:
            return
        self._ensure_real_runtime()
        self._siglip_processor = self._auto_processor.from_pretrained(self.settings.siglip_model_id)
        self._siglip = self._auto_model.from_pretrained(self.settings.siglip_model_id).to(self._device)
        self._siglip.eval()

    def _load_beit(self) -> None:
        if self._beit is not None:
            return
        self._ensure_real_runtime()
        # BEiT is used as a practical stand-in if a BEiT-3 checkpoint is unavailable.
        self._beit_processor = self._auto_image_processor.from_pretrained(self.settings.beit3_model_id)
        self._beit = self._auto_model.from_pretrained(self.settings.beit3_model_id).to(self._device)
        self._beit.eval()

    def encode_image(self, image_path: Path) -> tuple[list[float], list[float]]:
        if self.backend == "real":
            return self._encode_real(image_path)
        return self._encode_mock(image_path)

    def encode_text(self, text: str) -> list[float]:
        """Text → SigLIP space for cross-modal query."""
        if self.backend == "real":
            return self._encode_text_real(text)
        return self._hash_embed(text, self.settings.siglip_dim)

    def encode_keyframes(self, keyframes: list[KeyFrame]) -> list[VisualEmbedding]:
        out: list[VisualEmbedding] = []
        for kf in keyframes:
            siglip, beit3 = self.encode_image(kf.path)
            out.append(VisualEmbedding(keyframe=kf, siglip=siglip, beit3=beit3))
        return out

    def _encode_mock(self, image_path: Path) -> tuple[list[float], list[float]]:
        seed = f"{image_path.resolve()}:{image_path.stat().st_size}"
        return (
            self._hash_embed(seed + ":siglip", self.settings.siglip_dim),
            self._hash_embed(seed + ":beit3", self.settings.beit3_dim),
        )

    def _encode_real(self, image_path: Path) -> tuple[list[float], list[float]]:
        self._load_siglip()
        self._load_beit()
        image = Image.open(image_path).convert("RGB")
        torch = self._torch

        with torch.no_grad():
            siglip_inputs = self._siglip_processor(images=image, return_tensors="pt")
            siglip_inputs = {k: v.to(self._device) for k, v in siglip_inputs.items()}
            siglip_out = self._siglip.get_image_features(**siglip_inputs)
            siglip_vec = self._to_unit_vector(siglip_out)

            beit_inputs = self._beit_processor(images=image, return_tensors="pt")
            beit_inputs = {k: v.to(self._device) for k, v in beit_inputs.items()}
            beit_out = self._beit(**beit_inputs)
            beit_vec = self._to_unit_vector(beit_out)

        return siglip_vec, beit_vec

    def _encode_text_real(self, text: str) -> list[float]:
        self._load_siglip()
        torch = self._torch
        with torch.no_grad():
            inputs = self._siglip_processor(
                text=[text],
                padding="max_length",
                # SigLIP has a fixed, short text context (64 tokens for the
                # configured checkpoint).  Query planners can produce a
                # longer natural-language description, so trim it here rather
                # than fail the entire search request.
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            feats = self._siglip.get_text_features(**inputs)
            return self._to_unit_vector(feats)

    def _to_unit_vector(self, features: Any) -> list[float]:
        """Normalize image/text features whether they are a tensor or ModelOutput."""
        torch = self._torch
        tensor = self._as_embedding_tensor(features)
        vec = torch.nn.functional.normalize(tensor, dim=-1)[0]
        return vec.detach().cpu().tolist()

    def _as_embedding_tensor(self, features: Any):
        """Newer transformers may return BaseModelOutputWithPooling instead of a Tensor."""
        torch = self._torch
        if torch.is_tensor(features):
            tensor = features
        elif hasattr(features, "pooler_output") and features.pooler_output is not None:
            tensor = features.pooler_output
        elif hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
            # CLS / first token fallback (BEiT-style).
            tensor = features.last_hidden_state[:, 0]
        else:
            raise TypeError(f"Unsupported feature type: {type(features)!r}")

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _hash_embed(seed: str, dim: int) -> list[float]:
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        rng_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(rng_seed)
        vec = rng.normal(size=dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        return vec.tolist()
