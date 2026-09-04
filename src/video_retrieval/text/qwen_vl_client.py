from __future__ import annotations

import logging
from typing import Any

from video_retrieval.config import Settings, get_settings
from video_retrieval.text.content_parts import load_pil_image, normalize_parts

logger = logging.getLogger(__name__)

_client: QwenVLClient | None = None
_client_key: tuple[str, str, str, str, int] | None = None


class QwenVLClient:
    """Local Qwen2.5-VL / Qwen3-VL client (bf16 / fp16 / 4-bit)."""

    def __init__(self, settings: Settings):
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Qwen VL requires ML extras: pip install '.[ml]' "
                "(torch, transformers). For 4-bit also install bitsandbytes."
            ) from exc

        self.settings = settings
        self.model = settings.qwen_vl_model_id
        self._torch = torch
        self._max_new_tokens = max(16, int(settings.qwen_vl_max_new_tokens))
        dtype_name = (settings.qwen_vl_dtype or "bf16").strip().lower()
        device_map = (settings.qwen_vl_device or "auto").strip() or "auto"

        model_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": True,
        }
        if dtype_name in {"4bit", "int4", "bnb4"}:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(
                    "4-bit Qwen VL needs bitsandbytes + recent transformers"
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif dtype_name in {"fp16", "float16", "half"}:
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        logger.info(
            "Loading Qwen VL model=%s dtype=%s device_map=%s",
            self.model,
            dtype_name,
            device_map,
        )
        model_cls = _resolve_qwen_model_class()
        self._processor = AutoProcessor.from_pretrained(
            self.model,
            trust_remote_code=True,
        )
        self._model = model_cls.from_pretrained(self.model, **model_kwargs)
        self._model.eval()

    def generate_text(
        self,
        prompt: str,
        *,
        json_response: bool = False,
        component: str = "qwen_vl",
    ) -> str:
        return self.generate_parts(
            [{"type": "text", "text": prompt}],
            json_response=json_response,
            component=component,
        )

    def generate_parts(
        self,
        parts: list[Any],
        *,
        json_response: bool = False,
        component: str = "qwen_vl",
    ) -> str:
        content = _parts_to_qwen_content(normalize_parts(parts))
        if json_response:
            content = [
                {
                    "type": "text",
                    "text": (
                        "Respond with valid JSON only. "
                        "Do not wrap the JSON in markdown fences."
                    ),
                },
                *content,
            ]
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images = [
            item["image"]
            for item in content
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        kwargs: dict[str, Any] = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
        }
        if images:
            kwargs["images"] = images
        inputs = self._processor(**kwargs)
        device = next(self._model.parameters()).device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        decoded = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw = (decoded[0] if decoded else "").strip()
        logger.debug("Qwen VL %s generated %s chars", component, len(raw))
        return raw


def get_qwen_vl_client(
    settings: Settings | None = None,
    *,
    force: bool = False,
) -> QwenVLClient:
    """Return a process-wide Qwen VL client (heavyweights loaded once)."""
    global _client, _client_key
    cfg = settings or get_settings()
    key = (
        cfg.qwen_vl_model_id,
        (cfg.qwen_vl_dtype or "bf16").strip().lower(),
        (cfg.qwen_vl_device or "auto").strip(),
        str(cfg.qwen_vl_max_new_tokens),
        1,
    )
    if _client is not None and not force and _client_key == key:
        return _client
    _client = QwenVLClient(cfg)
    _client_key = key
    return _client


def reset_qwen_vl_client() -> None:
    global _client, _client_key
    _client = None
    _client_key = None


def _resolve_qwen_model_class():
    """Prefer Qwen3, then Qwen2.5, then generic auto classes."""
    try:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    except ImportError:
        pass
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration
    except ImportError:
        pass
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError:
        pass
    try:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq
    except ImportError as exc:
        raise ImportError(
            "Install transformers>=4.57 for Qwen3-VL "
            "(or >=4.45 for Qwen2.5-VL): pip install '.[ml]'"
        ) from exc


def _parts_to_qwen_content(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in parts:
        kind = part.get("type")
        if kind == "text":
            text = str(part.get("text") or "")
            if text:
                content.append({"type": "text", "text": text})
        elif kind == "image":
            content.append({"type": "image", "image": load_pil_image(part.get("image"))})
        elif kind == "image_bytes":
            content.append({"type": "image", "image": load_pil_image(part.get("data"))})
    if not content:
        content.append({"type": "text", "text": ""})
    return content
