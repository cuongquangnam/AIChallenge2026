from pathlib import Path
from unittest.mock import MagicMock
import json

import pytest

from video_retrieval.config import Settings
from video_retrieval.models import AudioTrack, FrameRole, KeyFrame
from video_retrieval.text.asr import ASREngine
from video_retrieval.text.ocr import OCREngine, _parse_batch_ocr_response
from video_retrieval.text.gemini_config import gemini_generate_config
from tests.helpers import write_dummy_image


def _mock_ocr_engine(settings: Settings) -> OCREngine:
    return OCREngine(settings, client=MagicMock())


@pytest.mark.unit
def test_mock_ocr_only_middle_frames(tmp_path: Path, settings: Settings) -> None:
    ocr = OCREngine(settings)
    middle = KeyFrame(
        video_id="clip",
        shot_index=0,
        role=FrameRole.MIDDLE,
        frame_index=5,
        timestamp_sec=0.5,
        path=write_dummy_image(tmp_path / "shot_0000_middle.jpg"),
    )
    start = KeyFrame(
        video_id="clip",
        shot_index=0,
        role=FrameRole.START,
        frame_index=0,
        timestamp_sec=0.0,
        path=write_dummy_image(tmp_path / "shot_0000_start.jpg"),
    )
    docs = ocr.extract_from_keyframes([start, middle])
    assert len(docs) == 1
    assert docs[0].source == "ocr"
    assert "middle" in docs[0].text


@pytest.mark.unit
def test_extract_from_keyframes_only_processes_middle_frames(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.ocr_backend = "gemini"
    settings.gemini_api_key = "test-key"
    called_batches: list[list[str]] = []

    def fake_extract_gemini_batch(self, keyframes: list[KeyFrame]) -> dict[str, str]:
        called_batches.append([kf.path.name for kf in keyframes])
        return {kf.path.name: "ocr text" for kf in keyframes}

    monkeypatch.setattr(OCREngine, "_extract_gemini_batch", fake_extract_gemini_batch)

    ocr = _mock_ocr_engine(settings)
    middle = KeyFrame(
        video_id="clip",
        shot_index=0,
        role=FrameRole.MIDDLE,
        frame_index=5,
        timestamp_sec=0.5,
        path=write_dummy_image(tmp_path / "shot_0000_middle.jpg"),
    )
    start = KeyFrame(
        video_id="clip",
        shot_index=0,
        role=FrameRole.START,
        frame_index=0,
        timestamp_sec=0.0,
        path=write_dummy_image(tmp_path / "shot_0000_start.jpg"),
    )
    docs = ocr.extract_from_keyframes([start, middle])
    assert len(docs) == 1
    assert called_batches == [["shot_0000_middle.jpg"]]


@pytest.mark.unit
def test_extract_from_keyframes_batches_gemini_requests(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.ocr_backend = "gemini"
    settings.gemini_api_key = "test-key"
    settings.gemini_batch_size = 2
    called_batches: list[list[str]] = []

    def fake_extract_gemini_batch(self, keyframes: list[KeyFrame]) -> dict[str, str]:
        called_batches.append([kf.path.name for kf in keyframes])
        return {kf.path.name: f"text from {kf.path.name}" for kf in keyframes}

    monkeypatch.setattr(OCREngine, "_extract_gemini_batch", fake_extract_gemini_batch)

    ocr = _mock_ocr_engine(settings)
    keyframes = [
        KeyFrame(
            video_id="clip",
            shot_index=index,
            role=FrameRole.MIDDLE,
            frame_index=index,
            timestamp_sec=float(index),
            path=write_dummy_image(tmp_path / f"shot_{index:04d}_middle.jpg"),
        )
        for index in range(3)
    ]
    docs = ocr.extract_from_keyframes(keyframes)
    assert len(docs) == 3
    assert called_batches == [
        ["shot_0000_middle.jpg", "shot_0001_middle.jpg"],
        ["shot_0002_middle.jpg"],
    ]


@pytest.mark.unit
def test_parse_batch_ocr_response_maps_image_ids() -> None:
    raw = json.dumps(
        {
            "results": [
                {"image_id": "a.jpg", "text": "hello"},
                {"image_id": "b.jpg", "text": ""},
            ]
        }
    )
    parsed = _parse_batch_ocr_response(raw, ["a.jpg", "b.jpg", "c.jpg"])
    assert parsed == {"a.jpg": "hello", "b.jpg": "", "c.jpg": ""}


@pytest.mark.unit
def test_parse_batch_ocr_response_handles_invalid_json() -> None:
    parsed = _parse_batch_ocr_response("not-json", ["a.jpg"])
    assert parsed == {"a.jpg": ""}


@pytest.mark.unit
def test_gemini_generation_config_for_v3_models() -> None:
    config = gemini_generate_config("gemini-3.5-flash", json_response=True)
    assert config is not None
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level is not None


@pytest.mark.unit
def test_gemini_generation_config_skips_pro_minimal() -> None:
    config = gemini_generate_config("gemini-3.1-pro-preview", json_response=True)
    assert config is not None
    assert config.thinking_config is None


@pytest.mark.unit
def test_gemini_generation_config_skips_older_models() -> None:
    assert gemini_generate_config("gemini-2.0-flash", json_response=False) is None


@pytest.mark.unit
def test_gemini_ocr_requires_api_key(settings: Settings) -> None:
    settings.ocr_backend = "gemini"
    settings.gemini_api_key = ""
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        OCREngine(settings)


@pytest.mark.unit
def test_mock_asr_returns_transcript(tmp_path: Path, settings: Settings) -> None:
    asr = ASREngine(settings)
    audio = AudioTrack(video_id="clip", path=tmp_path / "clip.wav", duration_sec=1.5)
    docs = asr.transcribe(audio)
    assert len(docs) == 1
    assert docs[0].source == "asr"
    assert "clip" in docs[0].text
