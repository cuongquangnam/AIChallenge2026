from pathlib import Path

import pytest

from video_retrieval.config import Settings
from video_retrieval.models import AudioTrack, FrameRole, KeyFrame
from video_retrieval.text.asr import ASREngine
from video_retrieval.text.ocr import OCREngine
from tests.helpers import write_dummy_image


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
