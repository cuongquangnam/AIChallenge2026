from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from video_retrieval.config import Settings
from video_retrieval.models import AudioTrack
from video_retrieval.text.asr import ASREngine


@pytest.mark.unit
def test_mock_asr_returns_transcript(tmp_path: Path, settings: Settings) -> None:
    asr = ASREngine(settings)
    audio = AudioTrack(video_id="clip", path=tmp_path / "clip.wav", duration_sec=1.5)

    docs = asr.transcribe(audio)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.doc_id == "clip:asr:0"
    assert doc.video_id == "clip"
    assert doc.source == "asr"
    assert doc.text == "mock transcription for clip"
    assert doc.start_sec == 0.0
    assert doc.end_sec == 1.5
    assert doc.metadata["audio_path"] == str(audio.path)


@pytest.mark.unit
def test_whisper_backend_loads_configured_model(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_whisper = SimpleNamespace(load_model=MagicMock(return_value=MagicMock()))
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)
    settings.asr_backend = "whisper"
    settings.whisper_model = "tiny"

    asr = ASREngine(settings)

    fake_whisper.load_model.assert_called_once_with("tiny")
    assert asr._model is fake_whisper.load_model.return_value


@pytest.mark.unit
def test_whisper_backend_requires_package(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "whisper":
            raise ImportError("No module named whisper")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    settings.asr_backend = "whisper"

    with pytest.raises(ImportError, match=r"Install ML extras"):
        ASREngine(settings)


@pytest.mark.unit
def test_whisper_transcribe_maps_segments(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.asr_backend = "whisper"
    monkeypatch.setattr(ASREngine, "_load_whisper", lambda self: None)

    asr = ASREngine(settings)
    asr._model = MagicMock()
    asr._model.transcribe.return_value = {
        "text": "hello world",
        "segments": [
            {"text": " hello ", "start": 0.0, "end": 0.8},
            {"text": "  ", "start": 0.8, "end": 1.0},
            {"text": "world", "start": 1.0, "end": 1.6},
        ],
    }
    audio = AudioTrack(video_id="news", path=tmp_path / "news.wav", duration_sec=2.0)

    docs = asr.transcribe(audio)

    asr._model.transcribe.assert_called_once_with(str(audio.path))
    assert [doc.doc_id for doc in docs] == ["news:asr:0", "news:asr:2"]
    assert [doc.text for doc in docs] == ["hello", "world"]
    assert docs[0].start_sec == 0.0
    assert docs[0].end_sec == 0.8
    assert docs[1].start_sec == 1.0
    assert docs[1].end_sec == 1.6
    assert all(doc.source == "asr" for doc in docs)
    assert all(doc.metadata["audio_path"] == str(audio.path) for doc in docs)


@pytest.mark.unit
def test_whisper_transcribe_falls_back_to_full_text(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.asr_backend = "whisper"
    monkeypatch.setattr(ASREngine, "_load_whisper", lambda self: None)

    asr = ASREngine(settings)
    asr._model = MagicMock()
    asr._model.transcribe.return_value = {
        "text": "  full transcript  ",
        "segments": [],
    }
    audio = AudioTrack(video_id="clip", path=tmp_path / "clip.wav", duration_sec=3.25)

    docs = asr.transcribe(audio)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.doc_id == "clip:asr:full"
    assert doc.text == "full transcript"
    assert doc.start_sec == 0.0
    assert doc.end_sec == 3.25


@pytest.mark.unit
def test_whisper_transcribe_returns_empty_when_no_text(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.asr_backend = "whisper"
    monkeypatch.setattr(ASREngine, "_load_whisper", lambda self: None)

    asr = ASREngine(settings)
    asr._model = MagicMock()
    asr._model.transcribe.return_value = {"text": "   ", "segments": []}
    audio = AudioTrack(video_id="silent", path=tmp_path / "silent.wav", duration_sec=1.0)

    assert asr.transcribe(audio) == []


@pytest.mark.unit
def test_whisper_transcribe_handles_missing_segment_times(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.asr_backend = "whisper"
    monkeypatch.setattr(ASREngine, "_load_whisper", lambda self: None)

    asr = ASREngine(settings)
    asr._model = MagicMock()
    asr._model.transcribe.return_value = {
        "segments": [{"text": "hello"}],
    }
    audio = AudioTrack(video_id="clip", path=tmp_path / "clip.wav", duration_sec=1.0)

    docs = asr.transcribe(audio)

    assert len(docs) == 1
    assert docs[0].text == "hello"
    assert docs[0].start_sec == 0.0
    assert docs[0].end_sec == 0.0
