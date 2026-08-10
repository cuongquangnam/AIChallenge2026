from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from video_retrieval.config import Settings
from video_retrieval.models import AudioTrack
from video_retrieval.text.asr import ASREngine

PHRASE = "hello world"


def _write_spoken_wav(path: Path, text: str = PHRASE) -> Path:
    """Synthesize short speech audio for live Whisper tests (macOS say + ffmpeg)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not shutil.which("say"):
        pytest.skip("macOS `say` is required to synthesize speech audio")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required to convert speech audio to WAV")

    aiff_path = path.with_suffix(".aiff")
    say_result = subprocess.run(
        ["say", "-o", str(aiff_path), text],
        capture_output=True,
        text=True,
    )
    if say_result.returncode != 0:
        pytest.skip(f"`say` failed: {say_result.stderr.strip()}")

    ffmpeg_result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(aiff_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    aiff_path.unlink(missing_ok=True)
    if ffmpeg_result.returncode != 0:
        pytest.skip(f"ffmpeg failed: {ffmpeg_result.stderr.strip()}")
    return path


@pytest.mark.integration
def test_whisper_asr_transcribes_spoken_audio(tmp_path: Path, settings: Settings) -> None:
    whisper = pytest.importorskip("whisper")

    try:
        whisper.load_model("tiny")
    except Exception as exc:  # noqa: BLE001 - network / download failures
        pytest.skip(f"Could not download Whisper tiny model: {exc}")

    wav_path = _write_spoken_wav(tmp_path / "hello.wav")
    settings.asr_backend = "whisper"
    settings.whisper_model = "tiny"

    asr = ASREngine(settings)
    docs = asr.transcribe(
        AudioTrack(video_id="spoken", path=wav_path, duration_sec=2.0)
    )

    assert docs, "expected Whisper to return at least one ASR document"
    combined = " ".join(doc.text for doc in docs).lower()
    assert "hello" in combined
    assert all(doc.source == "asr" for doc in docs)
    assert all(doc.video_id == "spoken" for doc in docs)
    assert all(doc.metadata["audio_path"] == str(wav_path) for doc in docs)
