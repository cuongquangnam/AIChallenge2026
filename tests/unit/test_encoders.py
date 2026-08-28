from pathlib import Path

import pytest

from video_retrieval.config import Settings
from video_retrieval.encoders.visual import VisualEncoder
from video_retrieval.models import FrameRole, KeyFrame
from tests.helpers import write_dummy_image


@pytest.mark.unit
def test_siglip_only_encoder_skips_beit_mock(settings: Settings, tmp_path: Path) -> None:
    encoder = VisualEncoder(settings, load_beit=False)
    image = write_dummy_image(tmp_path / "frame.jpg")
    siglip, beit = encoder.encode_image(image)
    assert len(siglip) == settings.siglip_dim
    assert beit == []


@pytest.mark.unit
def test_mock_encoder_dimensions_and_determinism(tmp_path: Path, settings: Settings) -> None:
    image = write_dummy_image(tmp_path / "frame.jpg")
    encoder = VisualEncoder(settings)

    a_sig, a_beit = encoder.encode_image(image)
    b_sig, b_beit = encoder.encode_image(image)

    assert len(a_sig) == settings.siglip_dim
    assert len(a_beit) == settings.beit3_dim
    assert a_sig == b_sig
    assert a_beit == b_beit


@pytest.mark.unit
def test_mock_encode_text_and_keyframes(tmp_path: Path, settings: Settings) -> None:
    encoder = VisualEncoder(settings)
    text_vec = encoder.encode_text("a person walking")
    assert len(text_vec) == settings.siglip_dim

    path = write_dummy_image(tmp_path / "shot_0000_start.jpg")
    kf = KeyFrame(
        video_id="v1",
        shot_index=0,
        role=FrameRole.START,
        frame_index=0,
        timestamp_sec=0.0,
        path=path,
    )
    embeddings = encoder.encode_keyframes([kf])
    assert len(embeddings) == 1
    assert embeddings[0].keyframe.video_id == "v1"
