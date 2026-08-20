from pathlib import Path

import pytest

from video_retrieval.config import Settings
from video_retrieval.encoders.visual import VisualEncoder, chunk_token_ids
from video_retrieval.models import FrameRole, KeyFrame
from video_retrieval.search.service import split_visual_clauses
from tests.helpers import write_dummy_image


@pytest.mark.unit
def test_mock_encoder_dimensions_and_determinism(tmp_path: Path, settings: Settings) -> None:
    image = write_dummy_image(tmp_path / "frame.jpg")
    encoder = VisualEncoder(settings)

    a_sig = encoder.encode_image(image)
    b_sig = encoder.encode_image(image)

    assert len(a_sig) == settings.siglip_dim
    assert a_sig == b_sig


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
    assert len(embeddings[0].siglip) == settings.siglip_dim


@pytest.mark.unit
def test_chunk_token_ids_covers_full_sequence() -> None:
    ids = list(range(90))
    chunks = chunk_token_ids(ids, chunk_size=62, overlap=8)
    assert all(len(chunk) <= 62 for chunk in chunks)
    flat_ends = {chunk[-1] for chunk in chunks}
    assert max(ids) in flat_ends
    assert chunks[0][0] == 0


@pytest.mark.unit
def test_split_visual_clauses_keeps_multi_sentence_query() -> None:
    text = (
        "Đây là phần giới thiệu việc phóng tàu vũ trụ tư nhân. "
        "Đoạn clip bắt đầu với hình ảnh 4 phi hành gia mặc áo đen. "
        "Một trong những nhiệm vụ dự kiến của tàu vũ trụ là nghiên cứu ánh sáng cực quang ở vùng cực"
    )
    clauses = split_visual_clauses(text)
    assert len(clauses) == 3
    assert "phi hành gia" in clauses[1]
    assert "cực quang" in clauses[2]
