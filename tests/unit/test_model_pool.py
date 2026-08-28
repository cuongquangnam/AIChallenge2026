"""Unit tests for exclusive model pools."""

import threading

import pytest

from video_retrieval.model_pool import ModelPool


@pytest.mark.unit
def test_model_pool_exclusive_borrow() -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    def factory(index: int) -> dict[str, int]:
        return {"id": index}

    pool: ModelPool[dict[str, int]] = ModelPool(factory, size=2, name="test")

    def worker() -> None:
        nonlocal active, peak
        with pool.borrow() as item:
            with lock:
                active += 1
                peak = max(peak, active)
            assert "id" in item
            threading.Event().wait(0.05)
            with lock:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak <= 2
