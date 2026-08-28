from __future__ import annotations

import logging
import queue
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ModelPool(Generic[T]):
    """Fixed-size pool; each borrow gets exclusive use of one model instance."""

    def __init__(
        self,
        factory: Callable[[int], T],
        *,
        size: int,
        name: str = "model",
    ):
        if size < 1:
            raise ValueError("pool size must be >= 1")
        self.name = name
        self.size = size
        self._queue: queue.Queue[T] = queue.Queue(maxsize=size)
        for index in range(size):
            logger.info("Loading %s pool instance %s/%s", name, index + 1, size)
            self._queue.put(factory(index))

    @contextmanager
    def borrow(self) -> Iterator[T]:
        model = self._queue.get()
        try:
            yield model
        finally:
            self._queue.put(model)
