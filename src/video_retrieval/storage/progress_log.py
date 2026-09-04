"""Stdout heartbeats for long Colab bootstrap waits."""

from __future__ import annotations

import threading
import time
from pathlib import Path


def tail_text(path: Path, *, max_chars: int = 1200) -> str:
    if not path.is_file():
        return "(no log file yet)"
    try:
        data = path.read_bytes()
    except OSError as exc:
        return f"(could not read log: {exc})"
    text = data.decode("utf-8", errors="replace")
    if not text.strip():
        return "(empty log)"
    return text[-max_chars:] if len(text) > max_chars else text


def print_log_heartbeat(
    prefix: str,
    log_path: Path,
    *,
    seconds_left: float | None = None,
    max_chars: int = 1200,
) -> None:
    left = f" ({int(seconds_left)}s left)" if seconds_left is not None else ""
    print(f"{prefix} still working…{left}", flush=True)
    print(f"{prefix} --- log tail ({log_path}) ---", flush=True)
    print(tail_text(log_path, max_chars=max_chars), flush=True)
    print(f"{prefix} --- end log tail ---", flush=True)


class Heartbeat:
    """Background prints while a blocking call runs (snapshot recover, downloads)."""

    def __init__(
        self,
        prefix: str,
        *,
        interval_sec: float = 20.0,
        log_path: Path | None = None,
        message: str = "still working…",
    ) -> None:
        self.prefix = prefix
        self.interval_sec = interval_sec
        self.log_path = log_path
        self.message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0

    def __enter__(self) -> Heartbeat:
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, name=f"hb-{self.prefix}", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        elapsed = time.monotonic() - self._started
        print(f"{self.prefix} done in {elapsed:.1f}s", flush=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            elapsed = time.monotonic() - self._started
            print(f"{self.prefix} {self.message} ({elapsed:.0f}s elapsed)", flush=True)
            if self.log_path is not None:
                print(f"{self.prefix} --- log tail ({self.log_path}) ---", flush=True)
                print(tail_text(self.log_path), flush=True)
                print(f"{self.prefix} --- end log tail ---", flush=True)
