from __future__ import annotations

"""Small dependency-free progress reporting for server experiments."""

import sys
import time
from typing import TextIO


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ProgressReporter:
    """Print progress to stderr without interfering with JSON stdout."""

    def __init__(
        self,
        label: str,
        total: int,
        *,
        unit: str = "items",
        expected_item_seconds: float | None = None,
        initial_elapsed_seconds: float = 0.0,
        interval_seconds: float = 5.0,
        stream: TextIO | None = None,
    ) -> None:
        self.label = label
        self.total = max(0, int(total))
        self.unit = unit
        self.expected_item_seconds = (
            None
            if expected_item_seconds is None
            else max(0.0, float(expected_item_seconds))
        )
        self.initial_elapsed_seconds = max(0.0, float(initial_elapsed_seconds))
        self.interval_seconds = interval_seconds
        self.stream = stream or sys.stderr
        self.started = time.perf_counter()
        self.last_report = 0.0
        self.last_completed = 0
        self.report(0, force=True)

    def _line(self, completed: int) -> str:
        active_elapsed = max(time.perf_counter() - self.started, 1e-6)
        elapsed = self.initial_elapsed_seconds + active_elapsed
        rate = completed / active_elapsed if completed else 0.0
        if completed:
            remaining = None if self.total <= completed else (self.total - completed) / rate
            projected_total = elapsed + (remaining or 0.0)
        elif self.expected_item_seconds is not None:
            remaining = self.total * self.expected_item_seconds
            projected_total = self.initial_elapsed_seconds + remaining
        else:
            remaining = None
            projected_total = None
        percent = 100.0 if self.total == 0 else min(100.0, completed * 100.0 / self.total)
        width = 24
        filled = width if self.total == 0 else min(width, int(width * completed / self.total))
        bar = "#" * filled + "." * (width - filled)
        return (
            f"[ALADDIN] {self.label}: [{bar}] {completed}/{self.total} "
            f"({percent:5.1f}%) | {rate:.2f} {self.unit}/s | "
            f"elapsed {_duration(elapsed)} | ETA {_duration(remaining)} | "
            f"total~{_duration(projected_total)}"
        )

    def report(self, completed: int, *, force: bool = False) -> None:
        completed = min(self.total, max(0, int(completed)))
        now = time.perf_counter()
        if not force and completed < self.total and now - self.last_report < self.interval_seconds:
            return
        self.last_report = now
        self.last_completed = completed
        if self.stream.isatty():
            print("\r" + self._line(completed), end="", file=self.stream, flush=True)
        else:
            print(self._line(completed), file=self.stream, flush=True)

    def finish(self) -> None:
        self.report(self.total, force=True)
        if self.stream.isatty():
            print(file=self.stream, flush=True)



