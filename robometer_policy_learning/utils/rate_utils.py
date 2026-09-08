"""
Utilities for rate limiting and performance tracking.
"""

import threading
import time
from typing import Optional


class RateMeter:
    """Tracks average seconds per update and exposes Hz. Thread-safe."""

    def __init__(self, name: str, alpha: float = 0.1):
        self.name = name
        self._ema_seconds = EmaValue(alpha=alpha)

    def record(self, seconds_for_update: float) -> None:
        self._ema_seconds.update(max(0.0, float(seconds_for_update)))

    @property
    def avg_seconds_per_update(self) -> float:
        return self._ema_seconds.avg

    @property
    def updates_per_second(self) -> float:
        avg = self.avg_seconds_per_update
        return (1.0 / avg) if avg > 0 else 0.0


class RateLimiter:
    """Simple wall-clock rate limiter to target a frequency (Hz)."""

    def __init__(self, target_hz: Optional[float]):
        self.target_hz = float(target_hz) if target_hz else 0.0
        self._period = (1.0 / self.target_hz) if self.target_hz > 0 else 0.0
        self._last_time: Optional[float] = None

    def throttle(self) -> None:
        if self._period <= 0:
            return
        now = time.time()
        if self._last_time is None:
            self._last_time = now
            return
        elapsed = now - self._last_time
        remaining = self._period - elapsed
        if remaining > 0:
            time.sleep(remaining)
            now = time.time()
        self._last_time = now


class EmaValue:
    """Thread-safe exponential moving average tracker."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self._avg = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def update(self, value: float) -> None:
        with self._lock:
            if self._count == 0:
                self._avg = float(value)
            else:
                self._avg = self.alpha * float(value) + (1.0 - self.alpha) * self._avg
            self._count += 1

    @property
    def avg(self) -> float:
        with self._lock:
            return float(self._avg)

    @property
    def count(self) -> int:
        with self._lock:
            return int(self._count)
