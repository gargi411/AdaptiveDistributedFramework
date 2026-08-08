"""Time utility functions for the Adaptive Distributed Framework.

High-resolution timing for scheduler overhead measurement and
ISO 8601 timestamp generation.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator


def now_utc_iso() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Returns:
        ISO 8601 UTC timestamp string (e.g., '2026-07-26T17:00:00.000000+00:00').

    Example:
        >>> ts = now_utc_iso()
        >>> print(ts)
        '2026-07-26T17:00:00.123456+00:00'
    """
    return datetime.now(timezone.utc).isoformat()


def monotonic_seconds() -> float:
    """Return the current value of the monotonic clock in seconds.

    Use for measuring elapsed time. Not affected by system clock changes.

    Returns:
        Float seconds from an arbitrary reference point.

    Example:
        >>> start = monotonic_seconds()
        >>> # ... do work ...
        >>> elapsed = monotonic_seconds() - start
    """
    return time.monotonic()


def perf_counter() -> float:
    """Return the high-resolution performance counter value.

    Use for measuring short intervals with maximum precision.
    Equivalent to time.perf_counter().

    Returns:
        Float seconds with nanosecond-level resolution.

    Example:
        >>> t0 = perf_counter()
        >>> # ... scheduler dispatch ...
        >>> scheduler_time = perf_counter() - t0
    """
    return time.perf_counter()


@contextmanager
def timer() -> Generator[list[float], None, None]:
    """Context manager that measures elapsed wall-clock time.

    Yields a single-element list. After the block exits, list[0] contains
    the elapsed time in seconds using perf_counter().

    Yields:
        A list that will contain the elapsed time in seconds after exit.

    Example:
        >>> with timer() as t:
        ...     time.sleep(0.1)
        >>> print(f"Elapsed: {t[0]:.3f}s")
        Elapsed: 0.100s
    """
    elapsed: list[float] = [0.0]
    start = perf_counter()
    try:
        yield elapsed
    finally:
        elapsed[0] = perf_counter() - start


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds (non-negative).

    Returns:
        Formatted string: e.g., '1h 23m 45.67s', '2m 10.00s', '0.845s'.

    Example:
        >>> print(format_duration(4425.67))
        '1h 13m 45.67s'
    """
    if seconds < 0:
        return "0.000s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s:.2f}s"
    if m > 0:
        return f"{m}m {s:.2f}s"
    return f"{s:.3f}s"


def compute_overhead_fraction(
    scheduler_time_seconds: float,
    total_time_seconds: float,
) -> float:
    """Compute scheduler overhead as a fraction of total execution time.

    Formula from architecture v2.0 §4.2:
        Scheduler Overhead = Scheduler Time / Total Execution Time

    Args:
        scheduler_time_seconds: Cumulative time spent inside the scheduler.
        total_time_seconds: End-to-end wall-clock execution time.

    Returns:
        Overhead fraction in [0.0, 1.0]. Returns 0.0 if total_time_seconds <= 0.

    Example:
        >>> fraction = compute_overhead_fraction(0.8, 120.0)
        >>> print(f"{fraction * 100:.3f}%")
        0.667%
    """
    if total_time_seconds <= 0:
        return 0.0
    return scheduler_time_seconds / total_time_seconds
