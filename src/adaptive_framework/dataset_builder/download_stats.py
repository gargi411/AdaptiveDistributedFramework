"""Download statistics dataclass for the Dataset Builder.

Tracks progress of the PubMed Central download session.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DownloadStats:
    """Tracks cumulative statistics for a dataset download session.

    Attributes:
        total_requested: Total articles requested from PubMed Central.
        total_downloaded: PDFs successfully downloaded.
        total_skipped: Articles skipped (already present, duplicate).
        total_failed: Articles that failed after all retries.
        total_bytes_downloaded: Cumulative bytes written to disk.
        session_start_time: Monotonic start time of the download session.
        errors: List of (article_id, error_message) tuples for failures.

    Example:
        >>> stats = DownloadStats(total_requested=100)
        >>> stats.record_success(file_size_bytes=1_048_576)
        >>> print(stats.success_rate)
        1.0
    """

    total_requested: int = 0
    total_downloaded: int = 0
    total_skipped: int = 0
    total_failed: int = 0
    total_bytes_downloaded: int = 0
    session_start_time: float = field(default_factory=time.monotonic)
    errors: list[tuple[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Mutators                                                             #
    # ------------------------------------------------------------------ #

    def record_success(self, file_size_bytes: int = 0) -> None:
        """Record a successful download.

        Args:
            file_size_bytes: Size of the downloaded file in bytes.
        """
        self.total_downloaded += 1
        self.total_bytes_downloaded += file_size_bytes

    def record_skip(self) -> None:
        """Record a skipped article (already downloaded or duplicate)."""
        self.total_skipped += 1

    def record_failure(self, article_id: str, reason: str) -> None:
        """Record a failed download attempt.

        Args:
            article_id: The PMC article identifier.
            reason: Human-readable failure reason.
        """
        self.total_failed += 1
        self.errors.append((article_id, reason))

    # ------------------------------------------------------------------ #
    # Derived Properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def success_rate(self) -> float:
        """Fraction of requested articles successfully downloaded.

        Returns:
            Float in [0.0, 1.0]. 0.0 if nothing was requested.
        """
        if self.total_requested == 0:
            return 0.0
        return self.total_downloaded / self.total_requested

    @property
    def total_mb_downloaded(self) -> float:
        """Total data downloaded in megabytes.

        Returns:
            Megabytes downloaded, rounded to 2 decimal places.
        """
        return round(self.total_bytes_downloaded / (1024 * 1024), 2)

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock time since session start.

        Returns:
            Elapsed time in seconds.
        """
        return time.monotonic() - self.session_start_time

    @property
    def download_rate_mb_per_second(self) -> float:
        """Average download speed in MB/s.

        Returns:
            MB per second. 0.0 if no time has elapsed.
        """
        elapsed = self.elapsed_seconds
        if elapsed <= 0:
            return 0.0
        return self.total_mb_downloaded / elapsed

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Serialize statistics to a plain dictionary.

        Returns:
            Dictionary suitable for JSON/CSV serialization.
        """
        return {
            "total_requested": self.total_requested,
            "total_downloaded": self.total_downloaded,
            "total_skipped": self.total_skipped,
            "total_failed": self.total_failed,
            "success_rate": round(self.success_rate, 4),
            "total_mb_downloaded": self.total_mb_downloaded,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "download_rate_mb_per_s": round(self.download_rate_mb_per_second, 4),
            "error_count": len(self.errors),
        }

    def __repr__(self) -> str:
        return (
            f"DownloadStats("
            f"downloaded={self.total_downloaded}/{self.total_requested}, "
            f"skipped={self.total_skipped}, "
            f"failed={self.total_failed}, "
            f"rate={self.download_rate_mb_per_second:.2f} MB/s)"
        )
