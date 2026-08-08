"""Workload Analyzer — Module 5 of Phase 2A.

Analyses a document dataset before scheduling to provide estimates and
recommendations. Consumed by the research evaluation engine and by the
CLI for pre-run planning output.

Responsibilities:
    - Compute total, average, min, max page counts.
    - Estimate processing cost (pages × processing_cost_per_page).
    - Estimate partition difficulty (coefficient of variation).
    - Estimate expected execution time per worker.
    - Generate a human-readable workload report.

Architecture note:
    This module contains ONLY analysis logic.
    It does NOT modify the dataset or trigger scheduling.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Any

from adaptive_framework.models.document import PDFMetadata


# ---------------------------------------------------------------------------
# Processing cost model
# ---------------------------------------------------------------------------

# Default seconds to process one page (for time estimation)
_DEFAULT_SECONDS_PER_PAGE: float = 2.0

# Difficulty thresholds (coefficient of variation)
_DIFFICULTY_LOW_CV: float = 0.2
_DIFFICULTY_MEDIUM_CV: float = 0.5


@dataclass
class WorkloadReport:
    """Structured workload analysis report.

    Attributes:
        total_documents: Number of documents in the dataset.
        total_pages: Sum of all page counts.
        avg_pages: Mean pages per document.
        median_pages: Median pages per document.
        min_pages: Minimum pages across all documents.
        max_pages: Maximum pages across all documents.
        std_pages: Standard deviation of pages.
        coefficient_of_variation: std / avg — measure of spread.
        total_size_mb: Total estimated file size.
        avg_size_mb: Mean file size per document.
        estimated_processing_cost: Total pages × cost_per_page seconds.
        expected_time_per_worker: Estimated seconds per worker (cost / workers).
        partition_difficulty: 'low', 'medium', or 'high'.
        recommended_workers: Suggested number of workers based on workload.
        analysis_timestamp_utc: ISO 8601 UTC timestamp of this analysis.

    Example:
        >>> report = WorkloadReport(total_documents=100, total_pages=5000, ...)
        >>> print(report.partition_difficulty)
        'medium'
    """

    total_documents: int
    total_pages: int
    avg_pages: float
    median_pages: float
    min_pages: int
    max_pages: int
    std_pages: float
    coefficient_of_variation: float
    total_size_mb: float
    avg_size_mb: float
    estimated_processing_cost: float
    expected_time_per_worker: float
    partition_difficulty: str
    recommended_workers: int
    analysis_timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary with all report fields.
        """
        return asdict(self)

    def format_report(self) -> str:
        """Format a human-readable workload report.

        Returns:
            Multi-line string suitable for console output or logging.
        """
        width = 62
        sep = "─" * width
        lines = [
            f"┌{sep}┐",
            f"│{'WORKLOAD ANALYSIS REPORT':^{width}}│",
            f"├{sep}┤",
            f"│  {'Documents:':<30} {self.total_documents:>10,d}          │",
            f"│  {'Total Pages:':<30} {self.total_pages:>10,d}          │",
            f"│  {'Average Pages/Document:':<30} {self.avg_pages:>10.1f}          │",
            f"│  {'Median Pages/Document:':<30} {self.median_pages:>10.1f}          │",
            f"│  {'Min Pages:':<30} {self.min_pages:>10,d}          │",
            f"│  {'Max Pages:':<30} {self.max_pages:>10,d}          │",
            f"│  {'Std Dev Pages:':<30} {self.std_pages:>10.2f}          │",
            f"│  {'Coeff. of Variation:':<30} {self.coefficient_of_variation:>10.4f}          │",
            f"├{sep}┤",
            f"│  {'Total Dataset Size:':<30} {self.total_size_mb:>10.2f} MB       │",
            f"│  {'Avg Document Size:':<30} {self.avg_size_mb:>10.2f} MB       │",
            f"├{sep}┤",
            f"│  {'Est. Processing Cost:':<30} {self.estimated_processing_cost:>10.1f} s        │",
            f"│  {'Expected Time/Worker:':<30} {self.expected_time_per_worker:>10.1f} s        │",
            f"│  {'Partition Difficulty:':<30} {self.partition_difficulty.upper():>10}          │",
            f"│  {'Recommended Workers:':<30} {self.recommended_workers:>10}          │",
            f"└{sep}┘",
        ]
        return "\n".join(lines)


class WorkloadAnalyzer:
    """Analyses a PDFMetadata dataset to produce a WorkloadReport.

    Attributes:
        _seconds_per_page: Assumed processing cost per page (in seconds).

    Example:
        >>> analyzer = WorkloadAnalyzer()
        >>> report = analyzer.analyze(dataset, num_workers=4)
        >>> print(report.format_report())
    """

    def __init__(
        self,
        seconds_per_page: float = _DEFAULT_SECONDS_PER_PAGE,
    ) -> None:
        """Initialise the WorkloadAnalyzer.

        Args:
            seconds_per_page: Assumed processing time per page in seconds.
                Used for cost and time estimates. Default: 2.0 s/page.
        """
        self._seconds_per_page = seconds_per_page

    def analyze(
        self,
        dataset: list[PDFMetadata],
        num_workers: int = 1,
    ) -> WorkloadReport:
        """Compute a WorkloadReport for the given dataset.

        Args:
            dataset: List of PDFMetadata records.
            num_workers: Number of workers (used for per-worker time estimate).

        Returns:
            WorkloadReport with all fields populated.

        Raises:
            ValueError: If dataset is empty.
        """
        if not dataset:
            raise ValueError("WorkloadAnalyzer.analyze() requires a non-empty dataset.")

        from datetime import datetime, timezone

        now_iso = datetime.now(timezone.utc).isoformat()

        page_counts = [m.pages for m in dataset]
        sizes_mb = [m.estimated_size_mb for m in dataset]

        total_pages = sum(page_counts)
        avg_pages = statistics.mean(page_counts)
        median_pages = statistics.median(page_counts)
        min_pages = min(page_counts)
        max_pages = max(page_counts)
        std_pages = statistics.stdev(page_counts) if len(page_counts) > 1 else 0.0
        cv = std_pages / avg_pages if avg_pages > 0 else 0.0

        total_size_mb = sum(sizes_mb)
        avg_size_mb = statistics.mean(sizes_mb)

        cost = total_pages * self._seconds_per_page
        effective_workers = max(1, num_workers)
        time_per_worker = cost / effective_workers

        difficulty = self._classify_difficulty(cv)
        recommended = self._recommend_workers(
            total_pages=total_pages,
            avg_pages=avg_pages,
            seconds_per_page=self._seconds_per_page,
        )

        return WorkloadReport(
            total_documents=len(dataset),
            total_pages=total_pages,
            avg_pages=round(avg_pages, 2),
            median_pages=float(median_pages),
            min_pages=min_pages,
            max_pages=max_pages,
            std_pages=round(std_pages, 2),
            coefficient_of_variation=round(cv, 6),
            total_size_mb=round(total_size_mb, 2),
            avg_size_mb=round(avg_size_mb, 4),
            estimated_processing_cost=round(cost, 2),
            expected_time_per_worker=round(time_per_worker, 2),
            partition_difficulty=difficulty,
            recommended_workers=recommended,
            analysis_timestamp_utc=now_iso,
        )

    def analyze_partitions(
        self,
        partition_page_counts: list[int],
    ) -> dict[str, Any]:
        """Compute balance metrics for a completed partition plan.

        Args:
            partition_page_counts: List of page counts (one per partition).

        Returns:
            Dictionary with balance_score, variance, cv, and imbalance_ratio.
        """
        if not partition_page_counts:
            return {}

        avg = sum(partition_page_counts) / len(partition_page_counts)
        variance = (
            sum((p - avg) ** 2 for p in partition_page_counts)
            / len(partition_page_counts)
        )
        std = math.sqrt(variance)
        cv = std / avg if avg > 0 else 0.0
        max_pages = max(partition_page_counts)
        min_pages = min(partition_page_counts)
        imbalance_ratio = max_pages / min_pages if min_pages > 0 else float("inf")
        balance_score = max(0.0, min(1.0, 1.0 - cv))

        return {
            "partition_count": len(partition_page_counts),
            "total_pages": sum(partition_page_counts),
            "avg_pages": round(avg, 2),
            "std_pages": round(std, 2),
            "coefficient_of_variation": round(cv, 6),
            "variance": round(variance, 2),
            "balance_score": round(balance_score, 6),
            "imbalance_ratio": round(imbalance_ratio, 4),
            "min_pages": min_pages,
            "max_pages": max_pages,
        }

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _classify_difficulty(cv: float) -> str:
        """Classify workload partitioning difficulty from coefficient of variation.

        Args:
            cv: Coefficient of variation (std / mean) of page counts.

        Returns:
            'low' if CV < 0.2, 'medium' if CV < 0.5, 'high' otherwise.
        """
        if cv < _DIFFICULTY_LOW_CV:
            return "low"
        if cv < _DIFFICULTY_MEDIUM_CV:
            return "medium"
        return "high"

    @staticmethod
    def _recommend_workers(
        total_pages: int,
        avg_pages: float,
        seconds_per_page: float,
        target_completion_minutes: float = 60.0,
    ) -> int:
        """Recommend an optimal number of workers.

        Targets a completion time of ``target_completion_minutes`` minutes.
        The recommendation is clamped to [1, 64].

        Args:
            total_pages: Total pages in the dataset.
            avg_pages: Average pages per document.
            seconds_per_page: Processing cost per page.
            target_completion_minutes: Desired maximum completion time.

        Returns:
            Recommended number of workers (integer >= 1).
        """
        total_cost_seconds = total_pages * seconds_per_page
        target_seconds = target_completion_minutes * 60
        if target_seconds <= 0:
            return 1
        raw = math.ceil(total_cost_seconds / target_seconds)
        return max(1, min(64, raw))
