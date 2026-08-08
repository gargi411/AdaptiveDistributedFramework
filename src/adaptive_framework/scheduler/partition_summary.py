"""Partition Summary — human-readable reporting for partition plans.

Generates formatted text and structured dictionary summaries of
partition plans, suitable for logging, console output, and research
paper tables.
"""

from __future__ import annotations

import math
from typing import Any

from adaptive_framework.models.scheduling import Partition, PartitionStatistics


class PartitionSummary:
    """Generates formatted summaries for partition plans.

    Attributes:
        _partitions: The list of Partition objects.
        _stats: The PartitionStatistics for this plan.

    Example:
        >>> summary = PartitionSummary(partitions, stats)
        >>> print(summary.format_table())
        ┌─────────────────────────────────────────────────────────────────┐
        │              Adaptive Page Count Partition Plan                 │
        ...
    """

    def __init__(
        self,
        partitions: list[Partition],
        stats: PartitionStatistics,
    ) -> None:
        """Initialise the PartitionSummary.

        Args:
            partitions: The partition plan to summarise.
            stats: Precomputed statistics for the plan.
        """
        self._partitions = partitions
        self._stats = stats

    # ------------------------------------------------------------------ #
    # Formatted output                                                     #
    # ------------------------------------------------------------------ #

    def format_table(self) -> str:
        """Format a text table showing the partition plan.

        Returns:
            Multi-line string showing each partition with its page count
            and work unit count, along with aggregate statistics.
        """
        width = 65
        sep = "─" * width
        lines: list[str] = []

        lines.append(f"┌{sep}┐")
        lines.append(
            f"│{'Adaptive Page Count Partition Plan':^{width}}│"
        )
        lines.append(f"├{sep}┤")
        lines.append(
            f"│  {'Worker':<12} {'Pages':>8} {'Work Units':>12} {'Bar':<30}│"
        )
        lines.append(f"├{sep}┤")

        max_pages = self._stats.max_pages_per_partition or 1
        for i, partition in enumerate(self._partitions):
            bar_filled = int(partition.total_pages / max_pages * 28)
            bar = "█" * bar_filled + "░" * (28 - bar_filled)
            worker_label = partition.worker_id or f"Worker {i + 1}"
            lines.append(
                f"│  {worker_label:<12} {partition.total_pages:>8,d} "
                f"{partition.total_work_units:>12,d} [{bar}]│"
            )

        lines.append(f"├{sep}┤")
        lines.append(
            f"│  {'Total Pages:':<20} {self._stats.total_pages:>10,d}"
            f"{'':<{width - 33}}│"
        )
        lines.append(
            f"│  {'Avg Pages/Partition:':<20} {self._stats.avg_pages_per_partition:>10.1f}"
            f"{'':<{width - 33}}│"
        )
        lines.append(
            f"│  {'Std Dev Pages:':<20} {self._stats.std_pages_per_partition:>10.1f}"
            f"{'':<{width - 33}}│"
        )
        balance_score = self.balance_score()
        lines.append(
            f"│  {'Balance Score:':<20} {balance_score:>10.4f}"
            f"{'':<{width - 33}}│"
        )
        lines.append(
            f"│  {'Variance:':<20} {self.variance():>10.2f}"
            f"{'':<{width - 33}}│"
        )
        lines.append(f"└{sep}┘")
        return "\n".join(lines)

    def format_compact(self) -> str:
        """Format a single-line compact summary.

        Returns:
            Compact summary string for log messages.
        """
        return (
            f"Partitions={self._stats.total_partitions}, "
            f"TotalPages={self._stats.total_pages:,}, "
            f"Avg={self._stats.avg_pages_per_partition:.0f}, "
            f"Std={self._stats.std_pages_per_partition:.1f}, "
            f"Balance={self.balance_score():.4f}"
        )

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def balance_score(self) -> float:
        """Compute a load balance score in [0.0, 1.0].

        A score of 1.0 means perfectly balanced (zero variance).
        A score approaching 0.0 indicates severe imbalance.

        Formula:
            balance_score = 1 - (std / avg)
            Clamped to [0, 1].

        Returns:
            Balance score. 1.0 if avg_pages == 0 (degenerate case).
        """
        avg = self._stats.avg_pages_per_partition
        std = self._stats.std_pages_per_partition
        if avg == 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - (std / avg)))

    def variance(self) -> float:
        """Return the variance of pages per partition.

        Returns:
            Variance (std²).
        """
        return self._stats.std_pages_per_partition ** 2

    def estimated_completion_time(
        self,
        pages_per_second_per_worker: float = 5.0,
    ) -> float:
        """Estimate total pipeline completion time in seconds.

        Based on the maximum pages across all partitions (the bottleneck).

        Args:
            pages_per_second_per_worker: Assumed processing throughput per worker.

        Returns:
            Estimated wall-clock completion time in seconds.
        """
        if pages_per_second_per_worker <= 0:
            return float("inf")
        bottleneck = self._stats.max_pages_per_partition
        return bottleneck / pages_per_second_per_worker

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full summary to a plain dictionary.

        Returns:
            Dictionary including partition details and aggregate statistics.
        """
        return {
            "statistics": self._stats.to_dict(),
            "balance_score": round(self.balance_score(), 6),
            "variance": round(self.variance(), 2),
            "partitions": [
                {
                    "partition_id": p.partition_id,
                    "worker_id": p.worker_id,
                    "total_pages": p.total_pages,
                    "work_unit_count": p.total_work_units,
                }
                for p in self._partitions
            ],
        }
