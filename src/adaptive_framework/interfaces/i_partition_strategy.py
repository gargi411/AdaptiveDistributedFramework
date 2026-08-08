"""IPartitionStrategy — Abstract partition strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from adaptive_framework.models.document import PDFMetadata
from adaptive_framework.models.scheduling import Partition, PartitionStatistics


class IPartitionStrategy(ABC):
    """Abstract interface for document partitioning strategies.

    A partition strategy takes the full dataset metadata and divides it
    into Partitions — groups of work units to be assigned to workers.

    The primary strategy (Phase 3) is page-count-based partitioning,
    but this interface allows alternative strategies (e.g., size-based,
    complexity-based) to be plugged in without modifying the scheduler.

    Example:
        >>> strategy: IPartitionStrategy = PageCountPartitionStrategy(cfg, logger)
        >>> partitions, stats = strategy.partition(dataset, num_workers=4)
        >>> print(stats.avg_pages_per_partition)
        62.5
    """

    @abstractmethod
    def partition(
        self,
        dataset: list[PDFMetadata],
        num_workers: int,
    ) -> tuple[list[Partition], PartitionStatistics]:
        """Divide the dataset into partitions for the given number of workers.

        Args:
            dataset: List of PDFMetadata, one per document.
            num_workers: Target number of worker partitions.

        Returns:
            Tuple of:
                - List of Partition objects (len may differ from num_workers
                  if the strategy decides to use fewer).
                - PartitionStatistics summarising the partition plan.

        Raises:
            SchedulerError: If partitioning fails (e.g., empty dataset).
        """

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the unique identifier of this partitioning strategy.

        Returns:
            Strategy name string (e.g., 'page_count').
        """
