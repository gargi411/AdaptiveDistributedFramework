"""IScheduler — Abstract adaptive scheduler interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from adaptive_framework.models.document import PDFMetadata
from adaptive_framework.models.scheduling import PageWorkUnit, PartitionStatistics
from adaptive_framework.models.runtime import RuntimeMetrics


class IScheduler(ABC):
    """Abstract interface for the Adaptive Scheduler.

    The Adaptive Scheduler:
        1. Receives the document dataset.
        2. Uses an IPartitionStrategy to create partitions.
        3. Builds a Priority Queue (ordered by page count).
        4. Dispatches PageWorkUnits to workers.
        5. Monitors and enforces Work Stealing.
        6. Tracks scheduler overhead (must be < 1%).

    This interface allows the scheduler implementation to be tested,
    replaced, or extended without affecting coordinators, workers,
    or the evaluation engine.

    Example:
        >>> scheduler: IScheduler = AdaptivePageCountScheduler(cfg, logger, strategy)
        >>> scheduler.submit_dataset(dataset)
        >>> scheduler.start()
    """

    @abstractmethod
    def submit_dataset(self, dataset: list[PDFMetadata]) -> PartitionStatistics:
        """Accept a document dataset and build the partition plan.

        Must be called before start(). Computes and stores the
        partition plan internally.

        Args:
            dataset: List of PDFMetadata for all documents to schedule.

        Returns:
            PartitionStatistics summarising the partition plan.

        Raises:
            SchedulerError: If the dataset is empty or partitioning fails.
        """

    @abstractmethod
    def start(self) -> None:
        """Start the scheduling loop.

        Begins dispatching work units from the Priority Queue to workers.
        Non-blocking: returns immediately; scheduling runs in background.

        Raises:
            SchedulerError: If submit_dataset() has not been called.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop the scheduling loop gracefully.

        Waits for in-flight dispatches to complete before stopping.
        """

    @abstractmethod
    def get_next_work_unit(self) -> PageWorkUnit | None:
        """Pop the highest-priority work unit from the queue.

        Returns:
            The next PageWorkUnit, or None if the queue is empty.
        """

    @abstractmethod
    def requeue_work_unit(self, work_unit: PageWorkUnit) -> None:
        """Return a work unit to the Priority Queue.

        Called by the Failure Recovery mechanism when a worker is lost.
        The work unit status is set to RETRYING before being re-queued.

        Args:
            work_unit: The work unit to return to the queue.
        """

    @abstractmethod
    def is_complete(self) -> bool:
        """Return True if all submitted work units have been processed.

        Returns:
            True when the queue is empty and no units are in-flight.
        """

    @abstractmethod
    def get_scheduler_overhead_fraction(self) -> float:
        """Return the current measured scheduler overhead fraction.

        Returns:
            Fraction in [0.0, 1.0] (e.g., 0.0067 = 0.67%).
            Architecture target: < 0.01 (< 1%).
        """
