"""IWorker — Abstract worker node interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from adaptive_framework.models.document import PageResult
from adaptive_framework.models.scheduling import PageWorkUnit
from adaptive_framework.models.runtime import WorkerStatus


class IWorker(ABC):
    """Abstract interface for a distributed worker node.

    A worker receives PageWorkUnits from the Scheduler, processes them
    through the Document Processing Engine, and returns PageResults
    to the ResultCollector.

    In Phase 2+, this will be a Ray remote actor. The interface ensures
    that the Coordinator communicates with workers only through this
    contract, never through Ray-specific internals.

    Example:
        >>> worker: IWorker = RayWorker(worker_id="w_01", cfg=cfg, logger=logger)
        >>> worker.initialize()
        >>> result = worker.process(work_unit)
    """

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the worker and its resources.

        Called once before process() is invoked. Sets up the Document
        Processing Engine and any required resources.

        Raises:
            ClusterError: If initialization fails.
        """

    @abstractmethod
    def process(self, work_unit: PageWorkUnit) -> list[PageResult]:
        """Process a PageWorkUnit and return per-page results.

        Args:
            work_unit: The work unit containing the page range to process.

        Returns:
            List of PageResult, one per page in the work unit.

        Raises:
            ProcessingError: If processing fails unrecoverably.
        """

    @abstractmethod
    def get_status(self) -> WorkerStatus:
        """Return the current status snapshot of this worker.

        Returns:
            WorkerStatus with current state, active units, and resource snapshot.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Gracefully shut down this worker.

        Completes any in-progress work unit before shutting down.
        """

    @abstractmethod
    def get_worker_id(self) -> str:
        """Return the unique identifier of this worker.

        Returns:
            Worker ID string.
        """
