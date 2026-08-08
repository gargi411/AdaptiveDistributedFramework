"""Scheduling data models for the Adaptive Distributed Framework.

Models:
    PageWorkUnit: Atomic unit of work dispatched to a worker.
    Partition: A named group of PageWorkUnits assigned to one worker.
    PartitionStatistics: Summary statistics about a partition set.

These models are consumed by the Adaptive Scheduler (Phase 3).
No scheduling logic is implemented here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

from adaptive_framework.core.exceptions import ValidationError


class WorkUnitStatus(str, Enum):
    """Lifecycle status of a PageWorkUnit.

    Values:
        PENDING: Not yet dispatched.
        DISPATCHED: Sent to a worker, awaiting completion.
        IN_PROGRESS: Currently being processed by a worker.
        COMPLETED: Processed successfully.
        FAILED: Processing failed; eligible for retry.
        RETRYING: Being retried after a failure or worker loss.
    """

    PENDING = "pending"
    DISPATCHED = "dispatched"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


# =============================================================
# PageWorkUnit
# =============================================================


@dataclass
class PageWorkUnit:
    """Atomic unit of work: a page range within a PDF document.

    The Adaptive Scheduler creates PageWorkUnits from a Partition plan
    and dispatches them to workers via the Priority Queue.

    Attributes:
        document_id: Parent document identifier.
        file_path: Absolute path to the source PDF.
        start_page: First page in this work unit (1-indexed, inclusive).
        end_page: Last page in this work unit (1-indexed, inclusive).
        page_count: Total pages in this work unit (end_page - start_page + 1).
        work_unit_id: Unique identifier. Auto-generated if not provided.
        priority: Scheduling priority. Higher = processed first.
        status: Current lifecycle status.
        assigned_worker_id: Worker currently holding this unit. None if unassigned.
        retry_count: Number of times this unit has been retried.

    Example:
        >>> wu = PageWorkUnit(document_id="abc-123",
        ...                   file_path="/data/paper.pdf",
        ...                   start_page=1, end_page=10)
        >>> wu.page_count
        10
    """

    document_id: str
    file_path: str
    start_page: int
    end_page: int
    work_unit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0
    status: WorkUnitStatus = WorkUnitStatus.PENDING
    assigned_worker_id: str | None = None
    retry_count: int = 0

    def __post_init__(self) -> None:
        """Validate work unit fields.

        Raises:
            ValidationError: If page range is invalid.
        """
        if self.start_page < 1:
            raise ValidationError(
                "PageWorkUnit.start_page must be >= 1 (1-indexed).",
                field="start_page",
                value=self.start_page,
            )
        if self.end_page < self.start_page:
            raise ValidationError(
                "PageWorkUnit.end_page must be >= start_page.",
                field="end_page",
                value=self.end_page,
            )
        if self.retry_count < 0:
            raise ValidationError(
                "PageWorkUnit.retry_count must be >= 0.",
                field="retry_count",
                value=self.retry_count,
            )

    @property
    def page_count(self) -> int:
        """Compute the number of pages in this work unit.

        Returns:
            Number of pages: end_page - start_page + 1.
        """
        return self.end_page - self.start_page + 1

    def is_terminal(self) -> bool:
        """Return True if this work unit is in a terminal state.

        A terminal state means no further scheduling actions will be taken.

        Returns:
            True if status is COMPLETED or FAILED.
        """
        return self.status in (WorkUnitStatus.COMPLETED, WorkUnitStatus.FAILED)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary including computed page_count.
        """
        d = asdict(self)
        d["page_count"] = self.page_count
        d["status"] = self.status.value
        return d

    def __repr__(self) -> str:
        return (
            f"PageWorkUnit(document_id='{self.document_id}', "
            f"pages={self.start_page}-{self.end_page} ({self.page_count}p), "
            f"status={self.status.value}, "
            f"priority={self.priority})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PageWorkUnit):
            return NotImplemented
        return self.work_unit_id == other.work_unit_id

    def __hash__(self) -> int:
        return hash(self.work_unit_id)


# =============================================================
# Partition
# =============================================================


@dataclass
class Partition:
    """A named group of PageWorkUnits assigned to one worker.

    The IPartitionStrategy creates Partitions from a list of documents.
    Each partition is ideally assigned to a single worker, though the
    Work Stealing algorithm may redistribute work units at runtime.

    Attributes:
        partition_id: Unique identifier. Auto-generated if not provided.
        worker_id: Target worker for this partition. None if unassigned.
        work_units: Ordered list of PageWorkUnits in this partition.
        total_pages: Sum of page_count across all work units.

    Example:
        >>> wu1 = PageWorkUnit("doc-1", "/data/a.pdf", 1, 20)
        >>> wu2 = PageWorkUnit("doc-2", "/data/b.pdf", 1, 15)
        >>> p = Partition(work_units=[wu1, wu2])
        >>> p.total_pages
        35
    """

    work_units: list[PageWorkUnit]
    partition_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    worker_id: str | None = None

    def __post_init__(self) -> None:
        if not self.work_units:
            raise ValidationError(
                "Partition.work_units must not be empty.",
                field="work_units",
            )

    @property
    def total_pages(self) -> int:
        """Sum of page counts across all work units.

        Returns:
            Total pages in this partition.
        """
        return sum(wu.page_count for wu in self.work_units)

    @property
    def total_work_units(self) -> int:
        """Return the number of work units in this partition.

        Returns:
            Count of work units.
        """
        return len(self.work_units)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary including computed totals.
        """
        return {
            "partition_id": self.partition_id,
            "worker_id": self.worker_id,
            "total_pages": self.total_pages,
            "total_work_units": self.total_work_units,
            "work_units": [wu.to_dict() for wu in self.work_units],
        }

    def __repr__(self) -> str:
        return (
            f"Partition(partition_id='{self.partition_id}', "
            f"worker_id={self.worker_id!r}, "
            f"work_units={self.total_work_units}, "
            f"total_pages={self.total_pages})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Partition):
            return NotImplemented
        return self.partition_id == other.partition_id

    def __hash__(self) -> int:
        return hash(self.partition_id)


# =============================================================
# PartitionStatistics
# =============================================================


@dataclass
class PartitionStatistics:
    """Summary statistics for a complete partition plan.

    Produced after all partitions are created, before scheduling begins.
    Used by the evaluation engine to compute load balance metrics.

    Attributes:
        total_partitions: Number of partitions created.
        total_work_units: Total PageWorkUnits across all partitions.
        total_pages: Grand total of pages across all partitions.
        min_pages_per_partition: Minimum pages in any single partition.
        max_pages_per_partition: Maximum pages in any single partition.
        avg_pages_per_partition: Mean pages per partition.
        std_pages_per_partition: Standard deviation of pages per partition.

    Example:
        >>> stats = PartitionStatistics(
        ...     total_partitions=4, total_work_units=12, total_pages=240,
        ...     min_pages_per_partition=55, max_pages_per_partition=65,
        ...     avg_pages_per_partition=60.0, std_pages_per_partition=3.5)
    """

    total_partitions: int
    total_work_units: int
    total_pages: int
    min_pages_per_partition: int
    max_pages_per_partition: int
    avg_pages_per_partition: float
    std_pages_per_partition: float

    def __post_init__(self) -> None:
        if self.total_partitions < 1:
            raise ValidationError(
                "PartitionStatistics.total_partitions must be >= 1.",
                field="total_partitions",
                value=self.total_partitions,
            )
        if self.min_pages_per_partition > self.max_pages_per_partition:
            raise ValidationError(
                "min_pages_per_partition must be <= max_pages_per_partition.",
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary representation of partition statistics.
        """
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"PartitionStatistics(partitions={self.total_partitions}, "
            f"total_pages={self.total_pages}, "
            f"avg_pages={self.avg_pages_per_partition:.1f})"
        )
