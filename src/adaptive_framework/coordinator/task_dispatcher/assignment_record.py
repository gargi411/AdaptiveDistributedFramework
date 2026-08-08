"""AssignmentRecord — Immutable record of a single task assignment.

Created by the TaskDispatcher whenever a PageWorkUnit is assigned to a worker.
Used for audit trail, retry tracking, and dashboard assignment history.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AssignmentStatus(str, Enum):
    """Lifecycle status of a task assignment.

    Values:
        ASSIGNED: Task has been dispatched; worker is processing it.
        COMPLETED: Worker reported successful completion.
        FAILED: Worker reported failure.
        RECOVERED: Assignment was recovered after worker loss and reassigned.
        CANCELLED: Assignment cancelled (e.g. graceful shutdown).
    """

    ASSIGNED = "assigned"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERED = "recovered"
    CANCELLED = "cancelled"


@dataclass
class AssignmentRecord:
    """Record of a single PageWorkUnit → Worker assignment.

    Attributes:
        assignment_id: Unique ID for this assignment event.
        work_unit_id: PageWorkUnit that was assigned.
        worker_id: Worker that received the assignment.
        document_id: Parent document of the work unit.
        page_count: Page count of the assigned work unit.
        assigned_at: ISO 8601 UTC timestamp of assignment.
        completed_at: ISO 8601 UTC timestamp of completion. None if pending.
        status: Current lifecycle status.
        retry_count: How many times this work unit has been reassigned.
        failure_reason: Error description if status is FAILED.

    Example:
        >>> rec = AssignmentRecord(
        ...     work_unit_id="wu_001",
        ...     worker_id="w_001",
        ...     document_id="doc_001",
        ...     page_count=42,
        ... )
    """

    work_unit_id: str
    worker_id: str
    document_id: str
    page_count: int
    assignment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    assigned_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None
    status: AssignmentStatus = AssignmentStatus.ASSIGNED
    retry_count: int = 0
    failure_reason: str | None = None

    # ------------------------------------------------------------------ #
    # State transitions                                                    #
    # ------------------------------------------------------------------ #

    def mark_completed(self) -> None:
        """Mark this assignment as COMPLETED and set completed_at timestamp."""
        self.status = AssignmentStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def mark_failed(self, reason: str) -> None:
        """Mark this assignment as FAILED.

        Args:
            reason: Human-readable failure description.
        """
        self.status = AssignmentStatus.FAILED
        self.failure_reason = reason
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def mark_recovered(self) -> None:
        """Mark this assignment as RECOVERED (work unit will be reassigned)."""
        self.status = AssignmentStatus.RECOVERED
        self.completed_at = datetime.now(timezone.utc).isoformat()

    @property
    def elapsed_seconds(self) -> float | None:
        """Compute assignment duration in seconds.

        Returns:
            Seconds from assigned_at to completed_at, or None if still pending.
        """
        if self.completed_at is None:
            return None
        try:
            from datetime import datetime as _dt
            start = _dt.fromisoformat(self.assigned_at)
            end = _dt.fromisoformat(self.completed_at)
            return (end - start).total_seconds()
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary including status value and elapsed_seconds.
        """
        d = asdict(self)
        d["status"] = self.status.value
        d["elapsed_seconds"] = self.elapsed_seconds
        return d

    def __repr__(self) -> str:
        return (
            f"AssignmentRecord(wu='{self.work_unit_id[:8]}', "
            f"worker='{self.worker_id[:8]}', "
            f"status={self.status.value}, "
            f"retry={self.retry_count})"
        )
