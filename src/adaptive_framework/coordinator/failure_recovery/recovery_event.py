"""RecoveryEvent — Data model for a single failure recovery event.

Recovery events record the full lifecycle of a worker failure:
from detection → task recovery → reassignment → completion.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class RecoveryEventType(str, Enum):
    """Classification of recovery events.

    Values:
        WORKER_LOST: Worker declared LOST by HeartbeatMonitor.
        TASKS_RECOVERED: In-flight tasks returned to the priority queue.
        WORKER_RECOVERED: Previously LOST worker reconnected.
        RETRY_SCHEDULED: A failed task has been scheduled for retry.
        RETRY_EXHAUSTED: Task exceeded max_retries — permanent failure.
        GRACEFUL_SHUTDOWN: Worker gracefully shut down (not a failure).
    """

    WORKER_LOST = "worker_lost"
    TASKS_RECOVERED = "tasks_recovered"
    WORKER_RECOVERED = "worker_recovered"
    RETRY_SCHEDULED = "retry_scheduled"
    RETRY_EXHAUSTED = "retry_exhausted"
    GRACEFUL_SHUTDOWN = "graceful_shutdown"


@dataclass
class RecoveryEvent:
    """A discrete failure/recovery event stored in the recovery engine history.

    Attributes:
        event_id: Unique identifier.
        event_type: Classification of this recovery event.
        worker_id: Worker involved in this event.
        work_unit_ids: IDs of PageWorkUnits affected (may be empty).
        retry_count: Current retry number (for RETRY_SCHEDULED events).
        message: Human-readable description for dashboard and logs.
        timestamp: ISO 8601 UTC timestamp.

    Example:
        >>> evt = RecoveryEvent(
        ...     event_type=RecoveryEventType.WORKER_LOST,
        ...     worker_id="w_002",
        ...     work_unit_ids=["wu_001", "wu_002"],
        ...     message="Worker w_002 declared LOST; 2 tasks recovered.",
        ... )
    """

    event_type: RecoveryEventType
    worker_id: str
    message: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    work_unit_ids: list[str] = field(default_factory=list)
    retry_count: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary representation of this event.
        """
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    def __repr__(self) -> str:
        return (
            f"RecoveryEvent(type={self.event_type.value}, "
            f"worker='{self.worker_id[:8]}', "
            f"tasks={len(self.work_unit_ids)}, "
            f"ts='{self.timestamp}')"
        )
