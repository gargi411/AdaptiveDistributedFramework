"""HeartbeatEvent — Data model for a single heartbeat occurrence.

Heartbeat events are emitted by workers and consumed by the HeartbeatMonitor
to track liveness, update resource metrics, and detect disconnections.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class HeartbeatEventType(str, Enum):
    """Classification of heartbeat events.

    Values:
        ALIVE: Normal periodic heartbeat — worker is healthy.
        TASK_UPDATE: Worker reports progress on current task.
        TASK_COMPLETED: Worker reports task finished successfully.
        TASK_FAILED: Worker reports task failure.
        TIMEOUT: Monitor detected a missed heartbeat.
        DISCONNECTED: Worker declared LOST after timeout threshold.
        RECONNECTED: Previously LOST worker sent a new heartbeat.
    """

    ALIVE = "alive"
    TASK_UPDATE = "task_update"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TIMEOUT = "timeout"
    DISCONNECTED = "disconnected"
    RECONNECTED = "reconnected"


@dataclass
class HeartbeatPayload:
    """Payload sent by a worker with every heartbeat.

    Attributes:
        worker_id: Sender worker identifier.
        cpu_percent: Current CPU utilization (%).
        ram_percent: Current RAM utilization (%).
        gpu_percent: Current GPU utilization (%). None if no GPU.
        queue_depth: Number of tasks in the worker's local queue.
        current_task_id: Active work unit ID. None if worker is idle.
        task_progress_percent: 0–100 progress of current task. None if idle.
        state: String value of current WorkerState.
        timestamp: ISO 8601 UTC timestamp.

    Example:
        >>> payload = HeartbeatPayload(
        ...     worker_id="w_001",
        ...     cpu_percent=72.5,
        ...     ram_percent=41.0,
        ...     queue_depth=3,
        ... )
    """

    worker_id: str
    cpu_percent: float
    ram_percent: float
    queue_depth: int
    state: str = "active"
    gpu_percent: float | None = None
    current_task_id: str | None = None
    task_progress_percent: float | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary representation of this payload.
        """
        return asdict(self)


@dataclass
class HeartbeatEvent:
    """A discrete heartbeat event stored in the monitor's history.

    Attributes:
        event_id: Unique identifier for this event.
        event_type: Classification (alive, timeout, disconnected, etc.).
        worker_id: Worker this event pertains to.
        payload: Full HeartbeatPayload if this is an ALIVE/TASK_UPDATE event.
        timestamp: ISO 8601 UTC timestamp of when the monitor processed this event.
        message: Human-readable description for dashboard logs.

    Example:
        >>> evt = HeartbeatEvent(
        ...     event_type=HeartbeatEventType.TIMEOUT,
        ...     worker_id="w_002",
        ...     message="Worker w_002 missed heartbeat (15s timeout).",
        ... )
    """

    event_type: HeartbeatEventType
    worker_id: str
    message: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: HeartbeatPayload | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary representation including payload if present.
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "worker_id": self.worker_id,
            "message": self.message,
            "timestamp": self.timestamp,
            "payload": self.payload.to_dict() if self.payload else None,
        }

    def __repr__(self) -> str:
        return (
            f"HeartbeatEvent(type={self.event_type.value}, "
            f"worker_id='{self.worker_id}', "
            f"ts='{self.timestamp}')"
        )
