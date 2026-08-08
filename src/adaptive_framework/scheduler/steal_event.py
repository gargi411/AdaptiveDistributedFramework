"""StealEvent — Data model for a single work-stealing occurrence.

Records the complete context of each steal: source, destination, tasks stolen,
and timestamp. Used by the dashboard's Work Stealing panel and evaluation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class StealEvent:
    """Record of a single work-stealing event.

    Attributes:
        event_id: Unique identifier.
        source_worker_id: Worker whose tasks were stolen (overloaded).
        destination_worker_id: Worker that received the stolen tasks (idle).
        stolen_work_unit_ids: IDs of PageWorkUnits that were transferred.
        source_queue_depth_before: Queue depth of source before steal.
        source_queue_depth_after: Queue depth of source after steal.
        destination_queue_depth_before: Queue depth of destination before steal.
        destination_queue_depth_after: Queue depth of destination after steal.
        timestamp: ISO 8601 UTC timestamp of the steal event.

    Example:
        >>> evt = StealEvent(
        ...     source_worker_id="w_002",
        ...     destination_worker_id="w_001",
        ...     stolen_work_unit_ids=["wu_010", "wu_011"],
        ...     source_queue_depth_before=5,
        ...     source_queue_depth_after=3,
        ...     destination_queue_depth_before=0,
        ...     destination_queue_depth_after=2,
        ... )
    """

    source_worker_id: str
    destination_worker_id: str
    stolen_work_unit_ids: list[str]
    source_queue_depth_before: int
    source_queue_depth_after: int
    destination_queue_depth_before: int
    destination_queue_depth_after: int
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def tasks_stolen(self) -> int:
        """Return the number of tasks transferred.

        Returns:
            Count of stolen work unit IDs.
        """
        return len(self.stolen_work_unit_ids)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary representation of this steal event.
        """
        d = asdict(self)
        d["tasks_stolen"] = self.tasks_stolen
        return d

    def __repr__(self) -> str:
        return (
            f"StealEvent("
            f"from='{self.source_worker_id[:8]}', "
            f"to='{self.destination_worker_id[:8]}', "
            f"count={self.tasks_stolen}, "
            f"ts='{self.timestamp}')"
        )
