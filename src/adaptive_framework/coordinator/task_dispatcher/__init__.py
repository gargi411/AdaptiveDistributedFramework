"""Task Dispatcher sub-package.

Provides:
    AssignmentRecord: Immutable audit record for a task assignment.
    AssignmentStatus: Lifecycle status enum for assignments.
    TaskDispatcher: Assigns work units to workers with overhead instrumentation.
"""

from adaptive_framework.coordinator.task_dispatcher.assignment_record import (
    AssignmentRecord,
    AssignmentStatus,
)
from adaptive_framework.coordinator.task_dispatcher.dispatcher import TaskDispatcher

__all__ = ["AssignmentRecord", "AssignmentStatus", "TaskDispatcher"]