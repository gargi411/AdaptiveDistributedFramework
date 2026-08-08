"""Failure Recovery sub-package.

Provides:
    RecoveryEvent: Discrete failure/recovery event record.
    RecoveryEventType: Classification enum for recovery events.
    FailureRecoveryEngine: Implements architecture §2.3 failure recovery flow.
"""

from adaptive_framework.coordinator.failure_recovery.recovery_event import (
    RecoveryEvent,
    RecoveryEventType,
)
from adaptive_framework.coordinator.failure_recovery.recovery_engine import (
    FailureRecoveryEngine,
)

__all__ = ["RecoveryEvent", "RecoveryEventType", "FailureRecoveryEngine"]