"""Worker Registry sub-package.

Provides:
    WorkerRecord: Persistent registration record for a cluster worker.
    WorkerRegistry: Thread-safe registry of all cluster workers.
"""

from adaptive_framework.coordinator.worker_registry.worker_record import WorkerRecord
from adaptive_framework.coordinator.worker_registry.registry import WorkerRegistry

__all__ = ["WorkerRecord", "WorkerRegistry"]