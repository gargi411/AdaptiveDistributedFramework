"""Distributed Coordinator package — Phase 2B Implementation.

Architecture v2.0 §2.3: Distributed Coordinator (Ray-based)

    DistributedCoordinator
    ├── ClusterManager         — Ray cluster lifecycle (start/stop/nodes)
    ├── WorkerRegistry         — Thread-safe worker registration & discovery
    ├── HeartbeatMonitor       — Worker liveness detection & timeout handling
    ├── TaskDispatcher         — Work unit assignment & completion tracking
    ├── FailureRecoveryEngine  — §2.3 task recovery flow
    ├── WorkStealingCoordinator — Research Algorithm #2
    └── ResourceOrchestrator   — Adaptive Runtime Resource Orchestration
"""

from adaptive_framework.coordinator.cluster_manager import ClusterManager
from adaptive_framework.coordinator.node_info import NodeInfo
from adaptive_framework.coordinator.distributed_coordinator import DistributedCoordinator
from adaptive_framework.coordinator.worker_registry import WorkerRecord, WorkerRegistry
from adaptive_framework.coordinator.heartbeat_monitor import (
    HeartbeatEvent,
    HeartbeatEventType,
    HeartbeatMonitor,
    HeartbeatPayload,
)
from adaptive_framework.coordinator.task_dispatcher import (
    AssignmentRecord,
    AssignmentStatus,
    TaskDispatcher,
)
from adaptive_framework.coordinator.failure_recovery import (
    FailureRecoveryEngine,
    RecoveryEvent,
    RecoveryEventType,
)
from adaptive_framework.coordinator.resource_orchestrator import ResourceOrchestrator

__all__ = [
    "ClusterManager",
    "NodeInfo",
    "DistributedCoordinator",
    "WorkerRecord",
    "WorkerRegistry",
    "HeartbeatEvent",
    "HeartbeatEventType",
    "HeartbeatMonitor",
    "HeartbeatPayload",
    "AssignmentRecord",
    "AssignmentStatus",
    "TaskDispatcher",
    "FailureRecoveryEngine",
    "RecoveryEvent",
    "RecoveryEventType",
    "ResourceOrchestrator",
]