# coordinator/failure_recovery

## Purpose

Placeholder package for the **Failure Recovery** sub-system of the Distributed
Coordinator (Architecture v2.0, §2.3).

Failure Recovery detects worker loss via heartbeat timeout and re-inserts
unfinished work units back into the priority queue for reassignment.

## Failure Recovery Flow (from architecture_v2.0_locked.md)

```
Worker Lost (e.g. Laptop 2 disconnects)
        ↓
Detect via Heartbeat Timeout
        ↓
Return Unfinished Work Units
        ↓
Re-insert into Priority Queue
        ↓
Assign to Available Worker
```

## Implementation Target

**Phase 4** — Distributed Coordinator

## Contents (Phase 4)

| Module | Description |
|--------|-------------|
| `recovery_manager.py` | Implements `IFailureRecovery`; handles timeout detection and task re-queue |

## Dependencies (Phase 4)

- `adaptive_framework.interfaces.i_scheduler` (IScheduler)
- `adaptive_framework.interfaces.i_worker` (IWorker)
- `adaptive_framework.models.scheduling` (PageWorkUnit)
- `adaptive_framework.core.constants` (DEFAULT_TASK_RETRY_ATTEMPTS)
- Ray actor model (external)
