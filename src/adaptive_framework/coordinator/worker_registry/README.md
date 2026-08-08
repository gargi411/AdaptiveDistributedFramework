# coordinator/worker_registry

## Purpose

Placeholder package for the **Worker Registry** sub-system of the Distributed
Coordinator (Architecture v2.0, §2.3).

The Worker Registry maintains the set of active Ray worker actors, their
capabilities (CPU count, memory, GPU), and their current workload status.

## Architecture Role

```
Coordinator
├── Worker Registry     ← This package
├── Task Dispatcher
├── Heartbeat Monitor
└── Failure Recovery
```

## Implementation Target

**Phase 4** — Distributed Coordinator

## Contents (Phase 4)

| Module | Description |
|--------|-------------|
| `worker_registry.py` | Maintains registered workers; supports add/remove/query operations |

## Dependencies (Phase 4)

- `adaptive_framework.interfaces.i_worker` (IWorker)
- `adaptive_framework.models.runtime` (WorkerStatus, ClusterStatus)
- `adaptive_framework.core.constants`
- Ray actor model (external)

## Notes

- Worker registration happens at cluster start-up via Ray actor discovery.
- The registry is the single source of truth for which workers are available.
- All queries from the Task Dispatcher and Heartbeat Monitor go through the registry.
