# coordinator/heartbeat_monitor

## Purpose

Placeholder package for the **Heartbeat Monitor** sub-system of the Distributed
Coordinator (Architecture v2.0, §2.3).

The Heartbeat Monitor periodically receives liveness signals from registered
workers. When a worker misses the heartbeat deadline, it triggers the Failure
Recovery sub-system.

## Architecture Role

```
Coordinator
├── Worker Registry
├── Task Dispatcher
├── Heartbeat Monitor     ← This package
└── Failure Recovery
```

## Implementation Target

**Phase 4** — Distributed Coordinator

## Contents (Phase 4)

| Module | Description |
|--------|-------------|
| `heartbeat_monitor.py` | Ray actor that tracks worker liveness and calls FailureRecovery on timeout |

## Configuration

Timeout parameters will be read from `configs/ray_cluster.yaml`:
- `heartbeat_interval_seconds` — how often workers send a ping
- `heartbeat_timeout_multiplier` — timeout = interval × multiplier

## Dependencies (Phase 4)

- `adaptive_framework.interfaces.i_worker` (IWorker)
- `adaptive_framework.models.runtime` (WorkerStatus)
- `adaptive_framework.core.constants`
- Ray actor model (external)
