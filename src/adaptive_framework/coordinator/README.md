# Distributed Coordinator Package

## Architecture (v2.0 §2.3)

```
Coordinator
├── Worker Registry       ← worker_registry/
├── Task Dispatcher       ← task_dispatcher/
├── Heartbeat Monitor     ← heartbeat_monitor/
└── Failure Recovery      ← failure_recovery/
```

## Status: Phase 4 Placeholder

## Transport

Ray actor model. Coordinator is a long-lived Ray actor.
Workers register themselves with the Worker Registry on startup.

## Failure Recovery Flow

```
Worker Lost (e.g. Laptop 2 disconnects)
        ↓
Detect via Heartbeat Timeout (configurable, default: 30s)
        ↓
Return Unfinished Work Units
        ↓
Re-insert into Priority Queue
        ↓
Assign to Available Worker
```

Retry attempts: configurable (default: 3).
Retry delay: configurable (default: 2s).