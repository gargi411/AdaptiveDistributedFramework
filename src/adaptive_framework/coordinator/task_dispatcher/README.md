# coordinator/task_dispatcher

## Purpose

Placeholder package for the **Task Dispatcher** sub-system of the Distributed
Coordinator (Architecture v2.0, §2.3).

The Task Dispatcher dequeues `PageWorkUnit` items from the priority queue and
assigns them to available workers via Ray remote calls.

## Architecture Role

```
Task Queue (Priority by Page Count)
        ↓
   Scheduler
  /    |    \
 W1    W2    W3

The Task Dispatcher bridges the Scheduler and the Workers.
```

## Implementation Target

**Phase 4** — Distributed Coordinator

## Contents (Phase 4)

| Module | Description |
|--------|-------------|
| `task_dispatcher.py` | Implements dispatch loop; dequeues, assigns, and tracks work units |

## Dependencies (Phase 4)

- `adaptive_framework.interfaces.i_scheduler` (IScheduler)
- `adaptive_framework.interfaces.i_worker` (IWorker)
- `adaptive_framework.interfaces.i_result_collector` (IResultCollector)
- `adaptive_framework.models.scheduling` (PageWorkUnit)
- `adaptive_framework.core.constants`
- Ray actor model (external)
