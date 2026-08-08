# Adaptive Scheduler Package

## Architecture (v2.0 §2.2)

```
Task Queue (Priority by Page Count)
        ↓
   Scheduler
  /    |    \
W1    W2    W3
```

## Status: Phase 3 Placeholder

## Key Algorithms (Phase 3)

1. **Page-Count Priority Queue** — documents sorted by descending page count
2. **Work Stealing** — idle workers steal from overloaded peers (threshold configurable)
3. **Zero-Copy** — shared memory buffers (Ray Plasma store)
4. **Overhead Monitoring** — instruments `time.perf_counter()` around dispatch loop

## Interface

- `interfaces/i_scheduler.py` → `IScheduler`
- `interfaces/i_partition_strategy.py` → `IPartitionStrategy`

## Overhead Target

Architecture §4.2: `Scheduler Overhead < 1%`
Formula: `scheduler_time / total_execution_time × 100`