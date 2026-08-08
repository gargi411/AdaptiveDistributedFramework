# Models Package

## Purpose

Pure Python **dataclass data models** for the Adaptive Distributed Framework.

No algorithms. No I/O. No business logic. Only structured data definitions.

## Modules

| Module | Models |
|--------|--------|
| `document.py` | `PDFMetadata`, `PageMetadata`, `PageResult`, `DocumentResult` |
| `scheduling.py` | `PageWorkUnit`, `Partition`, `PartitionStatistics` |
| `runtime.py` | `ResourceSnapshot`, `RuntimeMetrics`, `WorkerStatus`, `ClusterStatus`, `FrameworkStatus` |
| `evaluation.py` | `EvaluationResult` |

## Design Principles

- **Pure dataclasses**: No methods with side effects. No I/O.
- **Validation in `__post_init__`**: Invalid data raises `ValidationError` immediately.
- **Serialization via `to_dict()`**: Every model can be converted to a JSON-serializable dict.
- **Enums for state**: `WorkUnitStatus`, `WorkerState`, `FrameworkState` prevent magic string bugs.
- **Computed properties**: `success_rate`, `passes_overhead_target`, etc., are derived — never stored redundantly.

## Key Model: PDFMetadata (Architecture §2.4)

```python
meta = PDFMetadata(
    pages=42,
    estimated_size_mb=3.7,
    file_path="/data/raw/paper.pdf",
    resolution_dpi=300,       # Optional
    source_type="scanned",    # Optional: "scanned" | "digital"
    language="en",            # Optional: ISO 639-1
)
```

## Key Model: EvaluationResult (Architecture §4.1)

Records all 6 metrics:
1. Speedup
2. Throughput (pages/second)
3. CPU Utilization (%)
4. GPU Utilization (%)
5. Energy (Joules)
6. Scheduler Overhead (%) — must be < 1%

## Dependency Direction

```
models  →  core (exceptions, constants)
```

No imports from config, logging, interfaces, or any higher package.
