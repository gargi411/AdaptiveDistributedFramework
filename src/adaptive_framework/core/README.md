# Core Package

## Purpose

Foundation package for the Adaptive Distributed Framework. Contains the **exception hierarchy** and **framework-wide constants**.

This package is the **root of the dependency graph** — it imports nothing from within the framework.

## Modules

| Module | Contents |
|--------|----------|
| `exceptions.py` | Full 9-class exception hierarchy + 3 specialized sub-classes |
| `constants.py` | All framework-wide constants (no magic numbers) |

## Exception Hierarchy

```
Exception
└── FrameworkError
    ├── ConfigurationError
    ├── DatasetError
    ├── ClusterError
    │   └── WorkerLostError       ← triggers Failure Recovery
    ├── SchedulerError
    │   └── SchedulerOverheadError ← overhead > 1% target
    ├── PipelineError
    ├── EvaluationError
    ├── ValidationError
    └── ProcessingError
        └── OCRError
```

## Dependency Direction

```
core  →  (nothing — root package)
```

Every other package may import from `core`. `core` never imports from other framework packages.
