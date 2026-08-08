# Utils & DI Packages

## Utils Package

Pure utility functions. No business logic. No framework dependencies beyond `core`.

| Module | Contents |
|--------|----------|
| `file_utils.py` | PDF discovery, file size, MD5, read/write text |
| `yaml_utils.py` | Safe YAML load/dump/parse/merge |
| `path_utils.py` | Project root, configs/output/logs dir resolution |
| `system_utils.py` | CPU%, memory%, CPU count, hostname, platform info |
| `time_utils.py` | Timestamps, perf timers, `timer()` context manager, overhead fraction |
| `validation_utils.py` | validate_positive_int, validate_pdf_path, validate_page_count, etc. |

### Key: `time_utils.compute_overhead_fraction`

Implements the architecture §4.2 formula directly:

```python
fraction = compute_overhead_fraction(
    scheduler_time_seconds=0.8,
    total_time_seconds=120.0,
)
# → 0.00667 (0.667%)
```

---

## DI Package

Lightweight typed dependency injection container.

### Pattern

```python
# main.py (composition root only)
from adaptive_framework.di import DIContainer
from adaptive_framework.interfaces import ILogger, IConfigProvider

container = DIContainer()
container.register(ILogger, framework_logger)
container.register(IConfigProvider, config_manager)

# Inject into components via constructors
scheduler = AdaptiveScheduler(
    logger=container.resolve(ILogger),
    config=container.resolve(IConfigProvider).get_scheduler_config(),
    strategy=container.resolve(IPartitionStrategy),
)
```

### Rules

- ❌ Business logic (scheduler, OCR, RAG) **never** calls `container.resolve()` internally.
- ✅ Only `main.py` and factory functions wire dependencies.
- ✅ Tests override registrations with mocks via `container.register(ILogger, MockLogger())`.
