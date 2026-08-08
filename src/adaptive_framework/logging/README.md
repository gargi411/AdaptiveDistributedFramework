# Logging Package

## Purpose

Centralized logging system for the Adaptive Distributed Framework.

## Components

| File | Responsibility |
|------|---------------|
| `framework_logger.py` | `FrameworkLogger` — implements `ILogger`; configures 3 handlers |
| `formatters.py` | `JsonFormatter` — JSONL structured log format |
| `handlers.py` | `ContextInjectingHandler` — injects run_id, worker_id, node_id |

## Features

- ✅ **Console logging** — human-readable text format
- ✅ **Rotating text file** — `logs/framework.log`, 10 MB × 5 backups
- ✅ **Rotating JSON file** — `logs/framework.jsonl`, structured JSONL
- ✅ **Context fields** — run_id, worker_id, node_id, thread_id, process_id
- ✅ **bind()** — child logger with merged context (per-worker, per-module)
- ✅ **get_child()** — Python child logger for sub-modules

## Usage

```python
from adaptive_framework.config import ConfigManager
from adaptive_framework.logging import FrameworkLogger
from pathlib import Path

cfg = ConfigManager.get_instance()
cfg.load("configs/")
log_cfg = cfg.get_logging_config()

# Create root logger
logger = FrameworkLogger.from_config(log_cfg, output_dir=Path("outputs"),
                                     run_id="adf_run_001")
logger.info("Framework started.")

# Worker-specific child logger
worker_logger = logger.bind(worker_id="worker_01", node_id="laptop_02")
worker_logger.info("Worker initialized.")

# Sub-module logger
config_logger = logger.get_child("config")
config_logger.debug("Config loaded.")
```

## Dependency Direction

```
logging  →  interfaces (ILogger)
logging  →  core (constants)
logging  →  formatters
logging  →  handlers
```
