# Config Package

## Purpose

This package implements the **centralized configuration system** for the Adaptive Distributed Framework.

## Components

| File | Responsibility |
|------|---------------|
| `config_manager.py` | Singleton YAML loader with hot-reload support |
| `models.py` | Typed dataclass models for all 7 YAML configuration files |

## Design Decisions

- **Singleton**: Only one ConfigManager exists per process. Thread-safe creation.
- **Typed models**: Every section of every YAML file maps to a validated `@dataclass`.
- **Validation in `__post_init__`**: Invalid values raise `ConfigurationError` immediately at load time — never silently at runtime.
- **No global state in business logic**: Scheduler, OCR engine, and all other components receive config via constructor injection.
- **Hot reload**: Call `ConfigManager.get_instance().reload()` to pick up YAML changes without restarting.

## Configuration Files

| YAML File | Config Model |
|-----------|-------------|
| `configs/framework.yaml` | `FrameworkConfig` |
| `configs/logging.yaml` | `LoggingConfig` |
| `configs/ray_cluster.yaml` | `RayClusterConfig` |
| `configs/scheduler.yaml` | `SchedulerConfig` |
| `configs/ocr.yaml` | `DocumentProcessingEngineConfig` |
| `configs/evaluation.yaml` | `EvaluationConfig` |
| `configs/rag.yaml` | `RAGConfig` |

## Usage

```python
from adaptive_framework.config import ConfigManager

cfg = ConfigManager.get_instance()
cfg.load(config_dir="configs/")

framework_cfg = cfg.get_framework_config()
logging_cfg   = cfg.get_logging_config()
ray_cfg       = cfg.get_ray_cluster_config()
scheduler_cfg = cfg.get_scheduler_config()
dpe_cfg       = cfg.get_document_processing_engine_config()
eval_cfg      = cfg.get_evaluation_config()
rag_cfg       = cfg.get_rag_config()
```

## Dependency Direction

```
config_manager  →  models
config_manager  →  exceptions (core)
config_manager  →  yaml_utils (utils)
```

No higher-level package (scheduler, coordinator, OCR) is imported here.
