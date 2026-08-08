"""Sample configuration fixtures for tests.

Provides pre-built configuration objects for use in unit and integration tests
without loading actual YAML files from disk.
"""

from __future__ import annotations

from adaptive_framework.config.models import (
    ChunkerConfig,
    ConsoleLoggingConfig,
    DocumentProcessingEngineConfig,
    EmbedderConfig,
    EvaluationConfig,
    FileLoggingConfig,
    FrameworkConfig,
    LoggingConfig,
    OCRConfig,
    RAGConfig,
    RayClusterConfig,
    SchedulerConfig,
    SchedulerOverheadConfig,
    SchedulerOverheadMetricConfig,
    SharedMemoryConfig,
    VectorStoreConfig,
    WorkerConfig,
    WorkStealingConfig,
)


def make_framework_config(**overrides: object) -> FrameworkConfig:
    """Return a FrameworkConfig with sensible test defaults.

    Args:
        **overrides: Override any FrameworkConfig field.

    Returns:
        FrameworkConfig instance for testing.
    """
    defaults: dict[str, object] = {
        "name": "Test Framework",
        "version": "2.0.0",
        "run_id_prefix": "test_run",
        "output_dir": "test_outputs",
        "debug": True,
        "max_concurrent_jobs": 2,
        "stage_timeout_seconds": 60,
        "shutdown_timeout_seconds": 10,
    }
    defaults.update(overrides)
    return FrameworkConfig(**defaults)  # type: ignore[arg-type]


def make_logging_config(**overrides: object) -> LoggingConfig:
    """Return a LoggingConfig with test defaults (console only, no files).

    Args:
        **overrides: Override any LoggingConfig field.

    Returns:
        LoggingConfig for testing.
    """
    console = ConsoleLoggingConfig(
        enabled=True, level="DEBUG", use_rich=False, colorize=False
    )
    file_cfg = FileLoggingConfig(
        enabled=False, level="DEBUG", path="logs/test.log",
        max_bytes=1_000_000, backup_count=1, encoding="utf-8"
    )
    defaults: dict[str, object] = {
        "level": "DEBUG",
        "format": "text",
        "console": console,
        "file": file_cfg,
        "json_file": file_cfg,
        "context_fields": ["run_id", "worker_id"],
    }
    defaults.update(overrides)
    return LoggingConfig(**defaults)  # type: ignore[arg-type]


def make_scheduler_config(**overrides: object) -> SchedulerConfig:
    """Return a SchedulerConfig with test defaults.

    Args:
        **overrides: Override any SchedulerConfig field.

    Returns:
        SchedulerConfig for testing.
    """
    defaults: dict[str, object] = {
        "strategy": "page_count",
        "work_stealing": WorkStealingConfig(
            enabled=True, steal_threshold=2,
            steal_fraction=0.5, check_interval_seconds=1.0
        ),
        "overhead_monitoring": SchedulerOverheadConfig(
            enabled=True, target_max_overhead_fraction=0.01, warn_on_exceed=True
        ),
        "default_partition_count": 2,
        "min_pages_per_partition": 1,
    }
    defaults.update(overrides)
    return SchedulerConfig(**defaults)  # type: ignore[arg-type]
