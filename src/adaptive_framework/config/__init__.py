"""Configuration system for the Adaptive Distributed Framework.

This package provides a singleton ConfigManager that loads, validates,
and hot-reloads YAML configuration files. All framework components receive
configuration through dependency injection — never through global state.

Components:
    config_manager: Singleton YAML loader with hot reload support.
    models: Typed dataclass models for every configuration section.

Usage Example:
    >>> from adaptive_framework.config import ConfigManager
    >>> cfg = ConfigManager.get_instance()
    >>> cfg.load(config_dir="configs/")
    >>> framework_cfg = cfg.get_framework_config()
    >>> print(framework_cfg.name)
    'Adaptive Distributed Parallel Processing Framework'
"""

from adaptive_framework.config.config_manager import ConfigManager
from adaptive_framework.config.models import (
    DocumentProcessingEngineConfig,
    EvaluationConfig,
    FrameworkConfig,
    LoggingConfig,
    RAGConfig,
    RayClusterConfig,
    SchedulerConfig,
)

__all__ = [
    "ConfigManager",
    "DocumentProcessingEngineConfig",
    "EvaluationConfig",
    "FrameworkConfig",
    "LoggingConfig",
    "RAGConfig",
    "RayClusterConfig",
    "SchedulerConfig",
]
