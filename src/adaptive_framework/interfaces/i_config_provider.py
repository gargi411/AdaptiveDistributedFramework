"""IConfigProvider — Abstract configuration provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from adaptive_framework.config.models import (
    DocumentProcessingEngineConfig,
    EvaluationConfig,
    FrameworkConfig,
    LoggingConfig,
    RAGConfig,
    RayClusterConfig,
    SchedulerConfig,
)


class IConfigProvider(ABC):
    """Abstract interface for configuration providers.

    Allows the ConfigManager (concrete YAML implementation) to be
    replaced in tests with an in-memory provider, or in future phases
    with a remote config store (e.g., etcd, Consul), without changing
    any component that depends on this interface.

    Example:
        >>> class InMemoryConfigProvider(IConfigProvider):
        ...     def load(self, config_dir: Path) -> None: ...
        ...     def get_framework_config(self) -> FrameworkConfig: ...
    """

    @abstractmethod
    def load(self, config_dir: Path | str) -> None:
        """Load configuration from the given directory.

        Args:
            config_dir: Path to the configuration directory.
        """

    @abstractmethod
    def reload(self) -> None:
        """Hot-reload configuration without restarting the process."""

    @abstractmethod
    def get_framework_config(self) -> FrameworkConfig:
        """Return the typed FrameworkConfig.

        Returns:
            Validated FrameworkConfig instance.
        """

    @abstractmethod
    def get_logging_config(self) -> LoggingConfig:
        """Return the typed LoggingConfig.

        Returns:
            Validated LoggingConfig instance.
        """

    @abstractmethod
    def get_ray_cluster_config(self) -> RayClusterConfig:
        """Return the typed RayClusterConfig.

        Returns:
            Validated RayClusterConfig instance.
        """

    @abstractmethod
    def get_scheduler_config(self) -> SchedulerConfig:
        """Return the typed SchedulerConfig.

        Returns:
            Validated SchedulerConfig instance.
        """

    @abstractmethod
    def get_document_processing_engine_config(self) -> DocumentProcessingEngineConfig:
        """Return the typed DocumentProcessingEngineConfig.

        Returns:
            Validated DocumentProcessingEngineConfig instance.
        """

    @abstractmethod
    def get_evaluation_config(self) -> EvaluationConfig:
        """Return the typed EvaluationConfig.

        Returns:
            Validated EvaluationConfig instance.
        """

    @abstractmethod
    def get_rag_config(self) -> RAGConfig:
        """Return the typed RAGConfig.

        Returns:
            Validated RAGConfig instance.
        """

    @abstractmethod
    def get_raw(self) -> dict[str, Any]:
        """Return the raw merged configuration dictionary.

        Returns:
            Merged dictionary of all loaded configuration data.
        """
