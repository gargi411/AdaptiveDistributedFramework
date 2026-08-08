"""Core package for the Adaptive Distributed Framework.

Contains the exception hierarchy and framework-wide constants.
This package has zero dependencies on any other framework package,
making it the root of the dependency graph.
"""

from adaptive_framework.core.exceptions import (
    ClusterError,
    ConfigurationError,
    DatasetError,
    EvaluationError,
    FrameworkError,
    PipelineError,
    ProcessingError,
    SchedulerError,
    ValidationError,
)

__all__ = [
    "FrameworkError",
    "ConfigurationError",
    "DatasetError",
    "ClusterError",
    "SchedulerError",
    "PipelineError",
    "EvaluationError",
    "ValidationError",
    "ProcessingError",
]
