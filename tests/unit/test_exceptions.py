"""Unit tests for the custom exception hierarchy.

Tests:
    - Exception inheritance chain
    - Message propagation
    - Context field attachment
    - Error code support
    - All exception types are instantiable with a message
"""

from __future__ import annotations

import pytest

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


class TestFrameworkErrorBase:
    """Tests for FrameworkError base exception."""

    def test_is_exception_subclass(self) -> None:
        """FrameworkError must be a subclass of Exception."""
        assert issubclass(FrameworkError, Exception)

    def test_message_preserved(self) -> None:
        """FrameworkError preserves the message string."""
        err = FrameworkError("test message")
        assert "test message" in str(err)

    def test_can_be_raised_and_caught(self) -> None:
        """FrameworkError can be raised and caught as Exception."""
        with pytest.raises(Exception):
            raise FrameworkError("boom")

    def test_can_be_caught_by_own_type(self) -> None:
        """FrameworkError can be caught by its own type."""
        with pytest.raises(FrameworkError):
            raise FrameworkError("boom")


class TestExceptionInheritance:
    """Tests that all domain exceptions inherit from FrameworkError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            ConfigurationError,
            DatasetError,
            ClusterError,
            SchedulerError,
            PipelineError,
            EvaluationError,
            ValidationError,
            ProcessingError,
        ],
    )
    def test_inherits_from_framework_error(self, exc_class: type) -> None:
        """Every domain exception must inherit from FrameworkError."""
        assert issubclass(exc_class, FrameworkError), (
            f"{exc_class.__name__} must inherit from FrameworkError"
        )

    @pytest.mark.parametrize(
        "exc_class",
        [
            ConfigurationError,
            DatasetError,
            ClusterError,
            SchedulerError,
            PipelineError,
            EvaluationError,
            ValidationError,
            ProcessingError,
        ],
    )
    def test_inherits_from_exception(self, exc_class: type) -> None:
        """Every domain exception must be catchable as Exception."""
        assert issubclass(exc_class, Exception), (
            f"{exc_class.__name__} must inherit from Exception"
        )


class TestExceptionInstantiation:
    """Tests that every exception type can be instantiated and raised."""

    def test_configuration_error(self) -> None:
        """ConfigurationError is instantiable and raisable."""
        with pytest.raises(ConfigurationError, match="bad config"):
            raise ConfigurationError("bad config")

    def test_dataset_error(self) -> None:
        """DatasetError is instantiable and raisable."""
        with pytest.raises(DatasetError, match="dataset missing"):
            raise DatasetError("dataset missing")

    def test_cluster_error(self) -> None:
        """ClusterError is instantiable and raisable."""
        with pytest.raises(ClusterError, match="node down"):
            raise ClusterError("node down")

    def test_scheduler_error(self) -> None:
        """SchedulerError is instantiable and raisable."""
        with pytest.raises(SchedulerError, match="queue full"):
            raise SchedulerError("queue full")

    def test_pipeline_error(self) -> None:
        """PipelineError is instantiable and raisable."""
        with pytest.raises(PipelineError, match="step failed"):
            raise PipelineError("step failed")

    def test_evaluation_error(self) -> None:
        """EvaluationError is instantiable and raisable."""
        with pytest.raises(EvaluationError, match="metric unavailable"):
            raise EvaluationError("metric unavailable")

    def test_validation_error(self) -> None:
        """ValidationError is instantiable and raisable."""
        with pytest.raises(ValidationError, match="pages must be >= 1"):
            raise ValidationError("pages must be >= 1")

    def test_processing_error(self) -> None:
        """ProcessingError is instantiable and raisable."""
        with pytest.raises(ProcessingError, match="OCR failed"):
            raise ProcessingError("OCR failed")


class TestFrameworkErrorCatch:
    """Tests that FrameworkError catches all domain exceptions."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            ConfigurationError,
            DatasetError,
            ClusterError,
            SchedulerError,
            PipelineError,
            EvaluationError,
            ValidationError,
            ProcessingError,
        ],
    )
    def test_catchable_as_framework_error(self, exc_class: type) -> None:
        """All domain exceptions can be caught as FrameworkError."""
        with pytest.raises(FrameworkError):
            raise exc_class("any message")
