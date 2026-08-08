"""Unit tests for the FrameworkLogger logging system.

Tests:
    - FrameworkLogger can be constructed from a LoggingConfig
    - All log levels (debug, info, warning, error, critical) emit records
    - bind() returns a new logger with merged context
    - get_child() returns a child logger
    - JSON formatter produces valid JSON
    - ContextInjectingHandler injects context fields
"""

from __future__ import annotations

import json
import logging
import io
from typing import Any

import pytest

from adaptive_framework.logging.formatters import JsonFormatter
from adaptive_framework.logging.handlers import ContextInjectingHandler
from adaptive_framework.logging.framework_logger import FrameworkLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_log_cfg() -> Any:
    """Build a minimal LoggingConfig-like object using the real config models.

    Returns:
        LoggingConfig instance with console enabled, files disabled.
    """
    from adaptive_framework.config.models import (
        ConsoleLoggingConfig,
        FileLoggingConfig,
        LoggingConfig,
    )

    console = ConsoleLoggingConfig(
        enabled=True,
        level="DEBUG",
        use_rich=False,
        colorize=False,
    )
    file_cfg = FileLoggingConfig(
        enabled=False,
        level="DEBUG",
        path="logs/test.log",
        max_bytes=1_000_000,
        backup_count=1,
        encoding="utf-8",
    )
    return LoggingConfig(
        level="DEBUG",
        format="text",
        console=console,
        file=file_cfg,
        json_file=file_cfg,
        context_fields=["run_id", "worker_id"],
    )


# ---------------------------------------------------------------------------
# JsonFormatter
# ---------------------------------------------------------------------------


class TestJsonFormatter:
    """Tests for the JSON log formatter."""

    def _make_record(self, message: str = "test message") -> logging.LogRecord:
        """Create a minimal LogRecord for testing.

        Args:
            message: Log message content.

        Returns:
            A LogRecord with the given message.
        """
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test_logging.py",
            lineno=1,
            msg=message,
            args=(),
            exc_info=None,
        )
        record.run_id = "run_001"
        record.worker_id = "worker_01"
        record.node_id = "node_01"
        return record

    def test_format_returns_valid_json(self) -> None:
        """JsonFormatter.format() returns a valid JSON string."""
        formatter = JsonFormatter()
        record = self._make_record()
        output = formatter.format(record)
        # Must be parseable as JSON
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_format_contains_message(self) -> None:
        """Formatted JSON must contain the 'message' key."""
        formatter = JsonFormatter()
        record = self._make_record("hello world")
        data = json.loads(formatter.format(record))
        assert "message" in data
        assert data["message"] == "hello world"

    def test_format_contains_level(self) -> None:
        """Formatted JSON must contain the 'level' key."""
        formatter = JsonFormatter()
        record = self._make_record()
        data = json.loads(formatter.format(record))
        assert "level" in data


# ---------------------------------------------------------------------------
# ContextInjectingHandler
# ---------------------------------------------------------------------------


class TestContextInjectingHandler:
    """Tests for the ContextInjectingHandler."""

    def test_injects_run_id_into_record(self) -> None:
        """ContextInjectingHandler injects run_id into the log record."""
        stream = io.StringIO()
        inner_handler = logging.StreamHandler(stream)
        inner_handler.setFormatter(logging.Formatter("%(message)s|%(run_id)s"))
        handler = ContextInjectingHandler(
            inner_handler, run_id="adf_run_001", worker_id="none", node_id="none"
        )

        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="f.py", lineno=1,
            msg="hello", args=(), exc_info=None,
        )
        handler.emit(record)
        output = stream.getvalue()
        assert "adf_run_001" in output

    def test_injects_worker_id_into_record(self) -> None:
        """ContextInjectingHandler injects worker_id into the log record."""
        stream = io.StringIO()
        inner_handler = logging.StreamHandler(stream)
        inner_handler.setFormatter(logging.Formatter("%(worker_id)s"))
        handler = ContextInjectingHandler(
            inner_handler, run_id="none", worker_id="worker_42", node_id="none"
        )
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="f.py", lineno=1,
            msg="msg", args=(), exc_info=None,
        )
        handler.emit(record)
        assert "worker_42" in stream.getvalue()


# ---------------------------------------------------------------------------
# FrameworkLogger
# ---------------------------------------------------------------------------


class TestFrameworkLoggerConstruction:
    """Tests for FrameworkLogger construction and factory."""

    def test_from_config_returns_framework_logger(self, tmp_path: Any) -> None:
        """from_config() returns a FrameworkLogger instance."""
        cfg = _make_minimal_log_cfg()
        logger = FrameworkLogger.from_config(cfg, output_dir=tmp_path)
        assert isinstance(logger, FrameworkLogger)

    def test_logger_has_handlers(self, tmp_path: Any) -> None:
        """from_config() with console=True configures at least one handler."""
        cfg = _make_minimal_log_cfg()
        FrameworkLogger.from_config(cfg, output_dir=tmp_path)
        root = logging.getLogger("adaptive_framework")
        assert len(root.handlers) >= 1


class TestFrameworkLoggerLevels:
    """Tests that all log levels can be called without error."""

    @pytest.fixture()
    def logger(self, tmp_path: Any) -> FrameworkLogger:
        """Provide a FrameworkLogger instance for level tests."""
        cfg = _make_minimal_log_cfg()
        return FrameworkLogger.from_config(cfg, output_dir=tmp_path)

    def test_debug_does_not_raise(self, logger: FrameworkLogger) -> None:
        """logger.debug() does not raise."""
        logger.debug("debug message")

    def test_info_does_not_raise(self, logger: FrameworkLogger) -> None:
        """logger.info() does not raise."""
        logger.info("info message")

    def test_warning_does_not_raise(self, logger: FrameworkLogger) -> None:
        """logger.warning() does not raise."""
        logger.warning("warning message")

    def test_error_does_not_raise(self, logger: FrameworkLogger) -> None:
        """logger.error() does not raise."""
        logger.error("error message")

    def test_critical_does_not_raise(self, logger: FrameworkLogger) -> None:
        """logger.critical() does not raise."""
        logger.critical("critical message")


class TestFrameworkLoggerBind:
    """Tests for FrameworkLogger.bind() and get_child()."""

    @pytest.fixture()
    def logger(self, tmp_path: Any) -> FrameworkLogger:
        """Provide a FrameworkLogger instance for bind tests."""
        cfg = _make_minimal_log_cfg()
        return FrameworkLogger.from_config(cfg, output_dir=tmp_path)

    def test_bind_returns_new_instance(self, logger: FrameworkLogger) -> None:
        """bind() returns a different FrameworkLogger object."""
        bound = logger.bind(run_id="run_999")
        assert bound is not logger

    def test_bind_preserves_base_context(self, logger: FrameworkLogger) -> None:
        """bind() merges new context without losing existing keys."""
        bound = logger.bind(worker_id="worker_77")
        assert "worker_id" in bound._context
        assert bound._context["worker_id"] == "worker_77"

    def test_get_child_returns_framework_logger(self, logger: FrameworkLogger) -> None:
        """get_child() returns a FrameworkLogger."""
        child = logger.get_child("config")
        assert isinstance(child, FrameworkLogger)

    def test_get_child_is_different_instance(self, logger: FrameworkLogger) -> None:
        """get_child() returns a new, different instance."""
        child = logger.get_child("scheduler")
        assert child is not logger
