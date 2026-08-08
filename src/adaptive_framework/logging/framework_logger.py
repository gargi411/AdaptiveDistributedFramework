"""FrameworkLogger — Centralized logging system for the Adaptive Distributed Framework.

Implements ILogger with:
    - Console logging (with Rich formatting)
    - Rotating text file logging
    - Rotating JSON (JSONL) file logging
    - Context fields: run_id, worker_id, node_id, module_name,
      timestamp, thread_id, process_id
    - bind() for per-component child loggers with bound context
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Any

from adaptive_framework.core.constants import (
    DEFAULT_ENCODING,
    LOG_TIMESTAMP_FORMAT,
    ROOT_LOGGER_NAME,
    TEXT_LOG_FORMAT,
)
from adaptive_framework.interfaces.i_logger import ILogger
from adaptive_framework.logging.formatters import JsonFormatter
from adaptive_framework.logging.handlers import ContextInjectingHandler


# Sentinel for "not set" context values
_NOT_SET = "none"


class FrameworkLogger(ILogger):
    """Concrete centralized logger for the Adaptive Distributed Framework.

    Configures three handlers from a LoggingConfig:
        1. Console handler (StreamHandler, optionally Rich-enhanced).
        2. Rotating text file handler.
        3. Rotating JSON JSONL file handler.

    All records have run_id, worker_id, and node_id injected via
    ContextInjectingHandler before emission.

    Attributes:
        _logger: The underlying Python logging.Logger.
        _context: Bound context fields for this logger instance.

    Example:
        >>> from adaptive_framework.config import ConfigManager
        >>> from adaptive_framework.logging import FrameworkLogger
        >>> cfg = ConfigManager.get_instance()
        >>> cfg.load("configs/")
        >>> log_cfg = cfg.get_logging_config()
        >>> logger = FrameworkLogger.from_config(log_cfg, output_dir=Path("outputs"))
        >>> logger.info("Framework initialized.", run_id="adf_run_001")
    """

    def __init__(
        self,
        logger: logging.Logger,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize FrameworkLogger with a configured Logger and optional context.

        Args:
            logger: A fully configured Python logging.Logger instance.
            context: Initial context fields to bind to every log record.
        """
        self._logger = logger
        self._context: dict[str, Any] = context or {
            "run_id": _NOT_SET,
            "worker_id": _NOT_SET,
            "node_id": _NOT_SET,
        }

    @classmethod
    def from_config(
        cls,
        log_cfg: Any,
        output_dir: Path = Path("outputs"),
        run_id: str = _NOT_SET,
        worker_id: str = _NOT_SET,
        node_id: str = _NOT_SET,
    ) -> "FrameworkLogger":
        """Factory: create a fully configured FrameworkLogger from a LoggingConfig.

        Args:
            log_cfg: LoggingConfig dataclass instance.
            output_dir: Root output directory (logs go into output_dir/logs/).
            run_id: Current run identifier.
            worker_id: Current worker identifier.
            node_id: Current node identifier.

        Returns:
            Fully configured FrameworkLogger instance.
        """
        root_logger = logging.getLogger(ROOT_LOGGER_NAME)
        root_logger.setLevel(getattr(logging, log_cfg.level, logging.INFO))

        # Avoid duplicate handlers if called multiple times
        if root_logger.handlers:
            root_logger.handlers.clear()

        context = {
            "run_id": run_id,
            "worker_id": worker_id,
            "node_id": node_id,
        }

        text_formatter = logging.Formatter(
            fmt=TEXT_LOG_FORMAT,
            datefmt=LOG_TIMESTAMP_FORMAT,
        )
        json_formatter = JsonFormatter()

        # --- Console handler ---
        if log_cfg.console.enabled:
            console_level = getattr(logging, log_cfg.console.level, logging.INFO)
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(console_level)
            stream_handler.setFormatter(text_formatter)
            ctx_console = ContextInjectingHandler(stream_handler, **context)
            ctx_console.setLevel(console_level)
            root_logger.addHandler(ctx_console)

        # --- Rotating text file handler ---
        if log_cfg.file.enabled:
            log_path = output_dir / log_cfg.file.path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_level = getattr(logging, log_cfg.file.level, logging.DEBUG)
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_path),
                maxBytes=log_cfg.file.max_bytes,
                backupCount=log_cfg.file.backup_count,
                encoding=log_cfg.file.encoding,
            )
            file_handler.setLevel(file_level)
            file_handler.setFormatter(text_formatter)
            ctx_file = ContextInjectingHandler(file_handler, **context)
            ctx_file.setLevel(file_level)
            root_logger.addHandler(ctx_file)

        # --- Rotating JSON file handler ---
        if log_cfg.json_file.enabled:
            json_path = output_dir / log_cfg.json_file.path
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_level = getattr(logging, log_cfg.json_file.level, logging.DEBUG)
            json_file_handler = logging.handlers.RotatingFileHandler(
                filename=str(json_path),
                maxBytes=log_cfg.json_file.max_bytes,
                backupCount=log_cfg.json_file.backup_count,
                encoding=log_cfg.json_file.encoding,
            )
            json_file_handler.setLevel(json_level)
            json_file_handler.setFormatter(json_formatter)
            ctx_json = ContextInjectingHandler(json_file_handler, **context)
            ctx_json.setLevel(json_level)
            root_logger.addHandler(ctx_json)

        return cls(root_logger, context)

    def _make_extra(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge bound context with per-call kwargs into an ``extra`` dict.

        Args:
            kwargs: Additional fields from the log call.

        Returns:
            Merged extra dict with context fields taking precedence.
        """
        extra = {**self._context, **kwargs}
        return extra

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a DEBUG message.

        Args:
            message: The log message.
            **kwargs: Additional structured fields.
        """
        self._logger.debug(message, extra=self._make_extra(kwargs), stacklevel=2)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an INFO message.

        Args:
            message: The log message.
            **kwargs: Additional structured fields.
        """
        self._logger.info(message, extra=self._make_extra(kwargs), stacklevel=2)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a WARNING message.

        Args:
            message: The log message.
            **kwargs: Additional structured fields.
        """
        self._logger.warning(message, extra=self._make_extra(kwargs), stacklevel=2)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an ERROR message.

        Args:
            message: The log message.
            **kwargs: Additional structured fields.
        """
        self._logger.error(message, extra=self._make_extra(kwargs), stacklevel=2)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log a CRITICAL message.

        Args:
            message: The log message.
            **kwargs: Additional structured fields.
        """
        self._logger.critical(message, extra=self._make_extra(kwargs), stacklevel=2)

    def bind(self, **context: Any) -> "FrameworkLogger":
        """Return a child logger with additional context fields bound.

        The returned logger is a new FrameworkLogger instance that shares
        the same underlying Python logger but merges new context fields
        into every subsequent log record.

        Args:
            **context: Key-value pairs to bind.

        Returns:
            New FrameworkLogger with merged context.

        Example:
            >>> worker_logger = logger.bind(worker_id="worker_01", node_id="node_02")
            >>> worker_logger.info("Processing started.")
        """
        merged_context = {**self._context, **context}
        return FrameworkLogger(self._logger, merged_context)

    def get_child(self, name: str) -> "FrameworkLogger":
        """Return a child logger for a specific sub-module.

        Args:
            name: Sub-module name appended to the root logger name.

        Returns:
            New FrameworkLogger for the child module.

        Example:
            >>> config_logger = logger.get_child("config")
            >>> config_logger.info("Config loaded.")
        """
        child_logger = logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
        return FrameworkLogger(child_logger, dict(self._context))
