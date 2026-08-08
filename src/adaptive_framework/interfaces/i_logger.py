"""ILogger — Abstract logging interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ILogger(ABC):
    """Abstract interface for all loggers in the framework.

    Concrete implementations (FrameworkLogger) implement this interface.
    Components that need logging receive an ILogger via constructor injection
    and never import FrameworkLogger directly.

    Example:
        >>> class MyComponent:
        ...     def __init__(self, logger: ILogger) -> None:
        ...         self._logger = logger
        ...     def do_work(self) -> None:
        ...         self._logger.info("Work started.")
    """

    @abstractmethod
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a DEBUG-level message.

        Args:
            message: The log message.
            **kwargs: Additional structured fields to attach to the record.
        """

    @abstractmethod
    def info(self, message: str, **kwargs: Any) -> None:
        """Log an INFO-level message.

        Args:
            message: The log message.
            **kwargs: Additional structured fields.
        """

    @abstractmethod
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a WARNING-level message.

        Args:
            message: The log message.
            **kwargs: Additional structured fields.
        """

    @abstractmethod
    def error(self, message: str, **kwargs: Any) -> None:
        """Log an ERROR-level message.

        Args:
            message: The log message.
            **kwargs: Additional structured fields.
        """

    @abstractmethod
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log a CRITICAL-level message.

        Args:
            message: The log message.
            **kwargs: Additional structured fields.
        """

    @abstractmethod
    def bind(self, **context: Any) -> "ILogger":
        """Return a new logger with additional context fields bound.

        Args:
            **context: Key-value pairs to attach to all subsequent records.

        Returns:
            A new ILogger instance with the context applied.
        """
