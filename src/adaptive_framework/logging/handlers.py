"""Custom logging handlers for the Adaptive Distributed Framework.

Provides:
    ContextInjectingHandler: Injects run_id, worker_id, node_id into every
        LogRecord before it is emitted by the underlying handler.
"""

from __future__ import annotations

import logging
from typing import Any


class ContextInjectingHandler(logging.Handler):
    """A logging handler wrapper that injects context fields into every record.

    Wraps an underlying handler (StreamHandler, RotatingFileHandler, etc.)
    and injects ``run_id``, ``worker_id``, and ``node_id`` into every
    LogRecord before forwarding it.

    This allows downstream formatters (especially JsonFormatter) to always
    have these fields available without requiring callers to pass ``extra``
    on every log call.

    Attributes:
        _delegate: The underlying handler that performs the actual emit.
        _context: Context fields to inject into every record.

    Example:
        >>> import logging
        >>> from logging.handlers import RotatingFileHandler
        >>> delegate = RotatingFileHandler("logs/framework.log", maxBytes=10_000_000)
        >>> handler = ContextInjectingHandler(
        ...     delegate=delegate,
        ...     run_id="adf_run_001",
        ...     worker_id="none",
        ...     node_id="head",
        ... )
        >>> logger = logging.getLogger("adaptive_framework")
        >>> logger.addHandler(handler)
    """

    def __init__(
        self,
        delegate: logging.Handler,
        **context: Any,
    ) -> None:
        """Initialize the ContextInjectingHandler.

        Args:
            delegate: The underlying handler to wrap.
            **context: Key-value context fields to inject into every record.
        """
        super().__init__()
        self._delegate = delegate
        self._context: dict[str, Any] = context

        # Mirror the delegate's level and formatter
        self.setLevel(delegate.level)
        if delegate.formatter:
            self.setFormatter(delegate.formatter)

    def emit(self, record: logging.LogRecord) -> None:
        """Inject context into the record and forward to the delegate.

        Args:
            record: The LogRecord to emit.
        """
        for key, value in self._context.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        self._delegate.emit(record)

    def setFormatter(self, fmt: logging.Formatter | None) -> None:  # noqa: N802
        """Set formatter on both this handler and the delegate.

        Args:
            fmt: The formatter to apply.
        """
        super().setFormatter(fmt)
        self._delegate.setFormatter(fmt)

    def close(self) -> None:
        """Close both this handler and the delegate."""
        self._delegate.close()
        super().close()

    def flush(self) -> None:
        """Flush the delegate handler's buffer."""
        self._delegate.flush()

    def update_context(self, **new_context: Any) -> None:
        """Update or extend the injected context fields.

        Args:
            **new_context: New or updated key-value context fields.
        """
        self._context.update(new_context)
