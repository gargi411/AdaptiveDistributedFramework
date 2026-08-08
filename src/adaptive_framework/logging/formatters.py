"""JSON log record formatter for the Adaptive Distributed Framework.

Produces structured JSON log lines (JSONL format) with all context
fields defined in logging.yaml injected into every record.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects (JSONL).

    Each emitted line is a valid JSON object containing:
        - timestamp (ISO 8601, UTC)
        - level
        - logger name
        - message
        - run_id, worker_id, node_id (from LogRecord extra fields)
        - thread_id, process_id
        - exception info (if present)

    Example output line::

        {"timestamp": "2026-07-26T17:00:00.123456Z", "level": "INFO",
         "logger": "adaptive_framework.config", "message": "Config loaded",
         "run_id": "adf_run_001", "worker_id": "none", "node_id": "head"}

    Example:
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(JsonFormatter())
        >>> logger = logging.getLogger("adaptive_framework")
        >>> logger.addHandler(handler)
    """

    # Fields always present in the output record
    _ALWAYS_FIELDS: tuple[str, ...] = (
        "timestamp",
        "level",
        "logger",
        "message",
        "module",
        "thread_id",
        "process_id",
    )

    # Context fields injected from LogRecord extras
    _CONTEXT_FIELDS: tuple[str, ...] = (
        "run_id",
        "worker_id",
        "node_id",
    )

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a LogRecord to a JSON string.

        Args:
            record: The log record to format.

        Returns:
            Single-line JSON string terminated without a newline.
        """
        record_dict: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "thread_id": record.thread,
            "process_id": record.process,
        }

        # Inject context fields if present in the record's extra dict
        for field in self._CONTEXT_FIELDS:
            record_dict[field] = getattr(record, field, "none")

        # Attach exception info if present
        if record.exc_info:
            record_dict["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Attach any extra fields from ``logger.info(..., extra={...})``
        standard_keys = logging.LogRecord(
            "", 0, "", 0, "", (), None
        ).__dict__.keys()
        for key, value in record.__dict__.items():
            if key not in standard_keys and key not in record_dict:
                record_dict[key] = value

        return json.dumps(record_dict, default=str, ensure_ascii=False)
