"""Validation utility functions for the Adaptive Distributed Framework.

Pure validation functions that raise ValidationError on failure.
No side effects. No I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptive_framework.core.constants import (
    MAX_WORK_UNIT_SIZE_MB,
    MIN_SCHEDULABLE_PAGE_COUNT,
    PDF_EXTENSION,
    VALID_SOURCE_TYPES,
)
from adaptive_framework.core.exceptions import ValidationError


def validate_positive_int(value: int, field_name: str) -> None:
    """Validate that an integer value is strictly positive.

    Args:
        value: The integer to validate.
        field_name: Name of the field (used in error messages).

    Raises:
        ValidationError: If value <= 0.

    Example:
        >>> validate_positive_int(5, "page_count")   # OK
        >>> validate_positive_int(0, "page_count")   # raises
    """
    if value <= 0:
        raise ValidationError(
            f"'{field_name}' must be > 0, got {value}.",
            field=field_name,
            value=value,
        )


def validate_non_negative_float(value: float, field_name: str) -> None:
    """Validate that a float value is non-negative.

    Args:
        value: The float to validate.
        field_name: Name of the field.

    Raises:
        ValidationError: If value < 0.
    """
    if value < 0:
        raise ValidationError(
            f"'{field_name}' must be >= 0, got {value}.",
            field=field_name,
            value=value,
        )


def validate_fraction(value: float, field_name: str) -> None:
    """Validate that a value is a valid fraction in [0.0, 1.0].

    Args:
        value: The float to validate.
        field_name: Name of the field.

    Raises:
        ValidationError: If value not in [0.0, 1.0].
    """
    if not (0.0 <= value <= 1.0):
        raise ValidationError(
            f"'{field_name}' must be in [0.0, 1.0], got {value}.",
            field=field_name,
            value=value,
        )


def validate_pdf_path(file_path: Path) -> None:
    """Validate that a path points to an existing PDF file.

    Args:
        file_path: Path to validate.

    Raises:
        ValidationError: If the path does not exist, is not a file,
            or does not have a .pdf extension.

    Example:
        >>> validate_pdf_path(Path("/data/paper.pdf"))  # OK
        >>> validate_pdf_path(Path("/data/missing.pdf"))  # raises
    """
    if not file_path.exists():
        raise ValidationError(
            f"PDF file not found: '{file_path}'.",
            field="file_path",
            value=str(file_path),
        )
    if not file_path.is_file():
        raise ValidationError(
            f"'{file_path}' is not a file.",
            field="file_path",
            value=str(file_path),
        )
    if file_path.suffix.lower() != PDF_EXTENSION:
        raise ValidationError(
            f"'{file_path}' does not have a .pdf extension.",
            field="file_path",
            value=str(file_path),
        )


def validate_page_count(page_count: int) -> None:
    """Validate that a page count meets the minimum schedulable threshold.

    Args:
        page_count: Number of pages to validate.

    Raises:
        ValidationError: If page_count < MIN_SCHEDULABLE_PAGE_COUNT.
    """
    if page_count < MIN_SCHEDULABLE_PAGE_COUNT:
        raise ValidationError(
            f"page_count must be >= {MIN_SCHEDULABLE_PAGE_COUNT}, got {page_count}.",
            field="page_count",
            value=page_count,
        )


def validate_source_type(source_type: str | None) -> None:
    """Validate a document source_type value.

    Args:
        source_type: The source type string to validate. None is allowed
            (optional field per architecture §2.4).

    Raises:
        ValidationError: If source_type is not None and not in VALID_SOURCE_TYPES.
    """
    if source_type is not None and source_type not in VALID_SOURCE_TYPES:
        raise ValidationError(
            f"source_type must be one of {sorted(VALID_SOURCE_TYPES)}, got '{source_type}'.",
            field="source_type",
            value=source_type,
        )


def validate_non_empty_string(value: str, field_name: str) -> None:
    """Validate that a string is not empty or whitespace-only.

    Args:
        value: The string to validate.
        field_name: Field name for the error message.

    Raises:
        ValidationError: If value is empty or whitespace-only.
    """
    if not value or not value.strip():
        raise ValidationError(
            f"'{field_name}' must not be empty.",
            field=field_name,
            value=value,
        )


def validate_choices(
    value: Any,
    choices: frozenset[Any] | set[Any],
    field_name: str,
) -> None:
    """Validate that a value is within an allowed set.

    Args:
        value: The value to check.
        choices: Allowed values.
        field_name: Field name for the error message.

    Raises:
        ValidationError: If value is not in choices.
    """
    if value not in choices:
        raise ValidationError(
            f"'{field_name}' must be one of {sorted(str(c) for c in choices)}, "
            f"got '{value}'.",
            field=field_name,
            value=value,
        )
