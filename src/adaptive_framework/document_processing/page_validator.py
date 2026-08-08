"""Page Validator — Architecture improvement 2: Output validation before Page construction.

Validates extraction results before PageObjectBuilder assembles the Page.
Instead of silently accepting empty or low-quality results, the validator:
    - Checks OCR confidence thresholds
    - Detects suspiciously empty text on non-blank pages
    - Identifies duplicate text blocks
    - Validates bounding boxes
    - Emits structured warnings and a pass/fail decision

This prevents silent data quality failures from propagating into UnifiedDocument.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from adaptive_framework.models.page import BoundingBox, TextBlock

logger = logging.getLogger(__name__)

# Validation thresholds
_MIN_OCR_CONFIDENCE: float = 0.30      # Below this → warn or fail
_FAIL_OCR_CONFIDENCE: float = 0.10     # Below this → hard fail
_MIN_CHARS_NON_BLANK: int = 5          # Expect at least 5 chars on non-blank pages
_MIN_BBOX_DIMENSION: float = 1.0       # Bounding boxes smaller than 1pt → invalid
_DUPLICATE_WINDOW: int = 3             # Check sliding window for duplicate blocks


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation issue found during page validation.

    Attributes:
        severity: 'warning' or 'error'.
        code: Machine-readable issue code (e.g. 'LOW_OCR_CONFIDENCE').
        message: Human-readable description.
        field: The field/component that triggered this issue. None if global.
    """

    severity: str      # 'warning' | 'error'
    code: str
    message: str
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "field": self.field,
        }

    def __repr__(self) -> str:
        return f"ValidationIssue({self.severity.upper()} [{self.code}]: {self.message})"


@dataclass
class PageValidationResult:
    """Complete validation result for one page's extraction output.

    Attributes:
        page_number: 1-indexed page number.
        passed: True if no 'error' severity issues were found.
        issues: All issues (warnings and errors).
        validation_time_seconds: Time spent in validation.
        char_count_valid: True if char count passes threshold.
        confidence_valid: True if OCR confidence passes threshold.
        bbox_valid: True if all bounding boxes are geometrically valid.
        duplicates_found: Number of duplicate text blocks detected.
    """

    page_number: int
    passed: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    validation_time_seconds: float = 0.0
    char_count_valid: bool = True
    confidence_valid: bool = True
    bbox_valid: bool = True
    duplicates_found: int = 0

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Return only warning-level issues."""
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def errors(self) -> list[ValidationIssue]:
        """Return only error-level issues."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warning_messages(self) -> list[str]:
        """Return warning messages as strings."""
        return [i.message for i in self.warnings]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "page_number": self.page_number,
            "passed": self.passed,
            "issue_count": len(self.issues),
            "warning_count": len(self.warnings),
            "error_count": len(self.errors),
            "char_count_valid": self.char_count_valid,
            "confidence_valid": self.confidence_valid,
            "bbox_valid": self.bbox_valid,
            "duplicates_found": self.duplicates_found,
            "validation_time_seconds": self.validation_time_seconds,
            "issues": [i.to_dict() for i in self.issues],
        }

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"PageValidationResult({status}, "
            f"page={self.page_number}, "
            f"issues={len(self.issues)})"
        )


class PageValidator:
    """Validates extraction results before PageObjectBuilder assembles Page.

    Checks:
        1. OCR confidence threshold
        2. Empty text on non-blank scanned pages
        3. Duplicate text blocks
        4. Invalid bounding boxes (negative area or sub-pixel dimensions)

    Usage:
        >>> validator = PageValidator()
        >>> result = validator.validate(
        ...     page_number=3,
        ...     text="Introduction ...",
        ...     text_blocks=blocks,
        ...     ocr_confidence=0.85,
        ...     processing_method="ocr",
        ...     image_density=0.9,
        ... )
        >>> result.passed
        True

    Args:
        min_ocr_confidence: Warn below this threshold.
        fail_ocr_confidence: Fail (error) below this threshold.
        min_chars_non_blank: Minimum expected characters on non-blank pages.
        min_bbox_dimension: Minimum valid bbox width/height in PDF points.
    """

    def __init__(
        self,
        min_ocr_confidence: float = _MIN_OCR_CONFIDENCE,
        fail_ocr_confidence: float = _FAIL_OCR_CONFIDENCE,
        min_chars_non_blank: int = _MIN_CHARS_NON_BLANK,
        min_bbox_dimension: float = _MIN_BBOX_DIMENSION,
    ) -> None:
        self._min_ocr_confidence = min_ocr_confidence
        self._fail_ocr_confidence = fail_ocr_confidence
        self._min_chars = min_chars_non_blank
        self._min_bbox_dim = min_bbox_dimension

    def validate(
        self,
        page_number: int,
        text: str,
        text_blocks: list[TextBlock],
        ocr_confidence: float,
        processing_method: str,
        image_density: float = 0.0,
    ) -> PageValidationResult:
        """Run all validation checks on extraction output.

        Args:
            page_number: 1-indexed page number.
            text: Full extracted text.
            text_blocks: Extracted TextBlock list.
            ocr_confidence: Average OCR confidence [0.0, 1.0].
            processing_method: 'direct_text', 'ocr', 'mixed', etc.
            image_density: Fraction of page covered by images [0.0, 1.0].

        Returns:
            PageValidationResult with passed flag and issue list.
        """
        import time
        t0 = time.perf_counter()

        result = PageValidationResult(page_number=page_number)

        # 1. OCR confidence check
        self._check_confidence(ocr_confidence, processing_method, result)

        # 2. Empty text check
        self._check_empty_text(text, processing_method, image_density, result)

        # 3. Duplicate block check
        self._check_duplicates(text_blocks, result)

        # 4. Bounding box validity check
        self._check_bboxes(text_blocks, result)

        # Set passed = no errors
        result.passed = len(result.errors) == 0
        result.validation_time_seconds = time.perf_counter() - t0

        if not result.passed:
            logger.warning(
                "Page %d failed validation: %s",
                page_number,
                [e.code for e in result.errors],
            )
        elif result.warnings:
            logger.debug(
                "Page %d passed with %d warnings.",
                page_number, len(result.warnings),
            )

        return result

    # ── Individual checks ────────────────────────────────────────────────────

    def _check_confidence(
        self,
        ocr_confidence: float,
        processing_method: str,
        result: PageValidationResult,
    ) -> None:
        """Check OCR confidence threshold.

        Args:
            ocr_confidence: Average confidence score.
            processing_method: Extraction method used.
            result: Mutable PageValidationResult to append issues to.
        """
        if processing_method == "direct_text":
            return  # Direct extraction always has confidence=1.0

        if ocr_confidence < self._fail_ocr_confidence:
            result.confidence_valid = False
            result.issues.append(ValidationIssue(
                severity="error",
                code="OCR_CONFIDENCE_CRITICAL",
                message=(
                    f"OCR confidence {ocr_confidence:.2f} is critically low "
                    f"(threshold: {self._fail_ocr_confidence:.2f}). "
                    "Page text is likely garbage."
                ),
                field="ocr_confidence",
            ))
        elif ocr_confidence < self._min_ocr_confidence:
            result.confidence_valid = False
            result.issues.append(ValidationIssue(
                severity="warning",
                code="LOW_OCR_CONFIDENCE",
                message=(
                    f"OCR confidence {ocr_confidence:.2f} is below "
                    f"threshold {self._min_ocr_confidence:.2f}."
                ),
                field="ocr_confidence",
            ))

    def _check_empty_text(
        self,
        text: str,
        processing_method: str,
        image_density: float,
        result: PageValidationResult,
    ) -> None:
        """Check for empty text on pages that should have content.

        Args:
            text: Extracted text.
            processing_method: Extraction method.
            image_density: Image area fraction on this page.
            result: Mutable result to append issues to.
        """
        char_count = len(text.strip())
        is_likely_scanned = image_density > 0.3

        if char_count < self._min_chars and processing_method in ("ocr", "mixed"):
            if is_likely_scanned:
                result.char_count_valid = False
                result.issues.append(ValidationIssue(
                    severity="warning",
                    code="EMPTY_TEXT_SCANNED",
                    message=(
                        f"Scanned page produced only {char_count} characters. "
                        "OCR may have failed on this page."
                    ),
                    field="text",
                ))

        if char_count < self._min_chars and processing_method == "direct_text":
            result.issues.append(ValidationIssue(
                severity="warning",
                code="EMPTY_TEXT_DIGITAL",
                message=(
                    f"Digital page produced only {char_count} characters. "
                    "Page may be blank or contain only non-text elements."
                ),
                field="text",
            ))

    def _check_duplicates(
        self,
        text_blocks: list[TextBlock],
        result: PageValidationResult,
    ) -> None:
        """Detect duplicate consecutive text blocks.

        Args:
            text_blocks: Text blocks in reading order.
            result: Mutable result to append issues to.
        """
        if len(text_blocks) < 2:
            return

        seen_texts: dict[str, int] = {}
        duplicate_count = 0

        for block in text_blocks:
            normalized = block.text.strip().lower()
            if not normalized:
                continue
            if normalized in seen_texts:
                duplicate_count += 1
            else:
                seen_texts[normalized] = 1

        result.duplicates_found = duplicate_count

        if duplicate_count > 0:
            result.issues.append(ValidationIssue(
                severity="warning",
                code="DUPLICATE_TEXT_BLOCKS",
                message=(
                    f"Found {duplicate_count} duplicate text blocks. "
                    "This may indicate OCR double-rendering or overlapping regions."
                ),
                field="text_blocks",
            ))

    def _check_bboxes(
        self,
        text_blocks: list[TextBlock],
        result: PageValidationResult,
    ) -> None:
        """Validate bounding boxes for all text blocks.

        Args:
            text_blocks: Text blocks to validate.
            result: Mutable result to append issues to.
        """
        invalid_count = 0

        for block in text_blocks:
            bbox = block.bbox
            if (
                not bbox.is_valid()
                or bbox.width < self._min_bbox_dim
                or bbox.height < self._min_bbox_dim
            ):
                invalid_count += 1

        if invalid_count > 0:
            result.bbox_valid = False
            result.issues.append(ValidationIssue(
                severity="warning",
                code="INVALID_BOUNDING_BOXES",
                message=(
                    f"{invalid_count} text block(s) have invalid bounding boxes "
                    f"(zero or sub-pixel dimensions)."
                ),
                field="text_blocks",
            ))
