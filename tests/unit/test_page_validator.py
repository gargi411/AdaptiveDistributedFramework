"""Unit tests for PageValidator — Architecture improvement 2."""

from __future__ import annotations

import pytest

from adaptive_framework.document_processing.page_validator import (
    PageValidationResult,
    PageValidator,
    ValidationIssue,
)
from adaptive_framework.models.page import BoundingBox, TextBlock


def _make_blocks(texts: list[str], confidence: float = 1.0) -> list[TextBlock]:
    """Create a list of TextBlock objects for testing."""
    bb = BoundingBox(0, 0, 100, 20)
    return [
        TextBlock(text=t, bbox=bb, confidence=confidence, reading_order=i)
        for i, t in enumerate(texts)
    ]


class TestValidationIssue:
    def test_creation(self):
        issue = ValidationIssue(
            severity="warning",
            code="LOW_OCR_CONFIDENCE",
            message="Confidence too low.",
        )
        assert issue.severity == "warning"
        assert issue.code == "LOW_OCR_CONFIDENCE"

    def test_to_dict(self):
        issue = ValidationIssue(
            severity="error",
            code="OCR_CONFIDENCE_CRITICAL",
            message="Critical.",
            field="ocr_confidence",
        )
        d = issue.to_dict()
        assert d["severity"] == "error"
        assert d["field"] == "ocr_confidence"

    def test_repr(self):
        issue = ValidationIssue(severity="warning", code="EMPTY", message="Empty.")
        r = repr(issue)
        assert "WARNING" in r
        assert "EMPTY" in r


class TestPageValidationResult:
    def test_warnings_and_errors(self):
        result = PageValidationResult(
            page_number=1,
            issues=[
                ValidationIssue(severity="warning", code="W1", message="w1"),
                ValidationIssue(severity="error", code="E1", message="e1"),
            ],
        )
        assert len(result.warnings) == 1
        assert len(result.errors) == 1

    def test_passed_with_only_warnings(self):
        result = PageValidationResult(
            page_number=1,
            passed=True,
            issues=[ValidationIssue(severity="warning", code="W", message="w")],
        )
        assert result.passed

    def test_to_dict(self):
        result = PageValidationResult(page_number=3, passed=True)
        d = result.to_dict()
        assert d["page_number"] == 3
        assert d["passed"] is True

    def test_repr(self):
        result = PageValidationResult(page_number=5, passed=False)
        r = repr(result)
        assert "FAILED" in r


class TestPageValidator:
    def setup_method(self):
        self.validator = PageValidator(
            min_ocr_confidence=0.30,
            fail_ocr_confidence=0.10,
            min_chars_non_blank=5,
        )

    def test_valid_digital_page(self):
        blocks = _make_blocks(["Hello world this is a test paragraph."])
        result = self.validator.validate(
            page_number=1,
            text="Hello world this is a test paragraph.",
            text_blocks=blocks,
            ocr_confidence=1.0,
            processing_method="direct_text",
            image_density=0.0,
        )
        assert result.passed
        assert len(result.errors) == 0

    def test_valid_ocr_page_good_confidence(self):
        blocks = _make_blocks(["Some OCR text here."], confidence=0.92)
        result = self.validator.validate(
            page_number=2,
            text="Some OCR text here.",
            text_blocks=blocks,
            ocr_confidence=0.92,
            processing_method="ocr",
        )
        assert result.passed

    def test_low_ocr_confidence_warning(self):
        blocks = _make_blocks(["Low quality text"], confidence=0.25)
        result = self.validator.validate(
            page_number=3,
            text="Low quality text",
            text_blocks=blocks,
            ocr_confidence=0.25,
            processing_method="ocr",
        )
        assert result.passed  # only warning, not error
        codes = [i.code for i in result.issues]
        assert "LOW_OCR_CONFIDENCE" in codes

    def test_critical_ocr_confidence_error(self):
        blocks = _make_blocks(["garbage"], confidence=0.05)
        result = self.validator.validate(
            page_number=4,
            text="garbage",
            text_blocks=blocks,
            ocr_confidence=0.05,
            processing_method="ocr",
        )
        assert not result.passed
        codes = [i.code for i in result.errors]
        assert "OCR_CONFIDENCE_CRITICAL" in codes

    def test_empty_text_on_scanned_page_warns(self):
        result = self.validator.validate(
            page_number=5,
            text="",
            text_blocks=[],
            ocr_confidence=0.0,
            processing_method="ocr",
            image_density=0.8,
        )
        codes = [i.code for i in result.issues]
        assert "EMPTY_TEXT_SCANNED" in codes

    def test_empty_text_on_digital_page_warns(self):
        result = self.validator.validate(
            page_number=6,
            text="",
            text_blocks=[],
            ocr_confidence=1.0,
            processing_method="direct_text",
        )
        codes = [i.code for i in result.issues]
        assert "EMPTY_TEXT_DIGITAL" in codes

    def test_duplicate_blocks_detected(self):
        text = "Repeated block content"
        blocks = _make_blocks([text, text, text])
        result = self.validator.validate(
            page_number=7,
            text=text,
            text_blocks=blocks,
            ocr_confidence=0.90,
            processing_method="ocr",
        )
        assert result.duplicates_found > 0
        codes = [i.code for i in result.issues]
        assert "DUPLICATE_TEXT_BLOCKS" in codes

    def test_invalid_bboxes_detected(self):
        zero_bb = BoundingBox(0, 0, 0, 0)  # invalid
        blocks = [TextBlock(text="text", bbox=zero_bb)]
        result = self.validator.validate(
            page_number=8,
            text="text",
            text_blocks=blocks,
            ocr_confidence=1.0,
            processing_method="direct_text",
        )
        codes = [i.code for i in result.issues]
        assert "INVALID_BOUNDING_BOXES" in codes

    def test_direct_text_skips_confidence_check(self):
        """Direct extraction always has confidence=1.0 — no confidence warning."""
        result = self.validator.validate(
            page_number=9,
            text="Some extracted text",
            text_blocks=_make_blocks(["Some extracted text"]),
            ocr_confidence=0.0,  # would fail OCR check
            processing_method="direct_text",
        )
        confidence_codes = [
            i.code for i in result.issues
            if "CONFIDENCE" in i.code
        ]
        assert len(confidence_codes) == 0
