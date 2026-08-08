"""Unit tests for processing strategy pattern — Architecture improvement 1."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adaptive_framework.document_processing.processing_strategy import (
    DirectExtractionStrategy,
    MixedStrategy,
    PageExtractionResult,
    ProcessingStrategyFactory,
    SkippedStrategy,
)
from adaptive_framework.models.page import BoundingBox, PageType, TextBlock


def _make_mock_fitz_page(
    text: str = "Sample text content for testing.",
    image_count: int = 0,
) -> MagicMock:
    """Create a MagicMock that behaves like a PyMuPDF page."""
    mock_page = MagicMock()
    mock_page.get_text.return_value = text
    mock_page.get_images.return_value = []
    mock_page.rect = MagicMock(width=595.0, height=842.0)

    # Simulate get_text("dict") returning block structure
    block = {
        "type": 0,  # text block
        "bbox": (50.0, 100.0, 400.0, 120.0),
        "lines": [
            {
                "spans": [
                    {
                        "text": "Sample text content for testing.",
                        "font": "Arial",
                        "size": 11.0,
                    }
                ]
            }
        ],
    }
    mock_page.get_text.return_value = {"blocks": [block]}
    return mock_page


class TestDirectExtractionStrategy:
    def setup_method(self):
        self.strategy = DirectExtractionStrategy()

    def test_strategy_name(self):
        assert self.strategy.strategy_name == "direct_extraction"

    def test_process_returns_extraction_result(self):
        mock_page = MagicMock()
        # Return simple dict with blocks
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": (0, 0, 200, 20),
                    "lines": [
                        {"spans": [{"text": "Hello world.", "font": "Arial", "size": 12.0}]}
                    ],
                }
            ]
        }
        result = self.strategy.process(
            mock_page, page_number=1,
            document_id="doc-1", file_path="/tmp/test.pdf",
        )
        assert isinstance(result, PageExtractionResult)
        assert result.processing_method == "direct_text"
        assert result.ocr_confidence == 1.0

    def test_empty_page_returns_empty_text(self):
        mock_page = MagicMock()
        mock_page.get_text.return_value = {"blocks": []}
        result = self.strategy.process(
            mock_page, page_number=1,
            document_id="doc-1", file_path="/tmp/test.pdf",
        )
        assert result.text == ""
        assert result.processing_method == "direct_text"

    def test_fitz_error_returns_failed_method(self):
        mock_page = MagicMock()
        mock_page.get_text.side_effect = RuntimeError("Simulated fitz error")
        result = self.strategy.process(
            mock_page, page_number=1,
            document_id="doc-1", file_path="/tmp/test.pdf",
        )
        assert result.processing_method == "failed"
        assert result.error is not None


class TestSkippedStrategy:
    def setup_method(self):
        self.strategy = SkippedStrategy()

    def test_strategy_name(self):
        assert self.strategy.strategy_name == "skipped"

    def test_returns_skipped_result(self):
        result = self.strategy.process(
            MagicMock(), page_number=3,
            document_id="d", file_path="/f",
        )
        assert result.processing_method == "skipped"
        assert result.text == ""
        assert len(result.warnings) > 0


class TestProcessingStrategyFactory:
    def setup_method(self):
        self.factory = ProcessingStrategyFactory()

    def test_digital_returns_direct_strategy(self):
        s = self.factory.get_strategy(PageType.DIGITAL)
        assert s.strategy_name == "direct_extraction"

    def test_scanned_returns_ocr_strategy(self):
        s = self.factory.get_strategy(PageType.SCANNED)
        assert s.strategy_name == "ocr"

    def test_mixed_returns_mixed_strategy(self):
        s = self.factory.get_strategy(PageType.MIXED)
        assert s.strategy_name == "mixed"

    def test_unknown_falls_back_to_direct(self):
        s = self.factory.get_strategy(PageType.UNKNOWN)
        assert s.strategy_name == "direct_extraction"

    def test_available_strategies_list(self):
        strategies = self.factory.available_strategies()
        assert "direct_extraction" in strategies
        assert "ocr" in strategies
        assert "mixed" in strategies


class TestPageExtractionResult:
    def test_default_result(self):
        result = PageExtractionResult()
        assert result.text == ""
        assert result.processing_method == "direct_text"
        assert result.error is None
        assert result.ocr_confidence == 1.0

    def test_result_with_text_blocks(self):
        bb = BoundingBox(0, 0, 100, 20)
        tb = TextBlock(text="Hello", bbox=bb)
        result = PageExtractionResult(
            text="Hello",
            text_blocks=[tb],
            processing_method="direct_text",
        )
        assert len(result.text_blocks) == 1
        assert result.text_blocks[0].text == "Hello"
