"""Unit tests for Page and UnifiedDocument models — Batch 19."""

from __future__ import annotations

import pytest

from adaptive_framework.models.page import (
    BoundingBox,
    FigureData,
    LayoutElement,
    Page,
    PageStatistics,
    PageType,
    ProcessingMethod,
    TableData,
    TextBlock,
)
from adaptive_framework.models.unified_document import (
    DocumentLayout,
    DocumentStatistics,
    UnifiedDocument,
)


# ── BoundingBox ──────────────────────────────────────────────────────────────

class TestBoundingBox:
    def test_valid_bbox(self):
        bb = BoundingBox(0, 0, 100, 200)
        assert bb.width == 100
        assert bb.height == 200
        assert bb.area == 20_000
        assert bb.is_valid()

    def test_invalid_bbox_zero_width(self):
        bb = BoundingBox(50, 0, 50, 100)
        assert not bb.is_valid()

    def test_to_dict(self):
        bb = BoundingBox(1.0, 2.0, 3.0, 4.0)
        d = bb.to_dict()
        assert d == {"x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0}


# ── TextBlock ────────────────────────────────────────────────────────────────

class TestTextBlock:
    def test_basic_text_block(self):
        bb = BoundingBox(0, 0, 100, 20)
        tb = TextBlock(
            text="Introduction",
            bbox=bb,
            block_type="heading",
            confidence=0.98,
            reading_order=0,
        )
        assert tb.text == "Introduction"
        assert tb.confidence == 0.98

    def test_text_block_to_dict(self):
        bb = BoundingBox(0, 0, 50, 15)
        tb = TextBlock(text="Hello", bbox=bb)
        d = tb.to_dict()
        assert d["text"] == "Hello"
        assert "bbox" in d
        assert d["confidence"] == 1.0


# ── TableData ────────────────────────────────────────────────────────────────

class TestTableData:
    def test_table_data_creation(self):
        bb = BoundingBox(10, 20, 200, 100)
        td = TableData(
            page_number=2,
            bbox=bb,
            rows=3,
            cols=2,
            headers=("Name", "Value"),
            cells=(("A", "1"), ("B", "2"), ("C", "3")),
            markdown="| Name | Value |\n|---|\n| A | 1 |",
            csv="Name,Value\nA,1\n",
            confidence=0.92,
        )
        assert td.rows == 3
        assert td.cols == 2
        assert "table_id" in td.to_dict()

    def test_table_data_to_dict(self):
        bb = BoundingBox(0, 0, 100, 100)
        td = TableData(
            page_number=1,
            bbox=bb,
            rows=2,
            cols=2,
            cells=(("a", "b"), ("c", "d")),
        )
        d = td.to_dict()
        assert d["rows"] == 2
        assert d["page_number"] == 1


# ── FigureData ───────────────────────────────────────────────────────────────

class TestFigureData:
    def test_figure_data_creation(self):
        bb = BoundingBox(0, 0, 300, 200)
        fd = FigureData(
            page_number=5,
            bbox=bb,
            figure_type="chart",
            width_px=600,
            height_px=400,
            resolution_dpi=150,
            caption="Figure 1: Results",
        )
        assert fd.figure_type == "chart"
        assert fd.caption == "Figure 1: Results"

    def test_figure_to_dict_has_figure_id(self):
        bb = BoundingBox(0, 0, 100, 100)
        fd = FigureData(page_number=1, bbox=bb)
        d = fd.to_dict()
        assert "figure_id" in d
        assert d["page_number"] == 1


# ── PageStatistics ───────────────────────────────────────────────────────────

class TestPageStatistics:
    def test_default_statistics(self):
        stats = PageStatistics()
        assert stats.text_block_count == 0
        assert stats.ocr_confidence_avg == 1.0

    def test_statistics_to_dict(self):
        stats = PageStatistics(char_count=500, word_count=80)
        d = stats.to_dict()
        assert d["char_count"] == 500
        assert d["word_count"] == 80


# ── Page ─────────────────────────────────────────────────────────────────────

def _make_page(
    page_number: int = 1,
    text: str = "Sample text.",
    success: bool = True,
    page_type: PageType = PageType.DIGITAL,
    processing_method: ProcessingMethod = ProcessingMethod.DIRECT_TEXT,
) -> Page:
    return Page(
        document_id="doc-001",
        page_number=page_number,
        page_type=page_type,
        processing_method=processing_method,
        text=text,
        text_blocks=(),
        tables=(),
        figures=(),
        layout_elements=(),
        worker_id="worker-0",
        node_id="laptop-1",
        processing_time_seconds=0.1,
        ocr_confidence=1.0,
        success=success,
        error_message=None if success else "Something went wrong",
    )


class TestPage:
    def test_page_creation(self):
        page = _make_page()
        assert page.document_id == "doc-001"
        assert page.page_number == 1
        assert page.success

    def test_page_is_frozen(self):
        page = _make_page()
        with pytest.raises((AttributeError, TypeError)):
            page.page_number = 99  # type: ignore

    def test_page_word_count(self):
        page = _make_page(text="Hello world this is a test sentence.")
        assert page.word_count == 7

    def test_page_is_empty_true(self):
        page = _make_page(text="")
        assert page.is_empty

    def test_page_is_empty_false(self):
        page = _make_page(text="Some text content")
        assert not page.is_empty

    def test_page_to_dict(self):
        page = _make_page()
        d = page.to_dict()
        assert d["document_id"] == "doc-001"
        assert d["page_type"] == "digital"
        assert d["processing_method"] == "direct_text"

    def test_page_repr(self):
        page = _make_page()
        r = repr(page)
        assert "Page(" in r
        assert "page_number=1" in r

    def test_failed_page_has_error(self):
        page = _make_page(success=False)
        assert page.error_message == "Something went wrong"
        assert not page.success


# ── UnifiedDocument ──────────────────────────────────────────────────────────

def _make_unified_doc(pages: list[Page] | None = None) -> UnifiedDocument:
    pages = pages or [_make_page(1), _make_page(2)]
    bb = BoundingBox(0, 0, 100, 100)
    return UnifiedDocument(
        document_id="doc-001",
        file_path="/data/paper.pdf",
        pages=tuple(pages),
        full_text="Page 1 text. Page 2 text.",
        tables=(),
        figures=(),
        layout=DocumentLayout(title="My Paper"),
        statistics=DocumentStatistics(
            total_pages=len(pages),
            processed_pages=len(pages),
        ),
        processing_logs=("All good.",),
    )


class TestUnifiedDocument:
    def test_creation(self):
        doc = _make_unified_doc()
        assert doc.document_id == "doc-001"
        assert len(doc.pages) == 2

    def test_is_frozen(self):
        doc = _make_unified_doc()
        with pytest.raises((AttributeError, TypeError)):
            doc.document_id = "other"  # type: ignore

    def test_get_page_found(self):
        doc = _make_unified_doc()
        p = doc.get_page(1)
        assert p is not None
        assert p.page_number == 1

    def test_get_page_not_found(self):
        doc = _make_unified_doc()
        assert doc.get_page(99) is None

    def test_success_rate(self):
        doc = _make_unified_doc()
        assert doc.success_rate == 1.0

    def test_is_fully_processed(self):
        doc = _make_unified_doc()
        assert doc.is_fully_processed

    def test_to_dict(self):
        doc = _make_unified_doc()
        d = doc.to_dict()
        assert "document_id" in d
        assert "statistics" in d
        assert "pages" in d

    def test_get_tables_for_page_empty(self):
        doc = _make_unified_doc()
        tables = doc.get_tables_for_page(1)
        assert tables == ()

    def test_get_figures_for_page_empty(self):
        doc = _make_unified_doc()
        figures = doc.get_figures_for_page(1)
        assert figures == ()

    def test_repr(self):
        doc = _make_unified_doc()
        r = repr(doc)
        assert "UnifiedDocument(" in r

    def test_layout_title(self):
        doc = _make_unified_doc()
        assert doc.layout.title == "My Paper"
