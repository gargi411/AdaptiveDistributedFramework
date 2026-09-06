"""Unit tests for ContextBuilder and related models (Phase 4.5)."""

from __future__ import annotations

import pytest

from adaptive_framework.rag.generation.context_builder import (
    BuiltContext,
    ContextBuilder,
    EvidenceBlock,
)
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult


def _make_result(
    rank: int = 1,
    score: float = 0.9,
    text: str = "Evidence text content.",
    doc_id: str = "doc_001",
    source_file: str = "doc_001.pdf",
    page_numbers: tuple[int, ...] = (1,),
    chunk_id: str = "chk_001",
    section_heading: str | None = "Introduction",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=doc_id,
        source_file=source_file,
        page_numbers=page_numbers,
        text=text,
        section_heading=section_heading,
        document_type="biomedical",
        chunk_index=rank - 1,
        score=score,
        rank=rank,
    )


class TestContextBuilder:
    """Test suite for ContextBuilder bounding and filtering logic."""

    def test_build_empty_results(self) -> None:
        builder = ContextBuilder()
        context = builder.build([])
        assert isinstance(context, BuiltContext)
        assert len(context.evidence_blocks) == 0
        assert context.total_chars == 0
        assert not context.truncated
        assert len(context.retained_results) == 0

    def test_build_single_result(self) -> None:
        builder = ContextBuilder()
        res = _make_result(rank=1, score=0.92, text="Single document content.")
        context = builder.build([res])

        assert len(context.evidence_blocks) == 1
        block = context.evidence_blocks[0]
        assert isinstance(block, EvidenceBlock)
        assert block.rank == 1
        assert block.score == 0.92
        assert block.text == "Single document content."
        assert block.document_id == "doc_001"
        assert block.source_file == "doc_001.pdf"
        assert block.page_numbers == (1,)
        assert block.section_heading == "Introduction"
        assert context.total_chars == len("Single document content.")
        assert not context.truncated

    def test_evidence_ordering_by_rank(self) -> None:
        builder = ContextBuilder()
        r1 = _make_result(rank=2, score=0.85, text="Second", chunk_id="c2")
        r2 = _make_result(rank=1, score=0.95, text="First", chunk_id="c1")
        context = builder.build([r1, r2])

        assert len(context.evidence_blocks) == 2
        assert context.evidence_blocks[0].rank == 1
        assert context.evidence_blocks[0].text == "First"
        assert context.evidence_blocks[1].rank == 2
        assert context.evidence_blocks[1].text == "Second"

    def test_provenance_preserved(self) -> None:
        builder = ContextBuilder()
        res = _make_result(
            doc_id="clinical_trial_101",
            source_file="trial_report.pdf",
            page_numbers=(4, 5, 6),
            chunk_id="chk_trial_01",
            section_heading="Results",
        )
        context = builder.build([res])
        block = context.evidence_blocks[0]

        assert block.document_id == "clinical_trial_101"
        assert block.source_file == "trial_report.pdf"
        assert block.page_numbers == (4, 5, 6)
        assert block.chunk_id == "chk_trial_01"
        assert block.section_heading == "Results"

    def test_max_chunks_limit(self) -> None:
        builder = ContextBuilder(max_chunks=2)
        results = [
            _make_result(rank=1, score=0.95, text="Chunk 1", chunk_id="c1"),
            _make_result(rank=2, score=0.90, text="Chunk 2", chunk_id="c2"),
            _make_result(rank=3, score=0.85, text="Chunk 3", chunk_id="c3"),
        ]
        context = builder.build(results)
        assert len(context.evidence_blocks) == 2
        assert len(context.retained_results) == 2
        assert context.evidence_blocks[0].chunk_id == "c1"
        assert context.evidence_blocks[1].chunk_id == "c2"

    def test_max_context_chars_limit_and_truncation(self) -> None:
        builder = ContextBuilder(max_chunks=5, max_context_chars=120)
        # 100 chars
        r1 = _make_result(rank=1, score=0.9, text="A" * 100, chunk_id="c1")
        # 50 chars -> will be partially truncated (budget remaining: 20 < 50)
        r2 = _make_result(rank=2, score=0.8, text="B" * 50, chunk_id="c2")

        context = builder.build([r1, r2])
        assert context.truncated
        assert context.total_chars <= 120

    def test_min_score_filter(self) -> None:
        builder = ContextBuilder(min_score=0.7)
        r1 = _make_result(rank=1, score=0.85, text="Good", chunk_id="c1")
        r2 = _make_result(rank=2, score=0.65, text="Low score", chunk_id="c2")
        r3 = _make_result(rank=3, score=0.75, text="Acceptable", chunk_id="c3")

        context = builder.build([r1, r2, r3])
        assert len(context.evidence_blocks) == 2
        chunk_ids = [b.chunk_id for b in context.evidence_blocks]
        assert "c1" in chunk_ids
        assert "c3" in chunk_ids
        assert "c2" not in chunk_ids

    def test_all_results_below_min_score(self) -> None:
        builder = ContextBuilder(min_score=0.9)
        r1 = _make_result(rank=1, score=0.85, text="Good", chunk_id="c1")
        r2 = _make_result(rank=2, score=0.70, text="Fair", chunk_id="c2")

        context = builder.build([r1, r2])
        assert len(context.evidence_blocks) == 0
        assert context.total_chars == 0
        assert not context.truncated

    def test_to_dict_serialisable(self) -> None:
        builder = ContextBuilder()
        res = _make_result(rank=1, score=0.9, text="Test text.")
        context = builder.build([res])
        data = context.to_dict()

        assert "evidence_blocks" in data
        assert "total_chars" in data
        assert "truncated" in data
        assert "retained_results" in data
        assert len(data["evidence_blocks"]) == 1
        assert data["evidence_blocks"][0]["document_id"] == "doc_001"
        assert data["evidence_blocks"][0]["page_numbers"] == [1]

    def test_section_heading_none_handled(self) -> None:
        builder = ContextBuilder()
        res = _make_result(section_heading=None)
        context = builder.build([res])
        assert context.evidence_blocks[0].section_heading is None
        assert context.evidence_blocks[0].to_dict()["section_heading"] is None
