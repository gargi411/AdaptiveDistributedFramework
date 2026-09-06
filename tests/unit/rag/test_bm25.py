"""Unit tests for BM25Retriever and clinical tokenizer (Phase 4.8)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from adaptive_framework.models.chunk import Chunk
from adaptive_framework.rag.retrieval.bm25_retriever import (
    BM25Retriever,
    tokenize_clinical_text,
)


def _make_chunk(
    chunk_id: str,
    text: str,
    doc_id: str = "doc_1",
    section: str = "Clinical Summary",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        source_file="test.pdf",
        chunk_index=0,
        page_numbers=(1,),
        text=text,
        section_heading=section,
        document_type="clinical_note",
        character_count=len(text),
        word_count=len(text.split()),
        start_char_in_page=0,
        end_char_in_page=len(text),
    )


class TestBM25Tokenizer:
    """Test deterministic clinical tokenizer."""

    def test_empty_string(self) -> None:
        assert tokenize_clinical_text("") == []
        assert tokenize_clinical_text("   ") == []

    def test_case_folding(self) -> None:
        assert tokenize_clinical_text("Metformin T2DM") == ["metformin", "t2dm"]

    def test_alphanumeric_and_dosage_preservation(self) -> None:
        tokens = tokenize_clinical_text("Patient given 500mg Metformin BD; BP 120/80 mmHg.")
        assert "500mg" in tokens
        assert "metformin" in tokens
        assert "120" in tokens
        assert "80" in tokens
        assert "mmhg" in tokens

    def test_punctuation_stripping(self) -> None:
        tokens = tokenize_clinical_text("STEMI (acute): troponin-I elevation > 1.5!")
        assert "stemi" in tokens
        assert "acute" in tokens
        assert "troponin" in tokens
        assert "i" in tokens
        assert "elevation" in tokens


class TestBM25Retriever:
    """Test BM25Retriever mechanics and ranking."""

    def test_init_validation(self) -> None:
        with pytest.raises(ValueError, match="k1 must be non-negative"):
            BM25Retriever(k1=-0.5)
        with pytest.raises(ValueError, match="b must be between 0.0 and 1.0"):
            BM25Retriever(b=1.5)
        with pytest.raises(ValueError, match="b must be between 0.0 and 1.0"):
            BM25Retriever(b=-0.1)

    def test_empty_corpus(self) -> None:
        retriever = BM25Retriever()
        retriever.index_chunks([])
        assert retriever.doc_count == 0
        assert retriever.avgdl == 0.0
        results = retriever.retrieve("metformin")
        assert results == []

    def test_query_validation(self) -> None:
        retriever = BM25Retriever()
        with pytest.raises(ValueError, match="empty"):
            retriever.retrieve("")
        with pytest.raises(ValueError, match="empty"):
            retriever.retrieve("   ")
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            retriever.retrieve("metformin", top_k=0)

    def test_exact_term_match(self) -> None:
        c1 = _make_chunk("c1", "Patient diagnosed with acute myocardial infarction and elevated troponin.")
        c2 = _make_chunk("c2", "Patient diagnosed with uncomplicated type 2 diabetes mellitus.")
        retriever = BM25Retriever()
        retriever.index_chunks([c1, c2])

        results = retriever.retrieve("troponin", top_k=5)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert results[0].rank == 1
        assert results[0].score > 0.0

    def test_no_matching_terms(self) -> None:
        c1 = _make_chunk("c1", "Cardiology consultation notes.")
        retriever = BM25Retriever()
        retriever.index_chunks([c1])
        results = retriever.retrieve("neurosurgery")
        assert results == []

    def test_term_frequency_saturation(self) -> None:
        # Document with repeated term should score higher, but saturated
        c1 = _make_chunk("c1", "troponin troponin troponin cardiac marker")
        c2 = _make_chunk("c2", "troponin cardiac marker")
        retriever = BM25Retriever(k1=1.5, b=0.0)  # eliminate length effect
        retriever.index_chunks([c1, c2])

        res = retriever.retrieve("troponin", top_k=2)
        assert len(res) == 2
        assert res[0].chunk_id == "c1"
        assert res[1].chunk_id == "c2"
        assert res[0].score > res[1].score

    def test_document_frequency_idf(self) -> None:
        # Rare term should carry higher IDF weight than common term
        c1 = _make_chunk("c1", "patient admitted with rarepheochromocytoma hypertension")
        c2 = _make_chunk("c2", "patient admitted with essential hypertension")
        c3 = _make_chunk("c3", "patient admitted with pulmonary hypertension")
        retriever = BM25Retriever()
        retriever.index_chunks([c1, c2, c3])

        # Query with both rare and common terms
        res = retriever.retrieve("hypertension rarepheochromocytoma", top_k=3)
        assert res[0].chunk_id == "c1"

    def test_document_length_normalization(self) -> None:
        # Short document with term should score higher than long document with same term when b > 0
        c_short = _make_chunk("short", "troponin elevation")
        c_long = _make_chunk("long", "troponin elevation " + "word " * 50)
        retriever = BM25Retriever(k1=1.5, b=0.75)
        retriever.index_chunks([c_short, c_long])

        res = retriever.retrieve("troponin", top_k=2)
        assert res[0].chunk_id == "short"
        assert res[0].score > res[1].score

    def test_metadata_preservation(self) -> None:
        c = _make_chunk("chunk_xyz", "Cardiac arrest resuscitation", doc_id="doc_xyz", section="ICU")
        retriever = BM25Retriever()
        retriever.index_chunks([c])

        res = retriever.retrieve("cardiac", top_k=1)
        assert len(res) == 1
        r = res[0]
        assert r.chunk_id == "chunk_xyz"
        assert r.document_id == "doc_xyz"
        assert r.section_heading == "ICU"
        assert r.source_file == "test.pdf"
        assert r.page_numbers == (1,)

    def test_persistence_save_and_load(self) -> None:
        c1 = _make_chunk("c1", "Metformin 500mg daily", doc_id="doc_1")
        c2 = _make_chunk("c2", "Atorvastatin 20mg nightly", doc_id="doc_2")

        retriever = BM25Retriever(k1=1.2, b=0.8)
        retriever.index_chunks([c1, c2])

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "bm25_test.json"
            retriever.save(file_path)
            assert file_path.exists()

            loaded = BM25Retriever()
            loaded.load(file_path)

            assert loaded.k1 == 1.2
            assert loaded.b == 0.8
            assert loaded.doc_count == 2
            assert loaded.avgdl == retriever.avgdl

            res = loaded.retrieve("metformin", top_k=1)
            assert len(res) == 1
            assert res[0].chunk_id == "c1"

    def test_metrics(self) -> None:
        retriever = BM25Retriever()
        c = _make_chunk("c1", "Discharge medication list")
        retriever.index_chunks([c])
        retriever.retrieve("medication")

        metrics = retriever.get_metrics()
        assert metrics["total_queries"] == 1
        assert metrics["doc_count"] == 1
        assert "avg_retrieval_time_ms" in metrics
