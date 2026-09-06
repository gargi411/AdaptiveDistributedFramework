"""Integration test for RAG retrieval evaluation using real RetrievalEngine and FAISS."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from adaptive_framework.models.chunk import Chunk
from adaptive_framework.rag.embedding.bge_embedder import BGEEmbedder
from adaptive_framework.rag.evaluation.dataset import RAGEvaluationCase
from adaptive_framework.rag.evaluation.evaluator import RAGEvaluator
from adaptive_framework.rag.models.embedding import Embedding
from adaptive_framework.rag.retrieval.retrieval_engine import RetrievalEngine
from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager


def _make_vector(text: str, dim: int = 64) -> tuple[float, ...]:
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return tuple(float(x) for x in vec)


def test_rag_evaluation_integration_pipeline() -> None:
    """End-to-end integration test: EvaluationCases -> RetrievalEngine -> FAISS -> Evaluator -> Report."""
    dim = 64
    manager = FAISSManager(dim=dim, index_type="flat")

    # Sample clinical chunks
    c1_text = "Soumya Das was diagnosed with Ischemic Stroke. EEG showed abnormal spikes."
    c2_text = "Anitha Rao had right knee osteoarthritis and spine disc herniation at L4-L5."
    c3_text = "Rajesh Gupta presented with abdominal pain, endoscopy revealed acute gastritis."

    chunks = [
        Chunk(
            chunk_id="chk_01",
            document_id="doc_01",
            source_file="soumya.pdf",
            chunk_index=0,
            page_numbers=(1,),
            text=c1_text,
            section_heading="Discharge",
            document_type="clinical_note",
            character_count=len(c1_text),
            word_count=len(c1_text.split()),
            start_char_in_page=0,
            end_char_in_page=len(c1_text),
        ),
        Chunk(
            chunk_id="chk_02",
            document_id="doc_02",
            source_file="anitha.pdf",
            chunk_index=0,
            page_numbers=(1,),
            text=c2_text,
            section_heading="Discharge",
            document_type="clinical_note",
            character_count=len(c2_text),
            word_count=len(c2_text.split()),
            start_char_in_page=0,
            end_char_in_page=len(c2_text),
        ),
        Chunk(
            chunk_id="chk_03",
            document_id="doc_03",
            source_file="rajesh.pdf",
            chunk_index=0,
            page_numbers=(1,),
            text=c3_text,
            section_heading="Discharge",
            document_type="clinical_note",
            character_count=len(c3_text),
            word_count=len(c3_text.split()),
            start_char_in_page=0,
            end_char_in_page=len(c3_text),
        ),
    ]

    embeddings = [
        Embedding(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            vector=_make_vector(c.text, dim=dim),
            model_name="test-bge",
            embedding_time_seconds=0.001,
            created_at="2026-09-06T12:00:00+00:00",
        )
        for c in chunks
    ]

    manager.add_embeddings(embeddings)
    manager.register_chunks(chunks)

    mock_embedder = MagicMock(spec=BGEEmbedder)
    mock_embedder.embedding_dim = dim
    mock_embedder.embed_query.side_effect = lambda q: np.array(_make_vector(q, dim=dim), dtype=np.float32)

    engine = RetrievalEngine(
        embedding_provider=mock_embedder,
        faiss_manager=manager,
        default_top_k=3,
    )

    cases = [
        RAGEvaluationCase(
            case_id="eval_01",
            question="What findings were noted for Soumya Das?",
            relevant_document_ids=("doc_01",),
            relevant_chunk_ids=("chk_01",),
        ),
        RAGEvaluationCase(
            case_id="eval_02",
            question="What diagnosis was given to Anitha Rao?",
            relevant_document_ids=("doc_02",),
            relevant_chunk_ids=("chk_02",),
        ),
    ]

    evaluator = RAGEvaluator(retrieval_engine=engine, k_values=(1, 3, 5))
    report = evaluator.evaluate(cases)

    assert report.query_count == 2
    assert 1 in report.doc_metrics["precision"]
    assert 3 in report.doc_metrics["precision"]
    assert 5 in report.doc_metrics["precision"]
    assert report.latency_stats_ms["mean"] >= 0.0

    # Test saving reports
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir)
        summary_p, details_p = evaluator.save_report(report, out_path)
        assert summary_p.exists()
        assert details_p.exists()
        assert summary_p.stat().st_size > 0
        assert details_p.stat().st_size > 0
