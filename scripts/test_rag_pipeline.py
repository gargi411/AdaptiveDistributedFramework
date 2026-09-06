"""Standalone verification script for Phase 4.5 RAG Answer Generation.

Usage:
    .venv/Scripts/python scripts/test_rag_pipeline.py "What is the treatment for hypertension?"
"""

from __future__ import annotations

import sys
import tempfile
from unittest.mock import MagicMock
import numpy as np

from adaptive_framework.rag.embeddings.bge_embedder import BGEEmbedder
from adaptive_framework.rag.generation.context_builder import ContextBuilder
from adaptive_framework.rag.generation.fake_llm_provider import FakeLLMProvider
from adaptive_framework.rag.generation.prompt_builder import PromptBuilder
from adaptive_framework.rag.generation.rag_service import RAGService
from adaptive_framework.rag.retrieval.retrieval_engine import RetrievalEngine
from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager


def _make_vector(text: str, dim: int = 64) -> np.ndarray:
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def main() -> None:
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is the first-line pharmacotherapy for type 2 diabetes?"
    )

    print("============================================================")
    print("AdaptiveDistributedFramework - Phase 4.5 RAG Answer Demo")
    print("============================================================")
    print(f"Query: {query}\n")

    dim = 64
    temp_dir = tempfile.mkdtemp()
    manager = FAISSManager(dimension=dim, index_type="flat", persist_dir=temp_dir)

    docs = [
        {
            "chunk_id": "chk_dia_01",
            "document_id": "clinical_guideline_diabetes",
            "source_file": "diabetes_guidelines_2024.pdf",
            "page_numbers": [14, 15],
            "text": "Metformin is recommended as first-line pharmacological therapy for patients with type 2 diabetes mellitus unless contraindicated.",
            "section_heading": "First-Line Therapy",
            "document_type": "guideline",
            "chunk_index": 0,
        },
        {
            "chunk_id": "chk_htn_01",
            "document_id": "cardiology_consensus",
            "source_file": "cardiology_consensus.pdf",
            "page_numbers": [22],
            "text": "Initial antihypertensive treatment comprises thiazide diuretics, calcium channel blockers, or ACE inhibitors.",
            "section_heading": "Antihypertensive Agents",
            "document_type": "consensus",
            "chunk_index": 0,
        },
    ]

    vectors = np.array([_make_vector(d["text"], dim=dim) for d in docs], dtype=np.float32)
    manager.add_vectors(vectors=vectors, metadata=docs)

    embedder = MagicMock(spec=BGEEmbedder)
    embedder.embedding_dim = dim
    embedder.embed_query.side_effect = lambda q: _make_vector(q, dim=dim)

    retrieval_engine = RetrievalEngine(vector_store=manager, embedder=embedder, default_top_k=2)
    context_builder = ContextBuilder(max_chunks=3, max_context_chars=3000)
    prompt_builder = PromptBuilder()
    llm_provider = FakeLLMProvider(model_name="fake-llm-v1")

    service = RAGService(
        retrieval_engine=retrieval_engine,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_provider=llm_provider,
    )

    answer = service.answer(query)

    print(f"Generation Status: {answer.generation_status}")
    print(f"Retrieved Chunks : {len(answer.evidence)}")
    print(f"Total Latency    : {answer.total_latency_s * 1000:.2f} ms")
    print(f"Retrieval Latency: {answer.retrieval_latency_s * 1000:.2f} ms")
    print(f"Generation Latency: {answer.generation_latency_s * 1000:.2f} ms\n")

    print("--- GENERATED ANSWER ---")
    print(answer.answer_text)
    print("\n--- EVIDENCE PROVENANCE ---")
    for i, src in enumerate(answer.source_references, 1):
        pages = ", ".join(str(p) for p in src.page_numbers)
        print(f"[{i}] Document: {src.document_id} | Source: {src.source_file} | Pages: {pages}")
    print("============================================================")


if __name__ == "__main__":
    main()
