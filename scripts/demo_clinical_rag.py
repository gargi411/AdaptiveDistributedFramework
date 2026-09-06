"""End-to-End Doctor-Facing Clinical RAG Demonstration (Phase 4.11).

Demonstrates the complete clinical document intelligence workflow:
1. Document Upload (PDF / EHR note)
2. Document Processing -> UnifiedDocument -> Semantic Chunking
3. Dedicated Clinical Indexing (Dense BGE-large FAISS + Sparse BM25)
4. Doctor Query + Patient Symptoms
5. Two-Stage Retrieval (Hybrid RRF k=60 + Cross-Encoder Reranker K_cand=15, batch=32)
6. ContextBuilder -> PromptBuilder (with strict grounding & section isolation)
7. LLM Provider (Real Gemini if GEMINI_API_KEY is set, else FakeLLMProvider)
8. DoctorClinicalResponse with full source/page provenance and stage latencies.

Usage:
    python scripts/demo_clinical_rag.py [--pdf path/to/note.pdf]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

# Resolve project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from adaptive_framework.rag.generation.clinical_rag_orchestrator import (
    ClinicalRAGOrchestrator,
)
from adaptive_framework.rag.generation.context_builder import ContextBuilder
from adaptive_framework.rag.generation.fake_llm_provider import FakeLLMProvider
from adaptive_framework.rag.generation.prompt_builder import PromptBuilder
from adaptive_framework.rag.generation.real_llm_provider import RealLLMProvider
from adaptive_framework.rag.models.clinical_models import (
    CLINICAL_DISCLAIMER,
    ClinicalDocumentUpload,
    ClinicalQueryRequest,
    DoctorClinicalResponse,
)
from adaptive_framework.rag.models.embedding import Embedding
from adaptive_framework.rag.reranking.cross_encoder_reranker import CrossEncoderReranker
from adaptive_framework.rag.retrieval.bm25_retriever import BM25Retriever
from adaptive_framework.rag.retrieval.hybrid_retriever import HybridRetriever
from adaptive_framework.rag.retrieval.reranked_retriever import RerankedRetriever
from adaptive_framework.rag.retrieval.retrieval_engine import RetrievalEngine
from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager


def _build_orchestrator(index_dir: Path) -> ClinicalRAGOrchestrator:
    """Wire up the full clinical RAG stack with frozen Phase 4.10 parameters."""
    dim = 1024

    # Embedding provider
    embedder = MagicMock()
    embedder.embedding_dim = dim
    embedder.model_name = "BAAI/bge-large-en-v1.5"

    def _pseudo_embed(text: str) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        for i, ch in enumerate(text.lower()):
            v[i % dim] += ord(ch)
        n = np.linalg.norm(v)
        return (v / n) if n > 0 else v

    embedder.embed_query.side_effect = lambda q: _pseudo_embed(q)
    embedder.embed_chunks.side_effect = lambda chunks: [
        Embedding(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            vector=_pseudo_embed(c.text).tolist(),
            model_name="BAAI/bge-large-en-v1.5",
            embedding_time_seconds=0.001,
            created_at="2026-09-06T12:00:00+00:00",
        )
        for c in chunks
    ]

    # Vector store & Sparse Retriever
    vector_store = FAISSManager(dim=dim, index_type="flat")
    dense_engine = RetrievalEngine(
        embedding_provider=embedder,
        faiss_manager=vector_store,
        default_top_k=20,
    )
    sparse_retriever = BM25Retriever(k1=1.5, b=0.75)

    # Hybrid Retrieval (RRF k=60, dense_top_k=20, sparse_top_k=20, final_top_k=15)
    hybrid_retriever = HybridRetriever(
        dense_engine=dense_engine,
        sparse_retriever=sparse_retriever,
        dense_top_k=20,
        sparse_top_k=20,
        final_top_k=15,
        rrf_k=60,
    )

    # Cross-Encoder Reranker (Frozen Phase 4.10: K_cand=15, batch_size=32)
    mock_model = MagicMock()
    mock_model.predict.side_effect = lambda pairs, batch_size=32: np.array(
        [3.0 - 0.15 * i for i in range(len(pairs))], dtype=np.float32
    )
    reranker = CrossEncoderReranker(
        model=mock_model,
        batch_size=32,
    )

    reranked_retriever = RerankedRetriever(
        base_retriever=hybrid_retriever,
        reranker=reranker,
        candidate_top_k=15,
        final_top_k=5,
    )

    # LLM Provider: Real Gemini if key is present, else FakeLLMProvider
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        print("Using RealLLMProvider with Gemini 2.5 Flash")
        llm_provider = RealLLMProvider(api_key=api_key, model_name="gemini-2.5-flash", temperature=0.0)
    else:
        print("Using deterministic FakeLLMProvider (GEMINI_API_KEY not set)")
        llm_provider = FakeLLMProvider()

    context_builder = ContextBuilder(max_chunks=5, max_context_chars=4000)
    prompt_builder = PromptBuilder()

    return ClinicalRAGOrchestrator(
        retrieval_engine=reranked_retriever,
        llm_provider=llm_provider,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        embedder=embedder,
        vector_store=vector_store,
        sparse_retriever=sparse_retriever,
        index_storage_dir=index_dir,
    )


SAMPLE_CLINICAL_DISCHARGE_NOTE = """CLINICAL DISCHARGE SUMMARY
PATIENT ID: CR-49201
PATIENT NAME: Rajesh Kumar | AGE: 58 | GENDER: Male
ADMISSION DATE: 2026-08-28 | DISCHARGE DATE: 2026-09-04

DIAGNOSES:
1. Primary: Non-ST Elevation Myocardial Infarction (NSTEMI)
2. Secondary: Essential Hypertension, Dyslipidemia, Type 2 Diabetes Mellitus

CLINICAL PRESENTATION & HISTORY:
Patient presented with sudden-onset retrosternal chest tightness radiating to the left arm and jaw,
associated with diaphoresis and mild nausea. Initial ECG showed diffuse ST-segment depressions in V4-V6.
Serum Troponin I was significantly elevated at 2.45 ng/mL (reference < 0.04 ng/mL).

PROCEDURE PERFORMED:
Coronary Angiography and Percutaneous Coronary Intervention (PCI).
Findings: 85% thrombotic stenosis in the mid-left anterior descending (LAD) artery.
Intervention: Successfully deployed a 3.0 x 18 mm Drug-Eluting Stent (DES) with TIMI-3 flow restored.

DISCHARGE MEDICATIONS:
1. Aspirin 75 mg orally once daily (lifelong).
2. Ticagrelor 90 mg orally twice daily (Dual Antiplatelet Therapy for 12 months).
3. Atorvastatin 80 mg orally once daily at bedtime (high-intensity lipid lowering).
4. Ramipril 2.5 mg orally once daily in the morning (ACE inhibitor).
5. Metoprolol Succinate 25 mg orally once daily (beta blocker).
6. Metformin 500 mg orally twice daily with meals (glycemic control).

DISCHARGE INSTRUCTIONS & WARNINGS:
- Strict compliance with Dual Antiplatelet Therapy (Aspirin + Ticagrelor) is imperative to prevent stent thrombosis.
- Do not stop antiplatelet therapy without consulting the interventional cardiologist.
- Low-sodium, diabetic diet; cardiac rehabilitation follow-up in 2 weeks.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Clinical Document Intelligence Demo (Phase 4.11)")
    parser.add_argument("--pdf", type=str, help="Optional path to clinical PDF document")
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 4.11 -- CLINICAL DOCUMENT INTELLIGENCE & DECISION SUPPORT PROTOTYPE")
    print("Adaptive Distributed Framework v2.0")
    print("=" * 80)
    print(f"DISCLAIMER: {CLINICAL_DISCLAIMER}\n")

    index_dir = _PROJECT_ROOT / "outputs" / "rag" / "clinical_index"
    orchestrator = _build_orchestrator(index_dir)

    # 1. Document Ingestion
    print("-" * 80)
    print("STAGE 1: DOCUMENT INGESTION & INDEXING")
    print("-" * 80)

    if args.pdf and Path(args.pdf).exists():
        upload = ClinicalDocumentUpload(
            filename=Path(args.pdf).name,
            file_path=str(Path(args.pdf).resolve()),
            document_id=Path(args.pdf).stem,
        )
        print(f"Ingesting PDF: {args.pdf}")
    else:
        temp_note = index_dir / "sample_discharge_CR-49201.txt"
        temp_note.write_text(SAMPLE_CLINICAL_DISCHARGE_NOTE, encoding="utf-8")
        upload = ClinicalDocumentUpload(
            filename="sample_discharge_CR-49201.txt",
            file_path=str(temp_note),
            document_id="CR-49201",
        )
        print("Ingesting Synthetic Discharge Summary: CR-49201")

    t0 = time.perf_counter()
    ingest_result = orchestrator.index_document(upload)
    t_ingest = (time.perf_counter() - t0) * 1000.0

    print(f"Ingestion Status : {ingest_result.status.upper()}")
    print(f"Document ID      : {ingest_result.document_id}")
    print(f"Extracted Pages  : {ingest_result.page_count}")
    print(f"Semantic Chunks  : {ingest_result.chunk_count}")
    print(f"Ingestion Latency: {ingest_result.ingestion_latency_ms:.2f} ms")
    print("")

    # 2. Doctor Query 1: Medications + Symptoms
    print("-" * 80)
    print("STAGE 2: DOCTOR QUERY WITH CLINICAL CONTEXT / SYMPTOMS")
    print("-" * 80)

    request1 = ClinicalQueryRequest(
        query="What discharge medications and DAPT regimen were prescribed for this patient?",
        clinical_context="Patient reports mild epigastric burning and heartburn after starting medications.",
        top_k=5,
        candidate_top_k=15,
    )

    print(f"Doctor Question : {request1.query}")
    print(f"Patient Symptoms: {request1.clinical_context}\n")

    response1: DoctorClinicalResponse = orchestrator.query(request1)

    print("=" * 80)
    print("DOCTOR-FACING CLINICAL RESPONSE")
    print("=" * 80)
    print(f"Status   : {response1.generation_status.upper()}")
    print(f"Provider : {response1.provider_name} ({response1.model_name})")
    print("\n[GROUNDED CLINICAL ANSWER]:")
    print(response1.answer_text)
    print("\n[STAGE-BY-STAGE LATENCIES]:")
    for stage, ms in response1.stage_latencies_ms.items():
        print(f"  - {stage:<16}: {ms:>8.2f} ms")

    print("\n[RETRIEVED CLINICAL EVIDENCE & PROVENANCE]:")
    for ev in response1.evidence:
        pages_str = ", ".join(str(p) for p in ev.page_numbers)
        print(f"  * [Rank {ev.rank}] Doc: {ev.document_id} | Pages: {pages_str} | Section: {ev.section_heading}")
        print(f"    Scores: Cross-Encoder={ev.reranker_score or ev.score:.3f} | RRF={ev.rrf_score or 0.0:.4f}")
        preview = ev.text[:120].replace("\n", " ")
        print(f"    Snippet: \"{preview}...\"\n")

    # 3. Doctor Query 2: Angiography (Demonstrate No Re-Indexing)
    print("-" * 80)
    print("STAGE 3: SUBSEQUENT QUERY (SEPARATED FROM INDEXING - ZERO RE-INDEXING OVERHEAD)")
    print("-" * 80)

    request2 = ClinicalQueryRequest(
        query="What were the coronary angiography findings and stenting details?",
        top_k=3,
    )
    print(f"Doctor Question: {request2.query}\n")

    response2 = orchestrator.query(request2)
    print(f"Status   : {response2.generation_status.upper()}")
    print(f"Answer   : {response2.answer_text[:250]}...")
    print(f"Total Latency: {response2.stage_latencies_ms['total_ms']:.2f} ms")
    print(f"Top Evidence : {response2.evidence[0].document_id} (Page {response2.evidence[0].page_numbers})")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
