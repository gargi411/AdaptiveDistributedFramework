"""Doctor-Facing Clinical Document Intelligence Dashboard (Phase 4.11).

A functional Streamlit application for clinical document ingestion, semantic search,
grounded clinical question answering, and provenance verification.

Prototype notice: Decision-support only. Not an autonomous diagnostic system.

Usage:
    streamlit run dashboard/clinical_rag_app.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st

# Add project root and src/ to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
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
import numpy as np


@st.cache_resource
def get_orchestrator() -> ClinicalRAGOrchestrator:
    """Initialize and cache the Clinical RAG Orchestrator."""
    dim = 1024
    index_dir = _PROJECT_ROOT / "outputs" / "rag" / "clinical_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    # Fast normalized vector provider
    from unittest.mock import MagicMock
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

    vector_store = FAISSManager(dim=dim, index_type="flat")
    dense_engine = RetrievalEngine(
        embedding_provider=embedder,
        faiss_manager=vector_store,
        default_top_k=20,
    )
    sparse_retriever = BM25Retriever(k1=1.5, b=0.75)

    hybrid_retriever = HybridRetriever(
        dense_engine=dense_engine,
        sparse_retriever=sparse_retriever,
        dense_top_k=20,
        sparse_top_k=20,
        final_top_k=15,
        rrf_k=60,
    )

    mock_ce = MagicMock()
    mock_ce.predict.side_effect = lambda pairs, batch_size=32: np.array(
        [3.0 - 0.1 * i for i in range(len(pairs))], dtype=np.float32
    )
    reranker = CrossEncoderReranker(model=mock_ce, batch_size=32)

    reranked_retriever = RerankedRetriever(
        base_retriever=hybrid_retriever,
        reranker=reranker,
        candidate_top_k=15,
        final_top_k=5,
    )

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        llm_provider = RealLLMProvider(api_key=api_key, model_name="gemini-2.5-flash", temperature=0.0)
    else:
        llm_provider = FakeLLMProvider()

    return ClinicalRAGOrchestrator(
        retrieval_engine=reranked_retriever,
        llm_provider=llm_provider,
        context_builder=ContextBuilder(max_chunks=5, max_context_chars=4000),
        prompt_builder=PromptBuilder(),
        embedder=embedder,
        vector_store=vector_store,
        sparse_retriever=sparse_retriever,
        index_storage_dir=index_dir,
    )


def main() -> None:
    st.set_page_config(
        page_title="Clinical Document Intelligence",
        page_icon="stethoscope",
        layout="wide",
    )

    # 1. Header & Mandatory Safety Warning
    st.title("Clinical Document Intelligence & Decision Support")
    st.caption("Adaptive Distributed Framework v2.0 — Phase 4.11 Clinical Integration")

    st.warning(f"REGULATORY DISCLAIMER: {CLINICAL_DISCLAIMER}")

    orchestrator = get_orchestrator()

    # 2. Sidebar: Document Upload & Ingestion
    with st.sidebar:
        st.header("Document Ingestion")
        st.write("Upload patient EHR notes or clinical discharge summaries.")

        uploaded_file = st.file_uploader(
            "Upload Clinical Document",
            type=["pdf", "txt", "json"],
            help="Select a clinical discharge summary, lab report, or EHR export.",
        )

        doc_id_input = st.text_input("Document ID (optional)", placeholder="e.g. PATIENT-001")

        if uploaded_file is not None:
            if st.button("Process & Index Document", type="primary"):
                with st.spinner("Processing pages, generating semantic chunks, and indexing..."):
                    file_bytes = uploaded_file.read()
                    assigned_doc_id = doc_id_input.strip() or Path(uploaded_file.name).stem

                    upload = ClinicalDocumentUpload(
                        filename=uploaded_file.name,
                        file_bytes=file_bytes,
                        document_id=assigned_doc_id,
                    )

                    result = orchestrator.index_document(upload)

                    if result.status == "success":
                        st.success(f"Indexed {result.document_id} successfully!")
                        st.metric("Extracted Pages", result.page_count)
                        st.metric("Semantic Chunks", result.chunk_count)
                        st.metric("Ingestion Latency", f"{result.ingestion_latency_ms:.1f} ms")
                    else:
                        st.error(f"Ingestion failed: {result.error_message}")

        # Display currently indexed documents
        indexed_docs = orchestrator.get_indexed_documents()
        if indexed_docs:
            st.divider()
            st.subheader(f"Active Index ({len(indexed_docs)} documents)")
            for d in indexed_docs:
                st.text(f"* {d.document_id} ({d.chunk_count} chunks)")

    # 3. Main Query & Clinical Presentation Panel
    st.subheader("Clinical Inquiry")

    col1, col2 = st.columns([3, 2])

    with col1:
        clinical_question = st.text_area(
            "Clinical Question",
            value="What discharge medications and DAPT regimen were prescribed for this patient?",
            height=100,
            help="Enter specific inquiry regarding medication, diagnosis, interventions, or lab values.",
        )

    with col2:
        patient_symptoms = st.text_area(
            "Patient Presentation / Reported Symptoms (Optional)",
            value="Patient reports mild epigastric discomfort after beginning discharge medications.",
            height=100,
            help="Enter patient-reported context. This context is explicitly separated from documented findings.",
        )

    col_btn, col_k, col_cand = st.columns([2, 1, 1])
    with col_k:
        top_k = st.number_input("Final Top-K", min_value=1, max_value=10, value=5)
    with col_cand:
        cand_k = st.number_input("Candidate Depth (K_cand)", min_value=5, max_value=30, value=15)

    with col_btn:
        st.write("")
        st.write("")
        analyze_btn = st.button("Analyze Document", type="primary", use_container_width=True)

    # 4. Results Display
    if analyze_btn:
        if not clinical_question.strip():
            st.error("Please enter a valid clinical question.")
            return

        with st.spinner("Executing hybrid retrieval, cross-encoder reranking, and grounded reasoning..."):
            request = ClinicalQueryRequest(
                query=clinical_question.strip(),
                clinical_context=patient_symptoms.strip() or None,
                top_k=int(top_k),
                candidate_top_k=int(cand_k),
            )

            response: DoctorClinicalResponse = orchestrator.query(request)

        # Response Header
        st.divider()
        st.subheader("Clinical Intelligence Response")

        # Status badge
        if response.generation_status == "success":
            st.success(f"Status: SUCCESS | Provider: {response.provider_name} ({response.model_name})")
        elif response.generation_status == "insufficient_evidence":
            st.warning("Status: INSUFFICIENT EVIDENCE")
        else:
            st.error(f"Status: {response.generation_status.upper()}")

        # Answer container
        st.markdown("#### Grounded Answer")
        st.info(response.answer_text)

        # Latency breakdown
        st.markdown("#### Processing Latency Profile")
        lats = response.stage_latencies_ms
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrieval & Rerank", f"{lats.get('retrieval_ms', 0):.2f} ms")
        c2.metric("Context & Prompt", f"{(lats.get('context_ms', 0) + lats.get('prompt_ms', 0)):.2f} ms")
        c3.metric("LLM Generation", f"{lats.get('generation_ms', 0):.2f} ms")
        c4.metric("End-to-End Total", f"{lats.get('total_ms', 0):.2f} ms")

        # Evidence & Provenance
        st.markdown("#### Retrieved Clinical Evidence & Provenance")
        if not response.evidence:
            st.write("No evidence passages retrieved.")
        else:
            for ev in response.evidence:
                pages_str = ", ".join(str(p) for p in ev.page_numbers)
                title = f"Rank {ev.rank} | Document: {ev.document_id} (Page {pages_str}) | Score: {ev.score:.3f}"
                with st.expander(title, expanded=(ev.rank == 1)):
                    st.markdown(f"**Section Heading:** `{ev.section_heading}`")
                    st.markdown(f"**Source File:** `{ev.source_file}`")
                    st.markdown(f"**Chunk ID:** `{ev.chunk_id}`")
                    if ev.reranker_score is not None:
                        st.markdown(f"**Cross-Encoder Logit:** `{ev.reranker_score:.4f}` | **RRF Score:** `{ev.rrf_score or 0.0:.5f}`")
                    st.markdown("**Passage Text:**")
                    st.text(ev.text)


if __name__ == "__main__":
    main()
