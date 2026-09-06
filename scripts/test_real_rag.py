"""Real LLM Provider End-to-End Demonstration Script.

Demonstrates the complete Phase 4.6 RAG pipeline:
Doctor/User Question
        |
        v
RetrievalEngine
        |
        v
BGE Embeddings + FAISS Top-K
        |
        v
ContextBuilder
        |
        v
PromptBuilder
        |
        v
ILLMProvider (RealLLMProvider / FakeLLMProvider)
        |
        v
RAGAnswer (Answer + Evidence + Provenance)

Usage:
    # Using environment variable (GEMINI_API_KEY or OPENAI_API_KEY):
    .venv/Scripts/python scripts/test_real_rag.py "What findings are documented in Soumya Das's report?"

    # Specifying provider or key explicitly:
    .venv/Scripts/python scripts/test_real_rag.py "What is the diagnosis for Soumya Das?" --provider gemini --model gemini-1.5-flash
    .venv/Scripts/python scripts/test_real_rag.py "What are the discharge instructions?" --provider fake
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from adaptive_framework.models.chunk import Chunk
from adaptive_framework.rag.embedding.bge_embedder import BGEEmbedder
from adaptive_framework.rag.generation.context_builder import ContextBuilder
from adaptive_framework.rag.generation.fake_llm_provider import FakeLLMProvider
from adaptive_framework.rag.generation.prompt_builder import PromptBuilder
from adaptive_framework.rag.generation.rag_service import RAGService
from adaptive_framework.rag.generation.real_llm_provider import RealLLMProvider
from adaptive_framework.rag.interfaces.i_llm_provider import ILLMProvider
from adaptive_framework.rag.models.embedding import Embedding
from adaptive_framework.rag.retrieval.retrieval_engine import RetrievalEngine
from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager


def _make_vector(text: str, dim: int = 1024) -> tuple[float, ...]:
    """Generate a deterministic normalized vector tuple for text."""
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return tuple(float(x) for x in vec)


def _build_chunk(
    chunk_id: str,
    document_id: str,
    source_file: str,
    text: str,
    section_heading: str | None,
    document_type: str | None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_file=source_file,
        chunk_index=0,
        page_numbers=(1,),
        text=text,
        section_heading=section_heading,
        document_type=document_type,
        character_count=len(text),
        word_count=len(text.split()),
        start_char_in_page=0,
        end_char_in_page=len(text),
    )


def load_clinical_evidence_chunks(dataset_path: Path) -> list[Chunk]:
    """Load sample clinical notes from dataset or return default clinical evidence."""
    raw_notes: list[dict[str, Any]] = []

    if dataset_path.exists():
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_notes = json.load(f)[:10]
        except Exception:
            pass

    if raw_notes:
        chunks: list[Chunk] = []
        for i, item in enumerate(raw_notes):
            text = item.get("note", "")[:800]
            chunks.append(
                _build_chunk(
                    chunk_id=f"chk_synth_{item.get('id', i):04d}",
                    document_id=f"clinical_note_{item.get('id', i):04d}",
                    source_file="synthetic_notes_medium.json",
                    text=text,
                    section_heading=f"Specialty: {item.get('specialty', 'general')}",
                    document_type="discharge_summary",
                )
            )
        return chunks

    # Fallback clinical evidence
    c1_text = (
        "JIPMER Puducherry Discharge Summary\n"
        "Patient Name: Soumya Das | Age/Sex: 8 months/M | UHID: 582614792\n"
        "Chief Complaints: Severe headache with nausea for 4 days.\n"
        "History of Present Illness: Migraine-like photophobia.\n"
        "Physical Examination: Cranial nerves intact. PICCKLE: Pallor: +, Icterus: -\n"
        "Investigations: EEG shows abnormal spikes.\n"
        "Diagnosis: Ischemic Stroke (ICD: I63.9).\n"
        "Advice on Discharge: Tab. Clopidogrel 75mg OD. Condition at discharge: Stable, afebrile."
    )
    c2_text = (
        "Safdarjung Hospital Delhi Discharge Summary\n"
        "Patient Name: Anitha Rao | Age/Sex: 58/F | UHID: 224411653\n"
        "Chief Complaints: Lower back pain radiating to leg for 19 weeks.\n"
        "Investigations: MRI of the spine shows disc herniation at L4-L5.\n"
        "Diagnosis: Osteoarthritis of the right knee (ICD: M17.11).\n"
        "Advice on Discharge: Tab. Paracetamol 650mg SOS for pain. Physiotherapy advised."
    )
    return [
        _build_chunk(
            chunk_id="chk_jipmer_0001",
            document_id="JIPMER_Discharge_Summary_582614792",
            source_file="patient_soumya_das.pdf",
            text=c1_text,
            section_heading="Discharge Summary",
            document_type="clinical_note",
        ),
        _build_chunk(
            chunk_id="chk_safdarjung_0002",
            document_id="Safdarjung_Discharge_Summary_224411653",
            source_file="patient_anitha_rao.pdf",
            text=c2_text,
            section_heading="Discharge Summary",
            document_type="clinical_note",
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.6 Real LLM RAG Pipeline Demonstration")
    parser.add_argument(
        "query",
        nargs="?",
        default="What findings are documented in Soumya Das's report?",
        help="Doctor / user inquiry",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "real", "fake"],
        default=None,
        help="Provider to use (defaults to gemini if GEMINI_API_KEY set, else openai if OPENAI_API_KEY set, else fake)",
    )
    parser.add_argument("--model", default=None, help="Model identifier")
    parser.add_argument("--api-key", default=None, help="Explicit API key")
    parser.add_argument("--base-url", default=None, help="Custom base URL for OpenAI-compatible endpoint")
    args = parser.parse_args()

    # Determine provider selection
    selected_provider = args.provider
    if not selected_provider:
        if args.api_key or os.environ.get("GEMINI_API_KEY"):
            selected_provider = "gemini"
        elif os.environ.get("OPENAI_API_KEY"):
            selected_provider = "openai"
        else:
            selected_provider = "fake"

    print("============================================================")
    print("AdaptiveDistributedFramework - Phase 4.6 Real LLM RAG Demo")
    print("============================================================")
    print(f"Selected Provider : {selected_provider}")
    print(f"User Query        : {args.query}\n")

    # 1. Prepare clinical evidence and vector store
    dim = 1024
    vector_store = FAISSManager(dim=dim, index_type="flat")

    dataset_path = Path("dataset/Synthetic Indian Clinical Notes for Natural Langua/synthetic_notes_medium.json")
    chunks = load_clinical_evidence_chunks(dataset_path)

    embeddings = [
        Embedding(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            vector=_make_vector(c.text, dim=dim),
            model_name="BAAI/bge-large-en-v1.5",
            embedding_time_seconds=0.001,
            created_at="2026-09-06T12:00:00+00:00",
        )
        for c in chunks
    ]
    vector_store.add_embeddings(embeddings)
    vector_store.register_chunks(chunks)

    embedder = MagicMock(spec=BGEEmbedder)
    embedder.embedding_dim = dim
    embedder.embed_query.side_effect = lambda q: np.array(_make_vector(q, dim=dim), dtype=np.float32)

    # 2. Initialize RAG components
    retrieval_engine = RetrievalEngine(
        embedding_provider=embedder,
        faiss_manager=vector_store,
        default_top_k=2,
    )
    context_builder = ContextBuilder(max_chunks=3, max_context_chars=3000)
    prompt_builder = PromptBuilder()

    # 3. Initialize LLM Provider
    llm_provider: ILLMProvider
    if selected_provider == "fake":
        llm_provider = FakeLLMProvider(model_name=args.model or "fake-llm-v1")
    else:
        try:
            real_p = RealLLMProvider(
                backend=selected_provider,
                model_name=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                temperature=0.0,
                max_tokens=1024,
            )
            # Verify key resolution
            real_p._resolve_api_key()
            llm_provider = real_p
        except Exception as e:
            print("============================================================")
            print("PROVIDER CREDENTIAL NOTICE:")
            print(f"  {e}")
            print("  To execute a live real LLM generation, set your environment variable:")
            print("    Windows (PowerShell): $env:GEMINI_API_KEY=\"<your-api-key>\"")
            print("    Windows (CMD)       : set GEMINI_API_KEY=<your-api-key>")
            print("    Linux/macOS         : export GEMINI_API_KEY=\"<your-api-key>\"")
            print("  Or pass directly: --api-key \"<your-api-key>\"")
            print("  For OpenAI / Groq: set OPENAI_API_KEY=\"<your-api-key>\" --provider openai")
            print("============================================================\n")
            print("Proceeding with FakeLLMProvider for deterministic offline demonstration.\n")
            llm_provider = FakeLLMProvider(model_name="fake-llm-v1")

    # 4. Initialize RAGService
    service = RAGService(
        retrieval_engine=retrieval_engine,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_provider=llm_provider,
    )

    # 5. Process query
    try:
        answer = service.answer(args.query)
    except Exception as exc:
        print(f"Error during RAG execution: {exc}")
        return 1

    # 6. Display exact plain ASCII output requested
    print("Query:")
    print(answer.query)
    print("")
    print("Answer:")
    print(answer.answer_text)
    print("")
    print("Supporting Evidence:")
    for i, ev in enumerate(answer.evidence, 1):
        pages_str = ", ".join(str(p) for p in ev.page_numbers)
        print(f"- Document: {ev.document_id}")
        print(f"  Page: {pages_str}")
        print(f"  Chunk: {ev.chunk_id}")
        print(f"  Score: {ev.score:.4f}")
    print("")
    print("Latency:")
    print(f"- Retrieval: {answer.retrieval_latency_s * 1000:.2f} ms")
    print(f"- Context: {answer.context_latency_s * 1000:.2f} ms")
    print(f"- Prompt: {answer.prompt_latency_s * 1000:.2f} ms")
    print(f"- Generation: {answer.generation_latency_s * 1000:.2f} ms")
    print(f"- Total: {answer.total_latency_s * 1000:.2f} ms")
    print("============================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
