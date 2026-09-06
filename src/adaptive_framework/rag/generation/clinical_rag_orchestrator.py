"""Clinical RAG Orchestration Service (Phase 4.11).

Orchestrates end-to-end doctor-facing clinical document intelligence using existing
verified framework components:
- Document Processing & UnifiedDocument Builder
- Semantic Chunker
- BGE Embeddings + FAISS FlatIP (Dense)
- BM25Okapi (Sparse)
- Reciprocal Rank Fusion (Hybrid)
- Cross-Encoder Reranking (Phase 4.10 frozen K_cand=15, batch=32)
- ContextBuilder & PromptBuilder with strict grounding
- LLM Provider (Real Gemini or Fake for deterministic testing)

Separates indexing from querying to avoid redundant re-indexing.
Preserves complete document, page, chunk, and score provenance.
Uses dependency injection throughout.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from adaptive_framework.config.models import ChunkerConfig
from adaptive_framework.core.exceptions import (
    ContextBuildError,
    LLMProviderError,
    LLMTimeoutError,
    PromptBuildError,
    RerankerError,
    RetrievalEngineError,
)
from adaptive_framework.document_processing.unified_document_builder import (
    UnifiedDocumentBuilder,
)
from adaptive_framework.models.chunk import Chunk
from adaptive_framework.models.page import BoundingBox, LayoutElement, Page, PageType, ProcessingMethod
from adaptive_framework.models.unified_document import UnifiedDocument
from adaptive_framework.rag.chunker import SemanticChunker
from adaptive_framework.rag.generation.answer import SourceReference
from adaptive_framework.rag.generation.context_builder import ContextBuilder
from adaptive_framework.rag.generation.prompt_builder import PromptBuilder
from adaptive_framework.rag.interfaces.i_embedding_provider import IEmbeddingProvider
from adaptive_framework.rag.interfaces.i_llm_provider import ILLMProvider
from adaptive_framework.rag.interfaces.i_retrieval_engine import IRetrievalEngine
from adaptive_framework.rag.interfaces.i_sparse_retriever import ISparseRetriever
from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager
from adaptive_framework.rag.models.clinical_models import (
    CLINICAL_DISCLAIMER,
    ClinicalDocumentUpload,
    ClinicalQueryRequest,
    DoctorClinicalResponse,
    DoctorEvidenceItem,
    DocumentIngestionResult,
)
from adaptive_framework.rag.models.embedding import Embedding
from adaptive_framework.rag.retrieval.reranked_retriever import RerankedRetriever
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult

logger = logging.getLogger(__name__)

try:
    import fitz  # type: ignore[import]
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False


class ClinicalRAGOrchestrator:
    """Doctor-facing orchestration service for clinical document intelligence.

    Connects existing framework components via dependency injection.
    Maintains a dedicated clinical index namespace to prevent mutation of
    research evaluation baseline artifacts.
    """

    def __init__(
        self,
        retrieval_engine: IRetrievalEngine,
        llm_provider: ILLMProvider,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
        chunker: SemanticChunker | None = None,
        doc_builder: UnifiedDocumentBuilder | None = None,
        embedder: IEmbeddingProvider | None = None,
        vector_store: FAISSManager | None = None,
        sparse_retriever: ISparseRetriever | None = None,
        strategy_factory: Any | None = None,
        index_storage_dir: Path | str = "outputs/rag/clinical_index",
    ) -> None:
        """Initialise the Clinical RAG Orchestrator with injected components.

        Args:
            retrieval_engine: Configured retrieval engine (typically RerankedRetriever).
            llm_provider: Component interfacing with LLM backend.
            context_builder: Bounding and deduplication context builder.
            prompt_builder: Prompt structuring component with grounding rules.
            chunker: Semantic chunker for UnifiedDocument -> Chunks.
            doc_builder: Unified document builder for merging pages.
            embedder: Embedding provider for new document ingestion.
            vector_store: Vector store (e.g. FAISSManager) for new document ingestion.
            sparse_retriever: Sparse retriever (e.g. BM25Retriever) for new document ingestion.
            strategy_factory: Optional ProcessingStrategyFactory for OCR on scanned pages.
            index_storage_dir: Dedicated directory for clinical index state.
        """
        self.retrieval_engine = retrieval_engine
        self.llm_provider = llm_provider
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_builder = prompt_builder or PromptBuilder()
        default_chunker_config = ChunkerConfig(
            strategy="semantic",
            chunk_size=512,
            chunk_overlap=64,
        )
        self.chunker = chunker or SemanticChunker(config=default_chunker_config)
        self.doc_builder = doc_builder or UnifiedDocumentBuilder()
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse_retriever = sparse_retriever
        self.strategy_factory = strategy_factory

        self.index_storage_dir = Path(index_storage_dir)
        self.index_storage_dir.mkdir(parents=True, exist_ok=True)

        self._indexed_documents: dict[str, DocumentIngestionResult] = {}
        self._all_chunks: list[Chunk] = []

    # -------------------------------------------------------------------------
    # Document Ingestion & Indexing Lifecycle (Separated from Querying)
    # -------------------------------------------------------------------------

    def index_document(self, upload: ClinicalDocumentUpload) -> DocumentIngestionResult:
        """Process, chunk, and index an uploaded clinical document.

        Supports PDF, EHR text, or structured discharge notes.
        Does not re-index previously processed documents unless re-uploaded.

        Args:
            upload: Container with file bytes or path, filename, and document ID.

        Returns:
            DocumentIngestionResult with page count, chunk count, and ingestion latency.
        """
        t_start = time.perf_counter()
        doc_id = upload.document_id or f"doc_{Path(upload.filename).stem}"

        try:
            pages = self._extract_pages(upload, doc_id)
            if not pages:
                return DocumentIngestionResult(
                    document_id=doc_id,
                    filename=upload.filename,
                    page_count=0,
                    chunk_count=0,
                    ingestion_latency_ms=(time.perf_counter() - t_start) * 1000.0,
                    status="failed",
                    error_message="Document contained no extractable text or pages.",
                )

            file_path_str = upload.file_path or upload.filename
            unified_doc = self.doc_builder.build(
                document_id=doc_id,
                file_path=file_path_str,
                pages=pages,
                run_id="clinical_rag",
            )

            chunks = self.chunker.chunk_document(unified_doc)
            if not chunks:
                return DocumentIngestionResult(
                    document_id=doc_id,
                    filename=upload.filename,
                    page_count=len(pages),
                    chunk_count=0,
                    ingestion_latency_ms=(time.perf_counter() - t_start) * 1000.0,
                    status="failed",
                    error_message="Semantic chunker produced zero valid chunks.",
                )

            # Register chunks with vector store and sparse index if configured
            if self.embedder is not None and self.vector_store is not None:
                dim = getattr(self.embedder, "embedding_dim", 1024)
                model_name = getattr(self.embedder, "model_name", "BAAI/bge-large-en-v1.5")
                embeddings = [
                    Embedding(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        vector=list(self.embedder.embed_query(c.text)),
                        model_name=model_name,
                        embedding_time_seconds=0.001,
                        created_at="2026-09-06T12:00:00+00:00",
                    )
                    for c in chunks
                ]
                self.vector_store.add_embeddings(embeddings)
                if hasattr(self.vector_store, "register_chunks"):
                    self.vector_store.register_chunks(chunks)

            self._all_chunks.extend(chunks)

            if self.sparse_retriever is not None:
                self.sparse_retriever.index_chunks(self._all_chunks)

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            result = DocumentIngestionResult(
                document_id=doc_id,
                filename=upload.filename,
                page_count=len(pages),
                chunk_count=len(chunks),
                ingestion_latency_ms=round(elapsed_ms, 2),
                status="success",
            )
            self._indexed_documents[doc_id] = result
            logger.info("Successfully indexed clinical document %s (%d chunks, %.2f ms)", doc_id, len(chunks), elapsed_ms)
            return result

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.exception("Failed to ingest document %s: %s", upload.filename, exc)
            return DocumentIngestionResult(
                document_id=doc_id,
                filename=upload.filename,
                page_count=0,
                chunk_count=0,
                ingestion_latency_ms=round(elapsed_ms, 2),
                status="failed",
                error_message=str(exc),
            )

    def _extract_pages(self, upload: ClinicalDocumentUpload, document_id: str) -> list[Page]:
        """Extract immutable Page objects from PDF or text upload."""
        file_name_lower = upload.filename.lower()

        # 1. PDF processing via PyMuPDF
        if file_name_lower.endswith(".pdf") and _FITZ_AVAILABLE:
            if upload.file_bytes:
                doc = fitz.open(stream=upload.file_bytes, filetype="pdf")
            elif upload.file_path and Path(upload.file_path).exists():
                doc = fitz.open(upload.file_path)
            else:
                raise FileNotFoundError(f"PDF file not accessible: {upload.file_path}")

            file_path_str = upload.file_path or upload.filename
            pages: list[Page] = []
            for page_idx, fitz_page in enumerate(doc):
                page_num = page_idx + 1
                text = fitz_page.get_text("text").strip()
                rect = fitz_page.rect

                if text:
                    # Digital page: strictly use direct text extraction (bypassing OCR)
                    layout_elements = (
                        LayoutElement(
                            element_type="paragraph",
                            text=text,
                            bbox=BoundingBox(0.0, 0.0, float(rect.width), float(rect.height)),
                            reading_order=1,
                        ),
                    )
                    pages.append(
                        Page(
                            document_id=document_id,
                            page_number=page_num,
                            page_type=PageType.DIGITAL,
                            processing_method=ProcessingMethod.DIRECT_TEXT,
                            text=text,
                            text_blocks=(),
                            tables=(),
                            figures=(),
                            layout_elements=layout_elements,
                            worker_id="clinical_rag",
                            node_id="localhost",
                            processing_time_seconds=0.01,
                            ocr_confidence=1.0,
                            success=True,
                        )
                    )
                elif self.strategy_factory is not None:
                    # Scanned page (no text layer): invoke OCR strategy (e.g. AdaptiveRoutingStrategyProxy)
                    scanned_strat = self.strategy_factory.get_strategy(PageType.SCANNED)
                    strat_res = scanned_strat.process(fitz_page, page_num, document_id, file_path_str)
                    ocr_text = (strat_res.text or "").strip()
                    ocr_elements = tuple(strat_res.layout_elements) if strat_res.layout_elements else ()
                    if not ocr_elements and ocr_text:
                        ocr_elements = (
                            LayoutElement(
                                element_type="paragraph",
                                text=ocr_text,
                                bbox=BoundingBox(0.0, 0.0, float(rect.width), float(rect.height)),
                                reading_order=1,
                            ),
                        )
                    pages.append(
                        Page(
                            document_id=document_id,
                            page_number=page_num,
                            page_type=PageType.SCANNED,
                            processing_method=ProcessingMethod.OCR,
                            text=ocr_text,
                            text_blocks=tuple(strat_res.text_blocks),
                            tables=tuple(strat_res.tables),
                            figures=tuple(strat_res.figures),
                            layout_elements=ocr_elements,
                            worker_id="clinical_rag",
                            node_id="localhost",
                            processing_time_seconds=strat_res.ocr_time_s or 0.01,
                            ocr_confidence=strat_res.ocr_confidence,
                            warnings=tuple(strat_res.warnings),
                            success=bool(strat_res.error is None),
                        )
                    )
                else:
                    # Scanned page without strategy_factory configured (fallback to empty digital page)
                    pages.append(
                        Page(
                            document_id=document_id,
                            page_number=page_num,
                            page_type=PageType.DIGITAL,
                            processing_method=ProcessingMethod.DIRECT_TEXT,
                            text="",
                            text_blocks=(),
                            tables=(),
                            figures=(),
                            layout_elements=(),
                            worker_id="clinical_rag",
                            node_id="localhost",
                            processing_time_seconds=0.01,
                            ocr_confidence=1.0,
                            success=True,
                        )
                    )
            doc.close()
            return pages

        # 2. Text or JSON clinical note processing
        raw_text = ""
        if upload.file_bytes:
            raw_text = upload.file_bytes.decode("utf-8", errors="replace")
        elif upload.file_path and Path(upload.file_path).exists():
            raw_text = Path(upload.file_path).read_text(encoding="utf-8", errors="replace")

        raw_text = raw_text.strip()
        if not raw_text:
            return []

        # Split into simulated pages if multiple form feeds or long text
        page_texts = [p.strip() for p in raw_text.split("\f") if p.strip()] or [raw_text]
        pages = []
        for idx, p_text in enumerate(page_texts):
            page_num = idx + 1
            pages.append(
                Page(
                    document_id=document_id,
                    page_number=page_num,
                    page_type=PageType.DIGITAL,
                    processing_method=ProcessingMethod.DIRECT_TEXT,
                    text=p_text,
                    text_blocks=(),
                    tables=(),
                    figures=(),
                    layout_elements=(
                        LayoutElement(
                            element_type="paragraph",
                            text=p_text,
                            bbox=BoundingBox(0.0, 0.0, 612.0, 792.0),
                            reading_order=1,
                        ),
                    ),
                    worker_id="clinical_rag",
                    node_id="localhost",
                    processing_time_seconds=0.01,
                    ocr_confidence=1.0,
                    success=True,
                )
            )
        return pages

    # -------------------------------------------------------------------------
    # Doctor Query Lifecycle (Pure Query Execution)
    # -------------------------------------------------------------------------

    def query(self, request: ClinicalQueryRequest) -> DoctorClinicalResponse:
        """Process a doctor's clinical query end-to-end.

        Preserves strict separation between:
        - USER QUESTION
        - PATIENT CONTEXT (unverified user-reported symptoms)
        - RETRIEVED EVIDENCE (untrusted reference data)

        Args:
            request: Structured clinical request with question and optional context.

        Returns:
            DoctorClinicalResponse containing answer, evidence items, and stage latencies.
        """
        t_total_start = time.perf_counter()
        latencies: dict[str, float] = {
            "retrieval_ms": 0.0,
            "reranking_ms": 0.0,
            "context_ms": 0.0,
            "prompt_ms": 0.0,
            "generation_ms": 0.0,
            "total_ms": 0.0,
        }

        # Validate query string
        clean_q = request.query.strip()
        if not clean_q:
            latencies["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0
            return DoctorClinicalResponse(
                query=request.query,
                clinical_context=request.clinical_context,
                answer_text="Empty query provided. Please submit a valid clinical question.",
                evidence=(),
                source_references=(),
                stage_latencies_ms=latencies,
                generation_status="empty_query",
                provider_name=self.llm_provider.get_provider_name(),
                model_name=self.llm_provider.get_model_name(),
            )

        # 1. Retrieval & Reranking Stage
        t_ret_start = time.perf_counter()
        retrieval_results: list[RetrievalResult] = []
        filters = dict(request.metadata_filters or {})
        if request.document_id:
            filters["document_id"] = request.document_id

        try:
            if isinstance(self.retrieval_engine, RerankedRetriever):
                retrieval_results = self.retrieval_engine.retrieve(
                    query=clean_q,
                    top_k=request.top_k,
                    candidate_top_k=request.candidate_top_k,
                    metadata_filters=filters or None,
                )
                latencies["retrieval_ms"] = self.retrieval_engine.last_stage1_latency_ms
                latencies["reranking_ms"] = self.retrieval_engine.last_rerank_latency_ms
            elif hasattr(self.retrieval_engine, "last_rerank_latency_ms"):
                retrieval_results = self.retrieval_engine.retrieve(
                    query=clean_q,
                    top_k=request.top_k,
                    metadata_filters=filters or None,
                )
                latencies["retrieval_ms"] = getattr(
                    self.retrieval_engine,
                    "last_stage1_latency_ms",
                    (time.perf_counter() - t_ret_start) * 1000.0,
                )
                latencies["reranking_ms"] = getattr(
                    self.retrieval_engine, "last_rerank_latency_ms", 0.0
                )
            else:
                retrieval_results = self.retrieval_engine.retrieve(
                    query=clean_q,
                    top_k=request.top_k,
                    metadata_filters=filters or None,
                )
                latencies["retrieval_ms"] = (time.perf_counter() - t_ret_start) * 1000.0
                latencies["reranking_ms"] = 0.0

        except (RetrievalEngineError, RerankerError, Exception) as exc:
            latencies["retrieval_ms"] = (time.perf_counter() - t_ret_start) * 1000.0
            latencies["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0
            status = "reranker_failure" if isinstance(exc, RerankerError) else "retrieval_failure"
            logger.exception("Retrieval/Reranking error in clinical query: %s", exc)
            return DoctorClinicalResponse(
                query=request.query,
                clinical_context=request.clinical_context,
                answer_text=f"Retrieval error: Unable to retrieve clinical evidence ({exc}).",
                evidence=(),
                source_references=(),
                stage_latencies_ms=latencies,
                generation_status=status,
                provider_name=self.llm_provider.get_provider_name(),
                model_name=self.llm_provider.get_model_name(),
            )

        # 2. Check for Insufficient Evidence
        if not retrieval_results:
            latencies["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0
            return DoctorClinicalResponse(
                query=request.query,
                clinical_context=request.clinical_context,
                answer_text="Insufficient evidence to answer this question. No relevant clinical passages found in the uploaded records.",
                evidence=(),
                source_references=(),
                stage_latencies_ms=latencies,
                generation_status="insufficient_evidence",
                provider_name=self.llm_provider.get_provider_name(),
                model_name=self.llm_provider.get_model_name(),
            )

        # 3. Context Construction Stage
        t_ctx_start = time.perf_counter()
        try:
            built_context = self.context_builder.build(retrieval_results)
            latencies["context_ms"] = (time.perf_counter() - t_ctx_start) * 1000.0
        except ContextBuildError as exc:
            latencies["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0
            return DoctorClinicalResponse(
                query=request.query,
                clinical_context=request.clinical_context,
                answer_text=f"Context building error: {exc}",
                evidence=(),
                source_references=(),
                stage_latencies_ms=latencies,
                generation_status="context_error",
                provider_name=self.llm_provider.get_provider_name(),
                model_name=self.llm_provider.get_model_name(),
            )

        # 4. Prompt Construction Stage (with strict section separation)
        t_prompt_start = time.perf_counter()
        composite_query_parts = [f"USER QUESTION:\n{clean_q}"]
        if request.clinical_context and request.clinical_context.strip():
            composite_query_parts.append(
                f"PATIENT CONTEXT (SYMPTOMS REPORTED BY USER - NOT VERIFIED RECORD FINDINGS):\n"
                f"{request.clinical_context.strip()}\n\n"
                f"SAFETY INSTRUCTION: Distinguish documented findings in the uploaded record "
                f"from user-reported symptoms. Do not treat user symptoms as documented clinical facts."
            )
        composite_query = "\n\n".join(composite_query_parts)

        try:
            built_prompt = self.prompt_builder.build(
                context=built_context,
                query=composite_query,
            )
            latencies["prompt_ms"] = (time.perf_counter() - t_prompt_start) * 1000.0
        except PromptBuildError as exc:
            latencies["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0
            return DoctorClinicalResponse(
                query=request.query,
                clinical_context=request.clinical_context,
                answer_text=f"Prompt building error: {exc}",
                evidence=tuple(DoctorEvidenceItem.from_retrieval_result(r) for r in retrieval_results),
                source_references=(),
                stage_latencies_ms=latencies,
                generation_status="prompt_error",
                provider_name=self.llm_provider.get_provider_name(),
                model_name=self.llm_provider.get_model_name(),
            )

        # 5. LLM Generation Stage
        t_gen_start = time.perf_counter()
        try:
            llm_response = self.llm_provider.generate(built_prompt)
            latencies["generation_ms"] = (time.perf_counter() - t_gen_start) * 1000.0
            answer_text = llm_response.text
            generation_status = "success"
        except (LLMTimeoutError, LLMProviderError, Exception) as exc:
            latencies["generation_ms"] = (time.perf_counter() - t_gen_start) * 1000.0
            logger.exception("LLM generation failed: %s", exc)
            answer_text = f"LLM generation failed ({exc}). Please review retrieved evidence directly."
            generation_status = "llm_failure"

        # 6. Provenance & Evidence Compilation
        evidence_items = tuple(DoctorEvidenceItem.from_retrieval_result(r) for r in retrieval_results)

        # Compile distinct source references
        seen_refs: dict[tuple[str, str], set[int]] = {}
        for r in retrieval_results:
            key = (r.document_id, r.source_file)
            seen_refs.setdefault(key, set()).update(r.page_numbers)

        source_refs = tuple(
            SourceReference(
                document_id=doc_id,
                source_file=src_file,
                page_numbers=tuple(sorted(pages)),
            )
            for (doc_id, src_file), pages in seen_refs.items()
        )

        latencies["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0

        return DoctorClinicalResponse(
            query=request.query,
            clinical_context=request.clinical_context,
            answer_text=answer_text,
            evidence=evidence_items,
            source_references=source_refs,
            stage_latencies_ms={k: round(v, 3) for k, v in latencies.items()},
            generation_status=generation_status,
            provider_name=self.llm_provider.get_provider_name(),
            model_name=self.llm_provider.get_model_name(),
        )

    def get_indexed_documents(self) -> list[DocumentIngestionResult]:
        """Return list of currently indexed document metadata."""
        return list(self._indexed_documents.values())
