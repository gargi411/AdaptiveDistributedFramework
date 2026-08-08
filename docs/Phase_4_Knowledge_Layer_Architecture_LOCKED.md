# Phase 4 — Knowledge Layer Architecture (LOCKED)

> **Status: LOCKED ✅**
> Document prepared for review before implementation begins.
> Do NOT modify previous phases. Do NOT modify the scheduler, coordinator, or document processing engine.
> Phase 4 begins ONLY after an immutable `UnifiedDocument` has been produced.

---

## Table of Contents

1. [Research Objective](#1-research-objective)
2. [Role of Phase 4 within the Complete Framework](#2-role-of-phase-4-within-the-complete-framework)
3. [Architecture Pipeline Diagram](#3-architecture-pipeline-diagram)
4. [Folder Structure](#4-folder-structure)
5. [Package Layout](#5-package-layout)
6. [Data Models](#6-data-models)
7. [Interfaces](#7-interfaces)
8. [Batch-by-Batch Implementation Plan](#8-batch-by-batch-implementation-plan)
9. [Semantic Chunking Architecture](#9-semantic-chunking-architecture)
10. [Metadata Enrichment](#10-metadata-enrichment)
11. [Embedding Architecture](#11-embedding-architecture)
12. [FAISS Architecture](#12-faiss-architecture)
13. [Query Planner](#13-query-planner)
14. [Hybrid Retriever](#14-hybrid-retriever)
15. [Context Builder](#15-context-builder)
16. [Prompt Builder](#16-prompt-builder)
17. [LLM Provider Architecture](#17-llm-provider-architecture)
18. [Answer Generator](#18-answer-generator)
19. [Evaluation Architecture](#19-evaluation-architecture)
20. [Runtime Instrumentation](#20-runtime-instrumentation)
21. [Platform Compatibility](#21-platform-compatibility)
22. [Testing Strategy](#22-testing-strategy)
23. [Verification Plan](#23-verification-plan)
24. [Future Work](#24-future-work)

---

## 1. Research Objective

The primary research contribution of this project is **Adaptive Distributed Parallel Processing** — the scheduling, coordination, and parallel execution of large-scale biomedical document processing tasks across heterogeneous compute nodes using an intelligent, low-overhead workload scheduler.

Phase 4 is the **Knowledge Layer** that demonstrates the end-to-end value of the distributed framework. It transforms `UnifiedDocument` objects — produced by the distributed processing pipeline — into a queryable biomedical knowledge base using a **Retrieval-Augmented Generation (RAG)** pipeline.

**The RAG pipeline is NOT the research contribution. It is the application layer.**

The research contribution is measured by:
- Speedup achieved by the distributed scheduler over single-node execution.
- Work-stealing efficiency and idle-worker elimination.
- Scheduler overhead staying below 1% of total execution time.
- Failure recovery transparency and resilience.

Phase 4 proves that the distributed framework produces structured, high-quality `UnifiedDocument` objects suitable for downstream intelligent applications. It answers the question: *"What can we build with what the framework produces?"*

---

## 2. Role of Phase 4 within the Complete Framework

### Complete Phase Sequence

```
Phase 1 — Software Foundation
    ConfigManager, Interfaces, DI Container, Data Models, Logging

Phase 2 — Document Processing Engine
    OCR, Layout Analysis, Table Extraction, Figure Detection

Phase 3 — Adaptive Scheduler + Distributed Coordinator
    Page-Count Partitioning, Work Stealing, Ray Actor Coordination,
    Heartbeat Monitor, Failure Recovery → UnifiedDocument (immutable)

Phase 4 — Knowledge Layer (THIS DOCUMENT)
    Semantic Chunking → Embedding → FAISS → Hybrid Retrieval → LLM → Answer

Phase 5 — Evaluation Engine
    Speedup, Throughput, Scheduler Overhead, RAG Recall/Precision

Phase 6 — Dashboard + Reporting
    Visualisation of distributed metrics + RAG performance
```

### Phase 4 Exact Entry Point

Phase 4 receives a fully constructed, **frozen** `UnifiedDocument` object. It never reads PDFs. It never calls the scheduler. It never interacts with Ray workers directly.

```
UnifiedDocument
    document_id         → Chunk metadata anchor
    pages               → Page-level text, tables, figures, layout_elements
    full_text           → Full document text (fallback)
    tables              → TableData (structured content)
    figures             → FigureData (captions, references)
    layout              → DocumentLayout (title, authors, sections)
    statistics          → Processing provenance
```

Phase 4 outputs:
- A populated FAISS index with embedded, metadata-enriched chunks.
- A query interface that accepts natural language biomedical questions.
- A structured `Answer` containing text, citations, confidence, and sources.

---

## 3. Architecture Pipeline Diagram

```
+==============================================================================+
|                   PHASE 3 OUTPUT (LOCKED — DO NOT MODIFY)                   |
|                                                                              |
|   PDF -> [Distributed Workers] -> [Coordinator Merge] -> UnifiedDocument    |
+================================================+=============================+
                                                 |
                                                 v
+==============================================================================+
|                     PHASE 4 — KNOWLEDGE LAYER                               |
|                                                                              |
|  +------------------------------------------------------------------------+  |
|  |  INGESTION PIPELINE  (offline / batch)                                 |  |
|  |                                                                        |  |
|  |  UnifiedDocument                                                       |  |
|  |       |                                                                |  |
|  |       v                                                                |  |
|  |  +---------------------+                                              |  |
|  |  |  SemanticChunker    |  <- Headings, Sections, Paragraphs,          |  |
|  |  |  (IChunkBuilder)    |     Tables, Figures, Captions                |  |
|  |  +--------+------------+                                              |  |
|  |           |  list[Chunk]                                              |  |
|  |           v                                                           |  |
|  |  +---------------------+                                              |  |
|  |  |  MetadataEnricher   |  <- document_id, page, heading, section,     |  |
|  |  |                     |     journal, year, source, figure, table     |  |
|  |  +--------+------------+                                              |  |
|  |           |  list[Chunk] with ChunkMetadata                          |  |
|  |           v                                                           |  |
|  |  +---------------------+                                              |  |
|  |  |  EmbeddingEngine    |  <- BAAI/bge-large-en-v1.5                  |  |
|  |  |  (Parallel, Batch)  |     Parallel via Distributed Framework       |  |
|  |  +--------+------------+                                              |  |
|  |           |  list[Embedding]                                          |  |
|  |           v                                                           |  |
|  |  +---------------------+                                              |  |
|  |  |  FAISSIndexManager  |  <- IVF index, incremental updates,          |  |
|  |  |                     |     persistence, deletion                    |  |
|  |  +---------------------+                                              |  |
|  +------------------------------------------------------------------------+  |
|                                                                              |
|  +------------------------------------------------------------------------+  |
|  |  QUERY PIPELINE  (online / interactive)                                |  |
|  |                                                                        |  |
|  |  User Question (str)                                                   |  |
|  |       |                                                                |  |
|  |       v                                                                |  |
|  |  +---------------------+                                              |  |
|  |  |  QueryPlanner       |  <- Classify intent: entity, summary,        |  |
|  |  |                     |     comparison, timeline, medication...       |  |
|  |  +--------+------------+                                              |  |
|  |           |  QueryPlan                                                |  |
|  |           v                                                           |  |
|  |  +---------------------+                                              |  |
|  |  |  HybridRetriever    |  <- Vector similarity + Metadata filtering   |  |
|  |  |                     |     BM25 sparse + ranking + deduplication    |  |
|  |  +--------+------------+                                              |  |
|  |           |  list[RetrievalResult]                                    |  |
|  |           v                                                           |  |
|  |  +---------------------+                                              |  |
|  |  |  ContextBuilder     |  <- Token budgeting, ordering,               |  |
|  |  |                     |     deduplication, citation preservation     |  |
|  |  +--------+------------+                                              |  |
|  |           |  PromptContext                                            |  |
|  |           v                                                           |  |
|  |  +---------------------+                                              |  |
|  |  |  PromptBuilder      |  <- Biomedical templates, safety,            |  |
|  |  |                     |     citation formatting                      |  |
|  |  +--------+------------+                                              |  |
|  |           |  str (final prompt)                                       |  |
|  |           v                                                           |  |
|  |  +---------------------+                                              |  |
|  |  |  LLMProvider        |  <- Llama 3.1 8B Instruct (default)          |  |
|  |  |  (ILLMProvider)     |     Qwen, DeepSeek (swappable interface)     |  |
|  |  +--------+------------+                                              |  |
|  |           |  raw LLM output                                           |  |
|  |           v                                                           |  |
|  |  +---------------------+                                              |  |
|  |  |  AnswerGenerator    |  <- Final Answer, Citations, Confidence,     |  |
|  |  |                     |     Sources, EvaluationMetrics               |  |
|  |  +---------------------+                                              |  |
|  +------------------------------------------------------------------------+  |
+==============================================================================+
```

---

## 4. Folder Structure

The entire Phase 4 Knowledge Layer lives within the existing `rag/` package. No new top-level packages are created. No existing packages are modified.

```
src/adaptive_framework/rag/
├── __init__.py
├── README.md
├── models/
│   ├── __init__.py
│   ├── chunk.py                        [Chunk, ChunkMetadata]
│   ├── embedding.py                    [Embedding]
│   ├── retrieval.py                    [RetrievalResult, QueryPlan]
│   ├── context.py                      [PromptContext, Citation]
│   └── answer.py                       [Answer, RAGEvaluationMetrics]
├── interfaces/
│   ├── __init__.py
│   ├── i_embedding_provider.py
│   ├── i_retriever.py
│   ├── i_llm_provider.py
│   ├── i_chunk_builder.py
│   ├── i_faiss_vector_store.py
│   ├── i_query_planner.py
│   ├── i_prompt_builder.py
│   └── i_answer_generator.py
├── chunking/
│   ├── __init__.py
│   ├── chunk_builder.py
│   ├── semantic_chunker.py
│   ├── table_chunker.py
│   ├── figure_chunker.py
│   └── chunk_validator.py
├── metadata/
│   ├── __init__.py
│   ├── metadata_enricher.py
│   └── metadata_filter.py
├── embedding/
│   ├── __init__.py
│   ├── embedding_provider.py
│   ├── bge_embedder.py
│   ├── embedding_cache.py
│   ├── batch_embedding_worker.py
│   └── embedding_metrics.py
├── vector_store/
│   ├── __init__.py
│   ├── faiss_manager.py
│   ├── faiss_index_builder.py
│   ├── faiss_persistence.py
│   └── faiss_metrics.py
├── retrieval/
│   ├── __init__.py
│   ├── query_planner.py
│   ├── vector_retriever.py
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   └── retrieval_metrics.py
├── context/
│   ├── __init__.py
│   ├── context_builder.py
│   └── citation_tracker.py
├── prompt/
│   ├── __init__.py
│   ├── prompt_builder.py
│   ├── biomedical_template.py
│   ├── prompt_safety.py
│   └── prompt_metrics.py
├── llm/
│   ├── __init__.py
│   ├── llm_provider.py
│   ├── llama_provider.py
│   ├── qwen_provider.py
│   ├── deepseek_provider.py
│   ├── llm_factory.py
│   └── llm_metrics.py
├── answer/
│   ├── __init__.py
│   ├── answer_generator.py
│   └── confidence_scorer.py
├── evaluation/
│   ├── __init__.py
│   ├── rag_evaluator.py
│   ├── retrieval_evaluator.py
│   └── generation_evaluator.py
├── ingestion_pipeline.py
└── query_pipeline.py
```

**Test tree (mirrors existing tests/ structure):**

```
tests/
├── unit/rag/
│   ├── test_semantic_chunker.py
│   ├── test_table_chunker.py
│   ├── test_figure_chunker.py
│   ├── test_metadata_enricher.py
│   ├── test_bge_embedder.py
│   ├── test_embedding_cache.py
│   ├── test_faiss_manager.py
│   ├── test_query_planner.py
│   ├── test_hybrid_retriever.py
│   ├── test_context_builder.py
│   ├── test_prompt_builder.py
│   ├── test_answer_generator.py
│   └── test_rag_evaluator.py
├── integration/rag/
│   ├── test_ingestion_pipeline.py
│   └── test_query_pipeline.py
└── performance/rag/
    ├── test_embedding_throughput.py
    └── test_retrieval_latency.py
```

---

## 5. Package Layout

### Design Patterns Applied

| Pattern | Where Used | Why |
|---|---|---|
| **Interface-first** | All major subsystems | Enables swapping implementations without pipeline changes |
| **Factory Pattern** | `llm_factory.py`, `faiss_index_builder.py` | Decouple object creation from usage |
| **Strategy Pattern** | `IChunkBuilder`, `IEmbeddingProvider`, `ILLMProvider`, `IRetriever` | Each step is independently replaceable |
| **Builder Pattern** | `ContextBuilder`, `PromptBuilder` | Assemble complex objects step-by-step |
| **Adapter Pattern** | `bge_embedder.py`, `llama_provider.py`, `faiss_manager.py` | Wrap external libraries behind internal interfaces |
| **Provider Pattern** | `ILLMProvider` + concrete providers | Support multiple LLM backends without code changes |
| **Dependency Injection** | All modules receive dependencies via constructor | Testability, no global singletons in Phase 4 |

### Public API (rag/__init__.py exports)

```
KnowledgeIngestionPipeline   — Ingests UnifiedDocument -> FAISS
KnowledgeQueryPipeline       — Accepts question -> returns Answer
RAGEvaluator                 — Runs full evaluation suite
Chunk                        — Primary data model
Answer                       — Primary output model
```

---

## 6. Data Models

All Phase 4 data models are immutable frozen dataclasses consistent with `UnifiedDocument`, `Page`, and all Phase 1/3 models.

### 6.1 ChunkMetadata

```python
@dataclass(frozen=True)
class ChunkMetadata:
    """Structured metadata attached to every Chunk.

    Provides full provenance for citation, retrieval filtering,
    and answer generation. All fields are optional except document_id.
    """
    # Source provenance
    document_id: str                  # UnifiedDocument.document_id
    chunk_index: int                  # 0-indexed within document
    processing_run_id: str            # UnifiedDocument.run_id
    chunk_created_at: str             # ISO 8601 UTC

    # Structural provenance
    page_number: int | None           # 1-indexed; None for multi-page spans
    heading: str | None               # Nearest ancestor heading
    section: str | None               # Top-level section (e.g. "Methods")
    subsection: str | None            # Second-level heading

    # Document-level metadata
    source_file: str                  # UnifiedDocument.file_path
    title: str | None                 # DocumentLayout.title
    authors: tuple[str, ...]          # DocumentLayout.authors
    language: str | None              # ISO 639-1 code

    # Biomedical-specific
    journal: str | None               # Extracted journal name
    year: int | None                  # Publication year

    # Table and figure flags
    is_table: bool                    # True if from TableData
    table_id: str | None              # TableData.table_id
    is_figure: bool                   # True if from FigureData
    figure_id: str | None             # FigureData.figure_id
    caption: str | None               # Table or figure caption

    # Processing statistics
    ocr_confidence: float             # Avg OCR confidence from source pages
    char_count: int                   # Character count
    token_estimate: int               # chars // 4
```

### 6.2 Chunk

```python
@dataclass(frozen=True)
class Chunk:
    """A single semantic unit of text extracted from a UnifiedDocument.

    The primary unit processed by the embedding engine and stored in FAISS.
    Every Chunk carries full metadata for retrieval, citation, and filtering.
    """
    chunk_id: str                     # UUID4, globally unique
    text: str                         # Actual text content
    metadata: ChunkMetadata           # Full provenance and context
```

### 6.3 Embedding

```python
@dataclass(frozen=True)
class Embedding:
    """A dense vector representation of a Chunk.

    Produced by the EmbeddingEngine. Stored alongside Chunk in FAISS.
    """
    chunk_id: str                     # Matches Chunk.chunk_id
    document_id: str                  # Matches ChunkMetadata.document_id
    vector: tuple[float, ...]         # 1024-dimensional (bge-large-en-v1.5)
    model_name: str                   # "BAAI/bge-large-en-v1.5"
    embedding_time_seconds: float
    created_at: str                   # ISO 8601 UTC
```

### 6.4 RetrievalResult

```python
@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieved Chunk with relevance scores and retrieval method."""
    chunk: Chunk
    vector_score: float               # Cosine similarity [0, 1]
    bm25_score: float                 # BM25 sparse score [0, inf)
    hybrid_score: float               # Weighted combination
    rank: int                         # Final rank (1-indexed)
    retrieval_method: str             # "vector" | "bm25" | "hybrid"
```

### 6.5 Citation

```python
@dataclass(frozen=True)
class Citation:
    """A formatted bibliographic citation derived from ChunkMetadata."""
    citation_id: str                  # "[1]", "[2]", ...
    chunk_id: str
    document_id: str
    title: str | None
    authors: tuple[str, ...]
    journal: str | None
    year: int | None
    page_number: int | None
    section: str | None
    heading: str | None
    source_file: str
    inline_ref: str                   # Inline reference text e.g. "[1]"
```

### 6.6 PromptContext

```python
@dataclass(frozen=True)
class PromptContext:
    """Assembled context passed to the PromptBuilder."""
    question: str
    query_intent: str
    context_chunks: tuple[Chunk, ...]
    citations: tuple[Citation, ...]
    total_tokens: int
    token_budget_used: int
    token_budget_total: int
    retrieval_metadata: dict
```

### 6.7 Answer

```python
@dataclass(frozen=True)
class Answer:
    """Complete output of the RAG query pipeline."""
    question: str
    answer_text: str
    citations: tuple[Citation, ...]
    retrieved_chunks: tuple[Chunk, ...]
    confidence_score: float           # [0.0, 1.0]
    query_intent: str
    model_name: str
    embedding_model: str
    retrieval_method: str
    total_latency_seconds: float
    embedding_time_seconds: float
    retrieval_time_seconds: float
    generation_time_seconds: float
    created_at: str
```

### 6.8 RAGEvaluationMetrics

```python
@dataclass(frozen=True)
class RAGEvaluationMetrics:
    """Full evaluation metrics for one or more RAG pipeline runs."""
    # Retrieval quality
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    precision_at_10: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    # Latency (seconds, averaged over evaluated queries)
    avg_total_latency: float
    avg_embedding_time: float
    avg_retrieval_time: float
    avg_generation_time: float
    p95_total_latency: float
    p99_total_latency: float
    # Ingestion metrics
    total_chunks_indexed: int
    total_documents_indexed: int
    avg_embedding_throughput_chunks_per_sec: float
    index_build_time_seconds: float
    # Metadata
    evaluation_timestamp: str
    num_queries_evaluated: int
    llm_model: str
    embedding_model: str
```

### 6.9 QueryPlan

```python
@dataclass(frozen=True)
class QueryPlan:
    """Structured plan produced by QueryPlanner for one user question."""
    question: str
    intent: str
    top_k: int
    metadata_filters: dict
    use_vector: bool
    use_bm25: bool
    vector_weight: float
    bm25_weight: float
    require_table: bool
    require_figure: bool
    year_range: tuple[int, int] | None
    section_hint: str | None
```

---

## 7. Interfaces

All interfaces use Python abstract base classes (ABC). They extend existing Phase 1 interfaces where applicable.

### 7.1 IEmbeddingProvider

```python
class IEmbeddingProvider(ABC):
    """Abstract interface for text embedding models.
    
    Extends Phase 1 IEmbedder to operate on Chunk objects
    carrying full ChunkMetadata.
    """
    @abstractmethod
    def initialize(self) -> None: ...

    @abstractmethod
    def embed_chunks(self, chunks: list[Chunk]) -> list[Embedding]: ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]: ...
    # Note: BGE uses asymmetric encoding.
    # Query uses prefix; chunks do not.

    @abstractmethod
    def get_embedding_dim(self) -> int: ...

    @abstractmethod
    def get_model_name(self) -> str: ...

    @abstractmethod
    def get_metrics(self) -> dict[str, float]: ...

    @abstractmethod
    def shutdown(self) -> None: ...
```

### 7.2 IRetriever

```python
class IRetriever(ABC):
    """Abstract interface for any retrieval strategy."""
    @abstractmethod
    def retrieve(
        self,
        query_embedding: list[float],
        query_text: str,
        plan: QueryPlan,
    ) -> list[RetrievalResult]: ...

    @abstractmethod
    def get_strategy_name(self) -> str: ...

    @abstractmethod
    def get_metrics(self) -> dict[str, float]: ...
```

### 7.3 ILLMProvider

```python
class ILLMProvider(ABC):
    """Abstract interface for LLM backends.
    
    Any model (Llama, Qwen, DeepSeek, GPT) plugs in here
    without changing AnswerGenerator or PromptBuilder.
    """
    @abstractmethod
    def initialize(self) -> None: ...

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str: ...

    @abstractmethod
    def get_model_name(self) -> str: ...

    @abstractmethod
    def get_context_window(self) -> int: ...

    @abstractmethod
    def get_metrics(self) -> dict[str, float]: ...

    @abstractmethod
    def shutdown(self) -> None: ...
```

### 7.4 IChunkBuilder

```python
class IChunkBuilder(ABC):
    """Abstract interface for converting UnifiedDocument into Chunks."""
    @abstractmethod
    def build_chunks(self, document: UnifiedDocument) -> list[Chunk]: ...

    @abstractmethod
    def get_strategy_name(self) -> str: ...

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]: ...
```

### 7.5 IFAISSVectorStore

Extends the existing Phase 1 `IVectorStore` interface:

```python
class IFAISSVectorStore(IVectorStore):
    """Extends IVectorStore with FAISS-specific operations."""
    @abstractmethod
    def add_embeddings(self, embeddings: list[Embedding]) -> None: ...

    @abstractmethod
    def search_with_metadata(
        self,
        query_embedding: list[float],
        top_k: int,
        metadata_filters: dict,
    ) -> list[RetrievalResult]: ...

    @abstractmethod
    def delete_document(self, document_id: str) -> int: ...
    # Returns number of deleted chunks

    @abstractmethod
    def persist(self, index_path: str) -> None: ...

    @abstractmethod
    def load(self, index_path: str) -> None: ...

    @abstractmethod
    def get_index_stats(self) -> dict[str, Any]: ...
```

### 7.6 IQueryPlanner

```python
class IQueryPlanner(ABC):
    """Abstract interface for query intent classification and plan generation."""
    @abstractmethod
    def plan(self, question: str) -> QueryPlan: ...

    @abstractmethod
    def get_supported_intents(self) -> list[str]: ...
```

### 7.7 IPromptBuilder

```python
class IPromptBuilder(ABC):
    """Abstract interface for building LLM prompts from PromptContext."""
    @abstractmethod
    def build(self, context: PromptContext) -> str: ...

    @abstractmethod
    def get_template_name(self) -> str: ...

    @abstractmethod
    def estimate_tokens(self, context: PromptContext) -> int: ...
```

### 7.8 IAnswerGenerator

```python
class IAnswerGenerator(ABC):
    """Abstract interface for the final answer generation step."""
    @abstractmethod
    def generate(
        self,
        context: PromptContext,
        retrieval_results: list[RetrievalResult],
    ) -> Answer: ...
```

---

## 8. Batch-by-Batch Implementation Plan

Phase 4 is split into **5 implementation batches**. Each batch is independently testable before the next begins.

---

### Batch 4.1 — Data Models and Interfaces

**Purpose:** Establish all Phase 4 data contracts before any implementation begins.

**Files (all NEW):**
```
src/adaptive_framework/rag/models/
    __init__.py, chunk.py, embedding.py, retrieval.py, context.py, answer.py
src/adaptive_framework/rag/interfaces/
    __init__.py, i_embedding_provider.py, i_retriever.py, i_llm_provider.py,
    i_chunk_builder.py, i_faiss_vector_store.py, i_query_planner.py,
    i_prompt_builder.py, i_answer_generator.py
```

**Responsibilities:**
- Define all frozen dataclasses as documented in §6.
- Define all ABC interfaces as documented in §7.
- No implementation logic — only contracts.

**Dependencies:** Phase 1 models, Phase 3 `UnifiedDocument`. No new external libraries.

**Tests:** `tests/unit/rag/test_models.py` — instantiate all models, verify immutability, serialisation.

---

### Batch 4.2 — Semantic Chunking Engine

**Purpose:** Convert `UnifiedDocument` into semantically meaningful `Chunk` objects with full metadata.

**Files (all NEW):**
```
src/adaptive_framework/rag/chunking/
    __init__.py, chunk_builder.py, semantic_chunker.py,
    table_chunker.py, figure_chunker.py, chunk_validator.py
src/adaptive_framework/rag/metadata/
    __init__.py, metadata_enricher.py, metadata_filter.py
```

**Responsibilities:**
- `SemanticChunker`: Traverse `UnifiedDocument.pages`, read `LayoutElement` objects, split at heading/section/paragraph boundaries. Configurable `max_chunk_tokens` (default 512) and `min_chunk_tokens` (default 64).
- `TableChunker`: Convert each `TableData` into one Chunk (markdown + caption).
- `FigureChunker`: Convert each `FigureData` caption into one Chunk.
- `ChunkBuilder`: Coordinate all three chunkers, validate, sort, return full list.
- `MetadataEnricher`: Read `LayoutElement` tree and `DocumentLayout` to assign heading, section, subsection, title, authors, journal, year to each Chunk.
- `ChunkValidator`: Reject chunks shorter than `min_chunk_tokens`; flag empty chunks.

**Dependencies:** Batch 4.1. No new external libraries.

**Tests:** `test_semantic_chunker.py`, `test_table_chunker.py`, `test_figure_chunker.py`, `test_metadata_enricher.py`.

---

### Batch 4.3 — Embedding Engine and FAISS Manager

**Purpose:** Embed all `Chunk` objects in parallel and persist them in a FAISS index.

**Files (all NEW):**
```
src/adaptive_framework/rag/embedding/
    __init__.py, embedding_provider.py, bge_embedder.py,
    embedding_cache.py, batch_embedding_worker.py, embedding_metrics.py
src/adaptive_framework/rag/vector_store/
    __init__.py, faiss_manager.py, faiss_index_builder.py,
    faiss_persistence.py, faiss_metrics.py
```

**Responsibilities:**
- `BGEEmbedder`: Load `BAAI/bge-large-en-v1.5` via `sentence-transformers`. Support CPU and CUDA. Asymmetric encoding: query uses prefix instruction, chunks do not.
- `EmbeddingCache`: SHA-256(chunk_text) → cached vector on disk (`outputs/rag/cache/`).
- `BatchEmbeddingWorker`: Ray remote function. Accepts batch of `Chunk` objects, returns `list[Embedding]`. Uses the existing Ray infrastructure from Phase 3.
- `FAISSIndexBuilder`: Factory — `IndexFlatIP` (default), `IndexIVFFlat`, `IndexHNSWFlat`.
- `FAISSManager`: Wraps FAISS index. Maintains `dict[int, str]` (faiss_id → chunk_id) and `dict[str, Chunk]` (chunk_id → Chunk) as metadata map.
- `FAISSPersistence`: Saves index to `outputs/rag/index/index.faiss` and metadata to `outputs/rag/index/metadata.pkl`.

**Dependencies:** Batch 4.1, 4.2. External: `sentence-transformers`, `faiss-cpu`/`faiss-gpu`, `ray`.

**Tests:** `test_bge_embedder.py`, `test_embedding_cache.py`, `test_faiss_manager.py`.

---

### Batch 4.4 — Query Pipeline (Retrieval → Context → Prompt)

**Purpose:** Build the complete online query pipeline from raw question to `PromptContext`.

**Files (all NEW):**
```
src/adaptive_framework/rag/retrieval/
    __init__.py, query_planner.py, vector_retriever.py,
    bm25_retriever.py, hybrid_retriever.py, retrieval_metrics.py
src/adaptive_framework/rag/context/
    __init__.py, context_builder.py, citation_tracker.py
src/adaptive_framework/rag/prompt/
    __init__.py, prompt_builder.py, biomedical_template.py,
    prompt_safety.py, prompt_metrics.py
```

**Responsibilities:**
- `QueryPlanner`: Rule-based intent classifier (no LLM required). Keyword matching → 10 intent classes → `QueryPlan`.
- `VectorRetriever`: FAISS dense search. Applies metadata filters.
- `BM25Retriever`: BM25 over stored chunk texts (rank_bm25). Stateless.
- `HybridRetriever`: Merge, normalise, weighted combine, deduplicate, re-rank.
- `ContextBuilder`: Token-budget context. Order by document reading order. Remove near-duplicates. Assign citations via CitationTracker.
- `PromptBuilder`: Select biomedical template by intent. Format context + citations → final prompt string.
- `PromptSafety`: Injection detection, length enforcement, empty-context guard.

**Dependencies:** Batch 4.1, 4.2, 4.3. External: `rank_bm25`, `tiktoken`.

**Tests:** `test_query_planner.py`, `test_hybrid_retriever.py`, `test_context_builder.py`, `test_prompt_builder.py`.

---

### Batch 4.5 — LLM Provider, Answer Generator, and Evaluation

**Purpose:** Complete the query pipeline with LLM generation, structured answer production, and full evaluation harness.

**Files (all NEW):**
```
src/adaptive_framework/rag/llm/
    __init__.py, llm_provider.py, llama_provider.py, qwen_provider.py,
    deepseek_provider.py, llm_factory.py, llm_metrics.py
src/adaptive_framework/rag/answer/
    __init__.py, answer_generator.py, confidence_scorer.py
src/adaptive_framework/rag/evaluation/
    __init__.py, rag_evaluator.py, retrieval_evaluator.py, generation_evaluator.py
src/adaptive_framework/rag/
    ingestion_pipeline.py, query_pipeline.py
```

**Responsibilities:**
- `LlamaProvider`: `llama-cpp-python` (GGUF, fully offline). Configurable GPU layers.
- `LLMFactory`: Maps provider name string → concrete `ILLMProvider`.
- `AnswerGenerator`: `PromptBuilder` → safety check → `ILLMProvider.generate()` → parse → citations → confidence → `Answer`.
- `ConfidenceScorer`: Weighted combination of top-1 hybrid_score, coverage ratio, avg score.
- `IngestionPipeline`: `list[UnifiedDocument]` → FAISS fully populated.
- `QueryPipeline`: `question: str` → `Answer`.
- `RAGEvaluator`: Runs `QueryPipeline` for all ground truth questions → `RAGEvaluationMetrics` → writes reports.

**Dependencies:** All previous batches. External: `llama-cpp-python`, `transformers`, `numpy`.

**Tests:** `test_answer_generator.py`, `test_rag_evaluator.py`, `test_ingestion_pipeline.py` (integration), `test_query_pipeline.py` (integration).

---

## 9. Semantic Chunking Architecture

### Why Semantic Chunking Instead of Fixed-Size

Fixed-size chunking (e.g., split every 512 characters) is architecturally incompatible with biomedical documents:

| Problem | Fixed-Size | Semantic |
|---|---|---|
| Sentences cut mid-way | Frequent | Never |
| Table split across chunks | Always destroys structure | Table is one unit |
| Figure caption separated | Often | Preserved together |
| Section heading loses context | Heading in chunk A, body in chunk B | Heading anchors its section |
| Drug dosage sentence broken | Risk of incorrect retrieval | Sentence boundary respected |

Our `UnifiedDocument` provides **explicit structural signals** that fixed-size chunking wastes:
- `LayoutElement.element_type` in `{heading, paragraph, caption, footnote, table, figure}`
- `LayoutElement.level` for heading depth (1–6)
- `page.tables` and `page.figures` with fully structured content

### Conversion Algorithm: UnifiedDocument → Chunks

```
Input:  UnifiedDocument
Output: list[Chunk]

1. STRUCTURAL PASS  (page by page, page_number ascending)
   For each page in document.pages:
       For each layout_element in page.layout_elements (reading_order):
           If element_type == "heading":
               Push onto heading_stack
               Flush pending text buffer → emit Chunk

           If element_type in {paragraph, list_item, footnote}:
               Append element.text to buffer
               If buffer token_estimate >= max_chunk_tokens:
                   Emit Chunk from buffer (soft boundary: respect sentences)
                   Carry trailing sentence to next buffer

           If element_type == "caption":
               Emit as standalone Chunk (always, even if short)

       At end of page:
           Flush non-empty buffer → emit Chunk

2. TABLE PASS  (all document.tables)
   For each table in document.tables:
       Emit Chunk(text=table.markdown, metadata.is_table=True)

3. FIGURE PASS  (all document.figures)
   For each figure in document.figures:
       If figure.caption is not None:
           Emit Chunk(text="[Figure] " + figure.caption, metadata.is_figure=True)

4. VALIDATION PASS
   Remove chunks with token_estimate < min_chunk_tokens (default 32)
   Assign sequential chunk_index (0-indexed, document-scoped)
   Sort by (page_number, reading_order, chunk_index)

5. METADATA ENRICHMENT PASS
   For each chunk:
       Attach heading_stack top → chunk.metadata.heading
       Attach top-level section → chunk.metadata.section
       Attach document.layout.title, authors, language
       Attach journal, year (heuristic regex extraction from first page)
       Attach page_number, source_file, processing_run_id, ocr_confidence
```

### Chunk Boundary Rules (Priority Order)

| Boundary | Priority | Notes |
|---|---|---|
| Heading element | Highest | Always start new chunk at heading |
| Token limit reached | High | Soft split — respect sentence boundary |
| Paragraph element | Medium | Natural split point |
| Page end | Low | Flush buffer; allow cross-page sections |
| Table / Figure | Special | Always standalone chunk, never merged |
| Caption | Special | Always standalone, always adjacent to element |

---

## 10. Metadata Enrichment

Every `Chunk` carries a `ChunkMetadata` populated during the enrichment pass.

### Source Provenance

| Field | Source | Required |
|---|---|---|
| `document_id` | `UnifiedDocument.document_id` | Always |
| `source_file` | `UnifiedDocument.file_path` | Always |
| `processing_run_id` | `UnifiedDocument.run_id` | Always |
| `chunk_created_at` | Current UTC timestamp | Always |
| `ocr_confidence` | `Page.ocr_confidence` (avg of source pages) | Always |

### Structural Provenance

| Field | Source | Notes |
|---|---|---|
| `page_number` | `LayoutElement` source page | None for multi-page spans |
| `heading` | Top of `heading_stack` at chunk creation | Nearest ancestor heading |
| `section` | Top-level heading (level == 1) | e.g., "Introduction", "Methods" |
| `subsection` | Second-level heading (level == 2) | e.g., "Data Collection" |

### Document-Level Metadata

| Field | Source | Notes |
|---|---|---|
| `title` | `DocumentLayout.title` | May be None |
| `authors` | `DocumentLayout.authors` | Tuple of strings |
| `language` | `DocumentLayout.language` | ISO 639-1 |

### Biomedical-Specific Metadata

| Field | Extraction Method | Notes |
|---|---|---|
| `journal` | Regex on first-page header/footer text | None if not found |
| `year` | Regex `\b(19|20)\d{2}\b` on title page | None if not found |

Extraction is lightweight regex + heuristics — no LLM call for metadata extraction.

### Table and Figure Flags

| Field | Source | Purpose |
|---|---|---|
| `is_table` | True if from `TableData` | Enable table-only retrieval filtering |
| `table_id` | `TableData.table_id` | UUID from Phase 3 |
| `is_figure` | True if from `FigureData` | Enable figure-only retrieval filtering |
| `figure_id` | `FigureData.figure_id` | UUID from Phase 3 |
| `caption` | `TableData.caption` / `FigureData.caption` | Searchable caption text |

---

## 11. Embedding Architecture

### Model: BAAI/bge-large-en-v1.5

| Property | Value |
|---|---|
| Embedding dimension | 1024 |
| Max input tokens | 512 |
| Similarity metric | Cosine (normalised inner product) |
| Query prefix | `"Represent this sentence for searching relevant passages: "` |
| Chunk prefix | None (BGE asymmetric encoding) |

**Asymmetric encoding rule:** `embed_query()` applies the instruction prefix. `embed_chunks()` does not. This must be respected in all implementations.

### Parallel Embedding via Distributed Framework

```
IngestionPipeline receives list[UnifiedDocument]
    |
    v
ChunkBuilder produces list[Chunk]  (all documents)
    |
    v
EmbeddingBatchPartitioner splits into batches (default 64 chunks/batch)
    |
    v
[Scheduler submits batches as Ray remote tasks]
    ray.remote(batch_embedding_worker)(batch)  for each batch
    |
    v
Workers embed in parallel (existing Ray infrastructure from Phase 3)
    |
    v
Results collected -> list[Embedding]
    |
    v
FAISSManager.add_embeddings(embeddings)
```

**Key design rule:** The embedding step uses the **same Ray infrastructure** established in Phase 3. Embedding batches are submitted as Ray remote function calls — not new actors, not new infrastructure.

### Batching

- Default batch size: 64 chunks (configurable in `configs/rag.yaml`).
- GPU: 128–256 per batch.
- CPU: 32–64 per batch.
- `BGEEmbedder` uses `SentenceTransformer.encode(batch)` for efficient batched inference.

### Caching

- `EmbeddingCache`: SHA-256(chunk.text) → cached vector.
- Storage: `shelve` database at `outputs/rag/cache/embeddings.db`.
- Incremental ingestion skips re-embedding unchanged chunks.
- Cache invalidation: `--force-reembed` flag (not automatic).

### Incremental Updates

1. `FAISSManager.load()` restores existing index.
2. Only new `UnifiedDocument` objects are chunked and embedded.
3. New embeddings added via `FAISSManager.add_embeddings()`.
4. `FAISSManager.persist()` saves updated index.

For deletion: `delete_document(document_id)` removes chunks from metadata map, triggers index rebuild from remaining embeddings.

### Runtime Metrics (EmbeddingMetrics)

| Metric | Description |
|---|---|
| `chunks_embedded` | Total chunks embedded this session |
| `batches_processed` | Total batches submitted |
| `cache_hits` | Chunks served from cache |
| `cache_misses` | Chunks requiring fresh embedding |
| `total_embedding_time_s` | Total wall-clock embedding time |
| `avg_batch_time_s` | Average time per batch |
| `throughput_chunks_per_s` | Chunks embedded per second |

---

## 12. FAISS Architecture

### Index Types

| Index Type | Use Case | Notes |
|---|---|---|
| `IndexFlatIP` | < 10K chunks, exact search | Default for development |
| `IndexIVFFlat` | 10K–10M chunks, approximate | Requires training; recommended for production |
| `IndexHNSWFlat` | Large dataset, low latency | Graph-based; no training needed |

Similarity metric: inner product (cosine-equivalent after L2-normalisation applied by BGE).

### Metadata Mapping

FAISS stores dense vectors only. Metadata stored in two in-memory dicts, persisted together:
```
faiss_id (int)  ->  chunk_id (str)  ->  Chunk (with full ChunkMetadata)
```

### Index Creation Flow

```
FAISSIndexBuilder.build(index_type, embedding_dim) -> faiss.Index
FAISSManager.initialize(index_type, embedding_dim)
FAISSManager.add_embeddings(embeddings)
    -> Batch L2-normalise vectors
    -> faiss.index.add(vectors_np)
    -> Update metadata maps
FAISSManager.persist(index_path)
```

### Index Persistence

```
outputs/rag/index/
    index.faiss          Binary FAISS index (faiss.write_index)
    metadata.pkl         Pickle of {faiss_id_to_chunk_id, chunk_id_to_chunk}
    index_stats.json     Human-readable: chunk count, doc count, dim
```

### Search Flow

```
query_embedding (list[float], dim=1024)
    -> L2-normalise -> numpy array shape (1, 1024)
    -> faiss.index.search(query_np, top_k) -> (scores, faiss_ids)
    -> Lookup: faiss_id -> chunk_id -> Chunk
    -> Apply metadata_filters (section, year, is_table, is_figure, document_id)
    -> Return list[RetrievalResult] sorted by score descending
```

### Deletion Flow

```
FAISSManager.delete_document(document_id)
    -> Identify all chunk_ids where metadata.document_id == document_id
    -> Remove from chunk_id_to_chunk and faiss_id_to_chunk_id maps
    -> Trigger index rebuild from remaining chunks
    -> Persist updated index
    -> Return count of deleted chunks
```

---

## 13. Query Planner

The `QueryPlanner` is a **rule-based intent classifier**. No LLM required. Fast, deterministic, offline-compatible.

### Supported Intents

| Intent | Keyword Triggers | Default top_k | Filters Applied |
|---|---|---|---|
| `entity_lookup` | "what is", "define", "definition of" | 5 | None |
| `summarisation` | "summarise", "overview", "describe" | 10 | None |
| `comparison` | "compare", "difference between", "vs" | 10 | None |
| `timeline` | "when was", "history of", "first", "year" | 8 | `year_range` extracted |
| `medication_lookup` | "drug", "medication", "dose", "dosage" | 5 | `section_hint="pharmacology"` |
| `diagnosis_lookup` | "diagnose", "symptom", "criteria", "ICD" | 5 | None |
| `procedure_lookup` | "procedure", "surgery", "protocol" | 5 | `section_hint="methods"` |
| `table_query` | "table", "statistics", "rate", "number of" | 5 | `require_table=True` |
| `figure_query` | "figure", "graph", "chart", "diagram" | 5 | `require_figure=True` |
| `general` | (default — no keyword match) | 8 | None |

### QueryPlan Construction

```
question (str)
    -> Lowercase + tokenise
    -> Keyword matching (priority: medication > diagnosis > procedure >
       table > figure > timeline > comparison > summarisation >
       entity_lookup > general)
    -> Extract year mentions (regex: \b(19|20)\d{2}\b)
    -> Extract section hints from question text
    -> Construct QueryPlan(
           intent=matched_intent,
           top_k=intent_default_top_k,
           use_vector=True,
           use_bm25=True,
           vector_weight=0.7,   # Default; medication/diagnosis -> 0.5
           bm25_weight=0.3,     # Default; medication/diagnosis -> 0.5
           ...
       )
```

**Design note:** `medication_lookup` and `diagnosis_lookup` increase BM25 weight to 0.5 because exact medical term matching matters more than semantic similarity for drug and ICD code lookups.

---

## 14. Hybrid Retriever

### Retrieval Strategy

```
query -> IEmbeddingProvider.embed_query() -> query_vector

Parallel retrieval:
    VectorRetriever.retrieve(query_vector, plan) -> top_k*2 results
    BM25Retriever.retrieve(query_text, plan)   -> top_k*2 results

Merge:
    Union of chunk_ids from both retrievers

Score normalisation:
    vector_score_norm = vector_score           (cosine already in [0,1])
    bm25_score_norm   = bm25_score / max(all_bm25_scores)

Hybrid score:
    hybrid_score = (plan.vector_weight * vector_score_norm)
                 + (plan.bm25_weight   * bm25_score_norm)

Metadata filtering:
    Apply plan.metadata_filters:
        year_range, section_hint, require_table, require_figure

Deduplication:
    Exact: remove duplicate chunk_ids (keep highest score)
    Near-duplicate: Jaccard similarity > 0.85 -> remove lower-scored

Ranking:
    Sort by hybrid_score descending

Return: top_k RetrievalResult objects
```

### Deduplication Detail

Near-duplicate detection is O(n²) but bounded (n ≤ 2×top_k, max 20–40 chunks), so it is fast.

### Re-ranking (Optional)

- Default: disabled (adds latency; not required for PBL demo).
- When enabled: `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores top-k results.
- Configured via `configs/rag.yaml` (`retrieval.use_cross_encoder: false`).

---

## 15. Context Builder

### Token Budget

```
Total LLM context window (Llama 3.1 8B):     4096 tokens
Reserved for system prompt:                    256 tokens
Reserved for user question:                    128 tokens
Reserved for LLM answer generation:            512 tokens
---
Available for context chunks:                 3200 tokens
```

### Context Assembly Algorithm

```
Input:  list[RetrievalResult] sorted by hybrid_score desc
Output: PromptContext

1. Initialize token_budget = 3200
2. selected_chunks = []
3. For each result:
       cost = estimate_tokens(result.chunk.text)  (chars // 4)
       If token_budget - cost >= 0:
           selected_chunks.append(result.chunk)
           token_budget -= cost
       Else: break

4. Near-duplicate removal:
       Remove chunks with Jaccard similarity > 0.85
       (keep higher hybrid_score in each pair)

5. Citation assignment:
       citations = CitationTracker.assign(selected_chunks)

6. Ordering:
       Sort by (metadata.page_number, metadata.chunk_index)
       Present context in document reading order, not score order.
       Improves LLM coherence.

7. Return PromptContext(...)
```

---

## 16. Prompt Builder

### Biomedical Prompt Structure

```
[SYSTEM PROMPT]
You are a medical research assistant with expertise in biomedical literature.
Answer questions accurately based ONLY on the provided context.
If the context does not contain enough information, say so explicitly.
Do not fabricate drug names, dosages, or clinical data.
Cite every factual claim using [1], [2], etc.

[CONTEXT]
[1] (Source: {title}, {journal}, {year}, Page {page}, Section: {section})
{chunk_1_text}

[2] (Source: {title}, {journal}, {year}, Page {page}, Section: {section})
{chunk_2_text}

...

[QUESTION]
{user_question}

[ANSWER]
```

### Template Variants by Intent

| Intent | Additional System Instruction |
|---|---|
| `medication_lookup` | "Include drug class, mechanism, indications, contraindications if available." |
| `diagnosis_lookup` | "Include diagnostic criteria, ICD codes, differential diagnosis if available." |
| `comparison` | "Structure the answer as a comparative analysis with clear criteria." |
| `timeline` | "Order events chronologically. Include years when stated in the source." |
| `summarisation` | "Provide a concise summary. Use bullet points if appropriate." |
| `table_query` | "Reference the table data directly. Include table numbers." |

### Prompt Safety (PromptSafety.check)

1. **Injection detection:** Reject prompts containing `"ignore previous"`, `"disregard"`, `"you are now"`, `"pretend"`, `"roleplay"`.
2. **Length enforcement:** Reject prompts exceeding `MAX_PROMPT_TOKENS` (default 4096).
3. **Empty context guard:** If no context chunks retrieved → return safe "insufficient context" response without calling LLM.
4. **Personal data guard:** Flag if question matches medical record, SSN, or PHI patterns.

---

## 17. LLM Provider Architecture

### Interface-First Design

All LLM interaction is mediated through `ILLMProvider`. `AnswerGenerator` never calls a specific LLM directly. It receives an `ILLMProvider` via constructor dependency injection.

```
DI Container (or manual construction)
    -> LLMFactory.create(provider_name="llama")
    -> returns LlamaProvider(model_path, context_window, ...)
    -> injected into AnswerGenerator(llm=llm_provider, ...)
```

### Llama 3.1 8B Instruct (Default)

| Property | Value |
|---|---|
| Loader | `llama-cpp-python` (GGUF format) |
| Model file | `models/llama-3.1-8b-instruct.Q4_K_M.gguf` (user-provided) |
| Context window | 4096 tokens |
| Quantisation | Q4_K_M (4-bit) |
| GPU layers | Configurable (`n_gpu_layers`, default 0 for CPU) |
| Temperature | 0.1 (low for factual biomedical responses) |
| Max tokens | 512 |
| Offline | Fully offline — no network call at inference time |

### Provider Stubs

- `QwenProvider`: Implements `ILLMProvider`, raises `NotImplementedError`. Validates interface contract.
- `DeepSeekProvider`: Same — stub only in v1.0.

### Adding a New Provider (Zero Architecture Change)

```
1. Create new_provider.py implementing ILLMProvider.
2. Register in LLMFactory: "new_model" -> NewModelProvider.
3. Add configuration to configs/rag.yaml.
4. ZERO changes to AnswerGenerator, PromptBuilder, or any other module.
```

---

## 18. Answer Generator

### Generation Flow

```
PromptContext + list[RetrievalResult]
    |
    v
PromptBuilder.build(context) -> prompt (str)
    |
    v
PromptSafety.check(prompt) -> pass or return safe fallback
    |
    v
[t0 = perf_counter()]
ILLMProvider.generate(prompt, max_tokens=512, temperature=0.1) -> raw_text
[t1 = perf_counter()]
    |
    v
_parse_answer(raw_text) -> answer_text (str)
_extract_citations(raw_text, context.citations) -> cited_citations
    |
    v
ConfidenceScorer.score(retrieval_results, context) -> confidence (float)
    |
    v
Return Answer(
    answer_text=answer_text,
    citations=cited_citations,
    confidence_score=confidence,
    generation_time_seconds=(t1 - t0),
    ...
)
```

### Confidence Scoring Formula

```
confidence = (
    0.5 * top_1_hybrid_score        # Best retrieved chunk relevance
  + 0.3 * (n_chunks / top_k)        # Context coverage ratio
  + 0.2 * avg_hybrid_score          # Average quality of context
)
```

Confidence is derived from retrieval metrics — not LLM self-assessment. Measurable and calibratable.

---

## 19. Evaluation Architecture

### Retrieval Evaluation

Requires a ground truth dataset: list of `{question, relevant_chunk_ids}` pairs.

| Metric | Formula | Notes |
|---|---|---|
| **Recall@K** | `|relevant ∩ retrieved_K| / |relevant|` | Fraction of relevant chunks found |
| **Precision@K** | `|relevant ∩ retrieved_K| / K` | Fraction of retrieved that are relevant |
| **MRR** | `mean(1 / rank_of_first_relevant)` | Mean Reciprocal Rank |
| **nDCG@K** | `DCG@K / IDCG@K` | Normalised Discounted Cumulative Gain |

K values: 1, 3, 5, 10 (configurable in `configs/rag.yaml`).

### Latency Evaluation

Per-query latency decomposed into:

| Component | Instrument |
|---|---|
| Query embedding | `perf_counter()` around `embed_query()` |
| FAISS search | `perf_counter()` around `search_with_metadata()` |
| BM25 search | `perf_counter()` around BM25 ranking |
| Hybrid score + dedup | `perf_counter()` around merge |
| LLM generation | `perf_counter()` around `generate()` |
| Total | End-to-end around `QueryPipeline.run()` |

Aggregate: mean, p50, p95, p99 across all evaluated queries.

### Evaluation Output

```
outputs/rag/evaluation/
    evaluation_report.json     Full metric set
    evaluation_report.md       Human-readable formatted table
    per_query_results.csv      Row per question: scores, latency, intent
    retrieval_detail.json      Retrieved chunks per question
```

---

## 20. Runtime Instrumentation

Every Phase 4 module exposes `get_metrics() -> dict[str, float]`, consistent with the existing framework instrumentation pattern.

### Metrics Registry

| Module | Key Metrics |
|---|---|
| `BGEEmbedder` | `chunks_embedded`, `cache_hits`, `throughput_chunks_per_s`, `avg_batch_time_s` |
| `FAISSManager` | `total_chunks`, `total_documents`, `search_latency_ms`, `index_build_time_s` |
| `HybridRetriever` | `vector_score_avg`, `bm25_score_avg`, `hybrid_score_avg`, `dedup_removed` |
| `ContextBuilder` | `tokens_used`, `tokens_budget`, `chunks_selected`, `chunks_dropped_budget` |
| `PromptBuilder` | `prompt_length_tokens`, `safety_checks_failed`, `template_used` |
| `LlamaProvider` | `generation_time_s`, `tokens_generated`, `tokens_per_second` |
| `AnswerGenerator` | `confidence_score`, `citations_extracted`, `total_latency_s` |
| `RAGEvaluator` | All `RAGEvaluationMetrics` fields |

### Dashboard Integration

Metrics written as structured JSON after each pipeline run:

```
outputs/rag/metrics/
    ingestion_metrics_{run_id}.json
    query_metrics_{session_id}.json
```

These are consumed by the Phase 6 dashboard using the existing `outputs/` convention.

---

## 21. Platform Compatibility

| Platform | Support Level | Notes |
|---|---|---|
| **Windows** | Full | Primary development platform. All paths use `pathlib.Path`. |
| **Linux** | Full | CI and production target. Ray preferred on Linux. |
| **GPU (CUDA)** | Supported | BGE: torch.cuda. Llama: `n_gpu_layers > 0`. |
| **CPU-only** | Full | Default mode. BGE and Llama both run on CPU. |
| **Offline Mode** | Full | All models downloaded once. No runtime network calls. FAISS is local. |
| **macOS** | Best-effort | `llama-cpp-python` may require Metal build for MPS. Not tested. |

### Path Handling

All file paths use `pathlib.Path`. All outputs go to `outputs/rag/` (auto-created if not present).

### Model Files

| Model | Download | Stored At |
|---|---|---|
| `BAAI/bge-large-en-v1.5` | `sentence-transformers` auto-download | `~/.cache/huggingface/` |
| `Llama 3.1 8B Instruct Q4_K_M` | User downloads GGUF from HuggingFace | `models/` (project root) |

---

## 22. Testing Strategy

### Unit Tests (all modules mocked / isolated)

| Test File | What It Tests |
|---|---|
| `test_semantic_chunker.py` | Heading/paragraph/caption boundary conditions; no empty chunks |
| `test_table_chunker.py` | TableData -> Chunk; markdown formatting; caption preservation |
| `test_figure_chunker.py` | FigureData caption extraction; None caption handling |
| `test_metadata_enricher.py` | All metadata fields populated; journal/year extraction |
| `test_bge_embedder.py` | Output dim=1024; query prefix applied; batch handling |
| `test_embedding_cache.py` | Cache hit/miss; SHA-256 key; disk persistence round-trip |
| `test_faiss_manager.py` | Add/search/delete/persist/load cycle; metadata map consistency |
| `test_query_planner.py` | All 10 intent classes; year extraction; section hints |
| `test_hybrid_retriever.py` | Score normalisation; deduplication; metadata filtering |
| `test_context_builder.py` | Token budgeting; document-order output; citation assignment |
| `test_prompt_builder.py` | All 6 template variants; safety rejection cases |
| `test_answer_generator.py` | Mocked LLM; Answer field population; confidence range |
| `test_rag_evaluator.py` | Recall@K, MRR, nDCG from synthetic ground truth |

### Integration Tests

| Test File | What It Tests |
|---|---|
| `test_ingestion_pipeline.py` | Synthetic `UnifiedDocument` -> FAISS index populated end-to-end |
| `test_query_pipeline.py` | Question -> `Answer` with mocked LLM; all fields non-None |

### Benchmark Tests

| Test File | What It Tests |
|---|---|
| `test_embedding_throughput.py` | Chunks per second on CPU; scaling with batch size |
| `test_retrieval_latency.py` | Search latency at 1K, 10K, 100K indexed chunks |

### Retrieval Ground Truth Tests

Ground truth file: `tests/fixtures/rag/ground_truth.json`

```json
[
    {
        "question": "What is the mechanism of action of metformin?",
        "relevant_chunk_ids": ["<chunk_id_1>", "<chunk_id_2>"]
    }
]
```

### LLM Tests

All LLM tests use `MockLLMProvider` (returns fixed string, no model required). This allows testing `AnswerGenerator`, `PromptBuilder`, and `QueryPipeline` without local model files.

---

## 23. Verification Plan

### Verification Checkpoints

| Checkpoint | Criteria | Verification Command |
|---|---|---|
| **Batch 4.1** | All models instantiate correctly; all ABCs have correct signatures | `pytest tests/unit/rag/test_models.py` |
| **Batch 4.2** | All section headings preserved; no chunk < min_chunk_tokens; tables/figures standalone | `pytest tests/unit/rag/test_semantic_chunker.py` |
| **Batch 4.3** | Embedding dim=1024; FAISS round-trip correct; cache hit rate > 90% on re-run | `pytest tests/unit/rag/test_bge_embedder.py tests/unit/rag/test_faiss_manager.py` |
| **Batch 4.4** | All intents classified correctly; no duplicates in retrieval; safety rejects injection | `pytest tests/unit/rag/test_query_planner.py tests/unit/rag/test_hybrid_retriever.py` |
| **Batch 4.5** | `IngestionPipeline` ingests 5 test docs; `QueryPipeline` returns `Answer` with citations; evaluation report written | `pytest tests/integration/rag/` |

### Performance Targets

| Metric | Target | Notes |
|---|---|---|
| Embedding throughput | >= 50 chunks/s (CPU) | BGE batch 64 on CPU |
| FAISS search latency | <= 20 ms (10K index) | `IndexFlatIP`, CPU |
| Query latency (no LLM) | <= 500 ms | Embed + retrieve + context + prompt |
| Query latency (with LLM) | <= 30 s | CPU-only Llama 3.1 8B Q4_K_M |
| Recall@5 | >= 0.70 | On manually constructed ground truth |
| MRR | >= 0.60 | On manually constructed ground truth |

### Manual Verification Steps

1. Run `IngestionPipeline` on actual biomedical dataset (`dataset/`).
2. Inspect `outputs/rag/index/index_stats.json` — verify `total_chunks` > 0.
3. Run `QueryPipeline` with 5 representative biomedical questions.
4. Verify each `Answer`:
   - `answer_text` non-empty.
   - `citations` non-empty, mapping to real source documents.
   - `confidence_score` in `[0.0, 1.0]`.
5. Run `RAGEvaluator` — verify `evaluation_report.md` written to `outputs/rag/evaluation/`.

---

## 24. Future Work

Phase 4 is designed with explicit extension points. These capabilities are **out of scope for Phase 4** but integrate without architecture changes.

### 24.1 Knowledge Graphs

**Integration point:** After `MetadataEnricher`, a `KnowledgeGraphBuilder` module extracts entity relationships (disease → drug, gene → protein) and stores them in Neo4j or NetworkX. `HybridRetriever` can query the graph as a third retrieval source alongside vector and BM25, using the existing `IRetriever` interface.

**Why deferred:** Requires NER models and graph infrastructure. The `IRetriever` interface already supports adding `GraphRetriever` without changing merge logic.

### 24.2 Multi-Agent Retrieval

**Integration point:** Replace `QueryPipeline` with a `MultiAgentQueryOrchestrator`. Each agent specialises in a document subset (e.g., radiology papers, cardiology papers). Orchestrator dispatches sub-queries and merges results. `IRetriever` can be implemented as `AgentRetriever` dispatching to Ray actors.

**Why deferred:** Single-agent retrieval is sufficient for PBL scope. The interface makes multi-agent a plug-in.

### 24.3 Adaptive Chunking

**Integration point:** Replace `SemanticChunker` with `AdaptiveChunker` that uses `DocumentStatistics` (avg OCR confidence, table density, figure count) to dynamically choose chunk sizes. The `IChunkBuilder` interface already supports this. `get_strategy_name()` enables A/B comparison.

**Why deferred:** Semantic chunking is sufficient. Adaptive chunking requires evaluation harness (Phase 5) to measure its benefit first.

### 24.4 Cost-Based Retrieval

**Integration point:** Extend `QueryPlanner` to optimise retrieval strategy based on latency budget and accuracy targets. `QueryPlan.use_vector` and `use_bm25` flags already exist for a cost-aware planner to set.

**Why deferred:** Performance targets in §23 are achievable with the current fixed strategy.

### 24.5 Multimodal Retrieval

**Integration point:** `FigureChunker` currently indexes captions only. A `MultimodalEmbedder` (BioMedCLIP or CLIP) encodes figure images into embeddings, stored in a separate FAISS index. `HybridRetriever` merges text and image results. `FigureData.image_path` (already populated in Phase 3) preserves image paths for this future use.

**Why deferred:** Requires image storage and a multimodal model. `FigureData` is already structured for this extension.

---

## Configuration Reference

New entries for `configs/rag.yaml`:

```yaml
rag:
  chunking:
    strategy: "semantic"
    max_chunk_tokens: 512
    min_chunk_tokens: 32
    heading_boundary: true
    table_standalone: true
    figure_standalone: true

  embedding:
    model: "BAAI/bge-large-en-v1.5"
    batch_size: 64
    device: "cpu"                 # "cpu" | "cuda"
    cache_enabled: true
    cache_path: "outputs/rag/cache/embeddings.db"

  vector_store:
    backend: "faiss"
    index_type: "flat"            # "flat" | "ivf" | "hnsw"
    index_path: "outputs/rag/index"
    n_clusters: 100               # IVF only

  retrieval:
    top_k: 8
    vector_weight: 0.7
    bm25_weight: 0.3
    use_cross_encoder: false
    dedup_threshold: 0.85

  context:
    max_context_tokens: 3200
    min_chunk_tokens_for_context: 32

  prompt:
    template: "biomedical"
    max_prompt_tokens: 4096
    safety_enabled: true

  llm:
    provider: "llama"
    model_path: "models/llama-3.1-8b-instruct.Q4_K_M.gguf"
    context_window: 4096
    max_tokens: 512
    temperature: 0.1
    n_gpu_layers: 0               # 0 = CPU only

  evaluation:
    ground_truth_path: "tests/fixtures/rag/ground_truth.json"
    output_path: "outputs/rag/evaluation"
    eval_k_values: [1, 3, 5, 10]
```

---

## Dependency Summary

All additions to `requirements.txt` (no existing version changes):

| Library | Purpose | Version |
|---|---|---|
| `sentence-transformers` | BAAI/bge-large-en-v1.5 embedding | `>=2.7.0` |
| `faiss-cpu` | Vector index | `>=1.8.0` |
| `llama-cpp-python` | Llama 3.1 8B local inference | `>=0.2.90` |
| `rank_bm25` | BM25 sparse retrieval | `>=0.2.2` |
| `tiktoken` | Token counting | `>=0.7.0` |
| `numpy` | Vector operations | Already in requirements |
| `ray` | Parallel embedding dispatch | Already in requirements (Phase 3) |
| `torch` | Backend for sentence-transformers | `>=2.3.0` |

---

*Document locked: 2026-08-03 | Version: 4.0 | Prepared for implementation review.*
*Consistent with: architecture_v2.0_locked.md | Phase_3_Document_Processing_Architecture_LOCKED.md*
*Do NOT modify previous phases. Do NOT generate implementation before this document is approved.*
