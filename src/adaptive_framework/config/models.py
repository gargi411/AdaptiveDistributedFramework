"""Typed configuration data models for the Adaptive Distributed Framework.

All models are Python dataclasses with:
    - Full type hints
    - __post_init__ validation
    - to_dict() serialization
    - Meaningful __repr__

These models correspond 1-to-1 with YAML configuration files under configs/.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# =============================================================
# Framework Config  (framework.yaml)
# =============================================================


@dataclass
class FrameworkConfig:
    """Top-level framework configuration.

    Attributes:
        name: Human-readable framework name displayed in logs and reports.
        version: Semantic version of this configuration schema.
        run_id_prefix: Prefix for unique run identifiers.
        output_dir: Root directory for all runtime outputs.
        debug: Enable verbose debug logging and extra assertions.
        max_concurrent_jobs: Maximum concurrent pipeline jobs across the cluster.
        stage_timeout_seconds: Global timeout per pipeline stage (seconds).
        shutdown_timeout_seconds: Graceful shutdown timeout (seconds).

    Example:
        >>> cfg = FrameworkConfig(name="ADF", version="2.0.0",
        ...     run_id_prefix="adf_run", output_dir="outputs",
        ...     debug=False, max_concurrent_jobs=8,
        ...     stage_timeout_seconds=3600, shutdown_timeout_seconds=30)
        >>> cfg.name
        'ADF'
    """

    name: str
    version: str
    run_id_prefix: str
    output_dir: str
    debug: bool
    max_concurrent_jobs: int
    stage_timeout_seconds: int
    shutdown_timeout_seconds: int

    def __post_init__(self) -> None:
        """Validate framework configuration fields.

        Raises:
            ValueError: If any field has an invalid value.
        """
        if not self.name:
            raise ValueError("FrameworkConfig.name must not be empty.")
        if self.max_concurrent_jobs < 1:
            raise ValueError("FrameworkConfig.max_concurrent_jobs must be >= 1.")
        if self.stage_timeout_seconds < 1:
            raise ValueError("FrameworkConfig.stage_timeout_seconds must be >= 1.")
        if self.shutdown_timeout_seconds < 1:
            raise ValueError("FrameworkConfig.shutdown_timeout_seconds must be >= 1.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary representation of this config.
        """
        return asdict(self)


# =============================================================
# Logging Config  (logging.yaml)
# =============================================================


@dataclass
class ConsoleLoggingConfig:
    """Console handler configuration.

    Attributes:
        enabled: Whether to emit logs to the console.
        level: Minimum log level for this handler.
        use_rich: Use the Rich library for formatted output.
        colorize: Enable ANSI color codes.
    """

    enabled: bool
    level: str
    use_rich: bool
    colorize: bool

    def __post_init__(self) -> None:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level not in valid_levels:
            raise ValueError(
                f"ConsoleLoggingConfig.level must be one of {valid_levels}, got '{self.level}'."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FileLoggingConfig:
    """Rotating file handler configuration.

    Attributes:
        enabled: Whether to write logs to a rotating file.
        level: Minimum log level for this handler.
        path: Log file path (relative to framework output_dir).
        max_bytes: Maximum size per log file before rotation.
        backup_count: Number of rotated backup files to keep.
        encoding: File encoding.
    """

    enabled: bool
    level: str
    path: str
    max_bytes: int
    backup_count: int
    encoding: str

    def __post_init__(self) -> None:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level not in valid_levels:
            raise ValueError(
                f"FileLoggingConfig.level must be one of {valid_levels}, got '{self.level}'."
            )
        if self.max_bytes < 1024:
            raise ValueError("FileLoggingConfig.max_bytes must be >= 1024 bytes.")
        if self.backup_count < 0:
            raise ValueError("FileLoggingConfig.backup_count must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoggingConfig:
    """Top-level logging configuration.

    Attributes:
        level: Root log level.
        format: Output format — 'text' or 'json'.
        console: Console handler settings.
        file: Rotating text file handler settings.
        json_file: Rotating JSON file handler settings.
        context_fields: Field names automatically injected into every log record.
    """

    level: str
    format: str
    console: ConsoleLoggingConfig
    file: FileLoggingConfig
    json_file: FileLoggingConfig
    context_fields: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        valid_formats = {"text", "json"}
        if self.level not in valid_levels:
            raise ValueError(
                f"LoggingConfig.level must be one of {valid_levels}, got '{self.level}'."
            )
        if self.format not in valid_formats:
            raise ValueError(
                f"LoggingConfig.format must be one of {valid_formats}, got '{self.format}'."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================
# Ray Cluster Config  (ray_cluster.yaml)
# =============================================================


@dataclass
class WorkerConfig:
    """Ray worker node configuration.

    Attributes:
        heartbeat_interval_seconds: Interval between heartbeat pings.
        heartbeat_timeout_seconds: Timeout before marking a worker as lost.
        task_retry_attempts: Retries for reassigning tasks from failed workers.
        retry_delay_seconds: Delay between retry attempts.
    """

    heartbeat_interval_seconds: float
    heartbeat_timeout_seconds: float
    task_retry_attempts: int
    retry_delay_seconds: float

    def __post_init__(self) -> None:
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("WorkerConfig.heartbeat_interval_seconds must be > 0.")
        if self.heartbeat_timeout_seconds <= self.heartbeat_interval_seconds:
            raise ValueError(
                "WorkerConfig.heartbeat_timeout_seconds must be > heartbeat_interval_seconds."
            )
        if self.task_retry_attempts < 0:
            raise ValueError("WorkerConfig.task_retry_attempts must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SharedMemoryConfig:
    """Zero-copy shared memory configuration.

    Attributes:
        enabled: Enable zero-copy shared memory buffers.
        max_buffer_size: Maximum shared memory buffer size in bytes.
    """

    enabled: bool
    max_buffer_size: int

    def __post_init__(self) -> None:
        if self.max_buffer_size < 1:
            raise ValueError("SharedMemoryConfig.max_buffer_size must be >= 1.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RayClusterConfig:
    """Ray cluster configuration.

    Attributes:
        address: Ray cluster address ('local' or 'auto').
        num_cpus: CPUs available to Ray (None = auto-detect).
        num_gpus: GPUs available to Ray (None = auto-detect).
        object_store_memory: Object store memory limit (None = auto-detect).
        init_timeout_seconds: Ray init timeout.
        worker: Worker node settings.
        shared_memory: Zero-copy shared memory settings.
    """

    address: str
    num_cpus: int | None
    num_gpus: int | None
    object_store_memory: int | None
    init_timeout_seconds: int
    worker: WorkerConfig
    shared_memory: SharedMemoryConfig

    def __post_init__(self) -> None:
        if self.address not in ("local", "auto") and not self.address.startswith("ray://"):
            raise ValueError(
                "RayClusterConfig.address must be 'local', 'auto', or a 'ray://' URI."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================
# Scheduler Config  (scheduler.yaml)
# =============================================================


@dataclass
class WorkStealingConfig:
    """Work stealing sub-configuration.

    Attributes:
        enabled: Enable or disable work stealing.
        steal_threshold: Minimum tasks before stealing is allowed.
        steal_fraction: Maximum fraction of tasks stolen per operation.
        check_interval_seconds: Interval between stealing checks.
    """

    enabled: bool
    steal_threshold: int
    steal_fraction: float
    check_interval_seconds: float

    def __post_init__(self) -> None:
        if self.steal_threshold < 1:
            raise ValueError("WorkStealingConfig.steal_threshold must be >= 1.")
        if not (0.0 < self.steal_fraction <= 1.0):
            raise ValueError("WorkStealingConfig.steal_fraction must be in (0, 1].")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SchedulerOverheadConfig:
    """Scheduler overhead monitoring sub-configuration.

    Attributes:
        enabled: Enable overhead measurement.
        target_max_overhead_fraction: Maximum allowed scheduler overhead (fraction).
            Must be < 0.01 per architecture spec §4.2.
        warn_on_exceed: Emit WARNING log if overhead exceeds target.
    """

    enabled: bool
    target_max_overhead_fraction: float
    warn_on_exceed: bool

    def __post_init__(self) -> None:
        if not (0.0 < self.target_max_overhead_fraction < 1.0):
            raise ValueError(
                "SchedulerOverheadConfig.target_max_overhead_fraction must be in (0, 1)."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SchedulerConfig:
    """Adaptive scheduler configuration.

    Attributes:
        strategy: Scheduling strategy (e.g. 'page_count').
        work_stealing: Work stealing parameters.
        overhead_monitoring: Overhead measurement parameters.
        default_partition_count: Default number of partitions.
        min_pages_per_partition: Minimum pages per partition.
    """

    strategy: str
    work_stealing: WorkStealingConfig
    overhead_monitoring: SchedulerOverheadConfig
    default_partition_count: int
    min_pages_per_partition: int

    def __post_init__(self) -> None:
        if self.strategy not in ("page_count",):
            raise ValueError(
                f"SchedulerConfig.strategy '{self.strategy}' is not supported. "
                "Supported: ['page_count']."
            )
        if self.default_partition_count < 1:
            raise ValueError("SchedulerConfig.default_partition_count must be >= 1.")
        if self.min_pages_per_partition < 1:
            raise ValueError("SchedulerConfig.min_pages_per_partition must be >= 1.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================
# Document Processing Engine Config  (ocr.yaml)
# =============================================================

SUPPORTED_OCR_BACKENDS = frozenset({"paddleocr", "trocr", "nougat", "mineru", "docling", "openvino"})


@dataclass
class OCRConfig:
    """OCR sub-component configuration.

    Attributes:
        languages: Language codes for OCR.
        use_gpu: Enable GPU acceleration.
        confidence_threshold: Minimum confidence to accept OCR output.
        device: OpenVINO execution device ('AUTO', 'GPU', 'CPU').
        det_model_path: Optional path to text detection model.
        rec_model_path: Optional path to text recognition model.
    """

    languages: list[str]
    use_gpu: bool
    confidence_threshold: float
    device: str = "AUTO"
    det_model_path: str | None = None
    rec_model_path: str | None = None

    def __post_init__(self) -> None:
        if not self.languages:
            raise ValueError("OCRConfig.languages must not be empty.")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("OCRConfig.confidence_threshold must be in [0.0, 1.0].")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentProcessingEngineConfig:
    """Document Processing Engine configuration.

    Corresponds to architecture v2.0 §2.1. Wraps the OCR backend and all
    four Document Processing Engine sub-components.

    Attributes:
        ocr_backend: Active OCR backend identifier.
        ocr: OCR sub-component settings.
        page_timeout_seconds: Per-page processing timeout.
        num_threads: Parallel threads for CPU-bound operations.
    """

    ocr_backend: str
    ocr: OCRConfig
    page_timeout_seconds: int
    num_threads: int

    def __post_init__(self) -> None:
        if self.ocr_backend not in SUPPORTED_OCR_BACKENDS:
            raise ValueError(
                f"DocumentProcessingEngineConfig.ocr_backend '{self.ocr_backend}' is not "
                f"supported. Supported: {sorted(SUPPORTED_OCR_BACKENDS)}."
            )
        if self.page_timeout_seconds < 1:
            raise ValueError("DocumentProcessingEngineConfig.page_timeout_seconds must be >= 1.")
        if self.num_threads < 1:
            raise ValueError("DocumentProcessingEngineConfig.num_threads must be >= 1.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================
# Evaluation Config  (evaluation.yaml)
# =============================================================


@dataclass
class SchedulerOverheadMetricConfig:
    """Scheduler Overhead metric configuration.

    Attributes:
        enabled: Enable scheduler overhead measurement.
        target_max_percent: Target maximum overhead percentage (must be < 1.0).
        warn_on_exceed: Emit warning if overhead exceeds target.
    """

    enabled: bool
    target_max_percent: float
    warn_on_exceed: bool

    def __post_init__(self) -> None:
        if self.target_max_percent <= 0.0:
            raise ValueError(
                "SchedulerOverheadMetricConfig.target_max_percent must be > 0."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationConfig:
    """Evaluation engine configuration.

    Attributes:
        output_dir: Directory for evaluation reports.
        report_formats: List of report format strings.
        scheduler_overhead: Scheduler overhead metric settings.
        warmup_runs: Number of warm-up runs before recording.
        measurement_runs: Number of runs to average.
    """

    output_dir: str
    report_formats: list[str]
    scheduler_overhead: SchedulerOverheadMetricConfig
    warmup_runs: int
    measurement_runs: int

    def __post_init__(self) -> None:
        valid_formats = {"json", "csv", "markdown"}
        invalid = set(self.report_formats) - valid_formats
        if invalid:
            raise ValueError(
                f"EvaluationConfig.report_formats contains unsupported formats: {invalid}."
            )
        if self.warmup_runs < 0:
            raise ValueError("EvaluationConfig.warmup_runs must be >= 0.")
        if self.measurement_runs < 1:
            raise ValueError("EvaluationConfig.measurement_runs must be >= 1.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================
# RAG Config  (rag.yaml)
# =============================================================


@dataclass
class ChunkerConfig:
    """Text chunker configuration.

    Attributes:
        strategy: Chunking strategy identifier.
        chunk_size: Target chunk size in characters (not tokens) for the
            semantic strategy.  For fixed_size and sentence strategies this
            value remains the primary size signal.
        chunk_overlap: Character overlap carried from the end of one chunk
            into the start of the next.
        min_chunk_size: Minimum number of characters a chunk must contain.
            Chunks shorter than this are merged into the previous chunk
            rather than emitted as standalone fragments.
        max_chunk_size: Hard ceiling on chunk length in characters.
            A chunk that would exceed this limit is split at the nearest
            sentence boundary before the limit is reached.
    """

    strategy: str
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int = 50
    max_chunk_size: int = 2000

    def __post_init__(self) -> None:
        valid_strategies = {"fixed_size", "sentence", "semantic"}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"ChunkerConfig.strategy must be one of {valid_strategies}."
            )
        if self.chunk_size < 1:
            raise ValueError("ChunkerConfig.chunk_size must be >= 1.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "ChunkerConfig.chunk_overlap must be >= 0 and < chunk_size."
            )
        if self.min_chunk_size < 1:
            raise ValueError("ChunkerConfig.min_chunk_size must be >= 1.")
        if self.max_chunk_size < self.chunk_size:
            raise ValueError(
                "ChunkerConfig.max_chunk_size must be >= chunk_size."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EmbedderConfig:
    """Embedding model configuration.

    Attributes:
        model: Model identifier string.
        device: Compute device ('cpu' or 'cuda').
        batch_size: Embedding batch size.
        embedding_dim: Output embedding dimension.
        cache_enabled: Enable the SHA-256-keyed SQLite embedding cache.
        cache_path: Path to the SQLite cache file.
    """

    model: str
    device: str
    batch_size: int
    embedding_dim: int
    cache_enabled: bool = True
    cache_path: str = "outputs/rag/cache/embeddings.db"

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda"):
            raise ValueError("EmbedderConfig.device must be 'cpu' or 'cuda'.")
        if self.batch_size < 1:
            raise ValueError("EmbedderConfig.batch_size must be >= 1.")
        if self.embedding_dim < 1:
            raise ValueError("EmbedderConfig.embedding_dim must be >= 1.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VectorStoreConfig:
    """Vector store configuration.

    Attributes:
        backend: Storage backend identifier.
        persist_dir: Persistence directory path.
        collection_name: Collection name in the vector store.
        index_type: FAISS index type: 'flat', 'ivf', or 'hnsw'.
        index_path: Directory for FAISS index files.
        n_clusters: Number of IVF clusters (used only when index_type == 'ivf').
    """

    backend: str
    persist_dir: str
    collection_name: str
    index_type: str = "flat"
    index_path: str = "outputs/rag/index"
    n_clusters: int = 100

    def __post_init__(self) -> None:
        valid_backends = {"chromadb", "faiss", "qdrant"}
        if self.backend not in valid_backends:
            raise ValueError(
                f"VectorStoreConfig.backend must be one of {valid_backends}."
            )
        if not self.collection_name:
            raise ValueError("VectorStoreConfig.collection_name must not be empty.")
        valid_index_types = {"flat", "ivf", "hnsw"}
        if self.index_type not in valid_index_types:
            raise ValueError(
                f"VectorStoreConfig.index_type must be one of {valid_index_types}."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationConfig:
    """LLM Generation configuration settings.

    Attributes:
        provider: Provider identifier ('fake', 'real', 'gemini', 'openai', etc.).
        model: Model identifier.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in generated completion.
        api_key_env_var: Environment variable name holding provider API key.
        timeout_seconds: Request timeout in seconds.
        base_url: Optional custom base URL for OpenAI-compatible providers.
    """

    provider: str = "fake"
    model: str = "fake-llm-v1"
    temperature: float = 0.0
    max_tokens: int = 512
    api_key_env_var: str = "GEMINI_API_KEY"
    timeout_seconds: float = 30.0
    base_url: str | None = None

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("GenerationConfig.provider must not be empty.")
        if not self.model:
            raise ValueError("GenerationConfig.model must not be empty.")
        if self.temperature < 0.0:
            raise ValueError("GenerationConfig.temperature must be non-negative.")
        if self.max_tokens <= 0:
            raise ValueError("GenerationConfig.max_tokens must be positive.")
        if self.timeout_seconds <= 0.0:
            raise ValueError("GenerationConfig.timeout_seconds must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContextConfig:
    """Context assembly configuration settings.

    Attributes:
        max_chunks: Maximum chunks to include in context.
        max_context_chars: Maximum characters aggregate context text.
        min_score: Minimum similarity score to retain a chunk.
    """

    max_chunks: int = 5
    max_context_chars: int = 4000
    min_score: float = 0.0

    def __post_init__(self) -> None:
        if self.max_chunks <= 0:
            raise ValueError("ContextConfig.max_chunks must be positive.")
        if self.max_context_chars <= 0:
            raise ValueError("ContextConfig.max_context_chars must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BM25Config:
    """BM25 sparse retrieval configuration (Phase 4.8).

    Attributes:
        k1: Term frequency saturation parameter.
        b: Document length normalization parameter.
        index_path: Directory path for persisted BM25 index.
    """

    k1: float = 1.5
    b: float = 0.75
    index_path: str = "outputs/rag/index"

    def __post_init__(self) -> None:
        if self.k1 < 0.0:
            raise ValueError("BM25Config.k1 must be >= 0.0.")
        if not (0.0 <= self.b <= 1.0):
            raise ValueError("BM25Config.b must be between 0.0 and 1.0.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HybridConfig:
    """Hybrid retrieval configuration (Phase 4.8).

    Attributes:
        enabled: Whether hybrid retrieval is enabled.
        dense_top_k: Number of dense candidates to retrieve.
        sparse_top_k: Number of sparse candidates to retrieve.
        final_top_k: Final number of fused candidates.
        rrf_k: Reciprocal Rank Fusion constant.
    """

    enabled: bool = False
    dense_top_k: int = 20
    sparse_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if self.dense_top_k < 1:
            raise ValueError("HybridConfig.dense_top_k must be >= 1.")
        if self.sparse_top_k < 1:
            raise ValueError("HybridConfig.sparse_top_k must be >= 1.")
        if self.final_top_k < 1:
            raise ValueError("HybridConfig.final_top_k must be >= 1.")
        if self.rrf_k < 1:
            raise ValueError("HybridConfig.rrf_k must be >= 1.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RerankingConfig:
    """Reranking configuration settings (Phase 4.9).

    Attributes:
        enabled: Whether reranking is enabled.
        model_name: Cross-encoder model identifier.
        candidate_top_k: Number of candidates fetched from stage 1 for reranking.
        final_top_k: Number of reranked candidates to return.
        batch_size: Batch size for cross-encoder inference.
        device: Device identifier ('auto', 'cpu', 'cuda').
    """

    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_top_k: int = 20
    final_top_k: int = 5
    batch_size: int = 16
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.model_name or not self.model_name.strip():
            raise ValueError("RerankingConfig.model_name must not be empty.")
        if self.candidate_top_k < 1:
            raise ValueError("RerankingConfig.candidate_top_k must be >= 1.")
        if self.final_top_k < 1:
            raise ValueError("RerankingConfig.final_top_k must be >= 1.")
        if self.batch_size < 1:
            raise ValueError("RerankingConfig.batch_size must be >= 1.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalConfig:
    """Retrieval configuration settings (Phase 4.4, 4.8 & 4.9).

    Attributes:
        strategy: Retrieval strategy ('dense', 'hybrid', or 'hybrid_reranked'). Default: 'dense'.
        top_k: Default number of results to return.
        max_top_k: Maximum allowed top-K.
        similarity_metric: Dense similarity metric (e.g. 'cosine').
        min_score_threshold: Minimum score filter.
        hybrid: Hybrid retrieval configuration.
        bm25: BM25 configuration.
        reranking: Cross-encoder reranking configuration.
    """

    strategy: str = "dense"
    top_k: int = 5
    max_top_k: int = 50
    similarity_metric: str = "cosine"
    min_score_threshold: float = 0.0
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    reranking: RerankingConfig = field(default_factory=RerankingConfig)

    def __post_init__(self) -> None:
        valid_strategies = {"dense", "hybrid", "hybrid_reranked"}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"RetrievalConfig.strategy must be one of {valid_strategies}, got '{self.strategy}'."
            )
        if self.top_k < 1:
            raise ValueError("RetrievalConfig.top_k must be >= 1.")
        if self.max_top_k < self.top_k:
            raise ValueError("RetrievalConfig.max_top_k must be >= top_k.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RAGConfig:
    """RAG Demo configuration.

    Attributes:
        enabled: Enable or disable the RAG demo module.
        chunker: Text chunking settings.
        embedder: Embedding model settings.
        vector_store: Vector store backend settings.
        generation: Optional LLM generation settings.
        context: Optional Context building settings.
        retrieval: Optional Retrieval settings.
    """

    enabled: bool
    chunker: ChunkerConfig
    embedder: EmbedderConfig
    vector_store: VectorStoreConfig
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================
# Adaptive Routing Config (Phase G4)
# =============================================================


@dataclass
class GPUThresholdConfig:
    """Configurable thresholds for GPU work allocation.

    Attributes:
        min_workload_complexity: Minimum complexity [0.0, 1.0] to justify GPU execution.
        max_utilization_percent: Threshold above which GPU is considered saturated.
        min_memory_available_mb: Minimum free memory required before scheduling on GPU.
    """

    min_workload_complexity: float = 0.35
    max_utilization_percent: float = 85.0
    min_memory_available_mb: float = 256.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_workload_complexity <= 1.0):
            raise ValueError("GPUThresholdConfig.min_workload_complexity must be in [0.0, 1.0].")
        if not (0.0 <= self.max_utilization_percent <= 100.0):
            raise ValueError("GPUThresholdConfig.max_utilization_percent must be in [0.0, 100.0].")
        if self.min_memory_available_mb < 0.0:
            raise ValueError("GPUThresholdConfig.min_memory_available_mb must be >= 0.0.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CPUThresholdConfig:
    """Configurable thresholds for CPU work allocation.

    Attributes:
        max_utilization_percent: Threshold above which CPU is considered saturated.
    """

    max_utilization_percent: float = 80.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.max_utilization_percent <= 100.0):
            raise ValueError("CPUThresholdConfig.max_utilization_percent must be in [0.0, 100.0].")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SmallWorkloadConfig:
    """Configurable definition of small workloads favoring CPU execution.

    Attributes:
        max_pages: Maximum page count considered a small workload.
        max_complexity: Maximum complexity score considered simple.
    """

    max_pages: int = 1
    max_complexity: float = 0.20

    def __post_init__(self) -> None:
        if self.max_pages < 1:
            raise ValueError("SmallWorkloadConfig.max_pages must be >= 1.")
        if not (0.0 <= self.max_complexity <= 1.0):
            raise ValueError("SmallWorkloadConfig.max_complexity must be in [0.0, 1.0].")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HysteresisConfig:
    """Anti-oscillation hysteresis and cooldown settings.

    Attributes:
        enabled: Whether hysteresis dampening is active.
        cooldown_seconds: Minimum time between device switches.
        utilization_delta_threshold: Required margin above threshold to force a switch.
    """

    enabled: bool = True
    cooldown_seconds: float = 2.0
    utilization_delta_threshold: float = 5.0

    def __post_init__(self) -> None:
        if self.cooldown_seconds < 0.0:
            raise ValueError("HysteresisConfig.cooldown_seconds must be >= 0.0.")
        if self.utilization_delta_threshold < 0.0:
            raise ValueError("HysteresisConfig.utilization_delta_threshold must be >= 0.0.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveRoutingConfig:
    """Adaptive CPU/GPU Workload Routing Configuration (Phase G4).

    Attributes:
        enabled: Enable or disable adaptive work routing.
        policy_version: Version identifier for the routing policy.
        gpu: GPU capacity and eligibility thresholds.
        cpu: CPU capacity thresholds.
        small_workload: Small workload thresholds favoring CPU.
        hysteresis: Anti-oscillation hysteresis parameters.
    """

    enabled: bool = True
    policy_version: str = "v1.0"
    gpu: GPUThresholdConfig = field(default_factory=GPUThresholdConfig)
    cpu: CPUThresholdConfig = field(default_factory=CPUThresholdConfig)
    small_workload: SmallWorkloadConfig = field(default_factory=SmallWorkloadConfig)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


