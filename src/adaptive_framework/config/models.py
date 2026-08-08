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

SUPPORTED_OCR_BACKENDS = frozenset({"paddleocr", "trocr", "nougat", "mineru", "docling"})


@dataclass
class OCRConfig:
    """OCR sub-component configuration.

    Attributes:
        languages: Language codes for OCR.
        use_gpu: Enable GPU acceleration.
        confidence_threshold: Minimum confidence to accept OCR output.
    """

    languages: list[str]
    use_gpu: bool
    confidence_threshold: float

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
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Token overlap between consecutive chunks.
    """

    strategy: str
    chunk_size: int
    chunk_overlap: int

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
    """

    model: str
    device: str
    batch_size: int
    embedding_dim: int

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
    """

    backend: str
    persist_dir: str
    collection_name: str

    def __post_init__(self) -> None:
        valid_backends = {"chromadb", "faiss", "qdrant"}
        if self.backend not in valid_backends:
            raise ValueError(
                f"VectorStoreConfig.backend must be one of {valid_backends}."
            )
        if not self.collection_name:
            raise ValueError("VectorStoreConfig.collection_name must not be empty.")

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
    """

    enabled: bool
    chunker: ChunkerConfig
    embedder: EmbedderConfig
    vector_store: VectorStoreConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
