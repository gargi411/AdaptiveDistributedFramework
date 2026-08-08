"""Singleton ConfigManager for the Adaptive Distributed Framework.

Responsibilities:
    - Load all YAML configuration files from a directory.
    - Validate and parse YAML into typed configuration models.
    - Support hot reload via reload().
    - Expose typed accessors for each config section.
    - Enforce the Singleton pattern — one instance per process.

Design:
    This is a pure infrastructure component. It has zero knowledge of
    scheduling, OCR, Ray, or any business logic. All components receive
    configuration via constructor injection, never by calling
    ConfigManager directly inside business logic.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from adaptive_framework.config.models import (
    ChunkerConfig,
    ConsoleLoggingConfig,
    DocumentProcessingEngineConfig,
    EmbedderConfig,
    EvaluationConfig,
    FileLoggingConfig,
    FrameworkConfig,
    LoggingConfig,
    OCRConfig,
    RAGConfig,
    RayClusterConfig,
    SchedulerConfig,
    SchedulerOverheadConfig,
    SchedulerOverheadMetricConfig,
    SharedMemoryConfig,
    VectorStoreConfig,
    WorkerConfig,
    WorkStealingConfig,
)
from adaptive_framework.core.exceptions import ConfigurationError
from adaptive_framework.utils.yaml_utils import load_yaml_file


class ConfigManager:
    """Singleton configuration manager.

    Loads, validates, and exposes all YAML configuration sections as
    strongly-typed Python dataclass instances.

    The singleton is thread-safe: the first caller to ``get_instance()``
    creates the object; subsequent calls return the same object.

    Attributes:
        _instance: The single shared instance (class variable).
        _lock: Thread lock guarding instance creation.
        _config_dir: Path to the configs directory.
        _raw: Raw merged YAML data from all config files.
        _loaded: Whether configs have been loaded at least once.

    Example:
        >>> cfg = ConfigManager.get_instance()
        >>> cfg.load(config_dir=Path("configs"))
        >>> fw = cfg.get_framework_config()
        >>> print(fw.name)
        'Adaptive Distributed Parallel Processing Framework'
    """

    _instance: ConfigManager | None = None
    _lock: threading.Lock = threading.Lock()

    # Names of supported YAML config files
    _CONFIG_FILES: tuple[str, ...] = (
        "framework.yaml",
        "logging.yaml",
        "ray_cluster.yaml",
        "scheduler.yaml",
        "ocr.yaml",
        "evaluation.yaml",
        "rag.yaml",
    )

    def __init__(self) -> None:
        """Initialize an empty ConfigManager.

        Do not call this directly. Use ``ConfigManager.get_instance()``.
        """
        self._config_dir: Path | None = None
        self._raw: dict[str, Any] = {}
        self._loaded: bool = False

    @classmethod
    def get_instance(cls) -> ConfigManager:
        """Return the singleton ConfigManager instance.

        Creates the instance on the first call. Thread-safe.

        Returns:
            The singleton ConfigManager.

        Example:
            >>> cfg = ConfigManager.get_instance()
            >>> cfg is ConfigManager.get_instance()
            True
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load(self, config_dir: Path | str = "configs") -> None:
        """Load all YAML configuration files from the given directory.

        Reads each file listed in ``_CONFIG_FILES`` that exists in
        ``config_dir`` and merges them into a single internal dictionary.
        Missing optional files are silently skipped.

        Args:
            config_dir: Path to the directory containing YAML files.
                Defaults to 'configs' (relative to CWD).

        Raises:
            ConfigurationError: If the directory does not exist or any
                required YAML file fails to parse.

        Example:
            >>> cfg = ConfigManager.get_instance()
            >>> cfg.load(config_dir=Path("configs"))
        """
        config_path = Path(config_dir)
        if not config_path.is_dir():
            raise ConfigurationError(
                f"Configuration directory not found: '{config_path.resolve()}'. "
                "Create the directory or pass the correct path to load()."
            )

        self._config_dir = config_path
        self._raw = {}

        for filename in self._CONFIG_FILES:
            file_path = config_path / filename
            if file_path.exists():
                try:
                    data = load_yaml_file(file_path)
                    if data:
                        self._raw.update(data)
                except Exception as exc:
                    raise ConfigurationError(
                        f"Failed to parse configuration file '{file_path}': {exc}"
                    ) from exc

        self._loaded = True

    def reload(self) -> None:
        """Hot-reload all configuration files from the same directory.

        Re-reads all YAML files without restarting the process. Useful for
        updating configuration at runtime without downtime.

        Raises:
            ConfigurationError: If load() has never been called, or if
                reload fails for any file.

        Example:
            >>> cfg = ConfigManager.get_instance()
            >>> cfg.reload()   # re-reads all configs in-place
        """
        if not self._loaded or self._config_dir is None:
            raise ConfigurationError(
                "Cannot reload: ConfigManager.load() has not been called yet."
            )
        self.load(self._config_dir)

    def _require_loaded(self) -> None:
        """Assert that configuration has been loaded.

        Raises:
            ConfigurationError: If load() has not been called.
        """
        if not self._loaded:
            raise ConfigurationError(
                "Configuration has not been loaded. Call ConfigManager.load() first."
            )

    # ------------------------------------------------------------------
    # Typed config accessors
    # ------------------------------------------------------------------

    def get_framework_config(self) -> FrameworkConfig:
        """Return the typed FrameworkConfig.

        Returns:
            Validated FrameworkConfig dataclass instance.

        Raises:
            ConfigurationError: If configs not loaded or keys missing.
        """
        self._require_loaded()
        try:
            fw = self._raw["framework"]
            return FrameworkConfig(
                name=fw["name"],
                version=fw["version"],
                run_id_prefix=fw["run_id_prefix"],
                output_dir=fw["output_dir"],
                debug=fw["debug"],
                max_concurrent_jobs=fw["max_concurrent_jobs"],
                stage_timeout_seconds=fw["stage_timeout_seconds"],
                shutdown_timeout_seconds=fw["shutdown_timeout_seconds"],
            )
        except KeyError as exc:
            raise ConfigurationError(
                f"Missing required key in framework.yaml: {exc}"
            ) from exc

    def get_logging_config(self) -> LoggingConfig:
        """Return the typed LoggingConfig.

        Returns:
            Validated LoggingConfig dataclass instance.

        Raises:
            ConfigurationError: If configs not loaded or keys missing.
        """
        self._require_loaded()
        try:
            lg = self._raw["logging"]
            console_raw = lg["console"]
            file_raw = lg["file"]
            json_file_raw = lg["json_file"]
            return LoggingConfig(
                level=lg["level"],
                format=lg["format"],
                console=ConsoleLoggingConfig(
                    enabled=console_raw["enabled"],
                    level=console_raw["level"],
                    use_rich=console_raw["use_rich"],
                    colorize=console_raw["colorize"],
                ),
                file=FileLoggingConfig(
                    enabled=file_raw["enabled"],
                    level=file_raw["level"],
                    path=file_raw["path"],
                    max_bytes=file_raw["max_bytes"],
                    backup_count=file_raw["backup_count"],
                    encoding=file_raw["encoding"],
                ),
                json_file=FileLoggingConfig(
                    enabled=json_file_raw["enabled"],
                    level=json_file_raw["level"],
                    path=json_file_raw["path"],
                    max_bytes=json_file_raw["max_bytes"],
                    backup_count=json_file_raw["backup_count"],
                    encoding=json_file_raw["encoding"],
                ),
                context_fields=lg.get("context_fields", []),
            )
        except KeyError as exc:
            raise ConfigurationError(
                f"Missing required key in logging.yaml: {exc}"
            ) from exc

    def get_ray_cluster_config(self) -> RayClusterConfig:
        """Return the typed RayClusterConfig.

        Returns:
            Validated RayClusterConfig dataclass instance.

        Raises:
            ConfigurationError: If configs not loaded or keys missing.
        """
        self._require_loaded()
        try:
            rc = self._raw["ray_cluster"]
            worker_raw = rc["worker"]
            sm_raw = rc["shared_memory"]
            return RayClusterConfig(
                address=rc["address"],
                num_cpus=rc.get("num_cpus"),
                num_gpus=rc.get("num_gpus"),
                object_store_memory=rc.get("object_store_memory"),
                init_timeout_seconds=rc["init_timeout_seconds"],
                worker=WorkerConfig(
                    heartbeat_interval_seconds=worker_raw["heartbeat_interval_seconds"],
                    heartbeat_timeout_seconds=worker_raw["heartbeat_timeout_seconds"],
                    task_retry_attempts=worker_raw["task_retry_attempts"],
                    retry_delay_seconds=worker_raw["retry_delay_seconds"],
                ),
                shared_memory=SharedMemoryConfig(
                    enabled=sm_raw["enabled"],
                    max_buffer_size=sm_raw["max_buffer_size"],
                ),
            )
        except KeyError as exc:
            raise ConfigurationError(
                f"Missing required key in ray_cluster.yaml: {exc}"
            ) from exc

    def get_scheduler_config(self) -> SchedulerConfig:
        """Return the typed SchedulerConfig.

        Returns:
            Validated SchedulerConfig dataclass instance.

        Raises:
            ConfigurationError: If configs not loaded or keys missing.
        """
        self._require_loaded()
        try:
            sc = self._raw["scheduler"]
            ws_raw = sc["work_stealing"]
            oh_raw = sc["overhead_monitoring"]
            part_raw = sc["partition"]
            return SchedulerConfig(
                strategy=sc["strategy"],
                work_stealing=WorkStealingConfig(
                    enabled=ws_raw["enabled"],
                    steal_threshold=ws_raw["steal_threshold"],
                    steal_fraction=ws_raw["steal_fraction"],
                    check_interval_seconds=ws_raw["check_interval_seconds"],
                ),
                overhead_monitoring=SchedulerOverheadConfig(
                    enabled=oh_raw["enabled"],
                    target_max_overhead_fraction=oh_raw["target_max_overhead_fraction"],
                    warn_on_exceed=oh_raw["warn_on_exceed"],
                ),
                default_partition_count=part_raw["default_partition_count"],
                min_pages_per_partition=part_raw["min_pages_per_partition"],
            )
        except KeyError as exc:
            raise ConfigurationError(
                f"Missing required key in scheduler.yaml: {exc}"
            ) from exc

    def get_document_processing_engine_config(self) -> DocumentProcessingEngineConfig:
        """Return the typed DocumentProcessingEngineConfig.

        Returns:
            Validated DocumentProcessingEngineConfig dataclass instance.

        Raises:
            ConfigurationError: If configs not loaded or keys missing.
        """
        self._require_loaded()
        try:
            dpe = self._raw["document_processing_engine"]
            ocr_raw = dpe["ocr"]
            return DocumentProcessingEngineConfig(
                ocr_backend=dpe["ocr_backend"],
                ocr=OCRConfig(
                    languages=ocr_raw["languages"],
                    use_gpu=ocr_raw["use_gpu"],
                    confidence_threshold=ocr_raw["confidence_threshold"],
                ),
                page_timeout_seconds=dpe["page_timeout_seconds"],
                num_threads=dpe["num_threads"],
            )
        except KeyError as exc:
            raise ConfigurationError(
                f"Missing required key in ocr.yaml: {exc}"
            ) from exc

    def get_evaluation_config(self) -> EvaluationConfig:
        """Return the typed EvaluationConfig.

        Returns:
            Validated EvaluationConfig dataclass instance.

        Raises:
            ConfigurationError: If configs not loaded or keys missing.
        """
        self._require_loaded()
        try:
            ev = self._raw["evaluation"]
            so_raw = ev["metrics"]["scheduler_overhead"]
            bench_raw = ev["benchmark"]
            return EvaluationConfig(
                output_dir=ev["output_dir"],
                report_formats=ev["report_formats"],
                scheduler_overhead=SchedulerOverheadMetricConfig(
                    enabled=so_raw["enabled"],
                    target_max_percent=so_raw["target_max_percent"],
                    warn_on_exceed=so_raw["warn_on_exceed"],
                ),
                warmup_runs=bench_raw["warmup_runs"],
                measurement_runs=bench_raw["measurement_runs"],
            )
        except KeyError as exc:
            raise ConfigurationError(
                f"Missing required key in evaluation.yaml: {exc}"
            ) from exc

    def get_rag_config(self) -> RAGConfig:
        """Return the typed RAGConfig.

        Returns:
            Validated RAGConfig dataclass instance.

        Raises:
            ConfigurationError: If configs not loaded or keys missing.
        """
        self._require_loaded()
        try:
            rag = self._raw["rag"]
            chunker_raw = rag["chunker"]
            embedder_raw = rag["embedder"]
            vs_raw = rag["vector_store"]
            return RAGConfig(
                enabled=rag["enabled"],
                chunker=ChunkerConfig(
                    strategy=chunker_raw["strategy"],
                    chunk_size=chunker_raw["chunk_size"],
                    chunk_overlap=chunker_raw["chunk_overlap"],
                ),
                embedder=EmbedderConfig(
                    model=embedder_raw["model"],
                    device=embedder_raw["device"],
                    batch_size=embedder_raw["batch_size"],
                    embedding_dim=embedder_raw["embedding_dim"],
                ),
                vector_store=VectorStoreConfig(
                    backend=vs_raw["backend"],
                    persist_dir=vs_raw["persist_dir"],
                    collection_name=vs_raw["collection_name"],
                ),
            )
        except KeyError as exc:
            raise ConfigurationError(
                f"Missing required key in rag.yaml: {exc}"
            ) from exc

    def get_raw(self) -> dict[str, Any]:
        """Return the raw merged YAML dictionary.

        Useful for debugging. Prefer typed accessors for production code.

        Returns:
            Merged dictionary of all loaded YAML files.
        """
        self._require_loaded()
        return dict(self._raw)
