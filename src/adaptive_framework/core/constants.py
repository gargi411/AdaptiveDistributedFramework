"""Framework-wide constants for the Adaptive Distributed Framework.

Rules:
    - No magic numbers anywhere in the codebase — use these constants.
    - No mutable global state — all constants are immutable.
    - All constants are documented with purpose and units.
"""

from __future__ import annotations

# =============================================================
# Framework Identity
# =============================================================

FRAMEWORK_NAME: str = (
    "Adaptive Distributed Parallel Processing Framework for "
    "Large-Scale Biomedical Document Processing"
)
FRAMEWORK_VERSION: str = "2.0.0"
FRAMEWORK_SHORT_NAME: str = "ADF"

# =============================================================
# Architecture Constraints (from architecture_v2.0_locked.md)
# =============================================================

# §4.2: Scheduler overhead must be strictly less than this fraction.
SCHEDULER_OVERHEAD_TARGET_FRACTION: float = 0.01  # 1%

# Minimum valid page count for a document to be schedulable.
MIN_SCHEDULABLE_PAGE_COUNT: int = 1

# Maximum supported document size for a single work unit (MB).
MAX_WORK_UNIT_SIZE_MB: float = 500.0

# =============================================================
# Logging
# =============================================================

# Logger name used by the root framework logger.
ROOT_LOGGER_NAME: str = "adaptive_framework"

# Format string for human-readable (text) log records.
TEXT_LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "run=%(run_id)s | worker=%(worker_id)s | %(message)s"
)

# ISO 8601 timestamp format for log records.
LOG_TIMESTAMP_FORMAT: str = "%Y-%m-%dT%H:%M:%S"

# =============================================================
# File & Path
# =============================================================

# Extension for PDF documents.
PDF_EXTENSION: str = ".pdf"

# Extension for YAML configuration files.
YAML_EXTENSION: str = ".yaml"

# Default encoding for all text files written by the framework.
DEFAULT_ENCODING: str = "utf-8"

# Default output directory name (relative to CWD).
DEFAULT_OUTPUT_DIR: str = "outputs"

# Default configuration directory name (relative to CWD).
DEFAULT_CONFIG_DIR: str = "configs"

# =============================================================
# Scheduling
# =============================================================

# Identifier for the page-count scheduling strategy.
STRATEGY_PAGE_COUNT: str = "page_count"

# Default number of retry attempts when a worker is lost.
DEFAULT_TASK_RETRY_ATTEMPTS: int = 3

# Seconds to wait between task retry attempts.
DEFAULT_RETRY_DELAY_SECONDS: float = 2.0

# =============================================================
# Document Processing Engine
# =============================================================

# Supported OCR backend identifiers (must match ocr.yaml and config models).
OCR_BACKEND_PADDLEOCR: str = "paddleocr"
OCR_BACKEND_TROCR: str = "trocr"
OCR_BACKEND_NOUGAT: str = "nougat"
OCR_BACKEND_MINERU: str = "mineru"
OCR_BACKEND_DOCLING: str = "docling"

SUPPORTED_OCR_BACKENDS: frozenset[str] = frozenset(
    {
        OCR_BACKEND_PADDLEOCR,
        OCR_BACKEND_TROCR,
        OCR_BACKEND_NOUGAT,
        OCR_BACKEND_MINERU,
        OCR_BACKEND_DOCLING,
    }
)

# Minimum OCR confidence to accept a result (fraction, 0.0–1.0).
MIN_OCR_CONFIDENCE: float = 0.0

# Maximum OCR confidence (fraction, 0.0–1.0).
MAX_OCR_CONFIDENCE: float = 1.0

# =============================================================
# Metadata
# =============================================================

# Allowed source_type values in PDFMetadata.
SOURCE_TYPE_SCANNED: str = "scanned"
SOURCE_TYPE_DIGITAL: str = "digital"

VALID_SOURCE_TYPES: frozenset[str] = frozenset(
    {SOURCE_TYPE_SCANNED, SOURCE_TYPE_DIGITAL}
)

# Minimum valid DPI for resolution_dpi in PDFMetadata.
MIN_RESOLUTION_DPI: int = 1

# =============================================================
# RAG
# =============================================================

# Supported vector store backend identifiers.
VECTOR_STORE_CHROMADB: str = "chromadb"
VECTOR_STORE_FAISS: str = "faiss"
VECTOR_STORE_QDRANT: str = "qdrant"

SUPPORTED_VECTOR_STORES: frozenset[str] = frozenset(
    {VECTOR_STORE_CHROMADB, VECTOR_STORE_FAISS, VECTOR_STORE_QDRANT}
)

# =============================================================
# Evaluation
# =============================================================

# Supported report format identifiers.
REPORT_FORMAT_JSON: str = "json"
REPORT_FORMAT_CSV: str = "csv"
REPORT_FORMAT_MARKDOWN: str = "markdown"

SUPPORTED_REPORT_FORMATS: frozenset[str] = frozenset(
    {REPORT_FORMAT_JSON, REPORT_FORMAT_CSV, REPORT_FORMAT_MARKDOWN}
)

# Baseline number of nodes for Speedup calculation (single-node reference).
SPEEDUP_BASELINE_NODES: int = 1

# =============================================================
# Numeric Sentinels
# =============================================================

# Represents an unknown or not-yet-measured float value.
UNKNOWN_FLOAT: float = -1.0

# Represents an unknown or not-yet-measured integer value.
UNKNOWN_INT: int = -1
