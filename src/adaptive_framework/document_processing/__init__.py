"""Document Processing Engine — Phase 3 implementation.

Architecture v2.0 §2.1:
    PDF → PageWorkUnit → Adaptive Scheduler → Distributed Workers
    → ProcessingStrategy → ZeroCopyLoader → Direct Text / OCR
    → Layout Analysis → Tables → Figures → Validation
    → Page Objects → Coordinator Merge → UnifiedDocument

Public API:
    DocumentProcessingWorker: Worker-side entry point (receives PageWorkUnit)
    UnifiedDocumentBuilder: Coordinator-side merge (produces UnifiedDocument)
    PageObjectBuilder: Per-page pipeline orchestrator
    HealthChecker: System dependency health check
    BenchmarkLogger: Event bus → CSV performance logger
"""

from adaptive_framework.document_processing.benchmark_logger import BenchmarkLogger
from adaptive_framework.document_processing.cost_estimator import (
    IProcessingCostEstimator,
    NullCostEstimator,
    ProcessingCostEstimate,
)
from adaptive_framework.document_processing.document_type_detector import (
    DocumentClassificationSummary,
    DocumentTypeDetector,
    PageClassification,
)
from adaptive_framework.document_processing.event_bus import EventBus, get_default_bus
from adaptive_framework.document_processing.health_check import (
    HealthChecker,
    SystemHealthReport,
)
from adaptive_framework.document_processing.page_builder import PageObjectBuilder
from adaptive_framework.document_processing.page_validator import (
    PageValidationResult,
    PageValidator,
)
from adaptive_framework.document_processing.pdf_analyzer import (
    PDFAnalysisResult,
    PDFAnalyzer,
)
from adaptive_framework.document_processing.processing_strategy import (
    DirectExtractionStrategy,
    IProcessingStrategy,
    MixedStrategy,
    OCRStrategy,
    ProcessingStrategyFactory,
)
from adaptive_framework.document_processing.processing_worker import (
    DocumentProcessingWorker,
    WorkerProcessingResult,
)
from adaptive_framework.document_processing.unified_document_builder import (
    UnifiedDocumentBuilder,
)
from adaptive_framework.document_processing.zero_copy_loader import (
    LoadedPage,
    ZeroCopyPageLoader,
)

__all__ = [
    # Worker side
    "DocumentProcessingWorker",
    "WorkerProcessingResult",
    "PageObjectBuilder",
    # Coordinator side
    "UnifiedDocumentBuilder",
    # Strategy pattern
    "IProcessingStrategy",
    "DirectExtractionStrategy",
    "OCRStrategy",
    "MixedStrategy",
    "ProcessingStrategyFactory",
    # Analysis
    "PDFAnalyzer",
    "PDFAnalysisResult",
    "DocumentTypeDetector",
    "DocumentClassificationSummary",
    "PageClassification",
    "ZeroCopyPageLoader",
    "LoadedPage",
    # Validation
    "PageValidator",
    "PageValidationResult",
    # Event system
    "EventBus",
    "get_default_bus",
    "BenchmarkLogger",
    # Cost estimation interface
    "IProcessingCostEstimator",
    "NullCostEstimator",
    "ProcessingCostEstimate",
    # Health
    "HealthChecker",
    "SystemHealthReport",
]