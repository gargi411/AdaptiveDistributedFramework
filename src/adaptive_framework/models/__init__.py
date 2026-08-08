"""Models package for the Adaptive Distributed Framework.

Contains all domain data models as pure Python dataclasses.
No algorithms. No business logic. Only structured data.

Modules:
    document: PDFMetadata, PageMetadata, DocumentResult, PageResult
    scheduling: PageWorkUnit, Partition, PartitionStatistics
    runtime: ResourceSnapshot, RuntimeMetrics, WorkerStatus, ClusterStatus, FrameworkStatus
    evaluation: EvaluationResult
    page: Page, TextBlock, TableData, FigureData, LayoutElement, PageStatistics (Phase 3)
    unified_document: UnifiedDocument, DocumentLayout, DocumentStatistics (Phase 3)
    events: ProcessingEvent, EventType (Phase 3)
    processing_metrics: PageProcessingMetrics, StageMetrics (Phase 3)
"""

from adaptive_framework.models.document import (
    DocumentResult,
    PageMetadata,
    PageResult,
    PDFMetadata,
)
from adaptive_framework.models.evaluation import EvaluationResult
from adaptive_framework.models.events import EventType, ProcessingEvent
from adaptive_framework.models.page import (
    BoundingBox,
    FigureData,
    LayoutElement,
    Page,
    PageStatistics,
    PageType,
    ProcessingMethod,
    TableData,
    TextBlock,
)
from adaptive_framework.models.processing_metrics import (
    PageProcessingMetrics,
    StageMetrics,
)
from adaptive_framework.models.runtime import (
    ClusterStatus,
    FrameworkStatus,
    ResourceSnapshot,
    RuntimeMetrics,
    WorkerStatus,
)
from adaptive_framework.models.scheduling import (
    PageWorkUnit,
    Partition,
    PartitionStatistics,
)
from adaptive_framework.models.unified_document import (
    DocumentLayout,
    DocumentStatistics,
    UnifiedDocument,
)

__all__ = [
    # document (Phase 1/2)
    "PDFMetadata",
    "PageMetadata",
    "DocumentResult",
    "PageResult",
    # scheduling
    "PageWorkUnit",
    "Partition",
    "PartitionStatistics",
    # runtime
    "ResourceSnapshot",
    "RuntimeMetrics",
    "WorkerStatus",
    "ClusterStatus",
    "FrameworkStatus",
    # evaluation
    "EvaluationResult",
    # page models (Phase 3)
    "BoundingBox",
    "TextBlock",
    "TableData",
    "FigureData",
    "LayoutElement",
    "PageStatistics",
    "Page",
    "PageType",
    "ProcessingMethod",
    # unified document (Phase 3)
    "UnifiedDocument",
    "DocumentLayout",
    "DocumentStatistics",
    # events (Phase 3)
    "ProcessingEvent",
    "EventType",
    # metrics (Phase 3)
    "PageProcessingMetrics",
    "StageMetrics",
]
