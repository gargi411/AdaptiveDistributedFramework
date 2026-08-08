"""Abstract interfaces package for the Adaptive Distributed Framework.

Every major component is defined here as an ABC (Abstract Base Class).
Concrete implementations are in the respective module packages and are
registered/injected via the DI container.

This separation enforces the architecture rule:
    Scheduler must not know OCR implementation.
    OCR must not know Scheduler implementation.
    RAG must never depend on scheduling internals.
"""

from adaptive_framework.interfaces.i_chunker import IChunker
from adaptive_framework.interfaces.i_config_provider import IConfigProvider
from adaptive_framework.interfaces.i_dataset_builder import IDatasetBuilder
from adaptive_framework.interfaces.i_document_processor import IDocumentProcessor
from adaptive_framework.interfaces.i_embedder import IEmbedder
from adaptive_framework.interfaces.i_logger import ILogger
from adaptive_framework.interfaces.i_ocr_engine import IOCREngine
from adaptive_framework.interfaces.i_partition_strategy import IPartitionStrategy
from adaptive_framework.interfaces.i_report_generator import IReportGenerator
from adaptive_framework.interfaces.i_result_collector import IResultCollector
from adaptive_framework.interfaces.i_scheduler import IScheduler
from adaptive_framework.interfaces.i_vector_store import IVectorStore
from adaptive_framework.interfaces.i_worker import IWorker

__all__ = [
    "ILogger",
    "IConfigProvider",
    "IDatasetBuilder",
    "IPartitionStrategy",
    "IScheduler",
    "IWorker",
    "IResultCollector",
    "IOCREngine",
    "IDocumentProcessor",
    "IChunker",
    "IEmbedder",
    "IVectorStore",
    "IReportGenerator",
]
