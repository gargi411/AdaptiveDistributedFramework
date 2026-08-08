# Interfaces Package

## Purpose

All **Abstract Base Classes (ABCs)** for the Adaptive Distributed Framework.

Every major component is defined here as an interface. Concrete implementations live in their respective module packages and are wired together via the DI container.

## Design Principles

- **Dependency Inversion**: High-level components depend on abstractions, not concretions.
- **Single Responsibility**: Each interface defines exactly one component's contract.
- **Replaceability**: Any implementation can be swapped without affecting other components.

## Interface Inventory

| Interface | Purpose | Implemented By (future) |
|-----------|---------|------------------------|
| `ILogger` | Logging abstraction | `FrameworkLogger` |
| `IConfigProvider` | Configuration abstraction | `ConfigManager` |
| `IDatasetBuilder` | PDF dataset scanning | Phase 2 |
| `IPartitionStrategy` | Work partitioning | Phase 3 (PageCountStrategy) |
| `IScheduler` | Adaptive scheduling loop | Phase 3 |
| `IWorker` | Worker node processing | Phase 2 (RayWorker) |
| `IResultCollector` | Result aggregation | Phase 2 |
| `IOCREngine` | OCR backend abstraction | Phase 2 (PaddleOCR) |
| `IDocumentProcessor` | Document Processing Engine orchestration | Phase 2 |
| `IChunker` | Text chunking (RAG) | Phase 5 |
| `IEmbedder` | Text embedding (RAG) | Phase 5 |
| `IVectorStore` | Vector store (RAG) | Phase 5 |
| `IReportGenerator` | Evaluation report writing | Phase 6 |

## Architecture Rules Enforced by These Interfaces

```
Scheduler  →  IWorker        (not RayWorker directly)
Worker     →  IDocumentProcessor  (not PaddleOCR directly)
RAG        →  IVectorStore   (not ChromaDB directly)
Evaluation →  IReportGenerator   (not specific format)
```

## Dependency Direction

```
interfaces  →  models
interfaces  →  core (exceptions)
```

No imports from config, logging, or di packages.
