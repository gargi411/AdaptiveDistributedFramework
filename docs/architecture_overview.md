# Architecture Overview

## Adaptive Distributed Parallel Processing Framework — v2.0 (LOCKED)

---

## System Overview

The framework implements **Adaptive Runtime Resource Orchestration** across heterogeneous compute nodes to process large-scale biomedical PDF corpora with minimal latency and energy cost.

---

## Core Components

### 1. Document Processing Engine (§2.1)

```
Document Processing Engine
├── OCR                 → extracts text from PDF pages
├── Layout Analysis     → identifies regions (headers, body, footnotes)
├── Table Extraction    → parses tabular data
└── Figure Detection    → identifies and crops figures
```

- **Swappable backends**: PaddleOCR, TrOCR, Nougat, MinerU, Docling
- Interface: `IOCREngine`, `IDocumentProcessor`

### 2. Adaptive Scheduler (§2.2)

```
Task Queue (Priority by Page Count)
        ↓
   Scheduler
  /    |    \
W1    W2    W3
```

- **Page-count priority**: larger documents processed first
- **Work Stealing**: idle workers steal from overloaded peers
- **Zero-Copy**: Ray Plasma shared memory
- **Overhead target**: < 1% (measured and reported)
- Interface: `IScheduler`, `IPartitionStrategy`

### 3. Distributed Coordinator (§2.3)

```
Coordinator
├── Worker Registry
├── Task Dispatcher
├── Heartbeat Monitor
└── Failure Recovery
```

**Failure Recovery Flow**:
```
Worker Lost → Heartbeat Timeout → Return Work → Priority Queue → Another Worker
```

- Interface: `IWorker`

### 4. Metadata Generator (§2.4)

```json
{
  "document_id": "<uuid>",
  "pages": 42,
  "estimated_size_mb": 3.7,
  "resolution_dpi": 300,
  "source_type": "scanned | digital",
  "language": "en",
  "processing_timestamp": "<iso8601>"
}
```

### 5. RAG Demo (§2.5)

```
Document Text → Chunking → Embedding → Vector Store → Query
```

### 6. Evaluation Engine (§4)

| Metric | Target |
|--------|--------|
| Speedup | > 1.0 |
| Throughput | maximize |
| CPU/GPU Utilization | monitor |
| Energy | minimize |
| **Scheduler Overhead** | **< 1%** |

---

## Dependency Graph

```
utils          → core
core           → (root)
models         → core
interfaces     → models, core
config         → utils, core
logging        → interfaces, core
di             → core

Phase 2+:
document_processing → interfaces, models
scheduler           → interfaces, models
coordinator         → interfaces, models
dataset_builder     → interfaces, models
rag                 → interfaces, models
evaluation          → interfaces, models
```

---

## Transport & Memory Layer

| Mechanism | Purpose |
|-----------|---------|
| Ray | Distributed actor coordination |
| Shared Memory (Plasma) | Zero-copy inter-process buffers |
| Priority Queue | Page-count ordered task dispatch |

---

## Architecture Rules (Enforced)

1. Scheduler must not know OCR implementation.
2. OCR must not know Scheduler implementation.
3. RAG layer must never depend on scheduling internals.
4. Every module is replaceable without affecting others.
5. Evaluation depends only on public interfaces.
6. Infrastructure (logging, config, DI) is independent of business logic.
