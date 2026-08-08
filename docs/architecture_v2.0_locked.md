# Adaptive Distributed Framework — Architecture v2.0 (LOCKED)

> **Status: LOCKED ✅**
> Design finalized after mentor review. Stop redesigning — start implementing.

---

## Summary of Changes from Draft → v2.0

| Change | Description |
|--------|-------------|
| ✅ Change 1 | Renamed "OCR Engine" → "Document Processing Engine" |
| ✅ Change 2 | Added "Scheduler Overhead" metric to evaluation |
| ✅ Change 3 | Added "Failure Recovery" to Distributed Coordinator |
| ✅ Change 4 | Extended Metadata Generator fields |
| ✅ Change 5 | Renamed "Adaptive Runtime Scaling" → "Adaptive Runtime Resource Orchestration" |
| ❌ Removed | Parallel Token Prediction (moved to Future Work) |

---

## 1. System Overview

The framework implements **Adaptive Runtime Resource Orchestration** across heterogeneous compute nodes (laptops, desktops, edge devices) to process document-heavy workloads (PDF pipelines, RAG ingestion) with minimal latency and energy cost.

---

## 2. Core Components

### 2.1 Document Processing Engine *(formerly "OCR Engine")*

```
Document Processing Engine
├── OCR
├── Layout Analysis
├── Table Extraction
└── Figure Detection
```

**Design Rationale:**
The component is named "Document Processing Engine" rather than "OCR Engine" to remain future-proof. The OCR backend can be swapped without any architectural change:

| Potential Backend | Notes |
|-------------------|-------|
| PaddleOCR | Current default candidate |
| TrOCR | Transformer-based, high accuracy |
| Nougat | Scientific PDF specialized |
| MinerU | Structured document extraction |
| Docling | IBM's document understanding |

---

### 2.2 Adaptive Scheduler

**Algorithm:** Page-count-based scheduling with Work Stealing.

```
Task Queue (Priority by Page Count)
        ↓
   Scheduler
  /    |    \
W1    W2    W3
```

- **Work Stealing**: Idle workers steal tasks from overloaded peers.
- **Zero-Copy**: Shared memory regions avoid redundant data copying.
- **Scheduler Overhead Target**: < 1% of total execution time (see §4.2).

---

### 2.3 Distributed Coordinator

**Transport:** Ray

```
Coordinator
├── Worker Registry
├── Task Dispatcher
├── Heartbeat Monitor
└── Failure Recovery           ← NEW in v2.0
```

#### Failure Recovery Flow

```
Worker Lost (e.g. Laptop 2 disconnects)
        ↓
Detect via Heartbeat Timeout
        ↓
Return Unfinished Work Units
        ↓
Re-insert into Priority Queue
        ↓
Assign to Available Worker
```

- A basic retry mechanism with configurable timeout prevents a single node failure from crashing the entire pipeline.
- Ray's actor model handles the heartbeat and reassignment natively.

---

### 2.4 Metadata Generator

Each document produces a structured metadata record:

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

| Field | Status | Notes |
|-------|--------|-------|
| `pages` | Required | Core scheduler input |
| `estimated_size_mb` | Required | Scheduling weight |
| `resolution_dpi` | Optional | Populated if detectable |
| `source_type` | Optional | `scanned` vs `digital` |
| `language` | Optional | ISO 639-1 code |

> Fields marked Optional may be `null` in early versions. The schema is defined now so future extensions require no breaking changes.

---

### 2.5 RAG Demo

Demonstrates end-to-end pipeline: Document ingestion → Chunking → Embedding → Vector Store → Query.

*Keep as-is. No changes.*

---

## 3. Transport & Memory Layer

| Mechanism | Purpose |
|-----------|---------|
| Ray | Distributed actor coordination |
| Shared Memory | Zero-copy inter-process buffers |
| Priority Queue | Page-count ordered task dispatch |

---

## 4. Evaluation Plan

### 4.1 Primary Metrics

| Metric | Description |
|--------|-------------|
| Speedup | Multi-node vs single-node time ratio |
| Throughput | Pages processed per second |
| CPU Utilization | Per-node CPU % |
| GPU Utilization | Per-node GPU % (if applicable) |
| Energy Consumption | Joules per document batch |
| **Scheduler Overhead** | **% of total time spent in scheduler** ← NEW |

### 4.2 Scheduler Overhead — Definition

```
Scheduler Overhead (%) = (Scheduler Time / Total Execution Time) × 100
```

**Target:** < 1%
**Why it matters:** A scheduler that spends 30 seconds deciding on a 60-second workload wastes 50% of the run. This metric proves the scheduler is lightweight and does not eat its own gains.

**How to measure:**
1. Instrument the scheduler with `time.perf_counter()` around the dispatch loop.
2. Record cumulative scheduler time across the full batch.
3. Divide by wall-clock total execution time.

---

## 5. Terminology

| Old Term | New Term (v2.0) |
|----------|-----------------|
| OCR Engine | Document Processing Engine |
| Adaptive Runtime Scaling | Adaptive Runtime Resource Orchestration |

"Resource orchestration" aligns with established distributed systems and cloud computing literature, making the paper more precise and search-discoverable.

---

## 6. Future Work (Out of Scope for Implementation)

### Advanced Parallel Decoding
- **Parallel Token Prediction (PTP)** — Speculative decoding approach where multiple tokens are predicted in parallel and verified. This is an active research area on its own. Mention in the paper's future work section; do **not** implement.

### Other Future Directions
- Dynamic heterogeneous hardware profiling
- Auto-tuning chunk sizes based on document complexity
- GPU-accelerated layout analysis

---

## 7. What is NOT Changing

| Component | Decision |
|-----------|----------|
| Ray | ✅ Keep |
| Page Count Scheduling | ✅ Keep |
| Work Stealing | ✅ Keep |
| Zero Copy | ✅ Keep |
| RAG Demo | ✅ Keep |
| Evaluation Design | ✅ Keep |

---

## 8. Implementation Priorities (Next Phase)

> Architecture is locked. Now build.

| Week | Task |
|------|------|
| 1–2 | Ray cluster setup + Document Processing Engine shell with OCR backend |
| 3–4 | Scheduler with Work Stealing + Overhead instrumentation |
| 5–6 | Distributed Coordinator with Failure Recovery |
| 7–8 | Metadata Generator + RAG Demo integration |
| 9–10 | Evaluation runs, data collection, paper writing |

---

*Document locked: 2026-07-26 | Version: 2.0 | Do not modify architecture after this point.*
