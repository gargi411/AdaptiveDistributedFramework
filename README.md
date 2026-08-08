# Adaptive Distributed Parallel Processing Framework

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## Project Title

**Adaptive Distributed Parallel Processing Framework for Large-Scale Biomedical Document Processing using Intelligent Workload Scheduling**

---

## Research Context

This framework is developed as a **publication-quality research project** and **final-year project** targeting:

- Research publication in distributed systems / biomedical informatics
- Demonstrating adaptive runtime resource orchestration across heterogeneous nodes
- Intelligent page-count-based workload scheduling with Work Stealing
- End-to-end RAG (Retrieval-Augmented Generation) demo on biomedical documents

---

## Architecture Overview

```
AdaptiveDistributedFramework/
├── src/adaptive_framework/
│   ├── config/               — Configuration system (YAML, Singleton, Hot Reload)
│   ├── core/                 — Exceptions, constants
│   ├── models/               — Data models (PDFMetadata, WorkUnit, ClusterStatus, ...)
│   ├── interfaces/           — Abstract interfaces (IScheduler, IOCREngine, ...)
│   ├── logging/              — Centralized logging (Console, JSON, Rotating File)
│   ├── utils/                — Utility modules
│   ├── di/                   — Dependency Injection container
│   ├── document_processing/  — Document Processing Engine (OCR, Layout, Table, Figure)
│   ├── scheduler/            — Adaptive Scheduler (Work Stealing, Page-Count Priority)
│   ├── coordinator/          — Distributed Coordinator (Ray, Failure Recovery)
│   ├── dataset_builder/      — Dataset Builder
│   ├── rag/                  — RAG Demo (Chunking → Embedding → Vector Store → Query)
│   └── evaluation/           — Evaluation Engine (Speedup, Throughput, Scheduler Overhead)
├── configs/                  — YAML configuration files
├── tests/                    — Unit, Integration, Performance tests
├── scripts/                  — Helper scripts
└── docs/                     — Architecture, Developer Guide, Coding Standards
```

---

## Core Components (Architecture v2.0)

| Component | Description |
|-----------|-------------|
| **Document Processing Engine** | OCR, Layout Analysis, Table Extraction, Figure Detection |
| **Adaptive Scheduler** | Page-count priority queue, Work Stealing, Zero-Copy |
| **Distributed Coordinator** | Ray actors, Worker Registry, Heartbeat Monitor, Failure Recovery |
| **Metadata Generator** | PDF metadata extraction (pages, size, DPI, language, source type) |
| **RAG Demo** | Ingestion → Chunking → Embedding → Vector Store → Query |
| **Evaluation Engine** | Speedup, Throughput, CPU/GPU, Energy, Scheduler Overhead |

---

## Key Research Contributions

1. **Adaptive Runtime Resource Orchestration** — dynamic scheduling across heterogeneous nodes
2. **Page-Count-Based Work Stealing** — intelligent load balancing without over-provisioning
3. **Scheduler Overhead < 1%** — provably lightweight scheduler
4. **Failure Recovery** — worker node loss without pipeline crash

---

## Quick Start

### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

### 2. Verify environment

```bash
python scripts/check_environment.py
```

### 3. Validate configuration

```bash
python scripts/validate_configuration.py
```

### 4. Run framework

```bash
python main.py
```

---

## Development

### Code Quality

```bash
# Format
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Test
pytest
```

### Pre-commit hooks

```bash
pre-commit install
pre-commit run --all-files
```

---

## Implementation Phases

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Complete | Project skeleton, config, logging, models, interfaces |
| **Phase 2** | 🔲 Pending | Document Processing Engine + Ray cluster |
| **Phase 3** | 🔲 Pending | Adaptive Scheduler + Work Stealing |
| **Phase 4** | 🔲 Pending | Distributed Coordinator + Failure Recovery |
| **Phase 5** | 🔲 Pending | Dataset Builder + RAG Demo |
| **Phase 6** | 🔲 Pending | Evaluation Engine + Paper Data Collection |

---

## Documentation

- [Developer Guide](docs/developer_guide.md)
- [Architecture Overview](docs/architecture_overview.md)
- [Coding Standards](docs/coding_standards.md)
- [Contribution Guide](docs/contribution_guide.md)
- [Architecture v2.0 (Locked)](docs/architecture_v2.0_locked.md)

---

## License

MIT License — see [LICENSE](LICENSE)
