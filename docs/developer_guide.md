# Developer Guide

## Adaptive Distributed Parallel Processing Framework

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.12+ |
| pip | 23.0+ |
| Git | Any recent version |

---

## Setup

```bash
# Clone the repository
git clone <repository-url>
cd AdaptiveDistributedFramework

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# OR use the setup script
python scripts/setup_environment.py

# Install pre-commit hooks
pre-commit install

# Verify environment
python scripts/check_environment.py

# Validate configurations
python scripts/validate_configuration.py

# Run the framework
python main.py
```

---

## Project Structure

```
AdaptiveDistributedFramework/
├── src/adaptive_framework/     ← All source code
│   ├── config/                 ← Singleton ConfigManager + typed models
│   ├── core/                   ← Exception hierarchy + constants
│   ├── models/                 ← Dataclass data models
│   ├── interfaces/             ← Abstract interfaces (ABCs)
│   ├── logging/                ← Centralized logging
│   ├── utils/                  ← Pure utility functions
│   ├── di/                     ← DI container
│   ├── document_processing/    ← Document Processing Engine (Phase 2)
│   ├── scheduler/              ← Adaptive Scheduler (Phase 3)
│   ├── coordinator/            ← Distributed Coordinator (Phase 4)
│   ├── dataset_builder/        ← Dataset Builder (Phase 2)
│   ├── rag/                    ← RAG Demo (Phase 5)
│   └── evaluation/             ← Evaluation Engine (Phase 6)
├── configs/                    ← YAML configuration files
├── tests/                      ← Test suite
│   ├── unit/                   ← Unit tests (no external deps)
│   ├── integration/            ← Integration tests
│   ├── performance/            ← Benchmarks
│   └── fixtures/               ← Shared test data
├── scripts/                    ← Helper scripts
├── docs/                       ← Documentation
└── main.py                     ← Composition root
```

---

## Dependency Direction (Strict)

```
utils  →  core
core   →  (nothing)
models →  core
interfaces → models, core
config →  utils, core
logging → interfaces, core
di     →  core
```

**Higher layers (scheduler, coordinator, OCR, RAG) never imported in lower layers.**

---

## Code Quality Workflow

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Run tests
pytest

# Run tests with coverage
pytest --cov=adaptive_framework --cov-report=html
```

---

## Configuration System

All configuration is YAML-driven. Edit files in `configs/` and call `ConfigManager.get_instance().reload()` to pick up changes without restarting.

See [configs/README reference in config/README.md](../src/adaptive_framework/config/README.md).

---

## Adding a New Component (future phases)

1. Define the interface in `src/adaptive_framework/interfaces/i_<name>.py`.
2. Implement the interface in the appropriate module package.
3. Register in `main.py`'s composition root:
   ```python
   container.register(IMyComponent, ConcreteMyComponent(...))
   ```
4. Inject via constructor in consuming components:
   ```python
   class MyConsumer:
       def __init__(self, component: IMyComponent) -> None:
           self._component = component
   ```
5. Write unit tests in `tests/unit/test_<name>.py`.
