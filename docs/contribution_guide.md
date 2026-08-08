# Contribution Guide

## Adaptive Distributed Parallel Processing Framework

Thank you for contributing to this research project. This guide describes the
conventions, workflow, and quality standards expected from all contributors.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Branching Strategy](#branching-strategy)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Commit Messages](#commit-messages)
6. [Testing Requirements](#testing-requirements)
7. [Documentation Requirements](#documentation-requirements)
8. [Architecture Rules](#architecture-rules)
9. [Pull Request Checklist](#pull-request-checklist)

---

## Code of Conduct

- Treat all contributors with respect.
- Focus on technical merit in reviews.
- Assume good intent.

---

## Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, passing CI |
| `dev` | Integration of completed features |
| `feature/<name>` | New feature or module |
| `fix/<name>` | Bug fix |
| `phase/<N>` | Phase-scoped implementation branch |

> **Never commit directly to `main`.**  All changes go through pull requests.

---

## Development Workflow

```bash
# 1. Clone the repository
git clone <repository-url>
cd AdaptiveDistributedFramework

# 2. Install the package in editable mode with all dev dependencies
pip install -e ".[dev]"

# 3. Install pre-commit hooks
pre-commit install

# 4. Create a feature branch
git checkout -b feature/my-feature

# 5. Make changes, run quality checks
black src/ tests/
isort src/ tests/
ruff check src/ tests/
mypy src/
pytest

# 6. Commit and push
git add .
git commit -m "feat: add my feature"
git push origin feature/my-feature

# 7. Open a Pull Request against `dev`
```

---

## Coding Standards

All code must comply with **PEP 8** and the project's automated tooling.

### Formatter: Black + isort

```bash
black src/ tests/
isort src/ tests/
```

Black is the non-negotiable formatter. Never disable Black formatting.

### Linter: Ruff

```bash
ruff check src/ tests/
```

See `ruff.toml` for the active rule set. All warnings must be resolved before
merging. `# noqa` suppressions require a comment explaining why.

### Type Checker: mypy

```bash
mypy src/
```

All public functions and methods **must** have full type annotations. `Any` is
permitted only where genuinely necessary and must be documented.

### Naming Conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| Package | `snake_case` | `document_processing` |
| Module | `snake_case` | `file_utils.py` |
| Class | `PascalCase` | `ConfigManager` |
| Interface | `IPascalCase` | `IScheduler` |
| Function | `snake_case` | `find_pdf_files` |
| Constant | `UPPER_SNAKE_CASE` | `MAX_WORK_UNIT_SIZE_MB` |
| Private attribute | `_snake_case` | `_registry` |

### No Magic Numbers

Every numeric literal used as a threshold, limit, or sentinel **must** be
declared in `src/adaptive_framework/core/constants.py` with a descriptive name
and a comment explaining its meaning and units.

### Docstrings: Google Style

Every public class, method, and function must have a Google-style docstring
with the following sections where applicable:

```python
def my_function(arg: int) -> str:
    """One-line summary.

    Extended description if needed.

    Args:
        arg: Description of the argument.

    Returns:
        Description of the return value.

    Raises:
        ValueError: If arg is negative.

    Example:
        >>> result = my_function(42)
        >>> print(result)
        'forty-two'
    """
```

---

## Commit Messages

Follow the **Conventional Commits** specification:

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

### Types

| Type | When to use |
|------|------------|
| `feat` | New feature or module |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no logic change |
| `refactor` | Code restructuring, no feature change |
| `test` | Adding or updating tests |
| `chore` | Build system, dependencies |
| `perf` | Performance improvement |

### Examples

```
feat(scheduler): add page-count partitioning strategy
fix(config): handle missing optional YAML keys gracefully
docs(di): add usage example to DIContainer docstring
test(models): add PDFMetadata edge-case tests
```

---

## Testing Requirements

### Coverage Target

- **Unit tests**: >= 90% line coverage for all core modules.
- **Integration tests**: Must cover the happy-path end-to-end flow.
- **Performance tests**: Must include benchmarks for scheduler overhead
  (target: < 1% of total execution time per architecture_v2.0_locked.md section 4.2).

### Test File Naming

| Test type | Location | Naming |
|-----------|---------|--------|
| Unit | `tests/unit/` | `test_<module>.py` |
| Integration | `tests/integration/` | `test_<feature>_integration.py` |
| Performance | `tests/performance/` | `bench_<feature>.py` |

### Test Structure

```python
class TestMyComponent:
    """Tests for MyComponent."""

    def test_behaviour(self) -> None:
        """One-line description of what is being tested."""
        # Arrange
        ...
        # Act
        ...
        # Assert
        ...
```

Use `pytest.raises` for exception assertions. Never use bare `assert` without
a descriptive failure message for non-trivial checks.

---

## Documentation Requirements

Every new package **must** include a `README.md` describing:

1. **Purpose** -- what this package does.
2. **Contents** -- list of modules/sub-packages.
3. **Dependencies** -- what this package imports from other packages.
4. **Usage** -- one minimal usage example.

Every new module **must** include a module-level docstring describing:

- What the module contains.
- Any design decisions or constraints.

---

## Architecture Rules

> These rules are enforced and non-negotiable. They come from
> `docs/architecture_v2.0_locked.md`.

1. **Do not modify** the locked architecture document.
2. **Do not rename** packages, modules, or components defined in the
   architecture.
3. **Dependency direction is strict**:

   ```
   utils      -> core
   core       -> (nothing)
   models     -> core
   interfaces -> models, core
   config     -> utils, core
   logging    -> interfaces, core
   di         -> core
   ```

   Higher layers (scheduler, coordinator, document_processing, rag) **must
   never** be imported by lower layers.

4. **OCR must not know Scheduler internals.**
5. **Scheduler must not know OCR internals.**
6. **RAG layer must never depend on scheduling internals.**
7. Every future module must be replaceable without affecting the rest of the
   framework. Use interfaces, not concrete implementations.

---

## Pull Request Checklist

Before opening a PR, verify every item below:

- [ ] `black` applied -- no formatting errors
- [ ] `isort` applied -- imports sorted
- [ ] `ruff check` passes -- zero warnings
- [ ] `mypy src/` passes -- zero type errors
- [ ] `pytest` passes -- all existing tests green
- [ ] New code has corresponding unit tests
- [ ] Docstrings added to all public APIs
- [ ] `README.md` updated if a new package was added
- [ ] No magic numbers -- constants used
- [ ] No circular imports introduced
- [ ] Architecture rules respected (no layer violations)
- [ ] `project_progress.md` updated if a phase step was completed

---

*Contribution Guide -- Adaptive Distributed Framework v2.0*
