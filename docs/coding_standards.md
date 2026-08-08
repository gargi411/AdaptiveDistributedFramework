# Coding Standards

## Adaptive Distributed Parallel Processing Framework

---

## Language & Version

- **Python 3.12+** exclusively
- All source files must include `from __future__ import annotations`

---

## Style

| Tool | Rule |
|------|------|
| **Black** | Line length: 100 |
| **isort** | Profile: `black` |
| **Ruff** | All selected rules (see `ruff.toml`) |
| **mypy** | `strict = True` |

---

## Docstrings

**Google Style** for all public classes, methods, and functions.

```python
def compute_overhead_fraction(
    scheduler_time_seconds: float,
    total_time_seconds: float,
) -> float:
    """Compute scheduler overhead as a fraction of total execution time.

    Formula from architecture v2.0 §4.2:
        Scheduler Overhead = Scheduler Time / Total Execution Time

    Args:
        scheduler_time_seconds: Cumulative time in the scheduler.
        total_time_seconds: End-to-end wall-clock time.

    Returns:
        Overhead fraction in [0.0, 1.0].

    Example:
        >>> fraction = compute_overhead_fraction(0.8, 120.0)
        >>> print(f"{fraction * 100:.3f}%")
        0.667%
    """
```

---

## Type Hints

- **Everywhere** — no untyped functions, parameters, or return values.
- Use `from __future__ import annotations` for forward references.
- Use `X | None` (Python 3.10+ union) instead of `Optional[X]`.
- Use `list[X]`, `dict[K, V]`, `tuple[X, ...]` (lowercase generics).

---

## Principles

### SOLID
| Principle | How |
|-----------|-----|
| **S** — Single Responsibility | Each class/module does one thing |
| **O** — Open/Closed | Extend via new classes implementing interfaces |
| **L** — Liskov Substitution | All interface implementations are interchangeable |
| **I** — Interface Segregation | Interfaces are small and focused |
| **D** — Dependency Inversion | Depend on abstractions (ABCs), not concretions |

### Clean Architecture
- **No circular imports** — strictly enforced
- **Dependency direction**: `utils → core → models → interfaces → config → logging → di`
- **No magic numbers** — use `constants.py`
- **No global variables** — use dependency injection
- **No God classes** — split responsibilities

---

## Data Models

```python
@dataclass
class MyModel:
    """One-line summary.

    Attributes:
        field_name: Description of the field.

    Example:
        >>> m = MyModel(field_name="value")
    """
    field_name: str

    def __post_init__(self) -> None:
        if not self.field_name:
            raise ValidationError("field_name must not be empty.", field="field_name")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

---

## Tests

- **Marker**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.performance`
- **Test class naming**: `TestMyComponent`
- **Test method naming**: `test_<description>_<expected_outcome>`
- **No magic values in tests** — use fixtures or named constants
- **Coverage target**: ≥ 80% (enforced by `pyproject.toml`)

---

## Forbidden Patterns

```python
# ❌ Magic numbers
timeout = 30       # What is 30?
# ✅
timeout = STAGE_TIMEOUT_SECONDS

# ❌ Global mutable state
_config = {}       # Don't
# ✅ Use DI container

# ❌ Bare except
try: ...
except: ...
# ✅
except ConfigurationError as exc: ...

# ❌ Circular imports
# logging imports config, config imports logging
# ✅ Use interfaces to break cycles

# ❌ Untyped functions
def process(doc):    # No type hints
    pass
# ✅
def process(doc: PDFMetadata) -> DocumentResult:
    pass
```
