# DI Package

## Purpose

Lightweight typed **Dependency Injection** container.

## Design

- **No third-party DI framework** — minimal, transparent implementation.
- **Constructor injection only** — no property injection, no field injection.
- **Composition root** — the container is populated exclusively in `main.py`.
- **No service locator** — business logic never calls `container.resolve()` internally.

## Thread Safety

`DIContainer.register()` and `DIContainer.resolve()` are protected by a `threading.Lock`.

## Usage

```python
# main.py only
from adaptive_framework.di import DIContainer
from adaptive_framework.interfaces import ILogger, IConfigProvider, IScheduler

container = DIContainer()
container.register(ILogger, framework_logger)
container.register(IConfigProvider, config_manager)

# Components receive dependencies via constructors
scheduler = AdaptiveScheduler(
    logger=container.resolve(ILogger),
    config=container.resolve(IConfigProvider).get_scheduler_config(),
)
```

## Testing

```python
from adaptive_framework.di import DIContainer

def test_something():
    container = DIContainer()
    container.register(ILogger, MockLogger())   # override with mock
    ...
```
