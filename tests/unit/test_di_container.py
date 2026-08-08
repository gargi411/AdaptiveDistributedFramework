"""Unit tests for the DIContainer dependency injection container.

Tests:
    - register() stores an instance
    - resolve() returns the registered instance
    - resolve() raises ContainerError for unregistered interface
    - register() with non-type interface raises ContainerError
    - is_registered() returns correct boolean
    - unregister() removes a registration
    - registered_interfaces() lists all registered interface names
    - Thread safety: concurrent registration does not corrupt state
    - DIContainer.__repr__() includes interface names
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod

import pytest

from adaptive_framework.di.container import ContainerError, DIContainer


# ---------------------------------------------------------------------------
# Minimal test interfaces (declared locally to avoid importing real interfaces)
# ---------------------------------------------------------------------------


class ITestService(ABC):
    """Minimal test interface for DI container tests."""

    @abstractmethod
    def do_work(self) -> str:
        """Do some work and return a string result."""


class ITestLogger(ABC):
    """Minimal second test interface for multi-registration tests."""

    @abstractmethod
    def log(self, message: str) -> None:
        """Emit a log message."""


class ConcreteTestService(ITestService):
    """Concrete implementation of ITestService for testing."""

    def do_work(self) -> str:
        """Return a fixed result string."""
        return "work done"


class ConcreteTestLogger(ITestLogger):
    """Concrete implementation of ITestLogger for testing."""

    def log(self, message: str) -> None:
        """Silently discard the log message."""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDIContainerRegisterAndResolve:
    """Tests for basic register/resolve lifecycle."""

    def test_resolve_returns_registered_instance(self) -> None:
        """resolve() returns the exact instance that was registered."""
        container = DIContainer()
        service = ConcreteTestService()
        container.register(ITestService, service)
        resolved = container.resolve(ITestService)
        assert resolved is service

    def test_resolve_unregistered_raises_container_error(self) -> None:
        """resolve() raises ContainerError for an unregistered interface."""
        container = DIContainer()
        with pytest.raises(ContainerError, match="ITestService"):
            container.resolve(ITestService)

    def test_register_non_type_raises_container_error(self) -> None:
        """register() raises ContainerError if interface is not a type."""
        container = DIContainer()
        with pytest.raises(ContainerError):
            container.register("not_a_type", ConcreteTestService())  # type: ignore[arg-type]

    def test_register_overwrites_existing_registration(self) -> None:
        """Registering the same interface twice replaces the first registration."""
        container = DIContainer()
        service1 = ConcreteTestService()
        service2 = ConcreteTestService()
        container.register(ITestService, service1)
        container.register(ITestService, service2)
        resolved = container.resolve(ITestService)
        assert resolved is service2
        assert resolved is not service1

    def test_multiple_interfaces_independent(self) -> None:
        """Registering two interfaces stores them independently."""
        container = DIContainer()
        service = ConcreteTestService()
        logger = ConcreteTestLogger()
        container.register(ITestService, service)
        container.register(ITestLogger, logger)
        assert container.resolve(ITestService) is service
        assert container.resolve(ITestLogger) is logger


class TestDIContainerIsRegistered:
    """Tests for is_registered()."""

    def test_returns_false_before_registration(self) -> None:
        """is_registered() returns False before any registration."""
        container = DIContainer()
        assert container.is_registered(ITestService) is False

    def test_returns_true_after_registration(self) -> None:
        """is_registered() returns True after registration."""
        container = DIContainer()
        container.register(ITestService, ConcreteTestService())
        assert container.is_registered(ITestService) is True

    def test_returns_false_after_unregister(self) -> None:
        """is_registered() returns False after unregister()."""
        container = DIContainer()
        container.register(ITestService, ConcreteTestService())
        container.unregister(ITestService)
        assert container.is_registered(ITestService) is False


class TestDIContainerUnregister:
    """Tests for unregister()."""

    def test_unregister_removes_registration(self) -> None:
        """unregister() prevents resolve() from finding the interface."""
        container = DIContainer()
        container.register(ITestService, ConcreteTestService())
        container.unregister(ITestService)
        with pytest.raises(ContainerError):
            container.resolve(ITestService)

    def test_unregister_nonexistent_does_not_raise(self) -> None:
        """unregister() on a missing interface does not raise."""
        container = DIContainer()
        container.unregister(ITestService)  # should not raise


class TestDIContainerRegisteredInterfaces:
    """Tests for registered_interfaces()."""

    def test_empty_container_returns_empty_list(self) -> None:
        """Empty container returns an empty list."""
        container = DIContainer()
        assert container.registered_interfaces() == []

    def test_registered_interface_name_appears_in_list(self) -> None:
        """registered_interfaces() includes the name of a registered interface."""
        container = DIContainer()
        container.register(ITestService, ConcreteTestService())
        names = container.registered_interfaces()
        assert "ITestService" in names

    def test_multiple_registrations_all_appear(self) -> None:
        """registered_interfaces() lists all registered interface names."""
        container = DIContainer()
        container.register(ITestService, ConcreteTestService())
        container.register(ITestLogger, ConcreteTestLogger())
        names = container.registered_interfaces()
        assert "ITestService" in names
        assert "ITestLogger" in names


class TestDIContainerRepr:
    """Tests for DIContainer.__repr__()."""

    def test_repr_contains_registered_interface(self) -> None:
        """__repr__ includes the name of a registered interface."""
        container = DIContainer()
        container.register(ITestService, ConcreteTestService())
        assert "ITestService" in repr(container)

    def test_repr_empty_container(self) -> None:
        """__repr__ on an empty container contains 'DIContainer'."""
        container = DIContainer()
        assert "DIContainer" in repr(container)


class TestDIContainerThreadSafety:
    """Tests for thread safety of registration and resolution."""

    def test_concurrent_registration_does_not_raise(self) -> None:
        """Concurrent register() calls do not corrupt state or raise."""
        container = DIContainer()
        errors: list[Exception] = []

        def register_service() -> None:
            try:
                container.register(ITestService, ConcreteTestService())
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=register_service) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert container.is_registered(ITestService)

    def test_concurrent_resolve_is_consistent(self) -> None:
        """Concurrent resolve() calls always return the same instance."""
        container = DIContainer()
        service = ConcreteTestService()
        container.register(ITestService, service)

        results: list[ITestService] = []
        lock = threading.Lock()

        def resolve_service() -> None:
            resolved = container.resolve(ITestService)
            with lock:
                results.append(resolved)

        threads = [threading.Thread(target=resolve_service) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r is service for r in results), "Not all resolved to the same instance"
