"""Dependency Injection container for the Adaptive Distributed Framework.

Implements a lightweight, typed service registry with constructor injection.
No third-party DI framework is used — this is a clean, minimal implementation
that enforces the architecture's dependency rules.

Design:
    - Components register their interface type as the key.
    - Components request dependencies by interface type.
    - The container never instantiates components; callers pass pre-built instances.
    - No service locator pattern: components do not call the container at runtime.

Architecture Rule:
    The DI container is populated in main.py (composition root) and
    passed to components via constructors. Business logic never imports
    the container.
"""

from __future__ import annotations

import threading
from typing import Any, TypeVar

from adaptive_framework.core.exceptions import FrameworkError

T = TypeVar("T")


class ContainerError(FrameworkError):
    """Raised when the DI container cannot resolve a dependency."""


class DIContainer:
    """Lightweight typed dependency injection container.

    Stores registered instances keyed by their interface type (class object).
    Supports singleton registration only — all registered instances are shared.

    Thread-safe: registration and resolution are protected by a lock.

    Attributes:
        _registry: Maps interface type → concrete instance.
        _lock: Thread lock for concurrent access protection.

    Example:
        >>> container = DIContainer()
        >>> container.register(ILogger, framework_logger)
        >>> container.register(IConfigProvider, config_manager)
        >>>
        >>> # Resolve in main.py or factories only — never in business logic
        >>> logger = container.resolve(ILogger)
        >>> logger.info("Container initialized.")
    """

    def __init__(self) -> None:
        """Initialize an empty DIContainer."""
        self._registry: dict[type[Any], Any] = {}
        self._lock: threading.Lock = threading.Lock()

    def register(self, interface: type[T], instance: T) -> None:
        """Register a concrete instance for an interface type.

        If the interface is already registered, the existing registration
        is overwritten (allows overriding in tests).

        Args:
            interface: The abstract interface class (e.g., ILogger).
            instance: The concrete implementation instance.

        Raises:
            ContainerError: If interface is not a type.

        Example:
            >>> container.register(ILogger, FrameworkLogger.from_config(cfg))
        """
        if not isinstance(interface, type):
            raise ContainerError(
                f"DIContainer.register: 'interface' must be a type, got {type(interface)!r}."
            )
        with self._lock:
            self._registry[interface] = instance

    def resolve(self, interface: type[T]) -> T:
        """Retrieve the registered instance for an interface type.

        Args:
            interface: The abstract interface class to resolve.

        Returns:
            The registered concrete instance.

        Raises:
            ContainerError: If no instance is registered for this interface.

        Example:
            >>> logger = container.resolve(ILogger)
        """
        with self._lock:
            instance = self._registry.get(interface)
        if instance is None:
            raise ContainerError(
                f"DIContainer.resolve: No registration found for '{interface.__name__}'. "
                f"Call container.register({interface.__name__}, ...) first."
            )
        return instance  # type: ignore[return-value]

    def is_registered(self, interface: type[Any]) -> bool:
        """Check whether an interface has a registered implementation.

        Args:
            interface: The interface type to check.

        Returns:
            True if registered, False otherwise.

        Example:
            >>> if not container.is_registered(ILogger):
            ...     container.register(ILogger, fallback_logger)
        """
        with self._lock:
            return interface in self._registry

    def unregister(self, interface: type[Any]) -> None:
        """Remove a registration. Useful in tests to reset state.

        Args:
            interface: The interface type to unregister.
        """
        with self._lock:
            self._registry.pop(interface, None)

    def registered_interfaces(self) -> list[str]:
        """Return names of all currently registered interfaces.

        Returns:
            List of interface class names as strings.

        Example:
            >>> print(container.registered_interfaces())
            ['ILogger', 'IConfigProvider']
        """
        with self._lock:
            return [iface.__name__ for iface in self._registry]

    def __repr__(self) -> str:
        interfaces = self.registered_interfaces()
        return f"DIContainer(registered={interfaces})"
