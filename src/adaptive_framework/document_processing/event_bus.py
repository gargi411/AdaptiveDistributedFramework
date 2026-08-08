"""Internal event bus — Module 12: Publish/subscribe for processing events.

Thread-safe publish/subscribe system. Events are published by workers
during document processing and consumed by subscribers:
    - Engineering dashboard (live metrics panel)
    - BenchmarkLogger (every event → CSV performance data)

Design:
    - No polling: subscribers are called synchronously in the publisher thread.
    - Thread-safe: all operations protected by RLock.
    - Lightweight: no external dependencies.
    - Decoupled: publishers know nothing about subscribers.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Callable

from adaptive_framework.models.events import EventType, ProcessingEvent

logger = logging.getLogger(__name__)

# Type alias for event handler functions
EventHandler = Callable[[ProcessingEvent], None]


class EventBus:
    """Thread-safe publish/subscribe event bus for processing events.

    Subscribers register handlers per event type.
    Publishers emit ProcessingEvent objects.

    Handlers are called synchronously in the publisher's thread.
    For non-blocking behaviour, wrap handlers in a background thread.

    Usage:
        >>> bus = EventBus()
        >>> def on_page_finished(event):
        ...     print(f"Page {event.page_number} done in {event.wall_time_seconds:.3f}s")
        >>> bus.subscribe(EventType.PAGE_FINISHED, on_page_finished)
        >>> bus.publish(page_finished_event("doc-001", 3, "w0", "node1", 0.8, True))
        Page 3 done in 0.800s

    Thread safety:
        subscribe(), unsubscribe(), and publish() are all protected by RLock.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._wildcard_handlers: list[EventHandler] = []
        self._total_published: int = 0

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        """Register a handler for a specific event type.

        Args:
            event_type: The EventType to subscribe to.
            handler: Callable that accepts a ProcessingEvent.
        """
        with self._lock:
            self._handlers[event_type].append(handler)
            logger.debug(
                "Subscribed handler %r to %s.", handler.__name__, event_type.value
            )

    def subscribe_all(self, handler: EventHandler) -> None:
        """Register a handler for ALL event types (wildcard subscription).

        Args:
            handler: Callable that accepts any ProcessingEvent.
        """
        with self._lock:
            self._wildcard_handlers.append(handler)
            logger.debug("Wildcard subscriber: %r.", handler.__name__)

    def unsubscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        """Remove a previously registered handler.

        Args:
            event_type: The EventType to unsubscribe from.
            handler: The handler to remove.
        """
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)

    def publish(self, event: ProcessingEvent) -> None:
        """Publish an event to all registered subscribers.

        Calls all handlers registered for event.event_type and all
        wildcard handlers. Handler exceptions are caught and logged
        so one broken subscriber never crashes the pipeline.

        Args:
            event: The ProcessingEvent to broadcast.
        """
        with self._lock:
            handlers = list(self._handlers.get(event.event_type, []))
            wildcard = list(self._wildcard_handlers)
            self._total_published += 1

        all_handlers = handlers + wildcard

        for handler in all_handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.warning(
                    "EventBus handler %r raised exception for event %s: %s",
                    getattr(handler, "__name__", repr(handler)),
                    event.event_type.value,
                    exc,
                )

    def subscriber_count(self, event_type: EventType | None = None) -> int:
        """Return the number of subscribers.

        Args:
            event_type: Count only for this type. None = count all.

        Returns:
            Total subscriber count.
        """
        with self._lock:
            if event_type is not None:
                return len(self._handlers.get(event_type, []))
            return sum(len(v) for v in self._handlers.values()) + len(
                self._wildcard_handlers
            )

    @property
    def total_published(self) -> int:
        """Total number of events published on this bus."""
        with self._lock:
            return self._total_published

    def clear(self) -> None:
        """Remove all subscribers (useful for testing teardown)."""
        with self._lock:
            self._handlers.clear()
            self._wildcard_handlers.clear()
            logger.debug("EventBus cleared all subscribers.")


# Module-level default bus — used by workers and coordinator if no custom bus is injected.
_default_bus: EventBus | None = None
_bus_lock = threading.Lock()


def get_default_bus() -> EventBus:
    """Return the module-level default EventBus (singleton).

    Returns:
        Shared EventBus instance. Created on first call.
    """
    global _default_bus
    with _bus_lock:
        if _default_bus is None:
            _default_bus = EventBus()
        return _default_bus


def reset_default_bus() -> None:
    """Reset the module-level default EventBus (for test isolation)."""
    global _default_bus
    with _bus_lock:
        _default_bus = None
