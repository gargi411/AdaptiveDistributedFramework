"""Unit tests for processing events and event bus — Module 12."""

from __future__ import annotations

import threading
import time

import pytest

from adaptive_framework.document_processing.event_bus import (
    EventBus,
    get_default_bus,
    reset_default_bus,
)
from adaptive_framework.models.events import (
    EventType,
    ProcessingEvent,
    coordinator_merge_event,
    page_finished_event,
    page_started_event,
    worker_completed_event,
)


# ── Event model tests ────────────────────────────────────────────────────────

class TestProcessingEvent:
    def test_event_creation(self):
        event = ProcessingEvent(
            event_type=EventType.PAGE_STARTED,
            document_id="doc-001",
            page_number=3,
            worker_id="w0",
        )
        assert event.event_type == EventType.PAGE_STARTED
        assert event.page_number == 3

    def test_event_is_frozen(self):
        event = ProcessingEvent(event_type=EventType.PAGE_FINISHED)
        with pytest.raises((AttributeError, TypeError)):
            event.page_number = 99  # type: ignore

    def test_event_auto_id(self):
        e1 = ProcessingEvent(event_type=EventType.OCR_STARTED)
        e2 = ProcessingEvent(event_type=EventType.OCR_STARTED)
        assert e1.event_id != e2.event_id

    def test_to_dict(self):
        event = ProcessingEvent(
            event_type=EventType.TABLE_EXTRACTED,
            document_id="doc-1",
            page_number=2,
        )
        d = event.to_dict()
        assert d["event_type"] == "table_extracted"
        assert d["document_id"] == "doc-1"


class TestEventFactories:
    def test_page_started_event(self):
        ev = page_started_event("doc-1", 5, "w0", "node1", "direct_text")
        assert ev.event_type == EventType.PAGE_STARTED
        assert ev.page_number == 5
        assert ev.document_id == "doc-1"

    def test_page_finished_event(self):
        ev = page_finished_event("doc-1", 5, "w0", "node1", 0.85, True, 500, 2, 1)
        assert ev.event_type == EventType.PAGE_FINISHED
        assert ev.wall_time_seconds == pytest.approx(0.85)
        assert ev.payload["success"] is True
        assert ev.payload["char_count"] == 500

    def test_worker_completed_event(self):
        ev = worker_completed_event("w0", "node1", "doc-1", 10, 8.5)
        assert ev.event_type == EventType.WORKER_COMPLETED
        assert ev.payload["pages_processed"] == 10

    def test_coordinator_merge_event(self):
        ev = coordinator_merge_event("doc-1", 42, 0.3)
        assert ev.event_type == EventType.MERGE_COMPLETED
        assert ev.payload["page_count"] == 42


# ── EventBus tests ───────────────────────────────────────────────────────────

class TestEventBus:
    def setup_method(self):
        self.bus = EventBus()
        self.received: list[ProcessingEvent] = []

    def _handler(self, event: ProcessingEvent) -> None:
        self.received.append(event)

    def test_subscribe_and_publish(self):
        self.bus.subscribe(EventType.PAGE_STARTED, self._handler)
        event = page_started_event("doc-1", 1, "w0", "n1", "direct_text")
        self.bus.publish(event)
        assert len(self.received) == 1
        assert self.received[0].event_type == EventType.PAGE_STARTED

    def test_unsubscribed_type_not_received(self):
        self.bus.subscribe(EventType.PAGE_STARTED, self._handler)
        event = ProcessingEvent(event_type=EventType.OCR_STARTED)
        self.bus.publish(event)
        assert len(self.received) == 0

    def test_multiple_subscribers(self):
        received_b: list[ProcessingEvent] = []
        self.bus.subscribe(EventType.PAGE_FINISHED, self._handler)
        self.bus.subscribe(EventType.PAGE_FINISHED, lambda e: received_b.append(e))
        event = page_finished_event("d", 1, "w", "n", 1.0, True)
        self.bus.publish(event)
        assert len(self.received) == 1
        assert len(received_b) == 1

    def test_wildcard_subscriber(self):
        self.bus.subscribe_all(self._handler)
        self.bus.publish(ProcessingEvent(event_type=EventType.PAGE_STARTED))
        self.bus.publish(ProcessingEvent(event_type=EventType.OCR_FINISHED))
        assert len(self.received) == 2

    def test_broken_handler_does_not_crash_bus(self):
        def bad_handler(event):
            raise RuntimeError("Simulated handler crash")

        self.bus.subscribe(EventType.PAGE_STARTED, bad_handler)
        self.bus.subscribe(EventType.PAGE_STARTED, self._handler)
        event = page_started_event("d", 1, "w", "n", "ocr")
        self.bus.publish(event)  # should not raise
        assert len(self.received) == 1  # good handler still ran

    def test_unsubscribe(self):
        self.bus.subscribe(EventType.PAGE_FINISHED, self._handler)
        self.bus.unsubscribe(EventType.PAGE_FINISHED, self._handler)
        self.bus.publish(ProcessingEvent(event_type=EventType.PAGE_FINISHED))
        assert len(self.received) == 0

    def test_total_published_counter(self):
        self.bus.subscribe(EventType.PAGE_STARTED, self._handler)
        for _ in range(5):
            self.bus.publish(ProcessingEvent(event_type=EventType.PAGE_STARTED))
        assert self.bus.total_published == 5

    def test_subscriber_count(self):
        self.bus.subscribe(EventType.PAGE_STARTED, self._handler)
        self.bus.subscribe(EventType.OCR_STARTED, self._handler)
        assert self.bus.subscriber_count(EventType.PAGE_STARTED) == 1
        assert self.bus.subscriber_count() == 2

    def test_clear(self):
        self.bus.subscribe(EventType.PAGE_STARTED, self._handler)
        self.bus.clear()
        self.bus.publish(ProcessingEvent(event_type=EventType.PAGE_STARTED))
        assert len(self.received) == 0

    def test_thread_safety(self):
        """Multiple threads publishing concurrently should not crash."""
        results: list[ProcessingEvent] = []
        lock = threading.Lock()

        def safe_handler(e):
            with lock:
                results.append(e)

        self.bus.subscribe(EventType.PAGE_FINISHED, safe_handler)

        def publisher():
            for _ in range(20):
                self.bus.publish(ProcessingEvent(event_type=EventType.PAGE_FINISHED))

        threads = [threading.Thread(target=publisher) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 100


# ── Default bus singleton ────────────────────────────────────────────────────

class TestDefaultBus:
    def setup_method(self):
        reset_default_bus()

    def teardown_method(self):
        reset_default_bus()

    def test_default_bus_is_singleton(self):
        b1 = get_default_bus()
        b2 = get_default_bus()
        assert b1 is b2

    def test_reset_creates_new_bus(self):
        b1 = get_default_bus()
        reset_default_bus()
        b2 = get_default_bus()
        assert b1 is not b2
