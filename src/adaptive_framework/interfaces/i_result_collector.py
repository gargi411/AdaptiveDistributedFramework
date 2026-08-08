"""IResultCollector — Abstract result collector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from adaptive_framework.models.document import DocumentResult, PageResult


class IResultCollector(ABC):
    """Abstract interface for collecting and aggregating pipeline results.

    As workers complete PageWorkUnits, they submit PageResults to the
    ResultCollector. The collector aggregates them into DocumentResults.

    Example:
        >>> collector: IResultCollector = InMemoryResultCollector(logger)
        >>> collector.submit_page_result(page_result)
        >>> doc_result = collector.get_document_result("doc-001")
    """

    @abstractmethod
    def submit_page_result(self, result: PageResult) -> None:
        """Accept a single page result from a worker.

        Args:
            result: PageResult produced by a worker.
        """

    @abstractmethod
    def get_document_result(self, document_id: str) -> DocumentResult | None:
        """Return the aggregated result for a document, if complete.

        Args:
            document_id: Unique document identifier.

        Returns:
            DocumentResult if all pages have been submitted, else None.
        """

    @abstractmethod
    def get_all_results(self) -> list[DocumentResult]:
        """Return all completed DocumentResults.

        Returns:
            List of completed DocumentResult objects. Order is not guaranteed.
        """

    @abstractmethod
    def is_document_complete(self, document_id: str) -> bool:
        """Check whether all pages for a document have been collected.

        Args:
            document_id: Unique document identifier.

        Returns:
            True if all pages have been submitted and the DocumentResult is ready.
        """

    @abstractmethod
    def get_pending_document_ids(self) -> list[str]:
        """Return IDs of documents that are not yet fully collected.

        Returns:
            List of document_id strings still awaiting page results.
        """
