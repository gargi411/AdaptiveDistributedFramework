"""Adaptive Page Count Partitioner — Research Algorithm #1.

Implements ``IPartitionStrategy`` using the page-count-based partitioning
algorithm described in the research publication.

Algorithm: Adaptive Balanced Page Partitioning (ABPP)
------------------------------------------------------
Input:
    - dataset: list of PDFMetadata (each with a page count)
    - num_workers: number of worker partitions to create

Objective:
    Distribute documents across ``num_workers`` partitions such that the
    total page count per partition is as balanced as possible.
    The unit of distribution is the DOCUMENT, not the page.

Why page-count, not file-count?
    File-count partitioning (100 PDFs per worker) creates severe imbalance:
    Worker A: 100 × 5-page documents  =   500 pages  (light)
    Worker B: 100 × 50-page documents = 5,000 pages  (heavy)

    Page-count partitioning:
    Worker A: ~1,667 pages (balanced)
    Worker B: ~1,667 pages (balanced)
    Worker C: ~1,666 pages (balanced)

Algorithm steps:
    1. Sort documents by page count in descending order (greedy first-fit).
    2. Create ``num_workers`` empty buckets (partitions).
    3. For each document, assign it to the bucket with the FEWEST pages.
       (This is the Longest Processing Time (LPT) / Multiprocessor Scheduling
       heuristic, approximation ratio ≤ 4/3 OPT.)
    4. Wrap each bucket's documents in PageWorkUnit objects.
    5. Compute PartitionStatistics.
    6. Return (partitions, statistics).

Complexity: O(n log n) sort + O(n log k) heap (where k = num_workers).

Performance target:
    - Must complete in < 1% of total pipeline execution time.
    - This implementation is CPU-bound and deterministic.
"""

from __future__ import annotations

import heapq
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from adaptive_framework.core.exceptions import SchedulerError
from adaptive_framework.interfaces.i_partition_strategy import IPartitionStrategy
from adaptive_framework.models.document import PDFMetadata
from adaptive_framework.models.scheduling import (
    PageWorkUnit,
    Partition,
    PartitionStatistics,
)

logger = logging.getLogger("adaptive_framework.scheduler.page_count_partitioner")

# Minimum documents required to run the partitioner
_MIN_DOCUMENTS: int = 1


@dataclass
class _Bucket:
    """Internal bucket used during the LPT greedy assignment.

    Attributes:
        index: Bucket index (maps to partition/worker index).
        total_pages: Accumulated page count.
        documents: Documents assigned to this bucket.
    """

    total_pages: int = 0
    index: int = 0
    documents: list[PDFMetadata] = field(default_factory=list)

    def __lt__(self, other: "_Bucket") -> bool:
        """Enable heap comparison by total_pages ascending.

        Args:
            other: Another _Bucket to compare.

        Returns:
            True if this bucket has fewer pages than other.
        """
        return (self.total_pages, self.index) < (other.total_pages, other.index)


class PageCountPartitioner(IPartitionStrategy):
    """Adaptive Page Count Partitioner implementing IPartitionStrategy.

    Uses the LPT (Longest Processing Time) greedy heuristic to assign
    documents to partitions such that each partition's total page count
    is approximately equal.

    This is the primary scheduling algorithm for the research paper.

    Attributes:
        _min_pages_per_partition: Minimum pages threshold; partitions below
            this threshold are merged with a neighbour.
        _logger: Logger instance for scheduling events.

    Example:
        >>> partitioner = PageCountPartitioner()
        >>> partitions, stats = partitioner.partition(dataset, num_workers=4)
        >>> print(stats.std_pages_per_partition)  # near-zero = well balanced
        12.3
    """

    def __init__(
        self,
        min_pages_per_partition: int = 10,
        logger_instance: logging.Logger | None = None,
    ) -> None:
        """Initialise the PageCountPartitioner.

        Args:
            min_pages_per_partition: Discard partitions with fewer pages than
                this threshold (prevents overly small partitions).
            logger_instance: Optional custom logger (uses module logger by default).
        """
        self._min_pages_per_partition = min_pages_per_partition
        self._logger = logger_instance or logger

    # ------------------------------------------------------------------ #
    # IPartitionStrategy interface                                         #
    # ------------------------------------------------------------------ #

    def partition(
        self,
        dataset: list[PDFMetadata],
        num_workers: int,
    ) -> tuple[list[Partition], PartitionStatistics]:
        """Partition the dataset across workers by total page count (LPT).

        Args:
            dataset: List of PDFMetadata records (each must have pages >= 1).
            num_workers: Target number of partitions.

        Returns:
            Tuple of (partitions, statistics).

        Raises:
            SchedulerError: If dataset is empty or num_workers < 1.
        """
        start_ts = time.perf_counter()

        # ---- Validation ----
        if not dataset:
            raise SchedulerError(
                "PageCountPartitioner.partition() called with an empty dataset."
            )
        if num_workers < 1:
            raise SchedulerError(
                f"num_workers must be >= 1, got {num_workers}."
            )

        effective_workers = min(num_workers, len(dataset))

        self._logger.info(
            "Starting page-count partition.",
            extra={
                "total_documents": len(dataset),
                "num_workers": num_workers,
                "effective_workers": effective_workers,
            },
        )

        # ---- Step 1: Sort by page count descending (LPT greedy) ----
        sorted_docs = sorted(dataset, key=lambda m: m.pages, reverse=True)

        # ---- Step 2: Greedy LPT assignment using a min-heap ----
        buckets = [_Bucket(index=i) for i in range(effective_workers)]
        heap: list[_Bucket] = list(buckets)
        heapq.heapify(heap)

        for doc in sorted_docs:
            # Pop the bucket with fewest pages
            lightest = heapq.heappop(heap)
            lightest.documents.append(doc)
            lightest.total_pages += doc.pages
            heapq.heappush(heap, lightest)

        # ---- Step 3: Build Partition objects ----
        partitions: list[Partition] = []
        for bucket in sorted(buckets, key=lambda b: b.index):
            if not bucket.documents:
                continue
            if bucket.total_pages < self._min_pages_per_partition:
                self._logger.debug(
                    "Bucket %d has %d pages (< min %d); merging into first partition.",
                    bucket.index,
                    bucket.total_pages,
                    self._min_pages_per_partition,
                )
                if partitions:
                    # Merge into the first partition
                    partitions[0].work_units.extend(
                        self._docs_to_work_units(bucket.documents)
                    )
                    continue

            work_units = self._docs_to_work_units(bucket.documents)
            partition = Partition(
                work_units=work_units,
                worker_id=f"worker_{bucket.index:02d}",
            )
            partitions.append(partition)

        if not partitions:
            raise SchedulerError(
                "Partitioner produced zero valid partitions. "
                f"Dataset size: {len(dataset)}, num_workers: {num_workers}."
            )

        # ---- Step 4: Compute statistics ----
        stats = self._compute_statistics(partitions)
        elapsed = time.perf_counter() - start_ts

        self._logger.info(
            "Partition complete.",
            extra={
                "partitions": stats.total_partitions,
                "total_pages": stats.total_pages,
                "avg_pages": stats.avg_pages_per_partition,
                "std_pages": stats.std_pages_per_partition,
                "elapsed_seconds": round(elapsed, 6),
            },
        )
        return partitions, stats

    def get_strategy_name(self) -> str:
        """Return the unique identifier of this partitioning strategy.

        Returns:
            'page_count_lpt' — Longest Processing Time page-count strategy.
        """
        return "page_count_lpt"

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _docs_to_work_units(docs: list[PDFMetadata]) -> list[PageWorkUnit]:
        """Convert PDFMetadata records to PageWorkUnit objects.

        Each document becomes a single PageWorkUnit covering all its pages.
        Priority is set to the document's page count (higher = more urgent).

        Args:
            docs: Documents to convert.

        Returns:
            List of PageWorkUnit objects.
        """
        work_units: list[PageWorkUnit] = []
        for doc in docs:
            wu = PageWorkUnit(
                document_id=doc.document_id,
                file_path=doc.file_path,
                start_page=1,
                end_page=doc.pages,
                priority=doc.pages,  # page_count IS the priority
            )
            work_units.append(wu)
        return work_units

    @staticmethod
    def _compute_statistics(
        partitions: list[Partition],
    ) -> PartitionStatistics:
        """Compute PartitionStatistics from the completed partition plan.

        Args:
            partitions: The final list of Partition objects.

        Returns:
            PartitionStatistics with all aggregate metrics populated.
        """
        page_counts = [p.total_pages for p in partitions]
        total_pages = sum(page_counts)
        n = len(page_counts)
        avg = total_pages / n
        variance = sum((p - avg) ** 2 for p in page_counts) / n
        std = math.sqrt(variance)

        return PartitionStatistics(
            total_partitions=n,
            total_work_units=sum(p.total_work_units for p in partitions),
            total_pages=total_pages,
            min_pages_per_partition=min(page_counts),
            max_pages_per_partition=max(page_counts),
            avg_pages_per_partition=round(avg, 4),
            std_pages_per_partition=round(std, 4),
        )
