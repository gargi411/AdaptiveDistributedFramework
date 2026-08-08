"""Scheduler package — Phase 2A implementation.

Provides the core intelligent workload scheduling subsystem.

Public API:
    PageCountPartitioner       — IPartitionStrategy (primary research algorithm).
    PageCountPriorityQueue     — Thread-safe max-heap priority queue.
    PartitionSummary           — Formatted partition plan reporting.
    WorkloadAnalyzer           — Pre-scheduling dataset analysis.
    WorkloadReport             — Dataclass for analysis results.

Usage::

    from adaptive_framework.scheduler import (
        PageCountPartitioner,
        PageCountPriorityQueue,
        WorkloadAnalyzer,
        PartitionSummary,
    )

    # 1. Analyze workload
    analyzer = WorkloadAnalyzer()
    report = analyzer.analyze(dataset, num_workers=4)
    print(report.format_report())

    # 2. Partition by page count
    partitioner = PageCountPartitioner()
    partitions, stats = partitioner.partition(dataset, num_workers=4)

    # 3. Summarise
    summary = PartitionSummary(partitions, stats)
    print(summary.format_table())

    # 4. Enqueue work units
    queue = PageCountPriorityQueue(max_size=10000)
    for partition in partitions:
        for wu in partition.work_units:
            queue.insert(wu)
"""

from adaptive_framework.scheduler.page_count_partitioner import PageCountPartitioner
from adaptive_framework.scheduler.partition_summary import PartitionSummary
from adaptive_framework.scheduler.priority_queue import PageCountPriorityQueue
from adaptive_framework.scheduler.steal_event import StealEvent
from adaptive_framework.scheduler.work_stealing import WorkStealingCoordinator
from adaptive_framework.scheduler.workload_analyzer import WorkloadAnalyzer, WorkloadReport

__all__ = [
    "PageCountPartitioner",
    "PageCountPriorityQueue",
    "PartitionSummary",
    "StealEvent",
    "WorkloadAnalyzer",
    "WorkloadReport",
    "WorkStealingCoordinator",
]