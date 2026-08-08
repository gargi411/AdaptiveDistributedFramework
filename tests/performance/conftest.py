"""Shared fixtures for performance benchmarks.

Performance tests measure execution time of core algorithms.
They are tagged with @pytest.mark.performance and skipped in normal CI runs.
Run explicitly with: pytest tests/performance/ -m performance
"""

from __future__ import annotations
