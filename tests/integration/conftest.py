"""Shared fixtures for integration tests.

Integration tests may spin up real file I/O, cross-module wiring,
or configuration loading. They must not spin up Ray or external services.
"""

from __future__ import annotations
