"""Dataset Scanner — Phase 3.5.

Scans a configurable directory for PDF files, validates their readability,
and returns an alphabetically sorted list of resolved ``Path`` objects.

This module intentionally does NOT open PDF files or perform any extraction.
It is a pure filesystem scanner that feeds into ``DatasetLoader``.

Design:
    - Single public class ``DatasetScanner`` with a ``scan()`` method.
    - All configuration supplied at construction time (no globals/hardcoding).
    - Invalid or unreadable files are skipped with a logged warning.
    - Empty directories return an empty list (no exceptions).

Usage::

    from adaptive_framework.dataset_builder.dataset_scanner import DatasetScanner

    scanner = DatasetScanner(root="dataset/raw/pmc_pdfs")
    paths = scanner.scan()
    print(f"{len(paths)} PDFs found.")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("adaptive_framework.dataset_builder.scanner")


class DatasetScanner:
    """Scan a directory for readable PDF files.

    Produces a deterministic, alphabetically sorted ``list[Path]`` of absolute
    paths to every valid PDF found in the configured root directory.

    Attributes:
        _root: Root directory to scan.
        _recursive: Whether to recurse into sub-directories.
        _extensions: File extensions to include (lower-cased, with leading dot).

    Example:
        >>> scanner = DatasetScanner(root="dataset/raw/pmc_pdfs")
        >>> paths = scanner.scan()
        >>> len(paths)
        20
    """

    def __init__(
        self,
        root: str | Path,
        recursive: bool = False,
        extensions: list[str] | None = None,
    ) -> None:
        """Initialise the DatasetScanner.

        Args:
            root: Root directory to scan for PDFs. Relative paths are resolved
                against the current working directory at scan time.
            recursive: If True, scan all sub-directories as well. Defaults to False.
            extensions: List of file extensions to accept (e.g. ``['.pdf']``).
                Defaults to ``['.pdf']``. All comparisons are case-insensitive.
        """
        self._root = Path(root)
        self._recursive = recursive
        self._extensions: frozenset[str] = frozenset(
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in (extensions or ["pdf"])
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def scan(self) -> list[Path]:
        """Scan the root directory and return valid, readable PDF paths.

        Returns:
            Alphabetically sorted list of absolute ``Path`` objects for every
            valid PDF found. Returns an empty list if the directory does not
            exist or contains no matching files.
        """
        root = self._root.resolve()

        if not root.exists():
            logger.warning(
                "Dataset root does not exist: '%s'. Returning empty scan.", root
            )
            return []

        if not root.is_dir():
            logger.warning(
                "Dataset root is not a directory: '%s'. Returning empty scan.", root
            )
            return []

        logger.info("Scanning for PDFs in: '%s' (recursive=%s)", root, self._recursive)

        candidates = self._collect_candidates(root)
        valid_paths = self._validate(candidates)

        logger.info(
            "Scan complete: %d valid PDF(s) found in '%s'.",
            len(valid_paths),
            root,
        )
        return sorted(valid_paths)

    @property
    def root(self) -> Path:
        """Return the resolved root directory path.

        Returns:
            Absolute path to the scan root.
        """
        return self._root.resolve()

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _collect_candidates(self, root: Path) -> list[Path]:
        """Walk the directory and collect all files matching the extensions.

        Args:
            root: Resolved absolute path to the scan root.

        Returns:
            Unfiltered list of candidate paths (may include unreadable files).
        """
        candidates: list[Path] = []

        if self._recursive:
            for dirpath, _dirnames, filenames in os.walk(root):
                for fname in filenames:
                    fp = Path(dirpath) / fname
                    if fp.suffix.lower() in self._extensions:
                        candidates.append(fp)
        else:
            for fp in root.iterdir():
                if fp.is_file() and fp.suffix.lower() in self._extensions:
                    candidates.append(fp)

        return candidates

    def _validate(self, candidates: list[Path]) -> list[Path]:
        """Filter candidates to only those that are readable and non-empty.

        Args:
            candidates: Candidate file paths from ``_collect_candidates``.

        Returns:
            List of valid, readable, non-empty PDF paths.
        """
        valid: list[Path] = []
        for path in candidates:
            resolved = path.resolve()
            try:
                if not resolved.is_file():
                    logger.debug("Skipping non-file: '%s'", resolved)
                    continue
                size = resolved.stat().st_size
                if size == 0:
                    logger.warning(
                        "Skipping zero-byte file: '%s'", resolved.name
                    )
                    continue
                valid.append(resolved)
            except OSError as exc:
                logger.warning(
                    "Skipping unreadable file '%s': %s", resolved.name, exc
                )
        return valid
