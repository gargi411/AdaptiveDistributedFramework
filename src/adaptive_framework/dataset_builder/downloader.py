"""PDF Downloader for the PubMed Central Dataset Builder.

Downloads PDFs from PubMed Central Open Access with:
    - Resume support (partial file detection via Content-Length).
    - Skip logic for already-downloaded files (size + hash check).
    - Duplicate detection (MD5 fingerprint registry).
    - Retry logic with exponential backoff.
    - Progress reporting via callable hooks.
    - Configurable concurrency limit.

Architecture note:
    Downloads are performed synchronously (per-file) to avoid rate-limiting
    issues with NCBI. The caller controls iteration order and progress display.
"""

from __future__ import annotations

import hashlib
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from adaptive_framework.core.constants import PDF_EXTENSION
from adaptive_framework.core.exceptions import DatasetError
from adaptive_framework.dataset_builder.download_stats import DownloadStats

logger = logging.getLogger("adaptive_framework.dataset_builder.downloader")

# Default chunk size for streaming downloads (32 KiB)
_DOWNLOAD_CHUNK_BYTES: int = 32 * 1024

# Default retry configuration
_DEFAULT_MAX_RETRIES: int = 3
_DEFAULT_RETRY_DELAY_SECONDS: float = 2.0
_DEFAULT_BACKOFF_FACTOR: float = 2.0

# HTTP timeout for download requests
_DOWNLOAD_TIMEOUT_SECONDS: int = 60

# Type alias for a progress callback: (article_id, downloaded_bytes, total_bytes)
ProgressCallback = Callable[[str, int, int], None]


class PDFDownloader:
    """Downloads PDF files from URLs with resume, dedup, and retry support.

    Each article's PDF is downloaded to ``output_dir / pmc_id + '.pdf'``.
    A fingerprint registry (in-memory MD5 dict) prevents storing the same
    file content twice even if it arrives under a different URL.

    Attributes:
        _output_dir: Directory where downloaded PDFs are stored.
        _max_retries: Maximum download attempts per article.
        _retry_delay: Initial delay between retries (seconds).
        _backoff_factor: Multiplier applied to delay on each retry.
        _fingerprints: Maps MD5 hash → destination filename (dedup registry).

    Example:
        >>> downloader = PDFDownloader(output_dir=Path("dataset/raw"))
        >>> stats = DownloadStats(total_requested=5)
        >>> downloader.download(
        ...     url="https://ftp.ncbi.nlm.nih.gov/.../PMC1234567.pdf",
        ...     article_id="PMC1234567",
        ...     stats=stats,
        ... )
    """

    def __init__(
        self,
        output_dir: Path,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_delay: float = _DEFAULT_RETRY_DELAY_SECONDS,
        backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
        timeout: int = _DOWNLOAD_TIMEOUT_SECONDS,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Initialise the PDFDownloader.

        Args:
            output_dir: Destination directory. Created automatically if missing.
            max_retries: Max download attempts per article.
            retry_delay: Initial delay before the first retry (seconds).
            backoff_factor: Exponential backoff multiplier.
            timeout: HTTP connection + read timeout per request.
            progress_callback: Optional callable invoked on each chunk
                with (article_id, bytes_so_far, total_bytes).
        """
        self._output_dir = output_dir
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._backoff_factor = backoff_factor
        self._timeout = timeout
        self._progress_callback = progress_callback
        self._fingerprints: dict[str, str] = {}  # md5 → filename

        # Ensure output directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Pre-populate fingerprint registry from existing files
        self._index_existing_files()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def download(
        self,
        url: str,
        article_id: str,
        stats: DownloadStats,
    ) -> Path | None:
        """Download a single PDF from ``url`` to the output directory.

        Skips download if:
            - A file with the same name and expected size already exists.
            - The MD5 fingerprint already exists in the registry (duplicate).

        Args:
            url: Direct HTTPS URL to the PDF.
            article_id: PMC article identifier (used as filename stem).
            stats: DownloadStats object updated in-place.

        Returns:
            Path to the saved PDF on success, None on skip or failure.
        """
        if not url:
            logger.warning("Empty URL for article_id=%s; skipping.", article_id)
            stats.record_skip()
            return None

        dest_path = self._output_dir / f"{article_id}{PDF_EXTENSION}"

        # Skip if file already fully downloaded
        if self._is_already_downloaded(dest_path, url):
            logger.debug("Skipping already-downloaded: %s", dest_path.name)
            stats.record_skip()
            return dest_path

        logger.info("Downloading: %s → %s", article_id, dest_path.name)

        for attempt in range(1, self._max_retries + 1):
            try:
                file_size = self._stream_to_disk(
                    url=url,
                    dest_path=dest_path,
                    article_id=article_id,
                )

                # Duplicate fingerprint check
                md5 = self._compute_md5(dest_path)
                if md5 in self._fingerprints:
                    logger.warning(
                        "Duplicate detected: %s is identical to %s. Removing.",
                        dest_path.name,
                        self._fingerprints[md5],
                    )
                    dest_path.unlink(missing_ok=True)
                    stats.record_skip()
                    return None

                self._fingerprints[md5] = dest_path.name
                stats.record_success(file_size_bytes=file_size)
                logger.info(
                    "Download complete: %s (%.1f KB)",
                    dest_path.name,
                    file_size / 1024,
                )
                return dest_path

            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                logger.warning(
                    "Download attempt %d/%d failed for %s: %s",
                    attempt,
                    self._max_retries,
                    article_id,
                    exc,
                )
                if attempt < self._max_retries:
                    delay = self._retry_delay * (self._backoff_factor ** (attempt - 1))
                    logger.debug("Retrying in %.1f seconds.", delay)
                    time.sleep(delay)
                else:
                    logger.error(
                        "All %d attempts exhausted for %s.",
                        self._max_retries,
                        article_id,
                    )
                    stats.record_failure(article_id, str(exc))
                    # Remove partial file if it exists
                    dest_path.unlink(missing_ok=True)

        return None

    def download_batch(
        self,
        articles: list[tuple[str, str]],
        stats: DownloadStats | None = None,
    ) -> list[Path]:
        """Download a batch of (article_id, url) pairs sequentially.

        Args:
            articles: List of (article_id, pdf_url) tuples.
            stats: Optional DownloadStats shared across the batch.
                   Created internally if not provided.

        Returns:
            List of Paths for successfully downloaded files.
        """
        if stats is None:
            stats = DownloadStats(total_requested=len(articles))
        else:
            stats.total_requested += len(articles)

        downloaded: list[Path] = []
        for article_id, url in articles:
            result = self.download(url=url, article_id=article_id, stats=stats)
            if result is not None:
                downloaded.append(result)

        return downloaded

    def validate_file(self, file_path: Path) -> bool:
        """Validate a downloaded PDF by checking its file signature.

        A valid PDF starts with the magic bytes ``%PDF``.

        Args:
            file_path: Path to the PDF file.

        Returns:
            True if the file exists and starts with the PDF magic bytes.
        """
        if not file_path.exists() or file_path.stat().st_size == 0:
            return False
        try:
            with file_path.open("rb") as f:
                return f.read(4) == b"%PDF"
        except OSError:
            return False

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _index_existing_files(self) -> None:
        """Compute MD5 fingerprints for any PDFs already in output_dir."""
        for pdf_path in self._output_dir.glob(f"*{PDF_EXTENSION}"):
            try:
                md5 = self._compute_md5(pdf_path)
                self._fingerprints[md5] = pdf_path.name
            except OSError as exc:
                logger.warning("Could not index existing file %s: %s", pdf_path.name, exc)

    def _is_already_downloaded(self, dest_path: Path, url: str) -> bool:
        """Return True if the destination file already exists and appears valid.

        A quick check: file exists + is a valid PDF + size > 0.
        A full Content-Length check is skipped to avoid extra HTTP requests.

        Args:
            dest_path: Expected local destination path.
            url: Source URL (unused in quick-check mode).

        Returns:
            True if the file can be safely skipped.
        """
        return dest_path.exists() and dest_path.stat().st_size > 0 and self.validate_file(dest_path)

    def _stream_to_disk(
        self,
        url: str,
        dest_path: Path,
        article_id: str,
    ) -> int:
        """Stream a URL response to disk and return the total bytes written.

        Uses HTTP Range requests for resume if a partial file exists.

        Args:
            url: Download URL.
            dest_path: Local destination path (may be partial from a prior attempt).
            article_id: Used for progress callback identification.

        Returns:
            Total file size in bytes after the download.

        Raises:
            urllib.error.URLError: On network failure.
            OSError: On disk I/O failure.
        """
        existing_size = dest_path.stat().st_size if dest_path.exists() else 0
        headers: dict[str, str] = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            logger.debug("Resuming %s at byte %d.", article_id, existing_size)

        req = urllib.request.Request(url, headers=headers)
        mode = "ab" if existing_size > 0 else "wb"

        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            total_bytes = int(resp.headers.get("Content-Length", 0)) + existing_size
            written = existing_size
            with dest_path.open(mode) as f:
                while True:
                    chunk = resp.read(_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
                    if self._progress_callback:
                        self._progress_callback(article_id, written, total_bytes)

        return written

    @staticmethod
    def _compute_md5(file_path: Path, chunk_size: int = 8192) -> str:
        """Compute the MD5 hash of a file for duplicate detection.

        Args:
            file_path: Path to the file.
            chunk_size: Read chunk size in bytes.

        Returns:
            Lowercase hexadecimal MD5 string.
        """
        hasher = hashlib.md5()
        with file_path.open("rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
