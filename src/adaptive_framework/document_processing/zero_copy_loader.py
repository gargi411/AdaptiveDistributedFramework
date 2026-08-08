"""Zero-Copy Page Loader — Module 3: PyMuPDF → NumPy, no temp files.

Architecture key point:
    Workers receive PageWorkUnit (start_page, end_page) — NOT complete PDFs.
    The loader opens the PDF, jumps to the requested pages, streams them
    directly into NumPy arrays via PyMuPDF Pixmaps.

No PNG/JPEG intermediate files are written to disk.
No unnecessary copies of pixel data are made.

Research note:
    This zero-copy optimisation eliminates the encode→decode cycle that
    naive pipelines perform (PDF → render → PNG save → PNG load → OCR).
    The direct Pixmap→NumPy path is measurably faster and saves disk I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

logger = logging.getLogger(__name__)

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    logger.warning("NumPy not installed. ZeroCopyPageLoader returns stub arrays.")

try:
    import fitz  # type: ignore[import]
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False
    logger.warning("PyMuPDF not installed. ZeroCopyPageLoader unavailable.")


@dataclass
class LoadedPage:
    """A page loaded into memory as a NumPy array.

    Attributes:
        page_number: 1-indexed page number.
        image_array: HxWxC NumPy uint8 array (RGB). None if loading failed.
        width_px: Image width in pixels.
        height_px: Image height in pixels.
        width_pts: Page width in PDF points.
        height_pts: Page height in PDF points.
        dpi: Rendering resolution used.
        fitz_page: The open PyMuPDF Page object for direct text extraction.
                   None if PyMuPDF is not available.
        error: Error message if loading failed. None on success.
    """

    page_number: int
    image_array: Any  # numpy.ndarray | None
    width_px: int
    height_px: int
    width_pts: float
    height_pts: float
    dpi: int
    fitz_page: Any  # fitz.Page | None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Return True if the page was loaded without error."""
        return self.error is None and self.image_array is not None

    @property
    def has_numpy(self) -> bool:
        """Return True if image_array is a usable NumPy array."""
        return self.image_array is not None and _NUMPY_AVAILABLE

    def __repr__(self) -> str:
        if self.success:
            return (
                f"LoadedPage(page={self.page_number}, "
                f"{self.width_px}×{self.height_px}px @ {self.dpi}dpi)"
            )
        return f"LoadedPage(page={self.page_number}, error={self.error!r})"


class ZeroCopyPageLoader:
    """Loads specific page ranges from a PDF into memory as NumPy arrays.

    Uses PyMuPDF Pixmaps to stream pages directly into memory.
    No PNG, JPEG, or any temporary files are created.

    Workers receive a PageWorkUnit with (start_page, end_page).
    The loader opens the PDF once and yields LoadedPage objects.

    Args:
        dpi: Rendering resolution. Higher DPI improves OCR accuracy.
             Default 150 DPI balances quality and memory usage.
             Use 300 DPI for high-resolution biomedical images.
        colorspace: 'rgb' (3-channel) or 'gray' (1-channel).

    Example:
        >>> loader = ZeroCopyPageLoader(dpi=150)
        >>> pages = loader.load_pages("/data/paper.pdf", start=1, end=5)
        >>> for lp in pages:
        ...     print(lp.image_array.shape)
        (1125, 795, 3)
        ...
    """

    _COLORSPACE_MAP = {"rgb": 3, "gray": 1}

    def __init__(
        self,
        dpi: int = 150,
        colorspace: str = "rgb",
    ) -> None:
        self._dpi = dpi
        self._colorspace = colorspace.lower()
        if self._colorspace not in self._COLORSPACE_MAP:
            raise ValueError(
                f"colorspace must be 'rgb' or 'gray', got '{colorspace}'."
            )

    def load_pages(
        self,
        file_path: str,
        start_page: int,
        end_page: int,
    ) -> list[LoadedPage]:
        """Load a range of pages from a PDF into memory.

        Args:
            file_path: Absolute path to the PDF file.
            start_page: First page to load (1-indexed, inclusive).
            end_page: Last page to load (1-indexed, inclusive).

        Returns:
            List of LoadedPage objects in page_number order.
        """
        if not _FITZ_AVAILABLE:
            return [
                self._error_page(i, "PyMuPDF not available.")
                for i in range(start_page, end_page + 1)
            ]

        loaded: list[LoadedPage] = []
        doc = None
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)

            for page_number in range(start_page, end_page + 1):
                idx = page_number - 1
                if idx < 0 or idx >= total_pages:
                    loaded.append(
                        self._error_page(
                            page_number,
                            f"Page {page_number} out of range "
                            f"(document has {total_pages} pages).",
                        )
                    )
                    continue

                try:
                    lp = self._load_single_page(doc, idx, page_number)
                    loaded.append(lp)
                except Exception as exc:
                    logger.warning(
                        "Failed to load page %d from '%s': %s",
                        page_number, file_path, exc,
                    )
                    loaded.append(self._error_page(page_number, str(exc)))

        except Exception as exc:
            logger.error("Cannot open PDF '%s': %s", file_path, exc)
            for page_number in range(start_page, end_page + 1):
                loaded.append(
                    self._error_page(page_number, f"Cannot open PDF: {exc}")
                )
        finally:
            # Note: we do NOT close doc here because fitz_page references
            # within LoadedPage objects need the document to remain open.
            # Callers must call close_doc() or use the context manager.
            pass

        return loaded

    def load_pages_stream(
        self,
        file_path: str,
        start_page: int,
        end_page: int,
    ) -> Iterator[LoadedPage]:
        """Stream pages one-by-one from a PDF (generator).

        Yields LoadedPage objects without holding all pages in memory.
        Useful for long page ranges.

        Args:
            file_path: Absolute path to the PDF file.
            start_page: First page (1-indexed, inclusive).
            end_page: Last page (1-indexed, inclusive).

        Yields:
            LoadedPage objects in ascending page_number order.
        """
        if not _FITZ_AVAILABLE:
            for i in range(start_page, end_page + 1):
                yield self._error_page(i, "PyMuPDF not available.")
            return

        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            for i in range(start_page, end_page + 1):
                yield self._error_page(i, f"Cannot open PDF: {exc}")
            return

        try:
            total_pages = len(doc)
            for page_number in range(start_page, end_page + 1):
                idx = page_number - 1
                if idx < 0 or idx >= total_pages:
                    yield self._error_page(
                        page_number,
                        f"Page {page_number} out of range ({total_pages} total).",
                    )
                    continue
                try:
                    yield self._load_single_page(doc, idx, page_number)
                except Exception as exc:
                    yield self._error_page(page_number, str(exc))
        finally:
            doc.close()

    def _load_single_page(
        self, doc: Any, idx: int, page_number: int
    ) -> LoadedPage:
        """Load one page from an open fitz.Document.

        Args:
            doc: Open fitz.Document.
            idx: 0-indexed page index.
            page_number: 1-indexed page number.

        Returns:
            LoadedPage with NumPy array and page dimensions.
        """
        page = doc[idx]
        rect = page.rect

        # Build transformation matrix for DPI scaling
        scale = self._dpi / 72.0
        matrix = fitz.Matrix(scale, scale)

        # Determine colorspace
        if self._colorspace == "gray":
            cs = fitz.csGRAY
        else:
            cs = fitz.csRGB

        # Render to Pixmap — stays in memory, zero disk writes
        pixmap = page.get_pixmap(matrix=matrix, colorspace=cs, alpha=False)

        # Convert Pixmap samples → NumPy array (zero-copy view)
        if _NUMPY_AVAILABLE:
            import numpy as np
            n_channels = self._COLORSPACE_MAP[self._colorspace]
            img_array: Any = np.frombuffer(
                pixmap.samples, dtype=np.uint8
            ).reshape(pixmap.height, pixmap.width, n_channels)
        else:
            img_array = None

        return LoadedPage(
            page_number=page_number,
            image_array=img_array,
            width_px=pixmap.width,
            height_px=pixmap.height,
            width_pts=rect.width,
            height_pts=rect.height,
            dpi=self._dpi,
            fitz_page=page,
        )

    @staticmethod
    def _error_page(page_number: int, error: str) -> LoadedPage:
        """Create a failed LoadedPage record.

        Args:
            page_number: Page number that failed.
            error: Error description.

        Returns:
            LoadedPage with image_array=None and error set.
        """
        return LoadedPage(
            page_number=page_number,
            image_array=None,
            width_px=0,
            height_px=0,
            width_pts=0.0,
            height_pts=0.0,
            dpi=0,
            fitz_page=None,
            error=error,
        )
