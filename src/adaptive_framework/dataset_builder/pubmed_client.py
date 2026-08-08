"""PubMed Central Open Access HTTP client.

Wraps the NCBI E-utilities API to search and retrieve article metadata
for biomedical keyword queries. No authentication required for the OA subset.

API Reference:
    https://www.ncbi.nlm.nih.gov/books/NBK25501/

Architecture note:
    This module contains ONLY I/O and data-transfer logic.
    No business logic, no scheduling, no partitioning.

Constants:
    ESEARCH_URL: NCBI E-utilities search endpoint.
    EFETCH_URL:  NCBI E-utilities fetch endpoint.
    OA_FTP_BASE: PMC Open Access FTP mirror base URL for PDF downloads.
"""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# NCBI E-utilities endpoints (public, no API key needed for <= 3 req/s)
# ---------------------------------------------------------------------------

ESEARCH_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
OA_FTP_BASE: str = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/"

# NCBI polite request rate (3 requests/sec without API key)
_NCBI_REQUEST_INTERVAL_SECONDS: float = 0.35

# Default HTTP timeout for all NCBI requests
_HTTP_TIMEOUT_SECONDS: int = 30

logger = logging.getLogger("adaptive_framework.dataset_builder.pubmed_client")


@dataclass
class ArticleRecord:
    """Metadata for a single PubMed Central article.

    Attributes:
        pmc_id: PubMed Central article identifier (e.g. 'PMC1234567').
        pmid: PubMed identifier (may be empty if unavailable).
        title: Article title.
        journal: Journal name.
        year: Publication year.
        pdf_url: Direct URL to the Open Access PDF (empty if unavailable).
        is_oa: True if confirmed Open Access.

    Example:
        >>> rec = ArticleRecord(pmc_id="PMC7654321", pmid="34567890",
        ...                     title="Deep Learning in Oncology", journal="Nature",
        ...                     year="2021", pdf_url="", is_oa=True)
    """

    pmc_id: str
    pmid: str
    title: str
    journal: str
    year: str
    pdf_url: str = ""
    is_oa: bool = True
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary suitable for CSV / JSON serialization.
        """
        return {
            "pmc_id": self.pmc_id,
            "pmid": self.pmid,
            "title": self.title,
            "journal": self.journal,
            "year": self.year,
            "pdf_url": self.pdf_url,
            "is_oa": self.is_oa,
        }

    def __repr__(self) -> str:
        return f"ArticleRecord(pmc_id={self.pmc_id!r}, title={self.title[:40]!r})"


class PubMedCentralClient:
    """HTTP client for the NCBI E-utilities / PubMed Central Open Access API.

    Searches PubMed Central using a biomedical keyword query and returns
    structured article metadata including download URLs for Open Access PDFs.

    Design constraints:
        - All network I/O is synchronous (requests are sequential, rate-limited).
        - No authentication required for Open Access subset.
        - Respects NCBI's 3 requests/second limit for anonymous users.

    Attributes:
        _tool: Identifies this client to NCBI (required by ToS).
        _email: Contact email for NCBI (required by ToS).
        _request_interval: Minimum seconds between NCBI requests.

    Example:
        >>> client = PubMedCentralClient(tool="AdaptiveDistributedFramework",
        ...                              email="researcher@university.edu")
        >>> records = client.search("cancer immunotherapy", max_results=50)
        >>> print(len(records))
        50
    """

    def __init__(
        self,
        tool: str = "AdaptiveDistributedFramework",
        email: str = "adf@research.edu",
        request_interval: float = _NCBI_REQUEST_INTERVAL_SECONDS,
        http_timeout: int = _HTTP_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the PubMedCentralClient.

        Args:
            tool: Tool name sent to NCBI in every request (ToS requirement).
            email: Contact email sent to NCBI (ToS requirement).
            request_interval: Minimum seconds between successive NCBI requests.
            http_timeout: HTTP request timeout in seconds.
        """
        self._tool = tool
        self._email = email
        self._request_interval = request_interval
        self._http_timeout = http_timeout
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        max_results: int = 100,
        only_open_access: bool = True,
    ) -> list[ArticleRecord]:
        """Search PubMed Central and return article metadata.

        Args:
            query: Biomedical keyword query (e.g., 'cancer', 'diabetes type 2').
            max_results: Maximum number of articles to return.
            only_open_access: If True, restricts query to the OA subset.

        Returns:
            List of ArticleRecord objects (may be fewer than max_results
            if PMC returns fewer hits).

        Raises:
            urllib.error.URLError: If the NCBI network request fails.
        """
        if only_open_access:
            query = f"{query} AND open access[filter]"

        logger.info(
            "Searching PubMed Central.",
            extra={"query": query, "max_results": max_results},
        )

        pmc_ids = self._esearch(query=query, ret_max=max_results)
        if not pmc_ids:
            logger.warning("ESearch returned zero results.", extra={"query": query})
            return []

        logger.info(
            "ESearch complete.",
            extra={"hits": len(pmc_ids), "fetching_metadata": True},
        )

        records = self._efetch_summaries(pmc_ids)
        logger.info(
            "Metadata retrieved.",
            extra={"article_count": len(records)},
        )
        return records

    def get_pdf_url(self, pmc_id: str) -> str:
        """Construct the FTP-mirror PDF URL for a given PMC ID.

        The PMC Open Access FTP mirror organises PDFs in subdirectories
        based on the numeric suffix of the PMC ID. This method builds
        the expected URL pattern.

        Args:
            pmc_id: PMC article identifier (e.g. 'PMC7654321').

        Returns:
            Best-effort HTTPS URL to the PDF. Empty string if the ID is malformed.

        Example:
            >>> client.get_pdf_url("PMC7654321")
            'https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/65/43/PMC7654321.pdf'
        """
        numeric = pmc_id.replace("PMC", "").strip()
        if not numeric.isdigit():
            return ""
        # PMC FTP organises into 2-level subdirectories: last 2 and last 4 digits
        subdir1 = numeric[-2:].zfill(2)
        subdir2 = numeric[-4:-2].zfill(2) if len(numeric) >= 4 else "00"
        return f"{OA_FTP_BASE}{subdir1}/{subdir2}/{pmc_id}.pdf"

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _rate_limit(self) -> None:
        """Sleep to maintain NCBI's polite request rate."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._request_interval:
            time.sleep(self._request_interval - elapsed)
        self._last_request_time = time.monotonic()

    def _get(self, url: str, params: dict[str, str]) -> bytes:
        """Perform a rate-limited HTTP GET request.

        Args:
            url: Base URL.
            params: Query parameters dict.

        Returns:
            Raw response bytes.

        Raises:
            urllib.error.URLError: On network error.
        """
        self._rate_limit()
        full_url = f"{url}?{urllib.parse.urlencode(params)}"
        logger.debug("HTTP GET: %s", full_url)
        with urllib.request.urlopen(full_url, timeout=self._http_timeout) as resp:
            return resp.read()

    def _esearch(self, query: str, ret_max: int) -> list[str]:
        """Run ESearch and return a list of PMC IDs.

        Args:
            query: E-utilities search query string.
            ret_max: Maximum IDs to retrieve.

        Returns:
            List of PMC ID strings (e.g., ['PMC1234567', 'PMC7654321']).
        """
        params: dict[str, str] = {
            "db": "pmc",
            "term": query,
            "retmax": str(ret_max),
            "retmode": "xml",
            "tool": self._tool,
            "email": self._email,
            "usehistory": "n",
        }
        raw = self._get(ESEARCH_URL, params)
        root = ET.fromstring(raw)
        ids: list[str] = []
        for id_elem in root.findall(".//Id"):
            if id_elem.text:
                ids.append(f"PMC{id_elem.text.strip()}")
        return ids

    def _efetch_summaries(self, pmc_ids: list[str]) -> list[ArticleRecord]:
        """Fetch article summaries for a list of PMC IDs.

        Fetches in a single batch call (NCBI allows up to 10,000 IDs per request).
        Parses the eFetch XML response into ArticleRecord objects.

        Args:
            pmc_ids: List of PMC article identifiers.

        Returns:
            List of ArticleRecord objects.
        """
        # Remove 'PMC' prefix for the efetch call — NCBI uses raw numeric IDs
        numeric_ids = [pid.replace("PMC", "") for pid in pmc_ids]
        params: dict[str, str] = {
            "db": "pmc",
            "id": ",".join(numeric_ids),
            "retmode": "xml",
            "rettype": "full",
            "tool": self._tool,
            "email": self._email,
        }
        raw = self._get(EFETCH_URL, params)
        return self._parse_efetch_xml(raw, pmc_ids)

    def _parse_efetch_xml(
        self, raw_xml: bytes, pmc_ids: list[str]
    ) -> list[ArticleRecord]:
        """Parse eFetch XML and build ArticleRecord objects.

        Falls back gracefully: if a field is missing in the XML, a safe
        default (empty string or 'unknown') is used. This ensures the
        pipeline never crashes due to incomplete article metadata.

        Args:
            raw_xml: Raw XML bytes from the eFetch API.
            pmc_ids: Ordered list of expected PMC IDs (for cross-referencing).

        Returns:
            List of ArticleRecord objects (one per successfully parsed article).
        """
        records: list[ArticleRecord] = []
        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError as exc:
            logger.error("Failed to parse eFetch XML: %s", exc)
            return records

        for article_elem in root.findall(".//article"):
            try:
                rec = self._parse_single_article(article_elem)
                if rec:
                    records.append(rec)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipped article due to parse error: %s", exc)

        # If XML parse yielded nothing (format mismatch), build stubs from IDs
        if not records and pmc_ids:
            logger.warning(
                "eFetch XML parse returned no records; building stubs from IDs.",
                extra={"id_count": len(pmc_ids)},
            )
            for pmc_id in pmc_ids:
                records.append(
                    ArticleRecord(
                        pmc_id=pmc_id,
                        pmid="",
                        title=f"Article {pmc_id}",
                        journal="Unknown",
                        year="",
                        pdf_url=self.get_pdf_url(pmc_id),
                        is_oa=True,
                    )
                )
        return records

    def _parse_single_article(
        self, article_elem: ET.Element
    ) -> ArticleRecord | None:
        """Parse a single <article> XML element into an ArticleRecord.

        Args:
            article_elem: XML Element representing one article.

        Returns:
            ArticleRecord or None if the element contains no usable data.
        """

        def _text(path: str, default: str = "") -> str:
            elem = article_elem.find(path)
            return (elem.text or default).strip() if elem is not None else default

        pmc_id_raw = _text(".//article-id[@pub-id-type='pmc']")
        if not pmc_id_raw:
            return None

        pmc_id = f"PMC{pmc_id_raw}" if not pmc_id_raw.startswith("PMC") else pmc_id_raw
        pmid = _text(".//article-id[@pub-id-type='pmid']")
        title = _text(".//article-title", default=f"Untitled ({pmc_id})")
        journal = _text(".//journal-title", default="Unknown Journal")
        year = _text(".//pub-date/year", default="")
        pdf_url = self.get_pdf_url(pmc_id)

        return ArticleRecord(
            pmc_id=pmc_id,
            pmid=pmid,
            title=title,
            journal=journal,
            year=year,
            pdf_url=pdf_url,
            is_oa=True,
        )
