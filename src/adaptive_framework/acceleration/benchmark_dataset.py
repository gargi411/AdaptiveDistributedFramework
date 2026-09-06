"""Benchmark Dataset and Corpus Manager for Phase G5.

Prepares reproducible, standardized page batches across four workload classes:
  - CLASS_A: Small / simple scanned pages (complexity ~0.15 - 0.25)
  - CLASS_B: Medium-complexity scanned pages (complexity ~0.45 - 0.60)
  - CLASS_C: Large / high-complexity scanned pages (complexity ~0.65 - 0.85)
  - CLASS_D: Mixed-complexity workload (blend of A, B, and C)

Guarantees:
  - Identical pages provided to CPU_ONLY, GPU_ONLY, and ADAPTIVE policies
  - Zero patient identifiers, symptoms, or clinical notes in metadata
  - Deterministic reproducibility
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import numpy as np

from adaptive_framework.acceleration.benchmarking import WorkloadClass
from adaptive_framework.acceleration.workload_characterizer import (
    WorkloadCharacterizer,
    WorkloadProfile,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass(frozen=True)
class BenchmarkPageItem:
    """A single page item in the benchmark corpus."""
    item_id: str
    workload_class: str
    page_number: int
    source_pdf: str
    width_pts: float
    height_pts: float
    estimated_complexity: float
    is_scanned: bool
    image_density: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata (excluding raw image data or clinical text)."""
        d = asdict(self)
        d["width_pts"] = round(self.width_pts, 1)
        d["height_pts"] = round(self.height_pts, 1)
        d["estimated_complexity"] = round(self.estimated_complexity, 3)
        d["image_density"] = round(self.image_density, 3)
        return d


class BenchmarkCorpusManager:
    """Loads, curates, and caches benchmark workloads for G5."""

    def __init__(self, dataset_dir: Path | None = None, dpi: int = 150) -> None:
        self._dataset_dir = dataset_dir or (_PROJECT_ROOT / "dataset")
        self._dpi = dpi
        self._characterizer = WorkloadCharacterizer()
        self._cache: dict[str, list[tuple[BenchmarkPageItem, np.ndarray]]] = {}

    def get_workload_pages(
        self,
        workload_class: WorkloadClass | str,
        target_page_count: int,
    ) -> list[tuple[BenchmarkPageItem, np.ndarray]]:
        """Retrieve an exact number of pages for the given workload class.

        Args:
            workload_class: WorkloadClass enum or string name.
            target_page_count: Number of pages required (e.g. 5, 10, 20, 50).

        Returns:
            List of (BenchmarkPageItem metadata, rendered RGB image NumPy array).
        """
        w_class_str = (
            workload_class.value if isinstance(workload_class, WorkloadClass) else str(workload_class)
        )
        pool = self._load_class_pool(w_class_str)
        if not pool:
            # Fallback to generated synthetic pages if raw dataset files are absent
            pool = self._generate_synthetic_pool(w_class_str, max(target_page_count, 20))

        # Replicate or slice deterministically to match target_page_count
        selected: list[tuple[BenchmarkPageItem, np.ndarray]] = []
        for i in range(target_page_count):
            item, img = pool[i % len(pool)]
            # Ensure unique item_id per replicated index
            unique_item = BenchmarkPageItem(
                item_id=f"{w_class_str}_p{i+1}_{item.item_id}",
                workload_class=item.workload_class,
                page_number=i + 1,
                source_pdf=item.source_pdf,
                width_pts=item.width_pts,
                height_pts=item.height_pts,
                estimated_complexity=item.estimated_complexity,
                is_scanned=item.is_scanned,
                image_density=item.image_density,
            )
            selected.append((unique_item, img))

        return selected

    def export_metadata(self, output_path: Path | str) -> dict[str, Any]:
        """Export corpus metadata summary to JSON without clinical content."""
        meta = {
            "dpi": self._dpi,
            "classes": {},
        }
        for w_class in [WorkloadClass.CLASS_A, WorkloadClass.CLASS_B, WorkloadClass.CLASS_C, WorkloadClass.CLASS_D]:
            pool = self._load_class_pool(w_class.value)
            meta["classes"][w_class.value] = {
                "pool_size": len(pool),
                "items": [item.to_dict() for item, _ in pool[:10]],  # Sample 10 items
            }

        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return meta

    # ------------------------------------------------------------------ #
    # Internal Loading and Generation
    # ------------------------------------------------------------------ #

    def _load_class_pool(self, w_class: str) -> list[tuple[BenchmarkPageItem, np.ndarray]]:
        """Load or retrieve cached pool for the specified class."""
        if w_class in self._cache:
            return self._cache[w_class]

        pool: list[tuple[BenchmarkPageItem, np.ndarray]] = []

        if w_class == WorkloadClass.CLASS_A.value:
            # Class A: Small/Simple scanned pages (generated synthetic or small scanned)
            pool = self._generate_synthetic_pool(WorkloadClass.CLASS_A.value, 20)

        elif w_class == WorkloadClass.CLASS_B.value:
            # Class B: Medium-complexity scanned clinical reports from dataset/PDF_Original/Medium
            medium_dir = self._dataset_dir / "PDF_Original" / "Medium"
            pool = self._load_pdf_dir(medium_dir, w_class, max_pages=20)
            if not pool:
                pool = self._generate_synthetic_pool(w_class, 20)

        elif w_class == WorkloadClass.CLASS_C.value:
            # Class C: Large/Dense scanned documents from dataset/PDF_Original/Hard
            hard_dir = self._dataset_dir / "PDF_Original" / "Hard"
            pool = self._load_pdf_dir(hard_dir, w_class, max_pages=20)
            if not pool:
                pool = self._generate_synthetic_pool(w_class, 20)

        elif w_class == WorkloadClass.CLASS_D.value:
            # Class D: Mixed workload (interleaving A, B, and C)
            pool_a = self._load_class_pool(WorkloadClass.CLASS_A.value)
            pool_b = self._load_class_pool(WorkloadClass.CLASS_B.value)
            pool_c = self._load_class_pool(WorkloadClass.CLASS_C.value)
            max_len = max(len(pool_a), len(pool_b), len(pool_c))
            for i in range(max_len):
                if i < len(pool_a):
                    pool.append(pool_a[i])
                if i < len(pool_b):
                    pool.append(pool_b[i])
                if i < len(pool_c):
                    pool.append(pool_c[i])

        self._cache[w_class] = pool
        return pool

    def _load_pdf_dir(
        self,
        dir_path: Path,
        w_class: str,
        max_pages: int = 20,
    ) -> list[tuple[BenchmarkPageItem, np.ndarray]]:
        """Load and render real PDF pages from directory."""
        if not dir_path.exists():
            return []

        items: list[tuple[BenchmarkPageItem, np.ndarray]] = []
        pdf_files = sorted(dir_path.glob("*.pdf"))

        zoom = self._dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for pdf_path in pdf_files:
            try:
                doc = fitz.open(str(pdf_path))
                for page_idx in range(len(doc)):
                    if len(items) >= max_pages:
                        break
                    page = doc[page_idx]
                    rect = page.rect

                    # Render page to RGB image array
                    pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))

                    profile = self._characterizer.characterize_page(
                        page,
                        page_number=page_idx + 1,
                        document_id=pdf_path.name,
                        page_count=len(doc),
                    )

                    item = BenchmarkPageItem(
                        item_id=f"{pdf_path.stem}_p{page_idx+1}",
                        workload_class=w_class,
                        page_number=page_idx + 1,
                        source_pdf=pdf_path.name,
                        width_pts=float(rect.width),
                        height_pts=float(rect.height),
                        estimated_complexity=profile.estimated_complexity,
                        is_scanned=True,  # Benchmark treats all selected pages as scanned OCR units
                        image_density=profile.image_density if profile.image_density > 0 else 1.0,
                    )
                    items.append((item, img))
                doc.close()
            except Exception as exc:
                logger.debug("Failed loading PDF %s: %s", pdf_path, exc)

            if len(items) >= max_pages:
                break

        return items

    def _generate_synthetic_pool(
        self,
        w_class: str,
        count: int = 20,
    ) -> list[tuple[BenchmarkPageItem, np.ndarray]]:
        """Generate controlled synthetic scanned pages for reproducible benchmarks."""
        items: list[tuple[BenchmarkPageItem, np.ndarray]] = []

        if w_class == WorkloadClass.CLASS_A.value:
            # Small / simple scanned page (800x600, simple header + 2 lines)
            w, h = 600, 800
            complexity = 0.18
        elif w_class == WorkloadClass.CLASS_B.value:
            # Medium complexity (1200x900, report format with structured lines)
            w, h = 900, 1200
            complexity = 0.50
        else:
            # Class C: Large dense page (1600x1200, dense grid and text blocks)
            w, h = 1200, 1600
            complexity = 0.78

        for i in range(count):
            img = np.full((h, w, 3), 255, dtype=np.uint8)
            # Add synthetic line artifacts to represent scanned document layout
            if w_class == WorkloadClass.CLASS_A.value:
                img[50:60, 50:300, :] = 40  # Header bar
                img[100:110, 50:400, :] = 60
                img[150:160, 50:250, :] = 60
            elif w_class == WorkloadClass.CLASS_B.value:
                # Two column layout
                img[50:70, 50:w - 50, :] = 30
                for r in range(120, h - 100, 40):
                    img[r:r + 12, 50:w // 2 - 20, :] = 50
                    img[r:r + 12, w // 2 + 20:w - 50, :] = 50
            else:
                # Dense biomedical record
                img[40:65, 40:w - 40, :] = 20
                for r in range(90, h - 80, 25):
                    img[r:r + 10, 40:w - 40, :] = 45
                # Add vertical column dividers
                img[90:h - 80, w // 3, :] = 70
                img[90:h - 80, (2 * w) // 3, :] = 70

            item = BenchmarkPageItem(
                item_id=f"synth_{w_class.lower()}_{i+1}",
                workload_class=w_class,
                page_number=i + 1,
                source_pdf=f"synthetic_{w_class.lower()}.pdf",
                width_pts=float(w * 72.0 / self._dpi),
                height_pts=float(h * 72.0 / self._dpi),
                estimated_complexity=complexity,
                is_scanned=True,
                image_density=0.85 if w_class != WorkloadClass.CLASS_A.value else 0.20,
            )
            items.append((item, img))

        return items

    def get_corpus_metadata(self) -> dict[str, Any]:
        """Return metadata summary for all 4 workload classes."""
        metadata: dict[str, Any] = {}
        for wc in [WorkloadClass.CLASS_A, WorkloadClass.CLASS_B, WorkloadClass.CLASS_C, WorkloadClass.CLASS_D]:
            pool = self._load_class_pool(wc.value)
            metadata[wc.value] = {
                "pool_size": len(pool),
                "sample_items": [item.to_dict() for item, _ in pool[:5]],
            }
        return metadata

    def export_corpus_manifest(self, out_path: Path) -> None:
        """Export corpus metadata manifest to JSON."""
        meta = self.get_corpus_metadata()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

