"""IReportGenerator — Abstract evaluation report generator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from adaptive_framework.models.evaluation import EvaluationResult


class IReportGenerator(ABC):
    """Abstract interface for evaluation report generators.

    After a pipeline run, the Evaluation Engine calls the IReportGenerator
    to produce output files in one or more formats (JSON, CSV, Markdown).

    Example:
        >>> generator: IReportGenerator = MarkdownReportGenerator(logger)
        >>> path = generator.generate(result, output_dir=Path("evaluation_results/"))
        >>> print(path)
        PosixPath('evaluation_results/adf_run_001_report.md')
    """

    @abstractmethod
    def generate(
        self,
        result: EvaluationResult,
        output_dir: Path,
    ) -> Path:
        """Generate an evaluation report and write it to output_dir.

        Args:
            result: The completed EvaluationResult to report on.
            output_dir: Directory where the report file should be written.

        Returns:
            Path to the generated report file.

        Raises:
            EvaluationError: If the report cannot be written.
        """

    @abstractmethod
    def get_format(self) -> str:
        """Return the output format identifier for this generator.

        Returns:
            Format string: 'json' | 'csv' | 'markdown'.
        """
