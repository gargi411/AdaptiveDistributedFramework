"""Framework-wide exception hierarchy for the Adaptive Distributed Framework.

All exceptions derive from FrameworkError, which itself derives from
the built-in Exception. This gives callers fine-grained control:

    except FrameworkError:        # catch any framework error
    except ConfigurationError:    # catch only config errors
    except ClusterError:          # catch only cluster errors

Exception Hierarchy:

    Exception
    └── FrameworkError                  Base for all framework exceptions
        ├── ConfigurationError          YAML loading, missing keys, invalid values
        ├── DatasetError                Dataset building, missing files, corruption
        ├── ClusterError                Ray cluster init, worker connection issues
        │   └── WorkerLostError         A specific worker node disconnected
        ├── SchedulerError              Task queue, work stealing, partition errors
        │   └── SchedulerOverheadError  Scheduler overhead exceeded target threshold
        ├── PipelineError               Stage execution, stage timeout errors
        ├── EvaluationError             Metric collection, report generation errors
        ├── ValidationError             Input data validation failures
        └── ProcessingError             Document processing failures
            └── OCRError                OCR-backend-specific failures
"""


class FrameworkError(Exception):
    """Base exception for the Adaptive Distributed Framework.

    All framework-specific exceptions derive from this class.
    Catch this to handle any framework error generically.

    Args:
        message: Human-readable error description.
        context: Optional dictionary of additional debug context.

    Example:
        >>> try:
        ...     raise FrameworkError("Something went wrong", context={"run_id": "adf_001"})
        ... except FrameworkError as exc:
        ...     print(exc)
        Something went wrong
    """

    def __init__(self, message: str, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.context: dict[str, object] = context or {}

    def __repr__(self) -> str:
        ctx = f", context={self.context}" if self.context else ""
        return f"{self.__class__.__name__}('{self}'{ ctx})"


# =============================================================
# Tier-1 Exceptions  (direct children of FrameworkError)
# =============================================================


class ConfigurationError(FrameworkError):
    """Raised when configuration loading or validation fails.

    Triggers:
        - YAML file not found or unreadable.
        - Required configuration key is missing.
        - Configuration value fails ``__post_init__`` validation.

    Example:
        >>> raise ConfigurationError("Missing key 'name' in framework.yaml")
    """


class DatasetError(FrameworkError):
    """Raised when dataset building or loading fails.

    Triggers:
        - Dataset directory not found.
        - Corrupt or unreadable PDF files.
        - Dataset manifest is malformed.

    Example:
        >>> raise DatasetError("Dataset directory '/data/raw' not found.")
    """


class ClusterError(FrameworkError):
    """Raised when Ray cluster operations fail.

    Triggers:
        - Ray cluster initialization timeout.
        - Head node connection refused.
        - Worker node registration failure.

    Example:
        >>> raise ClusterError("Ray cluster failed to initialize within 60 seconds.")
    """


class SchedulerError(FrameworkError):
    """Raised when the Adaptive Scheduler encounters an error.

    Triggers:
        - Priority queue capacity exceeded.
        - Work stealing configuration is invalid.
        - Partition strategy returns zero partitions.

    Example:
        >>> raise SchedulerError("Priority queue capacity exceeded (max_size=10000).")
    """


class PipelineError(FrameworkError):
    """Raised when a pipeline stage fails.

    Triggers:
        - Stage execution throws an unhandled exception.
        - Stage timeout exceeded (stage_timeout_seconds).
        - Stage dependency not satisfied.

    Example:
        >>> raise PipelineError("OCR stage timed out after 3600 seconds.")
    """


class EvaluationError(FrameworkError):
    """Raised when metric collection or report generation fails.

    Triggers:
        - Metric sampling fails (e.g., psutil error).
        - Report directory cannot be created.
        - Unsupported report format requested.

    Example:
        >>> raise EvaluationError("Cannot write CSV report: permission denied.")
    """


class ValidationError(FrameworkError):
    """Raised when input data validation fails.

    Triggers:
        - PDF file is not a valid PDF.
        - Page count is zero or negative.
        - Required metadata field is missing.

    Args:
        message: Description of the validation failure.
        field: Optional name of the field that failed validation.
        value: Optional value that caused the failure.

    Example:
        >>> raise ValidationError("page_count must be > 0", field="page_count", value=-1)
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: object = None,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, context)
        self.field = field
        self.value = value

    def __repr__(self) -> str:
        parts = [f"'{self}'"]
        if self.field is not None:
            parts.append(f"field='{self.field}'")
        if self.value is not None:
            parts.append(f"value={self.value!r}")
        return f"ValidationError({', '.join(parts)})"


class ProcessingError(FrameworkError):
    """Raised when document processing fails.

    Triggers:
        - Document Processing Engine cannot open a PDF.
        - Layout analysis produces no output.
        - Table extraction returns malformed data.

    Args:
        message: Error description.
        document_id: Optional identifier of the document being processed.

    Example:
        >>> raise ProcessingError("Cannot open PDF", document_id="doc_0042")
    """

    def __init__(
        self,
        message: str,
        document_id: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, context)
        self.document_id = document_id


# =============================================================
# Tier-2 Exceptions  (specializations)
# =============================================================


class WorkerLostError(ClusterError):
    """Raised when a Ray worker node disconnects unexpectedly.

    Triggers the Failure Recovery flow in the Distributed Coordinator:
    the worker's unfinished tasks are returned to the Priority Queue
    and reassigned to another available worker.

    Args:
        message: Error description.
        worker_id: Identifier of the lost worker.
        unfinished_task_ids: IDs of tasks that were in-flight on the worker.

    Example:
        >>> raise WorkerLostError("Heartbeat timeout", worker_id="worker_02",
        ...                       unfinished_task_ids=["task_011", "task_012"])
    """

    def __init__(
        self,
        message: str,
        worker_id: str | None = None,
        unfinished_task_ids: list[str] | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, context)
        self.worker_id = worker_id
        self.unfinished_task_ids: list[str] = unfinished_task_ids or []


class SchedulerOverheadError(SchedulerError):
    """Raised when scheduler overhead exceeds the configured target threshold.

    Per architecture spec §4.2, scheduler overhead must be < 1%.
    This exception is raised (or a warning is emitted, depending on config)
    when the measured overhead exceeds ``target_max_overhead_fraction``.

    Args:
        message: Error description.
        measured_overhead_fraction: The measured overhead as a fraction.
        target_max_fraction: The configured target maximum.

    Example:
        >>> raise SchedulerOverheadError(
        ...     "Scheduler overhead 2.4% exceeds target 1.0%",
        ...     measured_overhead_fraction=0.024,
        ...     target_max_fraction=0.01,
        ... )
    """

    def __init__(
        self,
        message: str,
        measured_overhead_fraction: float = 0.0,
        target_max_fraction: float = 0.01,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, context)
        self.measured_overhead_fraction = measured_overhead_fraction
        self.target_max_fraction = target_max_fraction


class OCRError(ProcessingError):
    """Raised when an OCR backend fails to process a page.

    Triggers:
        - OCR backend not installed or not importable.
        - OCR confidence below threshold on all attempts.
        - Backend-specific runtime error.

    Args:
        message: Error description.
        document_id: Document being processed.
        page_number: Page number that caused the failure (1-indexed).
        backend: Name of the OCR backend that failed.

    Example:
        >>> raise OCRError("PaddleOCR inference failed", document_id="doc_001",
        ...                page_number=3, backend="paddleocr")
    """

    def __init__(
        self,
        message: str,
        document_id: str | None = None,
        page_number: int | None = None,
        backend: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, document_id, context)
        self.page_number = page_number
        self.backend = backend
