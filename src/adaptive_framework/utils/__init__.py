"""Utils package for the Adaptive Distributed Framework."""

from adaptive_framework.utils.file_utils import (
    compute_md5,
    ensure_directory,
    find_pdf_files,
    get_file_size_mb,
    read_text_file,
    safe_remove,
    write_text_file,
)
from adaptive_framework.utils.path_utils import (
    get_configs_dir,
    get_logs_dir,
    get_output_dir,
    get_project_root,
    is_subpath,
    resolve_path,
)
from adaptive_framework.utils.system_utils import (
    get_cpu_count,
    get_cpu_percent,
    get_hostname,
    get_memory_percent,
    get_platform_info,
    is_psutil_available,
)
from adaptive_framework.utils.time_utils import (
    compute_overhead_fraction,
    format_duration,
    monotonic_seconds,
    now_utc_iso,
    perf_counter,
    timer,
)
from adaptive_framework.utils.validation_utils import (
    validate_choices,
    validate_fraction,
    validate_non_empty_string,
    validate_non_negative_float,
    validate_page_count,
    validate_pdf_path,
    validate_positive_int,
    validate_source_type,
)
from adaptive_framework.utils.yaml_utils import (
    dump_yaml,
    load_yaml_file,
    load_yaml_string,
    merge_yaml_dicts,
)

__all__ = [
    # file_utils
    "find_pdf_files", "get_file_size_mb", "compute_md5",
    "ensure_directory", "safe_remove", "read_text_file", "write_text_file",
    # yaml_utils
    "load_yaml_file", "dump_yaml", "load_yaml_string", "merge_yaml_dicts",
    # path_utils
    "get_project_root", "get_configs_dir", "get_output_dir",
    "get_logs_dir", "resolve_path", "is_subpath",
    # system_utils
    "get_cpu_percent", "get_memory_percent", "get_cpu_count",
    "get_hostname", "get_platform_info", "is_psutil_available",
    # time_utils
    "now_utc_iso", "monotonic_seconds", "perf_counter",
    "timer", "format_duration", "compute_overhead_fraction",
    # validation_utils
    "validate_positive_int", "validate_non_negative_float",
    "validate_fraction", "validate_pdf_path", "validate_page_count",
    "validate_source_type", "validate_non_empty_string", "validate_choices",
]
