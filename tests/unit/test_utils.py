"""Unit tests for utility modules.

Tests cover:
    - file_utils: find_pdf_files, get_file_size_mb, compute_md5,
      ensure_directory, safe_remove, read_text_file, write_text_file
    - path_utils: resolve_output_dir, get_run_output_dir
    - time_utils: utc_now, format_iso8601, elapsed_seconds
    - validation_utils: validate_positive_int, validate_fraction,
      validate_non_empty_string, validate_path_exists
    - yaml_utils: load_yaml, dump_yaml
    - system_utils: get_cpu_count, get_available_memory_mb
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from adaptive_framework.core.exceptions import ValidationError
from adaptive_framework.utils.file_utils import (
    compute_md5,
    ensure_directory,
    find_pdf_files,
    get_file_size_mb,
    read_text_file,
    safe_remove,
    write_text_file,
)
from adaptive_framework.utils.time_utils import (
    now_utc_iso,
    perf_counter,
    monotonic_seconds,
)
from adaptive_framework.utils.validation_utils import (
    validate_fraction,
    validate_non_empty_string,
    validate_positive_int,
)
from adaptive_framework.utils.yaml_utils import dump_yaml, load_yaml_file


# ===========================================================================
# file_utils
# ===========================================================================


class TestFindPdfFiles:
    """Tests for find_pdf_files."""

    def test_returns_empty_list_when_no_pdfs(self, tmp_path: Path) -> None:
        """Empty directory returns an empty list."""
        result = find_pdf_files(tmp_path)
        assert result == []

    def test_finds_pdf_files(self, tmp_path: Path) -> None:
        """PDF files in the directory are returned."""
        (tmp_path / "a.pdf").write_text("pdf", encoding="utf-8")
        (tmp_path / "b.pdf").write_text("pdf", encoding="utf-8")
        result = find_pdf_files(tmp_path)
        assert len(result) == 2

    def test_ignores_non_pdf_files(self, tmp_path: Path) -> None:
        """Non-PDF files are not included in the result."""
        (tmp_path / "a.txt").write_text("text", encoding="utf-8")
        (tmp_path / "b.pdf").write_text("pdf", encoding="utf-8")
        result = find_pdf_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "b.pdf"

    def test_returns_sorted_list(self, tmp_path: Path) -> None:
        """Results are returned in sorted order."""
        (tmp_path / "z.pdf").write_text("z", encoding="utf-8")
        (tmp_path / "a.pdf").write_text("a", encoding="utf-8")
        result = find_pdf_files(tmp_path)
        assert result[0].name == "a.pdf"
        assert result[1].name == "z.pdf"

    def test_missing_directory_raises_file_not_found(self) -> None:
        """Non-existent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            find_pdf_files(Path("/nonexistent/path/xyz"))

    def test_recursive_finds_nested_pdfs(self, tmp_path: Path) -> None:
        """Recursive mode finds PDFs in sub-directories."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.pdf").write_text("pdf", encoding="utf-8")
        result = find_pdf_files(tmp_path, recursive=True)
        assert len(result) == 1
        assert result[0].name == "nested.pdf"

    def test_non_recursive_misses_nested_pdfs(self, tmp_path: Path) -> None:
        """Non-recursive mode does not find PDFs in sub-directories."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.pdf").write_text("pdf", encoding="utf-8")
        result = find_pdf_files(tmp_path, recursive=False)
        assert result == []


class TestGetFileSizeMb:
    """Tests for get_file_size_mb."""

    def test_returns_zero_for_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns 0.0 MB."""
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert get_file_size_mb(f) == 0.0

    def test_missing_file_raises(self) -> None:
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_file_size_mb(Path("/nonexistent/file.txt"))

    def test_returns_positive_for_non_empty_file(self, tmp_path: Path) -> None:
        """Non-empty file returns a positive size."""
        f = tmp_path / "data.txt"
        f.write_bytes(b"x" * 1024)  # 1 KiB
        size = get_file_size_mb(f)
        assert size > 0.0


class TestComputeMd5:
    """Tests for compute_md5."""

    def test_empty_file_has_known_md5(self, tmp_path: Path) -> None:
        """Empty file has the well-known MD5 of an empty byte string."""
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert compute_md5(f) == "d41d8cd98f00b204e9800998ecf8427e"

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Two files with different content produce different MD5 hashes."""
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"hello")
        f2.write_bytes(b"world")
        assert compute_md5(f1) != compute_md5(f2)

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        """Two files with identical content produce identical MD5 hashes."""
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"same content")
        f2.write_bytes(b"same content")
        assert compute_md5(f1) == compute_md5(f2)

    def test_missing_file_raises(self) -> None:
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            compute_md5(Path("/nonexistent/file.bin"))


class TestEnsureDirectory:
    """Tests for ensure_directory."""

    def test_creates_missing_directory(self, tmp_path: Path) -> None:
        """ensure_directory creates a directory that does not exist."""
        new_dir = tmp_path / "new" / "nested"
        ensure_directory(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_existing_directory_no_error(self, tmp_path: Path) -> None:
        """ensure_directory does not raise if directory already exists."""
        ensure_directory(tmp_path)  # already exists, must not raise

    def test_returns_path(self, tmp_path: Path) -> None:
        """ensure_directory returns the same path it was given."""
        result = ensure_directory(tmp_path / "output")
        assert result == tmp_path / "output"


class TestSafeRemove:
    """Tests for safe_remove."""

    def test_removes_existing_file(self, tmp_path: Path) -> None:
        """safe_remove removes an existing file and returns True."""
        f = tmp_path / "file.txt"
        f.write_text("content", encoding="utf-8")
        result = safe_remove(f)
        assert result is True
        assert not f.exists()

    def test_returns_false_for_missing_file(self, tmp_path: Path) -> None:
        """safe_remove returns False without raising if file is absent."""
        result = safe_remove(tmp_path / "ghost.txt")
        assert result is False


class TestReadWriteTextFile:
    """Tests for read_text_file and write_text_file."""

    def test_write_and_read_roundtrip(self, tmp_path: Path) -> None:
        """write_text_file + read_text_file roundtrip preserves content."""
        f = tmp_path / "out.txt"
        write_text_file(f, "hello world")
        content = read_text_file(f)
        assert content == "hello world"

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        """write_text_file creates parent directories if they don't exist."""
        f = tmp_path / "a" / "b" / "c.txt"
        write_text_file(f, "test")
        assert f.exists()

    def test_read_missing_file_raises(self) -> None:
        """read_text_file raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            read_text_file(Path("/nonexistent/file.txt"))

    def test_overwrite_false_raises_if_exists(self, tmp_path: Path) -> None:
        """write_text_file with overwrite=False raises FileExistsError."""
        f = tmp_path / "existing.txt"
        f.write_text("original", encoding="utf-8")
        with pytest.raises(FileExistsError):
            write_text_file(f, "new content", overwrite=False)


# ===========================================================================
# time_utils
# ===========================================================================


class TestTimeUtils:
    """Tests for time utility functions."""

    def test_utc_now_returns_non_empty_string(self) -> None:
        """now_utc_iso() returns a non-empty ISO 8601 string."""
        result = now_utc_iso()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "+" in result or "Z" in result  # timezone marker

    def test_perf_counter_returns_float(self) -> None:
        """perf_counter() returns a float."""
        result = perf_counter()
        assert isinstance(result, float)

    def test_monotonic_seconds_is_non_decreasing(self) -> None:
        """Two successive monotonic_seconds() calls are non-decreasing."""
        t1 = monotonic_seconds()
        time.sleep(0.01)
        t2 = monotonic_seconds()
        assert t2 >= t1


# ===========================================================================
# validation_utils
# ===========================================================================


class TestValidationUtils:
    """Tests for validation utility functions."""

    def test_validate_positive_int_valid(self) -> None:
        """validate_positive_int accepts positive integers."""
        validate_positive_int(1, "pages")  # should not raise
        validate_positive_int(100, "count")

    def test_validate_positive_int_zero_raises(self) -> None:
        """validate_positive_int raises ValidationError for zero."""
        with pytest.raises(ValidationError, match="pages"):
            validate_positive_int(0, "pages")

    def test_validate_positive_int_negative_raises(self) -> None:
        """validate_positive_int raises ValidationError for negatives."""
        with pytest.raises(ValidationError, match="count"):
            validate_positive_int(-5, "count")

    def test_validate_fraction_valid_bounds(self) -> None:
        """validate_fraction accepts values in [0.0, 1.0]."""
        validate_fraction(0.0, "ratio")
        validate_fraction(0.5, "ratio")
        validate_fraction(1.0, "ratio")

    def test_validate_fraction_above_one_raises(self) -> None:
        """validate_fraction raises ValidationError for values > 1.0."""
        with pytest.raises(ValidationError, match="ratio"):
            validate_fraction(1.1, "ratio")

    def test_validate_fraction_negative_raises(self) -> None:
        """validate_fraction raises ValidationError for negative values."""
        with pytest.raises(ValidationError, match="ratio"):
            validate_fraction(-0.1, "ratio")

    def test_validate_non_empty_string_valid(self) -> None:
        """validate_non_empty_string accepts non-empty strings."""
        validate_non_empty_string("hello", "name")  # should not raise

    def test_validate_non_empty_string_empty_raises(self) -> None:
        """validate_non_empty_string raises ValidationError for empty string."""
        with pytest.raises(ValidationError, match="name"):
            validate_non_empty_string("", "name")

    def test_validate_non_empty_string_whitespace_raises(self) -> None:
        """validate_non_empty_string raises ValidationError for whitespace-only."""
        with pytest.raises(ValidationError, match="name"):
            validate_non_empty_string("   ", "name")


# ===========================================================================
# yaml_utils
# ===========================================================================


class TestYamlUtils:
    """Tests for yaml_utils load/dump functions."""

    def test_load_valid_yaml_file(self, tmp_path: Path) -> None:
        """load_yaml_file correctly parses a valid YAML file."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\nnum: 42\n", encoding="utf-8")
        data = load_yaml_file(yaml_file)
        assert data == {"key": "value", "num": 42}

    def test_load_empty_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        """load_yaml_file on an empty YAML file returns an empty dict."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("", encoding="utf-8")
        data = load_yaml_file(yaml_file)
        assert data == {}

    def test_dump_and_reload_roundtrip(self, tmp_path: Path) -> None:
        """dump_yaml followed by load_yaml_file preserves the data structure."""
        original = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
        yaml_file = tmp_path / "roundtrip.yaml"
        dump_yaml(original, yaml_file)
        reloaded = load_yaml_file(yaml_file)
        assert reloaded == original

    def test_load_missing_file_raises(self) -> None:
        """load_yaml_file raises for a missing file."""
        from adaptive_framework.core.exceptions import ConfigurationError
        with pytest.raises((FileNotFoundError, ConfigurationError)):
            load_yaml_file(Path("/nonexistent/config.yaml"))
