"""File utility functions for the Adaptive Distributed Framework.

Pure functions with no side effects beyond file I/O.
No business logic, no framework dependencies.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from adaptive_framework.core.constants import DEFAULT_ENCODING, PDF_EXTENSION


def find_pdf_files(directory: Path, recursive: bool = False) -> list[Path]:
    """Return all PDF files in a directory.

    Args:
        directory: Path to the directory to search.
        recursive: If True, search subdirectories recursively.

    Returns:
        Sorted list of Path objects for each PDF file found.
            Empty list if no PDFs are present.

    Raises:
        FileNotFoundError: If directory does not exist.

    Example:
        >>> files = find_pdf_files(Path("/data/raw"))
        >>> print(len(files))
        42
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: '{directory}'")
    pattern = f"**/*{PDF_EXTENSION}" if recursive else f"*{PDF_EXTENSION}"
    return sorted(directory.glob(pattern))


def get_file_size_mb(file_path: Path) -> float:
    """Return the size of a file in megabytes.

    Args:
        file_path: Path to the file.

    Returns:
        File size in MB (rounded to 4 decimal places).

    Raises:
        FileNotFoundError: If the file does not exist.

    Example:
        >>> size = get_file_size_mb(Path("/data/paper.pdf"))
        >>> print(f"{size:.2f} MB")
        3.72 MB
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: '{file_path}'")
    size_bytes = file_path.stat().st_size
    return round(size_bytes / (1024 * 1024), 4)


def compute_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute the MD5 hash of a file.

    Args:
        file_path: Path to the file.
        chunk_size: Read chunk size in bytes. Defaults to 8192.

    Returns:
        Lowercase hexadecimal MD5 hash string.

    Raises:
        FileNotFoundError: If the file does not exist.

    Example:
        >>> md5 = compute_md5(Path("/data/paper.pdf"))
        >>> print(md5)
        'd41d8cd98f00b204e9800998ecf8427e'
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: '{file_path}'")
    hasher = hashlib.md5()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_directory(path: Path) -> Path:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path, now guaranteed to exist.

    Example:
        >>> log_dir = ensure_directory(Path("outputs/logs"))
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_remove(path: Path) -> bool:
    """Remove a file if it exists, without raising if absent.

    Args:
        path: Path to the file to remove.

    Returns:
        True if the file was removed, False if it did not exist.

    Example:
        >>> removed = safe_remove(Path("outputs/old_log.log"))
    """
    if path.exists() and path.is_file():
        path.unlink()
        return True
    return False


def read_text_file(file_path: Path, encoding: str = DEFAULT_ENCODING) -> str:
    """Read and return the full content of a text file.

    Args:
        file_path: Path to the text file.
        encoding: File encoding. Defaults to UTF-8.

    Returns:
        File content as a string.

    Raises:
        FileNotFoundError: If the file does not exist.

    Example:
        >>> content = read_text_file(Path("configs/framework.yaml"))
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: '{file_path}'")
    return file_path.read_text(encoding=encoding)


def write_text_file(
    file_path: Path,
    content: str,
    encoding: str = DEFAULT_ENCODING,
    overwrite: bool = True,
) -> None:
    """Write a string to a text file, creating parent directories as needed.

    Args:
        file_path: Destination file path.
        content: String content to write.
        encoding: File encoding. Defaults to UTF-8.
        overwrite: If False and file exists, raises FileExistsError.

    Raises:
        FileExistsError: If overwrite is False and the file already exists.

    Example:
        >>> write_text_file(Path("outputs/report.md"), "# Report")
    """
    if not overwrite and file_path.exists():
        raise FileExistsError(f"File already exists: '{file_path}'")
    ensure_directory(file_path.parent)
    file_path.write_text(content, encoding=encoding)
