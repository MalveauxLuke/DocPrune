"""Validate the immutable Poppler executables used for PDF rasterization."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

PDFTOOLS_ENV = "PDFTOOLS_DIR"
POPPLER_VERSION = "26.05.0"
PDFINFO_SHA256 = "5d0e1caa04f15391324c9e5f1d65753d8b925761eedc710c70e02f49b5080aae"
PDFTOPPM_SHA256 = "1102bc3f4a12f3d3d207ac8e39f463d0fb3c403511fb4c817e776e5251b53f33"


@dataclass(frozen=True)
class PDFToolsIdentity:
    """The exact Poppler executables used by a benchmark process."""

    directory: Path
    pdfinfo: Path
    pdftoppm: Path
    pdfinfo_sha256: str
    pdftoppm_sha256: str
    version: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_regular_executable(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink executable: {path}")
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise ValueError(f"{label} is missing: {path}") from error
    if not stat.S_ISREG(mode) or not os.access(path, os.X_OK):
        raise ValueError(f"{label} must be a regular executable: {path}")


def _version(path: Path, *, executable: str) -> str:
    try:
        result = subprocess.run(
            [str(path), "-v"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ValueError(f"unable to execute pinned {executable}: {path}") from error
    output = (result.stdout + result.stderr).splitlines()
    prefix = f"{executable} version "
    line = next((value.strip() for value in output if value.strip().startswith(prefix)), None)
    if result.returncode != 0 or line is None:
        raise ValueError(f"{executable} did not report a usable Poppler version: {path}")
    version = line.removeprefix(prefix)
    if version != POPPLER_VERSION:
        raise ValueError(
            f"{executable} version must be {POPPLER_VERSION}, got {version!r}: {path}"
        )
    return version


def validate_pdf_tools(directory: Path | str | None = None) -> PDFToolsIdentity:
    """Require the pinned Poppler installation and return its measured identity."""

    if directory is None:
        value = os.environ.get(PDFTOOLS_ENV)
        if not value:
            raise ValueError(f"required environment variable {PDFTOOLS_ENV} is not set")
        directory = value
    directory = Path(os.path.abspath(os.fspath(directory)))
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"{PDFTOOLS_ENV} must be a regular directory: {directory}")

    pdfinfo = directory / "bin" / "pdfinfo"
    pdftoppm = directory / "bin" / "pdftoppm"
    _require_regular_executable(pdfinfo, label="pdfinfo")
    _require_regular_executable(pdftoppm, label="pdftoppm")
    pdfinfo_sha256 = _sha256(pdfinfo)
    if pdfinfo_sha256 != PDFINFO_SHA256:
        raise ValueError(f"pdfinfo SHA-256 mismatch: {pdfinfo}")
    pdftoppm_sha256 = _sha256(pdftoppm)
    if pdftoppm_sha256 != PDFTOPPM_SHA256:
        raise ValueError(f"pdftoppm SHA-256 mismatch: {pdftoppm}")
    pdfinfo_version = _version(pdfinfo, executable="pdfinfo")
    pdftoppm_version = _version(pdftoppm, executable="pdftoppm")
    if pdfinfo_version != pdftoppm_version:
        raise ValueError("pdfinfo and pdftoppm report different Poppler versions")
    return PDFToolsIdentity(
        directory=directory,
        pdfinfo=pdfinfo,
        pdftoppm=pdftoppm,
        pdfinfo_sha256=pdfinfo_sha256,
        pdftoppm_sha256=pdftoppm_sha256,
        version=pdfinfo_version,
    )
