"""Tests for the pinned Poppler toolchain used by M3DocVQA rendering."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "examples" / "m3docvqa"))
from pdf_tools import validate_pdf_tools  # noqa: E402

PDFTOOLS_DIR = Path("/home/lmalveau/mamba-envs/m3docvqa-acquisition")
CORPUS_ROOT = Path("/scratch/lmalveau/docprune/datasets/m3docvqa")
PDFINFO_SHA256 = "5d0e1caa04f15391324c9e5f1d65753d8b925761eedc710c70e02f49b5080aae"
PDFTOPPM_SHA256 = "1102bc3f4a12f3d3d207ac8e39f463d0fb3c403511fb4c817e776e5251b53f33"


def _load_probe_image_module():
    """Load the helper by file path; tests must not require ``examples`` as a package."""

    path = ROOT / "examples" / "m3docvqa" / "make_probe_image.py"
    spec = importlib.util.spec_from_file_location("docprune_test_make_probe_image", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load probe helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_pdf_tools_accepts_the_pinned_acquisition_toolchain() -> None:
    if not PDFTOOLS_DIR.is_dir():
        pytest.skip("the SOL acquisition environment is not mounted")

    identity = validate_pdf_tools(PDFTOOLS_DIR)

    assert identity.directory == PDFTOOLS_DIR
    assert identity.pdfinfo_sha256 == PDFINFO_SHA256
    assert identity.pdftoppm_sha256 == PDFTOPPM_SHA256
    assert identity.version == "26.05.0"


def test_validate_pdf_tools_rejects_symlinked_executables(tmp_path: Path) -> None:
    if not PDFTOOLS_DIR.is_dir():
        pytest.skip("the SOL acquisition environment is not mounted")

    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "pdfinfo").symlink_to(PDFTOOLS_DIR / "bin" / "pdfinfo")
    (tmp_path / "bin" / "pdftoppm").symlink_to(PDFTOOLS_DIR / "bin" / "pdftoppm")

    with pytest.raises(ValueError, match="regular|symlink"):
        validate_pdf_tools(tmp_path)


def test_each_benchmark_launcher_has_the_same_pdf_tool_preflight() -> None:
    launcher_dir = ROOT / "examples" / "sbatch"
    for name in (
        "11_docprune_m3docvqa.sbatch",
        "12_docprune_m3docvqa_gate.sbatch",
        "13_docprune_m3docvqa_index.sbatch",
        "14_docprune_m3docvqa_eval_array.sbatch",
    ):
        text = (launcher_dir / name).read_text(encoding="utf-8")
        assert ': "${PDFTOOLS_DIR:?' in text
        assert "pdfinfo_sha256" in text
        assert "pdftoppm_sha256" in text
        assert 'PATH="$PDFTOOLS_DIR/bin:$ENV_DIR/bin:$PATH"' in text


def test_shell_pdf_tool_preflight_requires_the_directory() -> None:
    helper = ROOT / "examples" / "m3docvqa" / "pdf_tools_preflight.sh"
    env = os.environ.copy()
    env.pop("PDFTOOLS_DIR", None)
    result = subprocess.run(
        ["bash", "-c", f"source {helper}"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "PDFTOOLS_DIR" in result.stderr


def test_probe_helper_requires_pinned_pdf_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    make_probe_image = _load_probe_image_module()

    monkeypatch.delenv("PDFTOOLS_DIR", raising=False)
    with pytest.raises(ValueError, match="PDFTOOLS_DIR"):
        make_probe_image._require_pdf_tools()


def test_real_production_pdf_render_uses_pinned_tools() -> None:
    if not PDFTOOLS_DIR.is_dir() or not CORPUS_ROOT.is_dir():
        pytest.skip("the SOL acquisition environment or production corpus is not mounted")

    env = os.environ.copy()
    env["PDFTOOLS_DIR"] = str(PDFTOOLS_DIR)
    env["PATH"] = f"{PDFTOOLS_DIR / 'bin'}:{env['PATH']}"
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="docprune-pdf-smoke-") as root:
        output = Path(root) / "probe.png"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "examples/m3docvqa/make_probe_image.py"),
                "--corpus-root",
                str(CORPUS_ROOT),
                "--qid",
                "a33985b1e8b2502fc18cc8147dc27db8",
                "--output",
                str(output),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert output.is_file()
        assert output.stat().st_size > 0
