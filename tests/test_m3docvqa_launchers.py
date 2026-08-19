"""Regression checks for the executable M3DocVQA benchmark handoff."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
LAUNCHER_DIR = ROOT / "examples" / "sbatch"
LAUNCHERS = tuple(
    LAUNCHER_DIR / name
    for name in (
        "11_docprune_m3docvqa.sbatch",
        "12_docprune_m3docvqa_gate.sbatch",
        "13_docprune_m3docvqa_index.sbatch",
        "14_docprune_m3docvqa_eval_array.sbatch",
    )
)
HANDOFF = ROOT / "sol" / "handoffs" / "DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md"


def _python_heredocs(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    return tuple(re.findall(r"<<['\"]PY['\"]\n(.*?)\nPY\n", text, flags=re.DOTALL))


def test_embedded_python_heredocs_are_f821_clean() -> None:
    """Lint each embedded Python program independently, as bash executes it."""

    ruff = shutil.which("ruff")
    assert ruff, "ruff is required to lint launcher heredocs"
    for launcher in LAUNCHERS:
        snippets = _python_heredocs(launcher)
        assert snippets, f"launcher has no embedded Python: {launcher}"
        for number, snippet in enumerate(snippets, start=1):
            result = subprocess.run(
                [ruff, "check", "--select", "F821", "--stdin-filename", f"{launcher.name}.{number}.py", "-"],
                input=snippet,
                text=True,
                capture_output=True,
                check=False,
            )
            assert result.returncode == 0, f"{launcher} heredoc {number}: {result.stdout}{result.stderr}"


def test_gate_binds_qids_to_supporting_documents_and_checks_fixed_set() -> None:
    text = (LAUNCHER_DIR / "12_docprune_m3docvqa_gate.sbatch").read_text(encoding="utf-8")
    assert "supporting_context" in text
    assert "dataset.document_ids[0]" not in text
    assert "for qid in qids" in text
    assert "insufficient pages" in text


def test_eval_postcheck_uses_combined_summary_without_copy() -> None:
    text = (LAUNCHER_DIR / "11_docprune_m3docvqa.sbatch").read_text(encoding="utf-8")
    assert 'summary.get("efficiency")' in text
    assert 'efficiency["samples"]' in text
    assert 'summary.get("quality")' in text
    assert "cp --" not in text


def test_index_seals_resume_and_uses_production_manifest_loader() -> None:
    text = (LAUNCHER_DIR / "13_docprune_m3docvqa_index.sbatch").read_text(encoding="utf-8")
    assert ': "${RESUME:=0}"' in text
    assert 'case "$RESUME"' in text
    assert "--resume" in text
    assert "_load_index_manifest" in text
    assert "IndexManifest.validate_files" not in text


def test_launchers_validate_sealed_control_record() -> None:
    for launcher in LAUNCHERS:
        text = launcher.read_text(encoding="utf-8")
        assert 'CONTROL_RECORD' in text
        assert 'control.json' in text
        assert 'tree_sha' in text


def test_run_config_generator_is_executable_and_documented() -> None:
    generator = ROOT / "examples" / "m3docvqa" / "make_run_configs.py"
    assert generator.is_file()
    assert generator.stat().st_mode & 0o111
    handoff = HANDOFF.read_text(encoding="utf-8")
    assert "make_run_configs.py" in handoff
    assert "--dependency=\"afterok:" in handoff


def test_handoff_has_absolute_log_submission_and_no_control_placeholder() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "control.json" in text
    assert "rev-parse HEAD" in text
    assert "git rev-parse HEAD:" not in text
    assert "--output=\"$SLURM_LOG_DIR/" in text
    assert "--error=\"$SLURM_LOG_DIR/" in text
    assert "--chdir=\"$SLURM_LOG_DIR\"" in text
