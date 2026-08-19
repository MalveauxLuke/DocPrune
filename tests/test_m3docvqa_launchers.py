"""Regression checks for the executable M3DocVQA benchmark handoff."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "m3docvqa"))
from make_run_configs import (  # noqa: E402
    FIXED_GATE_SAMPLE_IDS,
    M3DOCRAG_COMMIT,
    RUNTIME_COMMIT,
    validate_gate_evidence,
)

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
    assert ",".join(FIXED_GATE_SAMPLE_IDS) in text
    assert 'test "$GATE_SAMPLE_IDS" = "$EXPECTED_GATE_SAMPLE_IDS"' in text
    assert 'source_positions != sorted(source_positions)' in text


def test_gate_lifecycle_and_post_gate_config_order_are_fail_closed() -> None:
    text = (LAUNCHER_DIR / "12_docprune_m3docvqa_gate.sbatch").read_text(encoding="utf-8")
    assert 'test ! -e "$GATE_ROOT"' in text
    assert 'mkdir -p "$GATE_ROOT"' in text
    assert 'RUN_CONFIG="$GATE_ROOT/' not in text
    assert 'RUN_CONFIG_ROOT' in text
    assert 'export PROJECT_DIR CONTROL_RECORD CORPUS_ROOT RUN_CONFIG_ROOT' in text
    assert 'make_run_configs.py' in text
    assert text.index('make_run_configs.py') < text.index('GATE_STATUS=passed')
    assert 'GATE_STATUS=passed' in text
    assert text.index('GATE_STATUS=passed') > text.index('python -m json.tool "$GATE_ROOT/gate.json"')


def test_gate_renders_probe_image_from_pinned_corpus_qid() -> None:
    text = (LAUNCHER_DIR / "12_docprune_m3docvqa_gate.sbatch").read_text(encoding="utf-8")
    assert 'PROBE_IMAGE="$GATE_ROOT/' in text
    assert "convert_from_path" in text
    assert "dpi=144" in text
    assert ': "${PROBE_IMAGE:' not in text
    assert ': "${CORPUS_ROOT:' in text


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


def test_submission_does_not_precreate_gate_or_run_final_generator() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    submission = text.split("## Exact Slurm submission commands", 1)[1].split(
        "## Pass conditions", 1
    )[0]
    assert 'mkdir -p "$SLURM_LOG_DIR" "$INPUT_ROOT"' in submission
    assert 'mkdir -p "$SLURM_LOG_DIR" "$GATE_ROOT"' not in submission
    assert "make_run_configs.py" not in submission
    assert 'RUN_CONFIG="$INPUT_ROOT/gate-top1.json"' in submission
    assert 'PROBE_IMAGE=' not in submission


def _gate_fixture(tmp_path: Path) -> tuple[Path, Path]:
    contract = {
        "schema_version": 2,
        "resources": {
            "qwen": {"model": "Qwen/Qwen2-VL-7B-Instruct", "revision": "eed13092ef92e448dd6875b2a00151bd3f7db0ac"},
            "colpali": {"model": "vidore/colpali-v1.2", "revision": "961b51745de3e9adb3468ac5c9ccca0ac626c217"},
            "colpali_backbone": {"model": "vidore/colpaligemma-3b-pt-448-base", "revision": "30ab955d073de4a91dc5a288e8c97226647e3e5a"},
        },
        "mapping_checks": {
            "colpali_visual_grid_inferred": True,
            "qwen_merge_groups_valid": True,
            "raster_order_verified": True,
        },
    }
    contract_path = tmp_path / "processor-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    semantic = {
        "status": "passed",
        "baseline_equivalence": True,
        "equivalence_qids": list(FIXED_GATE_SAMPLE_IDS),
        "supporting_documents": {qid: "doc" for qid in FIXED_GATE_SAMPLE_IDS},
        "docprune_traces": {str(page): [100, 80, 60, 40] for page in (1, 2, 4)},
    }
    semantic_path = tmp_path / "semantic-samples.json"
    semantic_path.write_text(json.dumps(semantic), encoding="utf-8")

    gate = {
        "schema_version": 1,
        "status": "passed",
        "runtime_commit": RUNTIME_COMMIT,
        "m3docrag_commit": M3DOCRAG_COMMIT,
        "processor_contract_path": str(contract_path.resolve()),
        "processor_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "sample_ids": list(FIXED_GATE_SAMPLE_IDS),
        "page_counts": [1, 2, 4],
        "semantic_samples_path": str(semantic_path.resolve()),
        "semantic_samples_sha256": hashlib.sha256(semantic_path.read_bytes()).hexdigest(),
    }
    unsigned_gate = dict(gate)
    gate["gate_sha256"] = hashlib.sha256(
        json.dumps(unsigned_gate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(__import__("json").dumps(gate), encoding="utf-8")
    return gate_path, contract_path


@pytest.mark.parametrize("mutation", ["gate_sha256", "runtime_commit", "sample_ids", "semantic", "traces"])
def test_run_config_generator_rejects_tampered_gate_evidence(tmp_path: Path, mutation: str) -> None:
    gate_path, contract_path = _gate_fixture(tmp_path)
    import json

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    semantic_path = Path(gate["semantic_samples_path"])
    if mutation == "gate_sha256":
        gate["gate_sha256"] = "0" * 64
    elif mutation == "runtime_commit":
        gate["runtime_commit"] = "1" * 40
    elif mutation == "sample_ids":
        gate["sample_ids"] = list(reversed(FIXED_GATE_SAMPLE_IDS))
    elif mutation == "semantic":
        semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
        semantic["baseline_equivalence"] = False
        semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
    else:
        semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
        semantic["docprune_traces"]["1"] = [1, 2, 3, 4]
        semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_gate_evidence(gate_path, contract_path)


def test_handoff_has_absolute_log_submission_and_no_control_placeholder() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "control.json" in text
    assert "rev-parse HEAD" in text
    assert "git rev-parse HEAD:" not in text
    assert "--output=\"$SLURM_LOG_DIR/" in text
    assert "--error=\"$SLURM_LOG_DIR/" in text
    assert "--chdir=\"$SLURM_LOG_DIR\"" in text
