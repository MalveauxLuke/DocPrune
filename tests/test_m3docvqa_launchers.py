"""Regression checks for the executable M3DocVQA benchmark handoff."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "m3docvqa"))
from make_probe_image import render_probe_image  # noqa: E402
from make_run_configs import (  # noqa: E402
    FIXED_GATE_SAMPLE_IDS,
    M3DOCRAG_COMMIT,
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
        "15_docprune_m3docvqa_compare.sbatch",
    )
)
HANDOFF = ROOT / "sol" / "handoffs" / "DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md"
TEST_RUNTIME_COMMIT = "a" * 40
HISTORICAL_RUNTIME_COMMIT = "3755812cc3dc1a6205671202894cdf7915bc95a9"
HISTORICAL_RUNTIME_DIR = "/home/lmalveau/DocPrune-runtime-3755812"
HISTORICAL_ATTEMPT_ROOT = "/scratch/lmalveau/docprune/benchmark-3755812/attempt-N"


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
                [
                    ruff,
                    "check",
                    "--select",
                    "F821",
                    "--stdin-filename",
                    f"{launcher.name}.{number}.py",
                    "-",
                ],
                input=snippet,
                text=True,
                capture_output=True,
                check=False,
            )
            assert result.returncode == 0, (
                f"{launcher} heredoc {number}: {result.stdout}{result.stderr}"
            )


def test_gate_binds_qids_to_supporting_documents_and_checks_fixed_set() -> None:
    text = (LAUNCHER_DIR / "12_docprune_m3docvqa_gate.sbatch").read_text(encoding="utf-8")
    assert "supporting_context" in text
    assert "MiniDataset" in text
    assert "build_index" in text
    assert "_load_index_manifest" in text
    assert "for qid in qids" in text
    assert "insufficient pages" in text
    assert ",".join(FIXED_GATE_SAMPLE_IDS) in text
    assert 'test "$GATE_SAMPLE_IDS" = "$EXPECTED_GATE_SAMPLE_IDS"' in text
    assert "source_positions != sorted(source_positions)" in text


def test_gate_trace_failure_reports_all_four_token_counts() -> None:
    text = (LAUNCHER_DIR / "12_docprune_m3docvqa_gate.sbatch").read_text(encoding="utf-8")
    assert 'counts = [trace[key] for key in ("original_visual_tokens", "post_btp_visual_tokens", "post_qtp_visual_tokens", "post_ctp_visual_tokens")]' in text
    assert 'f"non-monotonic or empty DocPrune trace for {qid} top-{pages}: counts={counts}"' in text


def test_gate_lifecycle_and_post_gate_config_order_are_fail_closed() -> None:
    text = (LAUNCHER_DIR / "12_docprune_m3docvqa_gate.sbatch").read_text(encoding="utf-8")
    assert 'test ! -e "$GATE_ROOT"' in text
    assert 'mkdir -p "$GATE_ROOT"' in text
    assert 'RUN_CONFIG="$GATE_ROOT/' not in text
    assert "RUN_CONFIG_ROOT" in text
    assert "export PROJECT_DIR CONTROL_RECORD CORPUS_ROOT RUN_CONFIG_ROOT" in text
    assert "make_run_configs.py" in text
    assert text.index("make_run_configs.py") < text.index("GATE_STATUS=passed")
    assert "GATE_STATUS=passed" in text
    assert text.index("GATE_STATUS=passed") > text.index(
        '"$ENV_DIR/bin/python" -m json.tool "$GATE_ROOT/gate.json"'
    )


def test_gate_renders_probe_image_from_pinned_corpus_qid() -> None:
    text = (LAUNCHER_DIR / "12_docprune_m3docvqa_gate.sbatch").read_text(encoding="utf-8")
    helper = (ROOT / "examples" / "m3docvqa" / "make_probe_image.py").read_text(encoding="utf-8")
    assert 'PROBE_IMAGE="$GATE_ROOT/' in text
    assert "make_probe_image.py" in text
    assert "convert_from_path" in helper
    assert "dpi=144" in helper
    assert ': "${PROBE_IMAGE:' not in text
    assert ': "${CORPUS_ROOT:' in text


def test_probe_helper_uses_first_pinned_supporting_pdf_and_no_pdf_path_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The helper must use CorpusIdentity.pdf_dir, not an invented pdf_path API."""

    import make_probe_image

    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps(
            {
                "qid": "fixed",
                "supporting_context": [
                    {"doc_id": "not-pinned"},
                    {"doc_id": "doc-a"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    document_ids = tmp_path / "dev_doc_ids.json"
    document_ids.write_text(json.dumps(["doc-a"]), encoding="utf-8")
    pdf_dir = tmp_path / "pdfs_dev"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "doc-a.pdf"
    pdf_path.write_bytes(b"pinned pdf")
    corpus = SimpleNamespace(
        questions_path=questions,
        document_ids_path=document_ids,
        pdf_dir=pdf_dir,
        validate=lambda: None,
    )
    monkeypatch.setattr(make_probe_image.CorpusIdentity, "from_root", lambda _: corpus)
    calls: dict[str, object] = {}

    class FakePage:
        def convert(self, mode: str) -> FakePage:
            calls["convert_mode"] = mode
            return self

        def save(self, path: Path, *, format: str, dpi: tuple[int, int]) -> None:
            calls["save"] = (Path(path), format, dpi)
            Path(path).write_bytes(b"png")

    def fake_convert(path: str, *, dpi: int, first_page: int, last_page: int) -> list[FakePage]:
        calls["convert"] = (Path(path), dpi, first_page, last_page)
        return [FakePage()]

    monkeypatch.setattr(make_probe_image, "convert_from_path", fake_convert)
    monkeypatch.setenv("PDFTOOLS_DIR", "/home/lmalveau/mamba-envs/m3docvqa-acquisition")
    output = tmp_path / "gate" / "probe-page.png"
    output.parent.mkdir()
    run_config = tmp_path / "gate-top1.json"
    run_config.write_text(json.dumps({"corpus": {"root": str(tmp_path)}}), encoding="utf-8")

    assert render_probe_image(tmp_path, "fixed", output, run_config=run_config) == output
    assert calls["convert"] == (pdf_path, 144, 1, 1)
    assert calls["convert_mode"] == "RGB"
    assert calls["save"] == (output, "PNG", (144, 144))
    assert output.read_bytes() == b"png"

    with pytest.raises(FileExistsError):
        render_probe_image(tmp_path, "fixed", output)


def test_eval_postcheck_uses_combined_summary_without_copy() -> None:
    text = (LAUNCHER_DIR / "11_docprune_m3docvqa.sbatch").read_text(encoding="utf-8")
    assert 'summary.get("efficiency")' in text
    assert 'efficiency["samples"]' in text
    assert 'summary.get("quality")' in text
    assert "cp --" not in text


def test_eval_array_delegates_via_project_dir_not_spooled_script() -> None:
    """The array must invoke the tracked cell launcher from the sealed checkout."""

    text = (LAUNCHER_DIR / "14_docprune_m3docvqa_eval_array.sbatch").read_text(
        encoding="utf-8"
    )
    launcher_assignment = (
        'CELL_LAUNCHER="$PROJECT_DIR/examples/sbatch/11_docprune_m3docvqa.sbatch"'
    )
    assert launcher_assignment in text
    assert 'test -f "$CELL_LAUNCHER"' in text
    assert 'exec "$CELL_LAUNCHER"' in text
    assert text.index("sealed control record project path mismatch") < text.index(
        launcher_assignment
    )
    assert "BASH_SOURCE" not in text
    assert "SCRIPT_DIR" not in text
    assert "dirname" not in text
    assert '"$0"' not in text
    assert "readlink" not in text
    assert "/var/spool/slurmd" not in text


def test_index_seals_resume_and_uses_production_manifest_loader() -> None:
    text = (LAUNCHER_DIR / "13_docprune_m3docvqa_index.sbatch").read_text(encoding="utf-8")
    assert ': "${RESUME:=0}"' in text
    assert 'case "$RESUME"' in text
    assert "--resume" in text
    assert "_load_index_manifest" in text
    assert "IndexManifest.validate_files" not in text


def test_index_surface_requires_schema_five_complete_sequence_and_raster_maps() -> None:
    text = (LAUNCHER_DIR / "13_docprune_m3docvqa_index.sbatch").read_text(encoding="utf-8")
    assert 'payload.get("schema_version") != 5' in text
    assert '"raster_indices"' in text
    assert '"page_offsets"' in text
    assert "complete unpadded ColPali" in text
    assert "schema 4" not in text


def test_gate_surface_requires_cuda_flash_and_context_reuse_checks() -> None:
    text = (LAUNCHER_DIR / "12_docprune_m3docvqa_gate.sbatch").read_text(encoding="utf-8")
    for required in (
        "torch.cuda.is_available()",
        'attn_implementation != "flash_attention_2"',
        "complete_colpali_equivalence",
        "upstream_retrieval_orders",
        "runtime_retrieval_orders",
        "colpali_counters",
        "after_qa",
        "schema5_mini_index",
        '"encoder_seconds"',
        '"decoder_seconds"',
        '"page_load_seconds"',
        '"total_sample_seconds"',
        '"retrieval_seconds"',
    ):
        assert required in text
    assert "retrieval_output=observed" in text
    assert "qtp_reencoding_count" not in text


def test_evaluation_completion_publishes_six_cell_comparison() -> None:
    compare = LAUNCHER_DIR / "15_docprune_m3docvqa_compare.sbatch"
    assert compare.is_file()
    text = compare.read_text(encoding="utf-8")
    assert "validate-run" in text
    assert "compare-runs" in text
    assert "--json-output" in text
    assert "--markdown-output" in text
    assert "test ! -e \"$COMPARISON_JSON\"" in text
    assert "test ! -e \"$COMPARISON_MARKDOWN\"" in text
    assert "all-kept-top1" in text
    assert "docprune-top4" in text


def test_launchers_validate_sealed_control_record() -> None:
    for launcher in LAUNCHERS:
        text = launcher.read_text(encoding="utf-8")
        assert "CONTROL_RECORD" in text
        assert "control.json" in text
        assert "tree_sha" in text


def test_launchers_validate_control_checkout_before_loading_helpers() -> None:
    """A mutable control checkout cannot supply code until its seal is verified."""

    for launcher in LAUNCHERS:
        text = launcher.read_text(encoding="utf-8")
        preflight = text.index('source "$PROJECT_DIR/examples/m3docvqa/pdf_tools_preflight.sh"')
        assert text.index('test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$CONTROL_COMMIT"') < preflight
        assert text.index('test -z "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=all)"') < preflight
        assert text.index('CONTROL_TREE="$(git -C "$PROJECT_DIR" rev-parse') < preflight
        assert text.index("sealed control record schema mismatch") < preflight
        assert text.index("sealed control record project path mismatch") < preflight


def test_run_config_generator_is_executable_and_documented() -> None:
    generator = ROOT / "examples" / "m3docvqa" / "make_run_configs.py"
    assert generator.is_file()
    assert generator.stat().st_mode & 0o111
    handoff = HANDOFF.read_text(encoding="utf-8")
    assert "make_run_configs.py" in handoff
    assert '--dependency="afterok:' in handoff


def test_active_benchmark_runtime_pin_is_sealed_to_approved_runtime() -> None:
    """Generators accept only the externally supplied detached runtime pin."""

    gate_config = (ROOT / "examples" / "m3docvqa" / "make_gate_config.py").read_text(
        encoding="utf-8"
    )
    run_config = (ROOT / "examples" / "m3docvqa" / "make_run_configs.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument("--runtime-commit", required=True)' in gate_config
    assert 'parser.add_argument("--runtime-commit", required=True)' in run_config
    assert 'parser.add_argument("--runtime-dir", type=Path, required=True)' in gate_config
    assert 'parser.add_argument("--runtime-dir", type=Path, required=True)' in run_config
    assert "a8d8ca6e32178a2468d729670d7219e3177a9c8b" not in gate_config
    assert "a8d8ca6e32178a2468d729670d7219e3177a9c8b" not in run_config


def test_corrected_handoff_records_diagnostic_attempt_and_schema_five_surface() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    runtime_sha = re.search(r"^runtime commit: ([0-9a-f]{40})$", text, flags=re.MULTILINE)
    assert runtime_sha is not None
    runtime7 = runtime_sha.group(1)[:7]
    assert "61830411" in text
    assert "61830405" in text and "61830410" in text
    assert "cannot be promoted" in text
    assert "schema 5" in text
    assert "complete unpadded ColPali" in text
    assert "15_docprune_m3docvqa_compare.sbatch" in text
    assert "afterok:$EVAL_JOB" in text
    assert f"/home/lmalveau/DocPrune-runtime-{runtime7}" in text
    assert f"/scratch/lmalveau/docprune/benchmark-{runtime7}/attempt-1" in text


def test_superseded_runtime_pin_and_paths_are_historical_only() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    active, historical = text.split("### Historical failures (not resumable or successful)", 1)
    assert HISTORICAL_RUNTIME_COMMIT not in active
    assert HISTORICAL_RUNTIME_DIR not in active
    assert HISTORICAL_ATTEMPT_ROOT not in active
    assert "benchmark-3755812/attempt-" in historical
    assert "none is a complete benchmark result or resumable active attempt" in historical


def test_reproduction_commands_use_literal_active_benchmark_root() -> None:
    """Active reproduction commands must target the pinned absolute artifact root."""

    text = (ROOT / "docs" / "reproduction" / "DOCPRUNE.md").read_text(encoding="utf-8")
    runtime_sha = re.search(
        r"^runtime commit: ([0-9a-f]{40})$", HANDOFF.read_text(encoding="utf-8"), flags=re.MULTILINE
    )
    assert runtime_sha is not None
    active_root = f"benchmark-{runtime_sha.group(1)[:7]}/attempt-1"
    assert active_root in text
    assert "benchmark-6c19bfc/attempt-2" not in text
    command_lines = tuple(
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("--") and active_root in line
    )
    assert command_lines
    assert all(f"/scratch/lmalveau/docprune/{active_root}/" in line for line in command_lines)
    assert all(f"/scratch/$USER/docprune/{active_root.split('/')[0]}/" not in line for line in command_lines)


def test_upstream_checkouts_require_strict_clean_status() -> None:
    for name in (
        "11_docprune_m3docvqa.sbatch",
        "12_docprune_m3docvqa_gate.sbatch",
        "13_docprune_m3docvqa_index.sbatch",
    ):
        text = (LAUNCHER_DIR / name).read_text(encoding="utf-8")
        assert (
            'test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain --untracked-files=all)"' in text
        )
        assert "__pycache__" not in text
    generator = (ROOT / "examples" / "m3docvqa" / "make_run_configs.py").read_text(encoding="utf-8")
    assert "if dirty:" in generator
    assert "non-bytecode" not in generator


def test_handoff_creates_fresh_dedicated_upstream_worktree() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "M3DOCRAG_SOURCE=/home/lmalveau/src/m3docrag-runtime-29e6ac2" in text
    assert "M3DOCRAG_DIR=/home/lmalveau/src/m3docrag-benchmark-29e6ac2" in text
    assert 'test ! -e "$M3DOCRAG_DIR"' in text
    assert 'git -C "$M3DOCRAG_SOURCE" worktree add --detach' in text
    assert "only the explicitly permitted untracked Python" not in text


def test_submission_does_not_precreate_gate_or_run_final_generator() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    submission = text.split("## Exact Slurm submission commands", 1)[1].split(
        "## Pass conditions", 1
    )[0]
    assert 'mkdir -p "$SLURM_LOG_DIR" "$INPUT_ROOT"' in submission
    assert 'mkdir -p "$SLURM_LOG_DIR" "$GATE_ROOT"' not in submission
    assert "make_run_configs.py" not in submission
    assert 'RUN_CONFIG="$INPUT_ROOT/gate-top1.json"' in submission
    assert "PROBE_IMAGE=" not in submission


def _gate_fixture(tmp_path: Path) -> tuple[Path, Path]:
    contract = {
        "schema_version": 2,
        "resources": {
            "qwen": {
                "model": "Qwen/Qwen2-VL-7B-Instruct",
                "revision": "eed13092ef92e448dd6875b2a00151bd3f7db0ac",
            },
            "colpali": {
                "model": "vidore/colpali-v1.2",
                "revision": "961b51745de3e9adb3468ac5c9ccca0ac626c217",
            },
            "colpali_backbone": {
                "model": "vidore/colpaligemma-3b-pt-448-base",
                "revision": "30ab955d073de4a91dc5a288e8c97226647e3e5a",
            },
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
        "complete_colpali_equivalence": True,
        "schema5_mini_index": {"status": "passed", "schema_version": 5, "fixture": True},
        "flash_attention_2": {
            "cuda": "12.4",
            "torch": "2.6.0",
            "transformers": "4.49.0",
            "attn_implementation": "flash_attention_2",
        },
        "upstream_retrieval_orders": {
            qid: {
                str(top_k): [{"doc_id": "doc", "page_index": index} for index in range(top_k)]
                for top_k in (1, 2, 4)
            }
            for qid in FIXED_GATE_SAMPLE_IDS
        },
        "runtime_retrieval_orders": {
            qid: {
                str(top_k): [{"doc_id": "doc", "page_index": index} for index in range(top_k)]
                for top_k in (1, 2, 4)
            }
            for qid in FIXED_GATE_SAMPLE_IDS
        },
        "colpali_counters": {
            "before_retrieval": {"processor_queries": 0, "processor_images": 20, "model_forwards": 20},
            "after_retrieval": {"processor_queries": 5, "processor_images": 20, "model_forwards": 25},
            "after_qa": {"processor_queries": 5, "processor_images": 20, "model_forwards": 25},
        },
        "timings": {
            qid: {
                field: 0.01
                for field in (
                    "retrieval_seconds",
                    "page_load_seconds",
                    "qa_seconds",
                    "total_sample_seconds",
                    "encoder_seconds",
                    "decoder_seconds",
                )
            }
            for qid in FIXED_GATE_SAMPLE_IDS
        },
        "docprune_traces": {str(page): [100, 80, 60, 40] for page in (1, 2, 4)},
    }
    semantic_path = tmp_path / "semantic-samples.json"
    # Gate production writes sort_keys=True; validator must not infer qid order
    # from the resulting supporting_documents mapping order.
    semantic_path.write_text(json.dumps(semantic, sort_keys=True), encoding="utf-8")

    gate = {
        "schema_version": 1,
        "status": "passed",
        "runtime_commit": TEST_RUNTIME_COMMIT,
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


def test_gate_evidence_accepts_sort_key_reordered_supporting_documents(tmp_path: Path) -> None:
    gate_path, contract_path = _gate_fixture(tmp_path)

    evidence = validate_gate_evidence(
        gate_path, contract_path, expected_runtime_commit=TEST_RUNTIME_COMMIT
    )

    assert evidence["sample_ids"] == list(FIXED_GATE_SAMPLE_IDS)


@pytest.mark.parametrize("kind", ["gate", "contract", "semantic"])
def test_gate_evidence_rejects_symlinked_declared_paths(tmp_path: Path, kind: str) -> None:
    gate_path, contract_path = _gate_fixture(tmp_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if kind == "gate":
        target = tmp_path / "gate-real.json"
        target.write_bytes(gate_path.read_bytes())
        gate_path.unlink()
        gate_path.symlink_to(target)
        with pytest.raises(ValueError, match="regular|symlink"):
            validate_gate_evidence(
                gate_path, contract_path, expected_runtime_commit=TEST_RUNTIME_COMMIT
            )
        return
    if kind == "contract":
        target = tmp_path / "contract-real.json"
        target.write_bytes(contract_path.read_bytes())
        contract_path.unlink()
        contract_path.symlink_to(target)
        gate["processor_contract_path"] = str(contract_path)
    else:
        semantic_path = Path(gate["semantic_samples_path"])
        target = tmp_path / "semantic-real.json"
        target.write_bytes(semantic_path.read_bytes())
        semantic_path.unlink()
        semantic_path.symlink_to(target)
        gate["semantic_samples_path"] = str(semantic_path)
    unsigned_gate = dict(gate)
    unsigned_gate.pop("gate_sha256", None)
    gate["gate_sha256"] = hashlib.sha256(
        json.dumps(unsigned_gate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="regular|symlink"):
        validate_gate_evidence(
            gate_path, contract_path, expected_runtime_commit=TEST_RUNTIME_COMMIT
        )


@pytest.mark.parametrize(
    "mutation", ["gate_sha256", "runtime_commit", "sample_ids", "semantic", "traces"]
)
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
        validate_gate_evidence(
            gate_path, contract_path, expected_runtime_commit=TEST_RUNTIME_COMMIT
        )


def test_handoff_has_absolute_log_submission_and_no_control_placeholder() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "control.json" in text
    assert "rev-parse HEAD" in text
    assert "git rev-parse HEAD:" not in text
    assert '--output="$SLURM_LOG_DIR/' in text
    assert '--error="$SLURM_LOG_DIR/' in text
    assert '--chdir="$SLURM_LOG_DIR"' in text


def test_launchers_pin_python_and_cli_when_pdf_tools_shadows_path(tmp_path: Path) -> None:
    """PDF-tools-first PATH must not change the benchmark interpreter or CLI."""

    pinned_pdf_tools = Path("/home/lmalveau/mamba-envs/m3docvqa-acquisition")
    if not all((pinned_pdf_tools / "bin" / name).is_file() for name in ("pdfinfo", "pdftoppm")):
        pytest.skip("the pinned Poppler environment is not mounted")

    shadow_bin = tmp_path / "pdf-tools" / "bin"
    env_bin = tmp_path / "docprune-sol" / "bin"
    shadow_bin.mkdir(parents=True)
    env_bin.mkdir(parents=True)
    for name in ("pdfinfo", "pdftoppm"):
        shutil.copy2(pinned_pdf_tools / "bin" / name, shadow_bin / name)
    shadow_marker = tmp_path / "shadow-python-ran"
    env_marker = tmp_path / "env-python-ran"
    (shadow_bin / "python").write_text(
        f"#!/usr/bin/env bash\nprintf shadow > {shadow_marker}\nexit 91\n",
        encoding="utf-8",
    )
    (env_bin / "python").write_text(
        f"#!/usr/bin/env bash\nprintf env > {env_marker}\n",
        encoding="utf-8",
    )
    (env_bin / "docprune-m3docvqa").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (shadow_bin / "python").chmod(0o755)
    (env_bin / "python").chmod(0o755)
    (env_bin / "docprune-m3docvqa").chmod(0o755)

    preflight = ROOT / "examples" / "m3docvqa" / "pdf_tools_preflight.sh"
    script = f"""
set -euo pipefail
export PDFTOOLS_DIR={shadow_bin.parent}
export ENV_DIR={env_bin.parent}
export LD_LIBRARY_PATH={pinned_pdf_tools}/lib
source {preflight}
test "$(command -v python)" = "$PDFTOOLS_DIR/bin/python"
set +e
python -c ignored
bare_rc=$?
set -e
test "$bare_rc" -eq 91
test -e {shadow_marker}
rm -f {shadow_marker}
"$ENV_DIR/bin/python" -c ignored
test ! -e {shadow_marker}
test -e {env_marker}
"""
    # The copied pinned tools let the preflight complete while the shadow
    # python models the real PDFTOOLS_DIR-first PATH collision.
    result = subprocess.run(["bash", "-c", script], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert not shadow_marker.exists()
    assert env_marker.exists()

    for launcher in LAUNCHERS:
        text = launcher.read_text(encoding="utf-8")
        assert not re.search(
            r"(?m)^\s*(?:[A-Z_][A-Z0-9_]*=\S+\s+)*(?:python|python3|pytest|docprune-m3docvqa)(?:\s|$)",
            text,
        )
        path_pos = text.find('export PATH="$PDFTOOLS_DIR/bin:$ENV_DIR/bin:$PATH"')
        if path_pos < 0:
            # The array wrapper delegates all Python/CLI work to launcher 11.
            assert launcher.name == "14_docprune_m3docvqa_eval_array.sbatch"
            continue
        post_path = text[path_pos:]
        assert not re.search(r"(?m)^\s*python(?:\s|$)", post_path)
        assert not re.search(r"(?m)^\s*docprune-m3docvqa(?:\s|$)", post_path)
        if launcher.name != "14_docprune_m3docvqa_eval_array.sbatch":
            assert '"$ENV_DIR/bin/python"' in post_path
            assert '"$ENV_DIR/bin/docprune-m3docvqa"' in post_path

    handoff = HANDOFF.read_text(encoding="utf-8")
    assert not re.search(
        r"(?m)^\s*(?:[A-Z_][A-Z0-9_]*=\S+\s+)*(?:python|python3|pytest|docprune-m3docvqa)(?:\s|$)",
        handoff,
    )
