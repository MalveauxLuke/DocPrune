"""Regression checks for the executable M3DocVQA benchmark handoff."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
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

from docprune.benchmark_seal import validate_attempt_root  # noqa: E402

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

    ruff = Path("/home/lmalveau/mamba-envs/docprune-sol/bin/ruff")
    assert ruff.is_file(), "the pinned environment Ruff is required to lint launcher heredocs"
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


def test_gate_semantic_wrapper_is_safe_without_ambient_pythonpath(tmp_path: Path) -> None:
    """The semantic wrapper must run when the caller did not set PYTHONPATH."""

    text = (LAUNCHER_DIR / "12_docprune_m3docvqa_gate.sbatch").read_text(encoding="utf-8")
    start = text.index(
        'PYTHONPATH="$PROJECT_DIR/examples/m3docvqa:$RUNTIME_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \\\n'
        '  "$ENV_DIR/bin/python" - "$RUN_CONFIG"'
    )
    end = text.index("\nPY\n", start) + len("\nPY")
    wrapper = text[start:end].replace(
        '"$ENV_DIR/bin/python"', shlex.quote(str(tmp_path / "env" / "bin" / "python")), 1
    )
    fake_python = tmp_path / "env" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/usr/bin/env bash\ncat >/dev/null\n", encoding="utf-8")
    fake_python.chmod(0o755)
    values = {
        "PROJECT_DIR": tmp_path,
        "RUNTIME_DIR": tmp_path / "runtime",
        "ENV_DIR": tmp_path / "env",
        "RUN_CONFIG": tmp_path / "run-config.json",
        "GATE_SAMPLE_IDS": "sample",
        "GATE_ROOT": tmp_path / "gate",
    }
    assignments = "\n".join(
        f"{name}={shlex.quote(str(value))}" for name, value in values.items()
    )
    result = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{assignments}\nunset PYTHONPATH\n{wrapper}\n"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


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


def test_nonmeasurement_gate_and_index_surfaces_use_unconstrained_htc_a100() -> None:
    gate = (LAUNCHER_DIR / "12_docprune_m3docvqa_gate.sbatch").read_text(encoding="utf-8")
    index = (LAUNCHER_DIR / "13_docprune_m3docvqa_index.sbatch").read_text(encoding="utf-8")
    for text, walltime in ((gate, "02:00:00"), (index, "04:00:00")):
        assert "#SBATCH --partition=htc" in text
        assert "#SBATCH --qos=public" not in text
        assert "#SBATCH --gres=gpu:a100:1" in text
        assert "#SBATCH --constraint=a100_80" not in text
        assert "#SBATCH --cpus-per-task=8" in text
        assert "#SBATCH --mem=128G" in text
        assert f"#SBATCH --time={walltime}" in text

    for name in ("11_docprune_m3docvqa.sbatch", "14_docprune_m3docvqa_eval_array.sbatch"):
        text = (LAUNCHER_DIR / name).read_text(encoding="utf-8")
        assert "#SBATCH --partition=public" in text
        assert "#SBATCH --gres=gpu:a100:1" in text
        assert "#SBATCH --constraint=a100_80" in text
        assert "#SBATCH --cpus-per-task=8" in text
        assert "#SBATCH --mem=128G" in text
        assert "#SBATCH --time=24:00:00" in text

    comparator = (LAUNCHER_DIR / "15_docprune_m3docvqa_compare.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=htc" in comparator
    assert "#SBATCH --cpus-per-task=8" in comparator
    assert "#SBATCH --mem=128G" in comparator


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


def test_handoff_active_attempt_root_runs_through_runtime_validator() -> None:
    """The active handoff path must be accepted by the actual runtime guard."""

    handoff = HANDOFF.read_text(encoding="utf-8")
    matches = re.findall(
        r"^export ATTEMPT_ROOT=(/scratch/lmalveau/docprune/benchmark-[0-9a-f]{7,40}/attempt-[1-9][0-9]*)$",
        handoff,
        flags=re.MULTILINE,
    )
    assert matches
    active_root = matches[0]
    assert validate_attempt_root(active_root, active_root) == active_root


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
    assert runtime_sha.group(1) == "384b330c72ce49ee2272d1602748307c973da37b"
    assert f"/scratch/lmalveau/docprune/benchmark-{runtime7}/attempt-1" in text
    assert "61943239" in text
    assert "61943240" in text and "61943247" in text


def test_handoff_hybrid_scheduling_overrides_preserve_hardware_contract() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    submission = text.split("## Exact Slurm submission commands", 1)[1].split(
        "## Pass conditions", 1
    )[0]
    assert "--partition=htc" in submission
    assert "--time=02:00:00" in submission
    assert "--time=04:00:00" in submission
    assert "--partition=public" in submission
    assert "--time=24:00:00" in submission
    assert "--time=01:00:00" in submission
    assert submission.count("--gres=gpu:a100:1") == 3
    for launcher in ("12_docprune_m3docvqa_gate.sbatch", "13_docprune_m3docvqa_index.sbatch"):
        assert launcher in submission
    assert "--constraint=a100_80" in text
    assert "--cpus-per-task=8" in text
    assert "--mem=128G" in text
    assert "/scratch/lmalveau/docprune/benchmark-384b330/attempt-1" in text
    assert "61943239" in text and "61943247" in text


def test_active_authority_docs_have_no_stale_attempt_one_root() -> None:
    authorities = (
        ROOT / "agent-context" / "CURRENT_TASK.md",
        ROOT / "sol" / "CURRENT_SOL_TASK.md",
        ROOT / "docs" / "reproduction" / "DOCPRUNE.md",
        ROOT / "docs" / "reproduction" / "RECONSTRUCTION_GAPS.md",
    )
    for path in authorities:
        text = path.read_text(encoding="utf-8")
        assert "benchmark-384b330/attempt-1" in text
    handoff = HANDOFF.read_text(encoding="utf-8")
    active = handoff.split("### Historical failures (not resumable or successful)", 1)[0]
    assert "benchmark-384b330/attempt-1" in active
    submission = handoff.split("## Exact Slurm submission commands", 1)[1]
    submission_code = submission.split("```bash", 1)[1].split("```", 1)[0]
    assert "benchmark-02385b3/attempt-2" not in submission_code
    assert "benchmark-02385b3/attempt-2" in handoff.split(
        "### Historical failures (not resumable or successful)", 1
    )[1]
    assert "/scratch/lmalveau/docprune/benchmark-6c19bfc/attempt-2" in handoff


def test_handoff_submission_graph_executes_against_fake_sbatch(tmp_path: Path) -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    submission = text.split("## Exact Slurm submission commands", 1)[1].split(
        "## Pass conditions", 1
    )[0]
    snippet = submission[submission.index("GATE_JOB=") : submission.index("printf 'gate=")]
    log = tmp_path / "sbatch.log"
    export_lines = []
    in_exports = False
    for line in submission.splitlines():
        if line.startswith("export PROJECT_DIR="):
            in_exports = True
        if in_exports and line.startswith("export "):
            export_lines.append(line)
        if in_exports and line.startswith("source "):
            break
    active_root = "/scratch/lmalveau/docprune/benchmark-384b330/attempt-1"
    assert f"export ATTEMPT_ROOT={active_root}" in export_lines
    assert 'export EXPECTED_ATTEMPT_ROOT="$ATTEMPT_ROOT"' in export_lines
    assert 'export CONTROL_RECORD="$ATTEMPT_ROOT/control.json"' in export_lines
    assert 'export GATE_ROOT="$ATTEMPT_ROOT/gate"' in export_lines
    assert 'export INPUT_ROOT="$ATTEMPT_ROOT/inputs"' in export_lines
    assert 'export RUN_CONFIG_ROOT="$ATTEMPT_ROOT/run-configs"' in export_lines
    assert 'export COMPARISON_JSON="$ATTEMPT_ROOT/comparison/six-cell.json"' in export_lines
    assert 'export COMPARISON_MARKDOWN="$ATTEMPT_ROOT/comparison/six-cell.md"' in export_lines
    control_record = tmp_path / "control.json"
    control_record.write_text(
        json.dumps({"control_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}),
        encoding="utf-8",
    )
    export_lines = [
        line.replace('$ATTEMPT_ROOT/control.json', str(control_record))
        for line in export_lines
    ]

    def run_variant(exports: list[str], launch: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        log.unlink(missing_ok=True)
        script = f'''set -e
{chr(10).join(exports)}
SBATCH_LOG={log}
ID_FILE={tmp_path / "next-id"}
printf '1\\n' > "$ID_FILE"
sbatch() {{ n=$(cat "$ID_FILE"); printf '%s\\n' "$((n + 1))" > "$ID_FILE"; printf 'MODE=%s PAGES=%s RUN_CONFIG=%s INDEX_ROOT=%s ATTEMPT_ROOT=%s EXPECTED_ATTEMPT_ROOT=%s CONTROL_RECORD=%s GATE_ROOT=%s INPUT_ROOT=%s RUN_CONFIG_ROOT=%s ALL_KEPT_INDEX_ROOT=%s DOCPRUNE_INDEX_ROOT=%s COMPARISON_JSON=%s COMPARISON_MARKDOWN=%s CONTROL_COMMIT=%s ARGS=%s\\n' "$MODE" "$PAGES" "$RUN_CONFIG" "$INDEX_ROOT" "$ATTEMPT_ROOT" "$EXPECTED_ATTEMPT_ROOT" "$CONTROL_RECORD" "$GATE_ROOT" "$INPUT_ROOT" "$RUN_CONFIG_ROOT" "$ALL_KEPT_INDEX_ROOT" "$DOCPRUNE_INDEX_ROOT" "$COMPARISON_JSON" "$COMPARISON_MARKDOWN" "$CONTROL_COMMIT" "$*" >> "$SBATCH_LOG"; printf 'job%s\\n' "$n"; }}
{launch}
'''
        result = subprocess.run(["bash", "-c", script], check=False, capture_output=True, text=True)
        return result, log.read_text(encoding="utf-8").splitlines() if log.exists() else []

    result, calls = run_variant(export_lines, snippet)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 9
    current_control = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    def call_fields(call: str) -> dict[str, str]:
        metadata = call.split(" ARGS=", 1)[0]
        return dict(item.split("=", 1) for item in shlex.split(metadata) if "=" in item)

    def option_multimap(args: list[str]) -> dict[str, list[str | None]]:
        options: dict[str, list[str | None]] = {}
        for arg in args:
            if not arg.startswith("--"):
                continue
            key, separator, value = arg.partition("=")
            options.setdefault(key, []).append(value if separator else None)
        return options

    expected_path_fields = {
        "ATTEMPT_ROOT": active_root,
        "EXPECTED_ATTEMPT_ROOT": active_root,
        "CONTROL_RECORD": str(control_record),
        "GATE_ROOT": f"{active_root}/gate",
        "INPUT_ROOT": f"{active_root}/inputs",
        "RUN_CONFIG_ROOT": f"{active_root}/run-configs",
        "COMPARISON_JSON": f"{active_root}/comparison/six-cell.json",
        "COMPARISON_MARKDOWN": f"{active_root}/comparison/six-cell.md",
    }
    expected_index_root_fields = {
        "ALL_KEPT_INDEX_ROOT": f"{active_root}/indexes/all-kept",
        "DOCPRUNE_INDEX_ROOT": f"{active_root}/indexes/docprune",
    }
    resource_keys = {
        "--partition",
        "--time",
        "--gres",
        "--constraint",
        "--cpus-per-task",
        "--mem",
    }
    allowed_option_keys = {
        "--parsable",
        *resource_keys,
        "--dependency",
        "--chdir",
        "--output",
        "--error",
        "--export",
    }
    gpu_resources = {
        "--gres": "gpu:a100:1",
        "--cpus-per-task": "8",
        "--mem": "128G",
    }
    eval_gpu_resources = {**gpu_resources, "--constraint": "a100_80"}
    expected_resources = [
        {"--partition": "htc", "--time": "02:00:00", **gpu_resources},
        *[
            {"--partition": "htc", "--time": "04:00:00", **gpu_resources}
            for _ in range(6)
        ],
        {"--partition": "public", "--time": "24:00:00", **eval_gpu_resources},
        {"--partition": "htc", "--time": "01:00:00", "--cpus-per-task": "8", "--mem": "128G"},
    ]
    expected_dependencies = [
        None,
        *["afterok:job1" for _ in range(6)],
        "afterok:job1:job2:job3:job4:job5:job6:job7",
        "afterok:job8",
    ]
    expected_call_paths = [
        {"RUN_CONFIG": f"{active_root}/inputs/gate-top1.json"},
        *[
            {
                "RUN_CONFIG": f"{active_root}/run-configs/{mode}-top{page}.json",
                "INDEX_ROOT": f"{active_root}/indexes/{mode}/top{page}",
            }
            for mode in ("all-kept", "docprune")
            for page in (1, 2, 4)
        ],
    ]

    def validate_graph(actual_calls: list[str]) -> None:
        assert len(actual_calls) == 9
        actual_args = [shlex.split(call.split("ARGS=", 1)[1]) for call in actual_calls]
        for index, (call, args) in enumerate(zip(actual_calls, actual_args)):
            fields = call_fields(call)
            assert fields["CONTROL_COMMIT"] == current_control
            assert {key: fields[key] for key in expected_path_fields} == expected_path_fields
            if index >= 7:
                assert {
                    key: fields[key] for key in expected_index_root_fields
                } == expected_index_root_fields
            options = option_multimap(args)
            assert set(options) <= allowed_option_keys
            assert options.get("--export") == ["ALL"]
            expected_dependency = expected_dependencies[index]
            assert options.get("--dependency", []) == (
                [] if expected_dependency is None else [expected_dependency]
            )
            expected_resource_map = expected_resources[index]
            assert {key for key in options if key in resource_keys} == set(expected_resource_map)
            for key in resource_keys:
                assert options.get(key, []) == (
                    [expected_resource_map[key]] if key in expected_resource_map else []
                )
            if index < len(expected_call_paths):
                for key, value in expected_call_paths[index].items():
                    assert fields[key] == value

        index_tuples = {
            (
                call_fields(call)["MODE"],
                call_fields(call)["PAGES"],
                call_fields(call)["RUN_CONFIG"],
                call_fields(call)["INDEX_ROOT"],
            )
            for call in actual_calls[1:7]
        }
        assert index_tuples == {
            (mode, str(page), f"{active_root}/run-configs/{mode}-top{page}.json", f"{active_root}/indexes/{mode}/top{page}")
            for mode in ("all-kept", "docprune")
            for page in (1, 2, 4)
        }

    validate_graph(calls)

    wrong_root = [line.replace(active_root, active_root.replace("attempt-1", "attempt-9")) for line in export_lines]
    _, wrong_root_calls = run_variant(wrong_root, snippet)
    with pytest.raises(AssertionError):
        validate_graph(wrong_root_calls)
    extra_dependency = snippet.replace(
        '--dependency="afterok:$EVAL_JOB"',
        '--dependency="afterok:$EVAL_JOB" --dependency=afterok:extra',
    )
    _, extra_dependency_calls = run_variant(export_lines, extra_dependency)
    with pytest.raises(AssertionError):
        validate_graph(extra_dependency_calls)
    conflicting_export = snippet.replace("--export=ALL", "--export=ALL --export=FOO=bar", 1)
    _, conflicting_export_calls = run_variant(export_lines, conflicting_export)
    with pytest.raises(AssertionError):
        validate_graph(conflicting_export_calls)
    duplicate_cell = snippet.replace("for PAGES in 1 2 4", "for PAGES in 1 1 4")
    _, duplicate_cell_calls = run_variant(export_lines, duplicate_cell)
    with pytest.raises(AssertionError):
        validate_graph(duplicate_cell_calls)
    duplicate_partition = snippet.replace(
        "--partition=htc --time=02:00:00",
        "--partition=htc --partition=htc --time=02:00:00",
        1,
    )
    _, duplicate_partition_calls = run_variant(export_lines, duplicate_partition)
    with pytest.raises(AssertionError):
        validate_graph(duplicate_partition_calls)
    conflicting_partition = snippet.replace(
        "--partition=htc --time=02:00:00",
        "--partition=htc --partition=public --time=02:00:00",
        1,
    )
    _, conflicting_partition_calls = run_variant(export_lines, conflicting_partition)
    with pytest.raises(AssertionError):
        validate_graph(conflicting_partition_calls)
    duplicate_mem = snippet.replace(
        "--cpus-per-task=8 --mem=128G",
        "--cpus-per-task=8 --mem=128G --mem=128G",
        1,
    )
    _, duplicate_mem_calls = run_variant(export_lines, duplicate_mem)
    with pytest.raises(AssertionError):
        validate_graph(duplicate_mem_calls)
    conflicting_gres = snippet.replace(
        "--gres=gpu:a100:1",
        "--gres=gpu:a100:1 --gres=gpu:a100:2",
        1,
    )
    _, conflicting_gres_calls = run_variant(export_lines, conflicting_gres)
    with pytest.raises(AssertionError):
        validate_graph(conflicting_gres_calls)
    wrong_expected_attempt_root_exports = [
        line.replace(
            'export EXPECTED_ATTEMPT_ROOT="$ATTEMPT_ROOT"',
            f"export EXPECTED_ATTEMPT_ROOT={active_root}/wrong-expected",
        )
        for line in export_lines
    ]
    _, wrong_expected_attempt_root_calls = run_variant(
        wrong_expected_attempt_root_exports, snippet
    )
    with pytest.raises(AssertionError):
        validate_graph(wrong_expected_attempt_root_calls)
    wrong_gate_root_exports = [
        line.replace(
            'export GATE_ROOT="$ATTEMPT_ROOT/gate"',
            f"export GATE_ROOT={active_root}/wrong-gate",
        )
        for line in export_lines
    ]
    _, wrong_gate_root_calls = run_variant(wrong_gate_root_exports, snippet)
    with pytest.raises(AssertionError):
        validate_graph(wrong_gate_root_calls)


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
    assert "benchmark-6c19bfc/attempt-2" in text
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


def _upstream_prepare_script(source: Path, destination: Path, commit: str) -> str:
    text = HANDOFF.read_text(encoding="utf-8")
    start = text.index("export M3DOCRAG_SOURCE=")
    block = text[start : text.index("```", start)]
    runtime_checks = (
        'if [[ ! -e "$RUNTIME_DIR" ]]; then\n'
        '  git -C "$PROJECT_DIR" worktree add --detach "$RUNTIME_DIR" "$EXPECTED_COMMIT"\n'
        'fi\n'
        'test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$EXPECTED_COMMIT"\n'
        'test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"\n'
    )
    block = block.replace(runtime_checks, "").removesuffix(")\n")
    return (
        "set -e\n"
        + block.replace("/home/lmalveau/src/m3docrag-runtime-29e6ac2", str(source))
        .replace("/home/lmalveau/src/m3docrag-benchmark-29e6ac2", str(destination))
        .replace("29e6ac2294d6b87075a1d45b8a8df175b214248a", commit)
    )


def _complete_prepare_script(source: Path, runtime: Path, destination: Path, commit: str) -> str:
    text = HANDOFF.read_text(encoding="utf-8")
    heading = text.index("## Prepare pinned runtime and upstream checkouts")
    start = text.index("(\nset -e\n", heading)
    end = text.index("```", start)
    return (
        text[start:end]
        .replace("/home/lmalveau/DocPrune-runtime-384b330", str(runtime))
        .replace("384b330c72ce49ee2272d1602748307c973da37b", commit)
        .replace("/home/lmalveau/src/m3docrag-runtime-29e6ac2", str(source))
        .replace("/home/lmalveau/src/m3docrag-benchmark-29e6ac2", str(destination))
        .replace("29e6ac2294d6b87075a1d45b8a8df175b214248a", commit)
    )


def _upstream_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    (source / "README").write_text("commit-a\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "README"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "--quiet", "-m", "commit-a"], check=True)
    commit_a = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    (source / "README").write_text("commit-b\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "README"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "--quiet", "-m", "commit-b"], check=True)
    commit_b = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return source, destination, commit_a, commit_b


def test_handoff_upstream_prep_creates_absent_destination(tmp_path: Path) -> None:
    source, destination, _, commit = _upstream_fixture(tmp_path)
    result = subprocess.run(["bash", "-c", _upstream_prepare_script(source, destination, commit)], check=False)
    assert result.returncode == 0
    assert destination.is_dir()
    assert not subprocess.run(
        ["git", "-C", str(destination), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_handoff_upstream_prep_reuses_exact_clean_destination(tmp_path: Path) -> None:
    source, destination, _, commit = _upstream_fixture(tmp_path)
    subprocess.run(["git", "-C", str(source), "worktree", "add", "--detach", str(destination), commit], check=True)
    before = (destination / "README").read_text(encoding="utf-8")
    result = subprocess.run(["bash", "-c", _upstream_prepare_script(source, destination, commit)], check=False)
    assert result.returncode == 0
    assert (destination / "README").read_text(encoding="utf-8") == before


def test_handoff_upstream_prep_rejects_wrong_sha_without_mutation(tmp_path: Path) -> None:
    source, destination, commit_a, commit_b = _upstream_fixture(tmp_path)
    subprocess.run(["git", "-C", str(source), "worktree", "add", "--detach", str(destination), commit_a], check=True)
    before = (destination / "README").read_text(encoding="utf-8")
    result = subprocess.run(["bash", "-c", _upstream_prepare_script(source, destination, commit_b)], check=False)
    assert result.returncode != 0
    assert (destination / "README").read_text(encoding="utf-8") == before


def test_handoff_upstream_prep_rejects_dirty_destination_without_mutation(tmp_path: Path) -> None:
    source, destination, _, commit = _upstream_fixture(tmp_path)
    subprocess.run(["git", "-C", str(source), "worktree", "add", "--detach", str(destination), commit], check=True)
    marker = destination / "untracked.txt"
    marker.write_text("preserve\n", encoding="utf-8")
    result = subprocess.run(["bash", "-c", _upstream_prepare_script(source, destination, commit)], check=False)
    assert result.returncode != 0
    assert marker.read_text(encoding="utf-8") == "preserve\n"


def test_handoff_complete_subshell_preserves_caller_state(tmp_path: Path) -> None:
    source, destination, _, commit = _upstream_fixture(tmp_path)
    runtime = tmp_path / "runtime"
    subprocess.run(["git", "-C", str(source), "worktree", "add", "--detach", str(runtime), commit], check=True)
    script = _complete_prepare_script(source, runtime, destination, commit)
    probe = (
        "set +e\n"
        "CALLER_SENTINEL=preserved\n"
        "RUNTIME_DIR=caller-runtime\n"
        "EXPECTED_COMMIT=caller-expected\n"
        "M3DOCRAG_SOURCE=caller-source\n"
        "M3DOCRAG_DIR=caller-dir\n"
        "M3DOCRAG_COMMIT=caller-commit\n"
        + script
    )
    probe += (
        "rc=$?\n"
        "printf 'sentinel=%s runtime=%s expected=%s source=%s dir=%s commit=%s errexit=%s rc=%s\\n' "
        "\"$CALLER_SENTINEL\" \"$RUNTIME_DIR\" \"$EXPECTED_COMMIT\" "
        "\"$M3DOCRAG_SOURCE\" \"$M3DOCRAG_DIR\" \"$M3DOCRAG_COMMIT\" \"$-\" \"$rc\"\n"
        "exit \"$rc\"\n"
    )
    result = subprocess.run(["bash", "-c", probe], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "sentinel=preserved" in result.stdout
    assert "runtime=caller-runtime" in result.stdout
    assert "expected=caller-expected" in result.stdout
    assert "source=caller-source" in result.stdout
    assert "dir=caller-dir" in result.stdout
    assert "commit=caller-commit" in result.stdout
    assert "errexit=" in result.stdout
    assert "e" not in result.stdout.split("errexit=", 1)[1].split()[0]


def test_handoff_documents_pinned_upstream_checkout_contract() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "M3DOCRAG_SOURCE=/home/lmalveau/src/m3docrag-runtime-29e6ac2" in text
    assert "M3DOCRAG_DIR=/home/lmalveau/src/m3docrag-benchmark-29e6ac2" in text
    assert 'if [[ ! -e "$M3DOCRAG_DIR" ]]; then' in text
    assert 'git -C "$M3DOCRAG_SOURCE" worktree add --detach' in text
    assert 'test "$(git -C "$M3DOCRAG_DIR" rev-parse HEAD)" = "$M3DOCRAG_COMMIT"' in text
    assert 'test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain --untracked-files=all)"' in text
    assert "only the explicitly permitted untracked Python" not in text


def test_handoff_secret_scan_is_precise_and_excludes_normal_token_names() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    scan_line = next(line for line in text.splitlines() if "git ls-files | rg -i" in line)
    assert "token|secret|password|credential" in scan_line
    assert ".*(token|secret|password|credential)" not in scan_line
    pattern = r"(^|/)(\.env(\..*)?|token|secret|password|credential)(\.[^/]*)?$"
    tracked = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True).stdout
    result = subprocess.run(["rg", "-i", pattern], input=tracked, check=False, capture_output=True, text=True)
    assert result.returncode == 1
    representative = "x/.env\nx/token.txt\nx/secret\nx/password.json\nx/credential\n"
    result = subprocess.run(["rg", "-i", pattern], input=representative, check=False, capture_output=True, text=True)
    assert result.returncode == 0
    assert set(result.stdout.splitlines()) == set(representative.splitlines())


def test_handoff_clean_smoke_uses_real_git_checkout() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert 'git clone --quiet --no-local "$PROJECT_DIR" "$SMOKE_ROOT"' in text
    assert 'git archive HEAD | tar -x -C "$SMOKE_ROOT"' not in text
    assert 'git -C "$SMOKE_ROOT" status --porcelain --untracked-files=all' in text


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
        "ctp_policy": {
            "head_aggregation": "arithmetic_mean",
            "score_scale": "current_visual_token_count",
            "raw_attention_semantics": "mean_head_attention_scores_before_visual_count_scaling",
            "transformed_attention_semantics": "raw_mean_times_current_visual_count",
            "prefill_query_token": "last_prompt_token",
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
