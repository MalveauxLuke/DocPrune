"""Tests for the fixed explicit-visual-state removal runtime identity."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from docprune import task7_runtime
from docprune.m3docrag import SampleInput
from docprune.m3docvqa_factory import _validate_result_record
from docprune.task6_runtime import (
    TASK6_RENDERER_CONTRACT,
    FixedPageFixture,
    FixedPageQuestion,
    FixedPageRecord,
)
from docprune.task7_runtime import task7_intervention_matrix

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _production_all_drop_record() -> dict[str, object]:
    return {
        "question_id": "q-1",
        "question": "Which value is shown?",
        "answers": ["42"],
        "predicted_answer": "42",
        "retrieved_pages": [
            {"doc_id": f"doc-{index}", "page_index": index, "score": 1.0 - index / 10}
            for index in range(4)
        ],
        "trace": {
            "original_visual_tokens": 16,
            "post_btp_visual_tokens": 12,
            "post_qtp_visual_tokens": 8,
            "post_ctp_visual_tokens": 0,
            "ctp_layer": None,
        },
        "timing": {
            "retrieval_seconds": 0.1,
            "page_load_seconds": 0.1,
            "qa_seconds": 0.2,
            "total_sample_seconds": 0.4,
            "encoder_seconds": 0.1,
            "decoder_seconds": 0.1,
            "profiler_enabled": False,
        },
        "forced_intervention": {
            "boundary": "B_input",
            "mode": "physical_delete",
            "selection_kind": "forced",
            "visual_population": 8,
            "requested_budget": 0,
            "achieved_budget": 0,
            "retained_visual_ids": [],
            "logical_retained_sequence_ids": [0, 1, 10, 11],
            "prefill_cache_lengths": [4] * 28,
            "retained_mrope_position_shape": [3, 1, 4],
            "retained_mrope_position_sha256": "a" * 64,
        },
    }


def test_task7_matrix_pairs_no_ctp_reference_with_exact_all_drop_boundaries() -> None:
    """Catch pruning a partial population, the wrong layer, or the reference arm."""

    cells = task7_intervention_matrix()

    assert [cell.name for cell in cells] == [
        "btp-qtp-no-ctp",
        "all-visual-drop-B_input",
        "all-visual-drop-B_0",
        "all-visual-drop-B_6",
        "all-visual-drop-B_13",
        "all-visual-drop-B_20",
        "all-visual-drop-B_23",
        "all-visual-drop-B_26",
    ]
    assert cells[0].ctp_policy is not None
    assert cells[0].ctp_policy.name == "btp-qtp-no-ctp"
    assert cells[0].forced_intervention is None
    assert [cell.ctp_policy for cell in cells[1:]] == [None] * 7
    assert [cell.forced_intervention.boundary for cell in cells[1:]] == [
        "input",
        0,
        6,
        13,
        20,
        23,
        26,
    ]
    assert all(cell.forced_intervention.mode == "physical_delete" for cell in cells[1:])
    assert all(cell.forced_intervention.retained_visual_ids == () for cell in cells[1:])
    assert cells[0].to_dict() == {
        "name": "btp-qtp-no-ctp",
        "analysis_family": "reference",
        "ctp_policy": cells[0].ctp_policy.to_dict(),
        "forced_intervention": None,
    }
    assert cells[1].to_dict() == {
        "name": "all-visual-drop-B_input",
        "analysis_family": "fixed-grid",
        "ctp_policy": None,
        "forced_intervention": {
            "boundary": "B_input",
            "mode": "physical_delete",
            "retained_visual_ids": [],
        },
    }
    assert cells[0].answerer_kwargs() == {
        "ctp_policy": cells[0].ctp_policy,
        "forced_intervention": None,
    }
    assert cells[1].answerer_kwargs() == {
        "ctp_policy": None,
        "forced_intervention": cells[1].forced_intervention,
    }


def test_task7_result_identity_cross_binds_reference_and_all_drop_records() -> None:
    """Catch relabeling a partial deletion or a CTP arm as fixed-grid all-drop."""

    cells = task7_intervention_matrix()
    common = {
        "fixed_page_fixture_sha256": "f" * 64,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "matrix_kind": "visual-state-fixed-grid",
    }
    reference = {
        **common,
        "matrix_cell": 0,
        "intervention_name": "btp-qtp-no-ctp",
        "trace": {
            "post_qtp_visual_tokens": 4,
            "post_ctp_visual_tokens": 4,
            "ctp_layer": None,
        },
        "policy_selection": {
            "policy": {
                "name": "btp-qtp-no-ctp",
                "family": "no-ctp",
                "selection_kind": "none",
            }
        },
    }
    all_drop = {
        **common,
        "matrix_cell": 1,
        "intervention_name": "all-visual-drop-B_input",
        "trace": {
            "post_qtp_visual_tokens": 4,
            "post_ctp_visual_tokens": 0,
            "ctp_layer": None,
        },
        "forced_intervention": {
            "boundary": "B_input",
            "mode": "physical_delete",
            "selection_kind": "forced",
            "visual_population": 4,
            "requested_budget": 0,
            "achieved_budget": 0,
            "retained_visual_ids": [],
            "logical_retained_sequence_ids": [0, 1, 6, 7],
            "prefill_cache_lengths": [4] * 28,
            "retained_mrope_position_shape": [3, 1, 4],
            "retained_mrope_position_sha256": "a" * 64,
        },
    }

    task7_runtime.validate_task7_result_record(reference, cells[0], fixture_sha256="f" * 64)
    task7_runtime.validate_task7_result_record(all_drop, cells[1], fixture_sha256="f" * 64)

    partial = deepcopy(all_drop)
    partial["forced_intervention"]["requested_budget"] = 1
    partial["forced_intervention"]["achieved_budget"] = 1
    partial["forced_intervention"]["retained_visual_ids"] = [0]
    partial["trace"]["post_ctp_visual_tokens"] = 1
    with pytest.raises(ValueError, match="all-drop identity"):
        task7_runtime.validate_task7_result_record(partial, cells[1], fixture_sha256="f" * 64)


@pytest.mark.parametrize(
    ("boundary", "cache_lengths"),
    [
        ("B_input", [4] * 28),
        ("B_0", [12] + [4] * 27),
        ("B_6", [12] * 7 + [4] * 21),
        ("B_26", [12] * 27 + [4]),
    ],
)
def test_task7_all_drop_requires_exact_physical_cache_topology(
    boundary: str, cache_lengths: list[int]
) -> None:
    """Catch relabeled or impossible cache vectors at each named deletion boundary."""

    record = _production_all_drop_record()
    cells = task7_intervention_matrix()
    cell = next(
        candidate
        for candidate in cells
        if candidate.forced_intervention is not None
        and (
            "B_input"
            if candidate.forced_intervention.boundary == "input"
            else f"B_{candidate.forced_intervention.boundary}"
        )
        == boundary
    )
    cell_index = cells.index(cell)
    record["forced_intervention"]["boundary"] = boundary
    record["forced_intervention"]["prefill_cache_lengths"] = cache_lengths
    task7_runtime.bind_task7_result_evidence(
        record,
        cell,
        cell_index=cell_index,
        fixture_sha256="f" * 64,
    )

    task7_runtime.validate_task7_result_record(record, cell, fixture_sha256="f" * 64)

    for invalid in (
        cache_lengths[:-1],
        [cache_lengths[0] + 1, *cache_lengths[1:]],
        [4] * 28 if boundary != "B_input" else [12] + [4] * 27,
    ):
        mutated = deepcopy(record)
        mutated["forced_intervention"]["prefill_cache_lengths"] = invalid
        with pytest.raises(ValueError, match="physical all-drop evidence"):
            task7_runtime.validate_task7_result_record(mutated, cell, fixture_sha256="f" * 64)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("logical_retained_sequence_ids", []),
        ("logical_retained_sequence_ids", [0, 2, 1, 11]),
        ("logical_retained_sequence_ids", [0, 1, 1, 11]),
        ("logical_retained_sequence_ids", [0, 1, 10, 12]),
        ("retained_mrope_position_shape", [3, 1, 0]),
        ("retained_mrope_position_shape", [1, 3, 4]),
    ],
)
def test_task7_all_drop_rejects_impossible_logical_or_mrope_evidence(
    field: str, invalid: list[int]
) -> None:
    """Catch empty/reordered text survivors or an M-RoPE length not bound to them."""

    record = _production_all_drop_record()
    cell = task7_intervention_matrix()[1]
    task7_runtime.bind_task7_result_evidence(
        record,
        cell,
        cell_index=1,
        fixture_sha256="f" * 64,
    )
    record["forced_intervention"][field] = invalid

    with pytest.raises(ValueError, match="physical all-drop evidence"):
        task7_runtime.validate_task7_result_record(record, cell, fixture_sha256="f" * 64)


def test_task7_result_evidence_is_manifest_derived_and_cannot_be_replaced() -> None:
    """Catch a runner overwriting fixed-page or intervention identity with result data."""

    cell = task7_intervention_matrix()[1]
    record: dict[str, object] = {}

    task7_runtime.bind_task7_result_evidence(
        record,
        cell,
        cell_index=1,
        fixture_sha256="f" * 64,
    )

    assert record == {
        "fixed_page_fixture_sha256": "f" * 64,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "matrix_kind": "visual-state-fixed-grid",
        "matrix_cell": 1,
        "intervention_name": "all-visual-drop-B_input",
    }
    with pytest.raises(ValueError, match="replace immutable evidence"):
        task7_runtime.bind_task7_result_evidence(
            {"matrix_cell": 2},
            cell,
            cell_index=1,
            fixture_sha256="f" * 64,
        )


def test_task7_all_drop_passes_generic_and_task7_validators_only_when_explicitly_allowed() -> None:
    """Catch the generic validator rejecting Task 7's intentional zero final budget."""

    record = _production_all_drop_record()
    cell = task7_intervention_matrix()[1]
    task7_runtime.bind_task7_result_evidence(
        record,
        cell,
        cell_index=1,
        fixture_sha256="f" * 64,
    )

    _validate_result_record(
        record,
        line_number=1,
        expected_page_count=4,
        production=True,
        forbid_policy=True,
        allowed_extra_fields={"matrix_cell", "matrix_kind", "intervention_name"},
        allow_zero_post_ctp=True,
    )
    task7_runtime.validate_task7_result_record(record, cell, fixture_sha256="f" * 64)

    with pytest.raises(ValueError, match="non-positive trace counts"):
        _validate_result_record(
            record,
            line_number=1,
            allowed_extra_fields={"matrix_cell", "matrix_kind", "intervention_name"},
        )

    invalid_early_zero = deepcopy(record)
    invalid_early_zero["trace"]["post_qtp_visual_tokens"] = 0
    with pytest.raises(ValueError, match="non-positive trace counts"):
        _validate_result_record(
            invalid_early_zero,
            line_number=1,
            allow_zero_post_ctp=True,
            allowed_extra_fields={"matrix_cell", "matrix_kind", "intervention_name"},
        )


def test_task7_source_identity_binds_question_answers_and_ordered_fixture_pages() -> None:
    """Catch relabeling cached outputs or changing any sealed retrieved page."""

    record = _production_all_drop_record()
    sample = SampleInput("q-1", "Which value is shown?", ("42",))
    fixture_question = SimpleNamespace(
        qid="q-1",
        pages=tuple(
            SimpleNamespace(
                doc_id=f"doc-{index}",
                page_index=index,
                score=1.0 - index / 10,
            )
            for index in range(4)
        ),
    )

    task7_runtime.validate_task7_source_identity(record, sample, fixture_question)

    for field, value in (
        ("question", "A different question"),
        ("answers", ["not 42"]),
        (
            "retrieved_pages",
            list(reversed(record["retrieved_pages"])),
        ),
    ):
        mutated = deepcopy(record)
        mutated[field] = value
        with pytest.raises(ValueError, match="source identity"):
            task7_runtime.validate_task7_source_identity(mutated, sample, fixture_question)


def test_task7_prefill_identity_binding_is_immutable() -> None:
    """Catch likelihood evidence being attached to a different prompt or tokenized input."""

    record = _production_all_drop_record()
    task7_runtime.bind_task7_prefill_identity(
        record,
        assistant_prompt_sha256="1" * 64,
        prefill_input_ids_shape=(1, 23),
        prefill_input_ids_sha256="2" * 64,
    )

    assert record["assistant_prompt_sha256"] == "1" * 64
    assert record["prefill_input_ids_shape"] == [1, 23]
    assert record["prefill_input_ids_sha256"] == "2" * 64
    with pytest.raises(ValueError, match="replace immutable prefill identity"):
        task7_runtime.bind_task7_prefill_identity(
            record,
            assistant_prompt_sha256="3" * 64,
            prefill_input_ids_shape=(1, 23),
            prefill_input_ids_sha256="2" * 64,
        )


def test_fixed_page_matrix_runner_embeds_task7_cells_without_a_task6_policy_gate() -> None:
    """Catch making Task 7 depend on nonexistent Task 6 policy-matrix fields."""

    path = ROOT / "examples" / "run_task6_matrix.py"
    spec = importlib.util.spec_from_file_location("task7_fixed_page_matrix_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cells = module._expected_cells({}, "visual-state-fixed-grid")

    assert cells == [
        {"cell": index, **cell.to_dict()} for index, cell in enumerate(task7_intervention_matrix())
    ]

    record = _production_all_drop_record()
    capture = SimpleNamespace(
        assistant_prompt_sha256="1" * 64,
        prefill_input_ids_shape=(1, 23),
        prefill_input_ids_sha256="2" * 64,
    )
    with pytest.raises(ValueError, match="target prompt differs"):
        module._bind_task7_likelihood_capture(
            record,
            capture,
            {"assistant_prompt_sha256": "3" * 64},
        )


def test_fixed_page_launcher_accepts_only_the_eight_cell_task7_grid() -> None:
    """Catch a CPU-only runner integration that the actual L40S launcher rejects."""

    launcher = (ROOT / "examples/sbatch/35_docprune_task6_l40s_matrix.sbatch").read_text(
        encoding="utf-8"
    )

    assert "native|native-extension|fixed|visual-state-fixed-grid" in launcher
    assert 'if [[ "$MATRIX_KIND" == visual-state-fixed-grid ]]; then EXPECTED=8; fi' in launcher
    assert "#SBATCH --constraint=l40s" in launcher
    assert "#SBATCH --no-requeue" in launcher
    assert "admit_task7_likelihood_pair_from_files" in launcher
    assert "global index" not in launcher.lower()
    assert "retrieve" not in launcher.lower()

    runner = (ROOT / "examples/run_task6_matrix.py").read_text(encoding="utf-8")
    assert runner.count("admit_task7_likelihood_pair_from_files(") >= 2


def test_task7_likelihood_record_uses_max_normalized_full_answer_adaptation() -> None:
    """Catch first-token scoring, EOS scoring, or averaging alternative gold items."""

    target = task7_runtime.build_task7_likelihood_target(
        ("Tommy", "Zhang Ling"),
        ((24732, 2408), (57, 20658, 50858)),
        assistant_prompt="<assistant>\n",
    )
    record = task7_runtime.build_task7_likelihood_record(
        qid="q-1",
        intervention_name="btp-qtp-no-ctp",
        target=target,
        per_reference_mean_loglikelihood=(-1.25, -0.5),
        fixture_sha256="a" * 64,
        run_manifest_sha256="b" * 64,
        source_result_sha256="c" * 64,
        prefill_input_ids_shape=(1, 17),
        prefill_input_ids_sha256="d" * 64,
    )

    assert target["target_name"] == "best-reference-full-gold-sequence"
    assert target["normalization"] == "mean-log-probability-per-supplied-answer-token"
    assert target["aggregation"] == "maximum-over-official-answer-items"
    assert target["eos_included"] is False
    assert target["tokenization"] == (
        "exact-assistant-prompt-prefix-suffix|tokenizer-encode|add-special-tokens-false|no-eos"
    )
    assert target["assistant_prompt_sha256"] == hashlib.sha256(b"<assistant>\n").hexdigest()
    assert target["answer_item_semantics"] == (
        "local-max-reference-adaptation-not-official-m3docvqa-multispan-scoring"
    )
    assert record["best_reference_index"] == 1
    assert record["best_reference_mean_loglikelihood"] == -0.5
    assert record["source_result_sha256"] == "c" * 64
    assert record["assistant_prompt_sha256"] == target["assistant_prompt_sha256"]
    assert record["prefill_input_ids_shape"] == [1, 17]
    assert record["prefill_input_ids_sha256"] == "d" * 64
    assert len(target["likelihood_target_sha256"]) == 64
    assert len(record["likelihood_record_sha256"]) == 64


def test_task7_gold_tokenization_preserves_items_without_special_tokens() -> None:
    """Catch standalone tokenization or chat/BOS/EOS tokens entering the gold target."""

    class RecordingTokenizer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            self.calls.append((text, add_special_tokens))
            return {
                "<assistant>\n": [151644, 77091, 198],
                "<assistant>\nTommy": [151644, 77091, 198, 24732, 2408],
                "<assistant>\nZhang Ling": [151644, 77091, 198, 57, 20658, 50858],
            }[text]

    tokenizer = RecordingTokenizer()

    target = task7_runtime.prepare_task7_likelihood_target(
        tokenizer,
        ("Tommy", "Zhang Ling"),
        assistant_prompt="<assistant>\n",
    )

    assert target["target_token_ids"] == [[24732, 2408], [57, 20658, 50858]]
    assert target["assistant_prompt_sha256"] == hashlib.sha256(b"<assistant>\n").hexdigest()
    assert tokenizer.calls == [
        ("<assistant>\n", False),
        ("<assistant>\nTommy", False),
        ("<assistant>\nZhang Ling", False),
    ]


def test_task7_gold_tokenization_rejects_a_retokenized_prompt_boundary() -> None:
    """Catch slicing a combined sequence whose answer changed the prompt tokenization."""

    class BoundaryMergingTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            del add_special_tokens
            return [1, 2] if text == "prompt" else [1, 9]

    with pytest.raises(ValueError, match="continuation prefix"):
        task7_runtime.tokenize_task7_gold_answers(
            BoundaryMergingTokenizer(),
            ("answer",),
            assistant_prompt="prompt",
        )


def _task7_artifact_fixture(tmp_path: Path) -> tuple[Path, str, FixedPageFixture]:
    reference = tmp_path / "reference.json"
    reference.write_text('{"selection_is_outcome_blind":true}\n', encoding="utf-8")
    eligible = tmp_path / "eligible.jsonl"
    eligible.write_text(
        json.dumps(
            {
                "qid": "q-1",
                "question": "Which value is shown?",
                "answers": [{"answer": "42"}, {"answer": "forty two"}],
                "supporting_context": [{"doc_id": "doc-1"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    feature_manifest = tmp_path / "features.json"
    feature_manifest.write_text("{}\n", encoding="utf-8")
    ledger = tmp_path / "ledger.json"
    ledger.write_text("[]\n", encoding="utf-8")
    pdf = tmp_path / "pages.pdf"
    pdf.write_bytes(b"pdf")
    feature = tmp_path / "pages.safetensors"
    feature.write_bytes(b"features")
    pages = tuple(
        FixedPageRecord(
            rank=index,
            doc_id=f"doc-{index}",
            page_index=index,
            score=1.0 - index / 10,
            source_pdf_path=pdf.resolve(),
            source_pdf_sha256=_sha256(pdf),
            rendered_rgb_width=2,
            rendered_rgb_height=2,
            rendered_rgb_sha256="a" * 64,
            renderer_contract=TASK6_RENDERER_CONTRACT,
            feature_shard_path=feature.resolve(),
            feature_shard_sha256=_sha256(feature),
            feature_page_index=index,
        )
        for index in range(4)
    )
    fixture = FixedPageFixture(
        fixture_version="task7-test-v1",
        reference_path=reference.resolve(),
        reference_sha256=_sha256(reference),
        eligible_questions_path=eligible.resolve(),
        eligible_questions_sha256=_sha256(eligible),
        feature_manifest_path=feature_manifest.resolve(),
        feature_manifest_sha256=_sha256(feature_manifest),
        completion_ledger_path=ledger.resolve(),
        completion_ledger_sha256=_sha256(ledger),
        questions=(
            FixedPageQuestion(
                "q-1",
                hashlib.sha256(b"Which value is shown?").hexdigest(),
                pages,
            ),
        ),
    )
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps(fixture.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return fixture_path, _sha256(fixture_path), fixture


def _task7_result_rows(fixture_sha256: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cells = task7_intervention_matrix()
    for index, cell in enumerate(cells):
        record = _production_all_drop_record()
        record["answers"] = ["42", "forty two"]
        if index == 0:
            record.pop("forced_intervention")
            record["trace"]["post_ctp_visual_tokens"] = 8
            from docprune.ctp_policy import no_crossing_selection

            record["policy_selection"] = no_crossing_selection(cell.ctp_policy, 8).to_dict()
        else:
            boundary = cell.forced_intervention.boundary
            boundary_name = "B_input" if boundary == "input" else f"B_{boundary}"
            record["forced_intervention"]["boundary"] = boundary_name
            compact, full = 4, 12
            record["forced_intervention"]["prefill_cache_lengths"] = (
                [compact] * 28
                if boundary == "input"
                else [full] * (boundary + 1) + [compact] * (27 - boundary)
            )
        task7_runtime.bind_task7_result_evidence(
            record,
            cell,
            cell_index=index,
            fixture_sha256=fixture_sha256,
        )
        if index < 2:
            task7_runtime.bind_task7_prefill_identity(
                record,
                assistant_prompt_sha256=hashlib.sha256(b"<assistant>\n").hexdigest(),
                prefill_input_ids_shape=(1, 17),
                prefill_input_ids_sha256="d" * 64,
            )
        rows.append(record)
    return rows


def _task7_run_manifest(
    tmp_path: Path, fixture_path: Path, fixture_sha256: str
) -> tuple[Path, str, str]:
    path = tmp_path / "run_manifest.json"
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "configured-task7-fixed-grid",
        "output": str(tmp_path),
        "matrix_kind": "visual-state-fixed-grid",
        "qid": "q-1",
        "cell_count": 8,
        "fixture_path": str(fixture_path),
        "fixture_sha256": fixture_sha256,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "task7_likelihood_output": str(tmp_path / "likelihood.jsonl"),
        "task7_likelihood_arms": ["btp-qtp-no-ctp", "all-visual-drop-B_input"],
    }
    payload["run_manifest_sha256"] = _canonical_sha256(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, _sha256(path), payload["run_manifest_sha256"]


def test_task7_opportunity_assembly_authenticates_sources_results_and_likelihoods(
    tmp_path: Path,
) -> None:
    """Catch trusted precomputed strata, support-ID drift, or an unbound likelihood drop."""

    fixture_path, fixture_sha256, _ = _task7_artifact_fixture(tmp_path)
    run_manifest_path, run_manifest_file_sha256, run_manifest_sha256 = _task7_run_manifest(
        tmp_path, fixture_path, fixture_sha256
    )
    results_path = tmp_path / "results.jsonl"
    results = _task7_result_rows(fixture_sha256)
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    target = task7_runtime.build_task7_likelihood_target(
        ("42", "forty two"),
        ((19, 17), (69, 70)),
        assistant_prompt="<assistant>\n",
    )
    likelihood_path = tmp_path / "likelihood.jsonl"
    likelihood_rows = [
        task7_runtime.build_task7_likelihood_record(
            qid="q-1",
            intervention_name="btp-qtp-no-ctp",
            target=target,
            per_reference_mean_loglikelihood=(-0.2, -0.7),
            fixture_sha256=fixture_sha256,
            run_manifest_sha256=run_manifest_sha256,
            source_result_sha256=_canonical_sha256(results[0]),
            prefill_input_ids_shape=(1, 17),
            prefill_input_ids_sha256="d" * 64,
        ),
        task7_runtime.build_task7_likelihood_record(
            qid="q-1",
            intervention_name="all-visual-drop-B_input",
            target=target,
            per_reference_mean_loglikelihood=(-0.5, -0.9),
            fixture_sha256=fixture_sha256,
            run_manifest_sha256=run_manifest_sha256,
            source_result_sha256=_canonical_sha256(results[1]),
            prefill_input_ids_shape=(1, 17),
            prefill_input_ids_sha256="d" * 64,
        ),
    ]
    likelihood_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in likelihood_rows),
        encoding="utf-8",
    )

    assembled = task7_runtime.assemble_task7_opportunity_row_from_artifacts(
        fixture_path=fixture_path,
        fixture_sha256=fixture_sha256,
        run_manifest_path=run_manifest_path,
        run_manifest_file_sha256=run_manifest_file_sha256,
        results_path=results_path,
        results_sha256=_sha256(results_path),
        likelihood_path=likelihood_path,
        likelihood_sha256=_sha256(likelihood_path),
    )
    analysis_bundle = task7_runtime.assemble_task7_analysis_bundle_from_artifacts(
        fixture_path=fixture_path,
        fixture_sha256=fixture_sha256,
        run_manifest_path=run_manifest_path,
        run_manifest_file_sha256=run_manifest_file_sha256,
        results_path=results_path,
        results_sha256=_sha256(results_path),
        likelihood_path=likelihood_path,
        likelihood_sha256=_sha256(likelihood_path),
    )

    assert assembled["qid"] == "q-1"
    assert analysis_bundle["qid"] == "q-1"
    assert analysis_bundle["curve"]["row"] == {
        "qid": "q-1",
        "reference_f1": 50.0,
        "all_drop_f1": {
            "B_input": 50.0,
            "B_0": 50.0,
            "B_6": 50.0,
            "B_13": 50.0,
            "B_20": 50.0,
            "B_23": 50.0,
            "B_26": 50.0,
        },
    }
    assert analysis_bundle["opportunity"]["row"] == assembled
    assert analysis_bundle["curve"]["provenance"] == analysis_bundle["opportunity"]["provenance"]
    assert analysis_bundle["curve"]["provenance"] == {
        "fixture_sha256": fixture_sha256,
        "run_manifest_file_sha256": run_manifest_file_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "results_file_sha256": _sha256(results_path),
        "likelihood_file_sha256": _sha256(likelihood_path),
        "reference_result_sha256": _canonical_sha256(results[0]),
        "input_all_drop_result_sha256": _canonical_sha256(results[1]),
    }
    bundle_without_hash = dict(analysis_bundle)
    assert bundle_without_hash.pop("analysis_bundle_sha256") == _canonical_sha256(
        bundle_without_hash
    )
    assert assembled["supporting_document_ids"] == ["doc-1"]
    assert assembled["retrieved_document_ids"] == ["doc-0", "doc-1", "doc-2", "doc-3"]
    assert assembled["reference"]["em_correct"] is False
    assert assembled["reference"]["f1"] == 50.0
    assert assembled["input_all_drop"]["best_reference_loglikelihood_drop_per_token"] == (
        pytest.approx(0.3)
    )
    assert (
        assembled["input_all_drop"]["likelihood_target_sha256"]
        == target["likelihood_target_sha256"]
    )

    results[2]["assistant_prompt_sha256"] = hashlib.sha256(b"<assistant>\n").hexdigest()
    results[2]["prefill_input_ids_shape"] = [1, 17]
    results[2]["prefill_input_ids_sha256"] = "d" * 64
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown fields"):
        task7_runtime.assemble_task7_opportunity_row_from_artifacts(
            fixture_path=fixture_path,
            fixture_sha256=fixture_sha256,
            run_manifest_path=run_manifest_path,
            run_manifest_file_sha256=run_manifest_file_sha256,
            results_path=results_path,
            results_sha256=_sha256(results_path),
            likelihood_path=likelihood_path,
            likelihood_sha256=_sha256(likelihood_path),
        )
    for key in (
        "assistant_prompt_sha256",
        "prefill_input_ids_shape",
        "prefill_input_ids_sha256",
    ):
        results[2].pop(key)
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results), encoding="utf-8"
    )

    substituted_likelihood = tmp_path / "substituted-likelihood.jsonl"
    substituted_likelihood.write_bytes(likelihood_path.read_bytes())
    with pytest.raises(ValueError, match="run manifest does not bind"):
        task7_runtime.assemble_task7_opportunity_row_from_artifacts(
            fixture_path=fixture_path,
            fixture_sha256=fixture_sha256,
            run_manifest_path=run_manifest_path,
            run_manifest_file_sha256=run_manifest_file_sha256,
            results_path=results_path,
            results_sha256=_sha256(results_path),
            likelihood_path=substituted_likelihood,
            likelihood_sha256=_sha256(substituted_likelihood),
        )

    bad_manifest_path = tmp_path / "bad-run-manifest.json"
    bad_manifest = json.loads(run_manifest_path.read_text())
    bad_manifest["task7_likelihood_arms"] = [
        "all-visual-drop-B_input",
        "btp-qtp-no-ctp",
    ]
    bad_manifest.pop("run_manifest_sha256")
    bad_manifest["run_manifest_sha256"] = _canonical_sha256(bad_manifest)
    bad_manifest_path.write_text(
        json.dumps(bad_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="run manifest does not bind"):
        task7_runtime.assemble_task7_opportunity_row_from_artifacts(
            fixture_path=fixture_path,
            fixture_sha256=fixture_sha256,
            run_manifest_path=bad_manifest_path,
            run_manifest_file_sha256=_sha256(bad_manifest_path),
            results_path=results_path,
            results_sha256=_sha256(results_path),
            likelihood_path=likelihood_path,
            likelihood_sha256=_sha256(likelihood_path),
        )

    likelihood_path.write_text(
        likelihood_path.read_text().replace("-0.5", "-0.4"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="likelihood artifact checksum"):
        task7_runtime.assemble_task7_opportunity_row_from_artifacts(
            fixture_path=fixture_path,
            fixture_sha256=fixture_sha256,
            run_manifest_path=run_manifest_path,
            run_manifest_file_sha256=run_manifest_file_sha256,
            results_path=results_path,
            results_sha256=_sha256(results_path),
            likelihood_path=likelihood_path,
            likelihood_sha256=likelihood_rows[0]["likelihood_record_sha256"],
        )


def test_task7_opportunity_assembly_rejects_wrong_prompt_and_stale_result(
    tmp_path: Path,
) -> None:
    """Catch a self-hashed likelihood being relabeled onto a different prompt or QA row."""

    fixture_path, fixture_sha256, _ = _task7_artifact_fixture(tmp_path)
    run_path, run_file_sha, run_sha = _task7_run_manifest(tmp_path, fixture_path, fixture_sha256)
    results = _task7_result_rows(fixture_sha256)
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results), encoding="utf-8"
    )

    def write_likelihood(prompt: str, source_results: list[dict[str, object]]) -> Path:
        target = task7_runtime.build_task7_likelihood_target(
            ("42", "forty two"), ((19, 17), (69, 70)), assistant_prompt=prompt
        )
        path = tmp_path / "likelihood.jsonl"
        rows = [
            task7_runtime.build_task7_likelihood_record(
                qid="q-1",
                intervention_name=name,
                target=target,
                per_reference_mean_loglikelihood=values,
                fixture_sha256=fixture_sha256,
                run_manifest_sha256=run_sha,
                source_result_sha256=_canonical_sha256(source_results[index]),
                prefill_input_ids_shape=(1, 17),
                prefill_input_ids_sha256="d" * 64,
            )
            for index, (name, values) in enumerate(
                (
                    ("btp-qtp-no-ctp", (-0.2, -0.7)),
                    ("all-visual-drop-B_input", (-0.5, -0.9)),
                )
            )
        ]
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    wrong_prompt = write_likelihood("WRONG PROMPT", results)
    with pytest.raises(ValueError, match="prompt or prefill identity"):
        task7_runtime.assemble_task7_opportunity_row_from_artifacts(
            fixture_path=fixture_path,
            fixture_sha256=fixture_sha256,
            run_manifest_path=run_path,
            run_manifest_file_sha256=run_file_sha,
            results_path=results_path,
            results_sha256=_sha256(results_path),
            likelihood_path=wrong_prompt,
            likelihood_sha256=_sha256(wrong_prompt),
        )

    valid_likelihood = write_likelihood("<assistant>\n", results)
    results[1]["predicted_answer"] = "stale result mutation"
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source result identity"):
        task7_runtime.assemble_task7_opportunity_row_from_artifacts(
            fixture_path=fixture_path,
            fixture_sha256=fixture_sha256,
            run_manifest_path=run_path,
            run_manifest_file_sha256=run_file_sha,
            results_path=results_path,
            results_sha256=_sha256(results_path),
            likelihood_path=valid_likelihood,
            likelihood_sha256=_sha256(valid_likelihood),
        )


def test_task7_likelihood_artifact_is_atomic_ordered_and_no_replace(tmp_path: Path) -> None:
    """Catch exposing a partial pair, swapping arms, or overwriting admitted evidence."""

    target = task7_runtime.build_task7_likelihood_target(
        ("42",), ((19,),), assistant_prompt="<assistant>\n"
    )
    rows = tuple(
        task7_runtime.build_task7_likelihood_record(
            qid="q-1",
            intervention_name=name,
            target=target,
            per_reference_mean_loglikelihood=(value,),
            fixture_sha256="a" * 64,
            run_manifest_sha256="b" * 64,
            source_result_sha256=digest * 64,
            prefill_input_ids_shape=(1, 17),
            prefill_input_ids_sha256="d" * 64,
        )
        for name, value, digest in (
            ("btp-qtp-no-ctp", -0.2, "1"),
            ("all-visual-drop-B_input", -0.5, "2"),
        )
    )
    path = (tmp_path / "likelihood.jsonl").resolve()

    digest = task7_runtime.write_task7_likelihood_artifact(path, rows)

    assert digest == _sha256(path)
    assert tuple(json.loads(line) for line in path.read_text().splitlines()) == rows
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        task7_runtime.write_task7_likelihood_artifact(path, rows)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="reference and B_input in order"):
        task7_runtime.write_task7_likelihood_artifact(
            (tmp_path / "swapped.jsonl").resolve(), tuple(reversed(rows))
        )


def test_task7_likelihood_artifact_rejects_a_symlinked_ancestor(tmp_path: Path) -> None:
    """Catch output publication being redirected through an ancestor symlink."""

    real_parent = tmp_path / "real" / "nested"
    real_parent.mkdir(parents=True)
    (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
    target = task7_runtime.build_task7_likelihood_target(
        ("42",), ((19,),), assistant_prompt="<assistant>\n"
    )
    rows = tuple(
        task7_runtime.build_task7_likelihood_record(
            qid="q-1",
            intervention_name=name,
            target=target,
            per_reference_mean_loglikelihood=(value,),
            fixture_sha256="a" * 64,
            run_manifest_sha256="b" * 64,
            source_result_sha256=digest * 64,
            prefill_input_ids_shape=(1, 17),
            prefill_input_ids_sha256="d" * 64,
        )
        for name, value, digest in (
            ("btp-qtp-no-ctp", -0.2, "1"),
            ("all-visual-drop-B_input", -0.5, "2"),
        )
    )

    with pytest.raises(ValueError, match="symlink|non-directory"):
        task7_runtime.write_task7_likelihood_artifact(
            tmp_path / "alias" / "nested" / "likelihood.jsonl", rows
        )
    assert not (real_parent / "likelihood.jsonl").exists()


def test_task7_authenticated_parse_uses_the_exact_hashed_bytes(tmp_path: Path) -> None:
    """Catch authenticating one inode and then parsing replacement path bytes."""

    path = (tmp_path / "rows.jsonl").resolve()
    path.write_text('{"value":"authenticated"}\n', encoding="utf-8")
    authenticated = task7_runtime._authenticated_file(path, _sha256(path), "test rows")

    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text('{"value":"replacement"}\n', encoding="utf-8")
    replacement.replace(path)

    assert task7_runtime._jsonl_mappings_bytes(authenticated, "test rows") == (
        {"value": "authenticated"},
    )


def _write_task7_likelihood_pair(
    path: Path,
    *,
    fixture_sha256: str,
    run_manifest_sha256: str,
    results: list[dict[str, object]],
) -> None:
    target = task7_runtime.build_task7_likelihood_target(
        ("42", "forty two"),
        ((19, 17), (69, 70)),
        assistant_prompt="<assistant>\n",
    )
    rows = tuple(
        task7_runtime.build_task7_likelihood_record(
            qid="q-1",
            intervention_name=name,
            target=target,
            per_reference_mean_loglikelihood=values,
            fixture_sha256=fixture_sha256,
            run_manifest_sha256=run_manifest_sha256,
            source_result_sha256=_canonical_sha256(results[index]),
            prefill_input_ids_shape=(1, 17),
            prefill_input_ids_sha256="d" * 64,
        )
        for index, (name, values) in enumerate(
            (
                ("btp-qtp-no-ctp", (-0.2, -0.7)),
                ("all-visual-drop-B_input", (-0.5, -0.9)),
            )
        )
    )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_task7_likelihood_pair_admission_binds_manifest_and_first_two_results(
    tmp_path: Path,
) -> None:
    """Catch a final/resumed run exiting successfully with unbound likelihood evidence."""

    fixture_path, fixture_sha256, _ = _task7_artifact_fixture(tmp_path)
    run_path, _, run_sha = _task7_run_manifest(tmp_path, fixture_path, fixture_sha256)
    results = _task7_result_rows(fixture_sha256)
    results_path = (tmp_path / "results.jsonl").resolve()
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results), encoding="utf-8"
    )
    likelihood_path = (tmp_path / "likelihood.jsonl").resolve()
    _write_task7_likelihood_pair(
        likelihood_path,
        fixture_sha256=fixture_sha256,
        run_manifest_sha256=run_sha,
        results=results,
    )

    digest = task7_runtime.admit_task7_likelihood_pair_from_files(
        run_manifest_path=run_path,
        results_path=results_path,
        likelihood_path=likelihood_path,
    )

    assert digest == _sha256(likelihood_path)


@pytest.mark.parametrize("attack", ["stale", "swapped", "malformed"])
def test_task7_likelihood_pair_admission_rejects_invalid_evidence(
    tmp_path: Path, attack: str
) -> None:
    """Catch stale, swapped, or malformed likelihood evidence on resume/postflight."""

    fixture_path, fixture_sha256, _ = _task7_artifact_fixture(tmp_path)
    run_path, _, run_sha = _task7_run_manifest(tmp_path, fixture_path, fixture_sha256)
    results = _task7_result_rows(fixture_sha256)
    results_path = (tmp_path / "results.jsonl").resolve()
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results), encoding="utf-8"
    )
    likelihood_path = (tmp_path / "likelihood.jsonl").resolve()
    _write_task7_likelihood_pair(
        likelihood_path,
        fixture_sha256=fixture_sha256,
        run_manifest_sha256=run_sha,
        results=results,
    )
    if attack == "stale":
        results[1]["predicted_answer"] = "stale mutation"
        results_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in results),
            encoding="utf-8",
        )
    elif attack == "swapped":
        lines = likelihood_path.read_text(encoding="utf-8").splitlines(keepends=True)
        likelihood_path.write_text("".join(reversed(lines)), encoding="utf-8")
    else:
        likelihood_path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source result|in order|valid JSONL"):
        task7_runtime.admit_task7_likelihood_pair_from_files(
            run_manifest_path=run_path,
            results_path=results_path,
            likelihood_path=likelihood_path,
        )


def test_task7_likelihood_pair_admission_rejects_a_symlinked_input_ancestor(
    tmp_path: Path,
) -> None:
    """Catch postflight admission following a redirected ancestor component."""

    real = tmp_path / "real" / "nested"
    real.mkdir(parents=True)
    run_path = real / "run_manifest.json"
    run_path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|non-directory"):
        task7_runtime.admit_task7_likelihood_pair_from_files(
            run_manifest_path=tmp_path / "alias" / "nested" / "run_manifest.json",
            results_path=real / "results.jsonl",
            likelihood_path=real / "likelihood.jsonl",
        )
