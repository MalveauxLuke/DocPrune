"""Tests for the fixed explicit-visual-state removal runtime identity."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from docprune import task7_runtime
from docprune.m3docrag import SampleInput
from docprune.m3docvqa_factory import _validate_result_record
from docprune.task7_runtime import task7_intervention_matrix

ROOT = Path(__file__).resolve().parents[1]


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


def test_fixed_page_launcher_accepts_only_the_eight_cell_task7_grid() -> None:
    """Catch a CPU-only runner integration that the actual L40S launcher rejects."""

    launcher = (ROOT / "examples/sbatch/35_docprune_task6_l40s_matrix.sbatch").read_text(
        encoding="utf-8"
    )

    assert "native|native-extension|fixed|visual-state-fixed-grid" in launcher
    assert 'if [[ "$MATRIX_KIND" == visual-state-fixed-grid ]]; then EXPECTED=8; fi' in launcher
    assert "#SBATCH --constraint=l40s" in launcher
    assert "#SBATCH --no-requeue" in launcher
    assert "global index" not in launcher.lower()
    assert "retrieve" not in launcher.lower()
