"""Tests for the fixed explicit-visual-state removal runtime identity."""

from __future__ import annotations

from docprune.task7_runtime import task7_intervention_matrix


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
