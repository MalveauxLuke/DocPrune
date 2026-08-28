"""Fail-closed runtime identities for the visual-state removal curve."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from docprune.ctp_policy import CTPPolicy, btp_qtp_no_ctp_policy
from docprune.m3docrag import SampleInput
from docprune.qwen2vl.decoder import ForcedVisualIntervention
from docprune.task6_runtime import FixedPageQuestion


@dataclass(frozen=True, slots=True)
class Task7InterventionCell:
    """One mutually exclusive reference or fixed-boundary all-drop arm."""

    name: str
    ctp_policy: CTPPolicy | None
    forced_intervention: ForcedVisualIntervention | None

    def answerer_kwargs(self) -> dict[str, object]:
        """Return the mutually exclusive arguments consumed by the Qwen answerer."""

        return {
            "ctp_policy": self.ctp_policy,
            "forced_intervention": self.forced_intervention,
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize the mutually exclusive fixed-grid arm identity."""

        forced = self.forced_intervention
        return {
            "name": self.name,
            "analysis_family": "reference" if forced is None else "fixed-grid",
            "ctp_policy": None if self.ctp_policy is None else self.ctp_policy.to_dict(),
            "forced_intervention": (
                None
                if forced is None
                else {
                    "boundary": (
                        "B_input" if forced.boundary == "input" else f"B_{forced.boundary}"
                    ),
                    "mode": forced.mode,
                    "retained_visual_ids": list(forced.retained_visual_ids),
                }
            ),
        }


def task7_intervention_matrix() -> tuple[Task7InterventionCell, ...]:
    """Return the fixed reference/all-drop matrix."""

    boundaries: tuple[str | int, ...] = ("input", 0, 6, 13, 20, 23, 26)
    return (
        Task7InterventionCell("btp-qtp-no-ctp", btp_qtp_no_ctp_policy(), None),
        *(
            Task7InterventionCell(
                f"all-visual-drop-B_{boundary}",
                None,
                ForcedVisualIntervention(boundary, "physical_delete", ()),
            )
            for boundary in boundaries
        ),
    )


def bind_task7_result_evidence(
    record: dict[str, object],
    cell: Task7InterventionCell,
    *,
    cell_index: int,
    fixture_sha256: str,
) -> None:
    """Attach only manifest-derived Task 7 evidence before validation and write."""

    matrix = task7_intervention_matrix()
    if (
        type(cell_index) is not int
        or not 0 <= cell_index < len(matrix)
        or matrix[cell_index] != cell
    ):
        raise ValueError("Task 7 result uses an invalid matrix cell")
    if len(fixture_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in fixture_sha256
    ):
        raise ValueError("Task 7 fixture checksum must be a lowercase SHA-256")
    evidence = {
        "fixed_page_fixture_sha256": fixture_sha256,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "matrix_kind": "visual-state-fixed-grid",
        "matrix_cell": cell_index,
        "intervention_name": cell.name,
    }
    for key, value in evidence.items():
        if key in record and record[key] != value:
            raise ValueError(f"Task 7 result attempts to replace immutable evidence: {key}")
        record[key] = value


def validate_task7_source_identity(
    record: Mapping[str, object],
    sample: SampleInput,
    fixture_question: FixedPageQuestion,
) -> None:
    """Bind a Task 7 result to the sealed sample and ordered fixed-page tuple."""

    expected_pages = [
        {
            "doc_id": page.doc_id,
            "page_index": page.page_index,
            "score": float(page.score),
        }
        for page in fixture_question.pages
    ]
    if (
        sample.question_id != fixture_question.qid
        or record.get("question_id") != sample.question_id
        or record.get("question") != sample.question
        or record.get("answers") != list(sample.answers)
        or record.get("retrieved_pages") != expected_pages
    ):
        raise ValueError("Task 7 result does not match the sealed source identity")


def validate_task7_result_record(
    record: Mapping[str, object],
    cell: Task7InterventionCell,
    *,
    fixture_sha256: str,
) -> None:
    """Cross-bind one generic validated result to its fixed-grid Task 7 arm."""

    if (
        len(fixture_sha256) != 64
        or any(character not in "0123456789abcdef" for character in fixture_sha256)
        or record.get("fixed_page_fixture_sha256") != fixture_sha256
        or record.get("fixed_page_provenance") is not True
        or record.get("global_index_loaded") is not False
        or record.get("matrix_kind") != "visual-state-fixed-grid"
    ):
        raise ValueError("Task 7 result has invalid fixed-page identity")
    matrix = task7_intervention_matrix()
    try:
        cell_index = matrix.index(cell)
    except ValueError as error:
        raise ValueError("Task 7 result uses an unknown intervention cell") from error
    if record.get("matrix_cell") != cell_index or record.get("intervention_name") != cell.name:
        raise ValueError("Task 7 result has invalid matrix-cell identity")
    trace = record.get("trace")
    if not isinstance(trace, Mapping):
        raise ValueError("Task 7 result has invalid pruning trace")
    population = trace.get("post_qtp_visual_tokens")
    retained = trace.get("post_ctp_visual_tokens")
    if (
        not isinstance(population, int)
        or isinstance(population, bool)
        or population < 0
        or not isinstance(retained, int)
        or isinstance(retained, bool)
        or retained < 0
        or trace.get("ctp_layer") is not None
    ):
        raise ValueError("Task 7 result has invalid pruning trace")

    if cell.forced_intervention is None:
        selection = record.get("policy_selection")
        policy = selection.get("policy") if isinstance(selection, Mapping) else None
        if (
            "forced_intervention" in record
            or not isinstance(policy, Mapping)
            or policy.get("name") != "btp-qtp-no-ctp"
            or policy.get("family") != "no-ctp"
            or policy.get("selection_kind") != "none"
            or retained != population
        ):
            raise ValueError("Task 7 result has invalid reference identity")
        return

    forced = record.get("forced_intervention")
    expected_boundary = (
        "B_input"
        if cell.forced_intervention.boundary == "input"
        else f"B_{cell.forced_intervention.boundary}"
    )
    if (
        "policy_selection" in record
        or not isinstance(forced, Mapping)
        or forced.get("boundary") != expected_boundary
        or forced.get("mode") != "physical_delete"
        or forced.get("selection_kind") != "forced"
        or forced.get("visual_population") != population
        or forced.get("requested_budget") != 0
        or forced.get("achieved_budget") != 0
        or forced.get("retained_visual_ids") != []
        or retained != 0
    ):
        raise ValueError("Task 7 result has invalid all-drop identity")
