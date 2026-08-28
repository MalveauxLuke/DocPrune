"""Fail-closed runtime identities for the visual-state removal curve."""

from __future__ import annotations

from dataclasses import dataclass

from docprune.ctp_policy import CTPPolicy, btp_qtp_no_ctp_policy
from docprune.qwen2vl.decoder import ForcedVisualIntervention


@dataclass(frozen=True, slots=True)
class Task7InterventionCell:
    """One mutually exclusive reference or fixed-boundary all-drop arm."""

    name: str
    ctp_policy: CTPPolicy | None
    forced_intervention: ForcedVisualIntervention | None


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
