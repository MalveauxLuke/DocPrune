"""Measured outcome preferences, preserving correctness strata and identity."""

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .policy import mask_scores


@dataclass(frozen=True)
class Outcome:
    g: float
    s: float

    def __post_init__(self):
        if not math.isfinite(self.g) or not math.isfinite(self.s):
            raise ValueError("Teacher outcomes must be finite")

    @property
    def c(self):
        return self.g - self.s


def preference(a, b, reference, *, baseline_correct, mode, epsilon, margin):
    """+1 favors a, -1 favors b, 0 is unordered. No hash tie-break training labels."""
    if mode not in ("g_only", "gold_aware", "pure_contrast"):
        raise ValueError("Unknown supervision mode")
    if epsilon < 0 or margin < 0 or not math.isfinite(epsilon + margin):
        raise ValueError("Invalid preference thresholds")

    def direction(delta):
        return int(delta > margin) - int(delta < -margin)

    if baseline_correct or mode == "g_only":
        return direction(a.g - b.g)
    if mode == "pure_contrast":
        return direction(a.c - b.c)
    threshold = reference.g - epsilon
    a_ok, b_ok = a.g >= threshold, b.g >= threshold
    if a_ok != b_ok:
        if min(abs(a.g - threshold), abs(b.g - threshold)) <= margin:
            return 0
        return 1 if a_ok else -1
    return direction(a.c - b.c if a_ok else a.g - b.g)


@dataclass
class TeacherBank:
    instance_key: str
    masks: torch.Tensor
    outcomes: tuple[Outcome, ...]
    reference: Outcome
    baseline_correct: bool
    proposal_uses_s: bool
    family: str

    def validate(self, inputs, *, require_s_free=True):
        if not isinstance(self.baseline_correct, bool) or not isinstance(
            self.proposal_uses_s, bool
        ):
            raise ValueError(
                "Teacher stratum and acquisition flags must be explicit booleans"
            )
        if (
            self.instance_key != inputs.identity.key
            or self.family != inputs.identity.document_family
        ):
            raise ValueError("Teacher/input identity mismatch")
        if require_s_free and self.proposal_uses_s:
            raise ValueError(
                "Central Stage 2 comparison requires proposals independent of S"
            )
        n = len(inputs.layout.region_ids)
        if self.masks.ndim != 2 or self.masks.shape != (len(self.outcomes), n):
            raise ValueError("Teacher masks/outcomes mismatch")
        if self.masks.dtype != torch.bool or len(self.outcomes) < 2:
            raise ValueError("Need at least two boolean masks")
        if torch.unique(self.masks, dim=0).shape[0] != len(self.outcomes):
            raise ValueError("Duplicate teacher masks")
        from .policy import achievable_budget

        cost = inputs.layout.costs
        target = achievable_budget(cost, inputs.budget)
        if not (self.masks.to(cost) @ cost == target).all():
            raise ValueError(
                "Policy comparisons must share the achievable token budget"
            )

    def pairs(self, mode, epsilon, margin):
        pairs = []
        for i, a in enumerate(self.outcomes):
            for j in range(i):
                pref = preference(
                    a,
                    self.outcomes[j],
                    self.reference,
                    baseline_correct=self.baseline_correct,
                    mode=mode,
                    epsilon=epsilon,
                    margin=margin,
                )
                if pref:
                    pairs.append((i, j) if pref > 0 else (j, i))
        return pairs


def question_loss(
    region_scores,
    bank,
    *,
    mode,
    epsilon,
    margin,
    temperature=1.0,
    correction_scores=None,
    auxiliary_weight=1.0,
):
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Temperature must be finite and positive")
    if auxiliary_weight < 0 or not math.isfinite(auxiliary_weight):
        raise ValueError("Auxiliary weight must be finite and nonnegative")
    if correction_scores is not None and correction_scores.shape != (
        len(bank.outcomes),
    ):
        raise ValueError("Correction/teacher mask count mismatch")
    pairs = bank.pairs(mode, epsilon, margin)
    if not pairs:
        return None
    values = mask_scores(region_scores, bank.masks)
    plus, minus = torch.tensor(pairs, device=values.device).T

    def rank(scores):
        return F.softplus(-(scores[plus] - scores[minus]) / temperature).mean()

    direct = rank(values)
    if correction_scores is None or auxiliary_weight == 0:
        return direct
    total = values + correction_scores.to(values)
    return direct + auxiliary_weight * rank(total)


def grouped_loss(losses, families, *, weighting="question"):
    active = [
        (loss, family) for loss, family in zip(losses, families) if loss is not None
    ]
    if len(losses) != len(families) or not active:
        raise ValueError("No ordered supervision or mismatched groups")
    if weighting == "question":
        return torch.stack([x for x, _ in active]).mean()
    if weighting != "family":
        raise ValueError("Unknown training weighting")
    groups = {}
    for loss, family in active:
        groups.setdefault(family, []).append(loss)
    return torch.stack([torch.stack(group).mean() for group in groups.values()]).mean()
