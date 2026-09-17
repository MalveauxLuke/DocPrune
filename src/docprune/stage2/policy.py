"""Direct Head 1, optional set correction, and exact-cost 0/1 allocation."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from .cost import measure


class PolicyHead(nn.Module):
    def __init__(self, width, hidden=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden or width), nn.GELU(), nn.Linear(hidden or width, 1)
        )

    def forward(self, regions):
        return self.net(regions).squeeze(-1)


def mask_scores(scores, masks):
    if masks.ndim != 2 or masks.shape[1] != scores.numel():
        raise ValueError("Mask/region shape mismatch")
    if not ((masks == 0) | (masks == 1)).all():
        raise ValueError("Masks must be binary")
    return masks.to(scores) @ scores


@dataclass(frozen=True)
class PolicyEncoding:
    """Mask-independent representations and Head 1 scores from one selector call."""

    regions: torch.Tensor
    scores: torch.Tensor


class SetCorrection(nn.Module):
    """Optional Head 2: nonlinear kept/dropped set pooling over stored regions.

    No visual re-encoding and no positional encoding of the arbitrary region list.
    Geometry, question and budget context already belong to the regional states.
    This correction is not an identified main-effect/interaction decomposition.
    """

    def __init__(self, width):
        super().__init__()
        self.elements = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.readout = nn.Sequential(
            nn.Linear(2 * width + 1, width), nn.GELU(), nn.Linear(width, 1)
        )

    def forward(self, regions, masks):
        if regions.ndim != 2 or regions.shape[0] == 0:
            raise ValueError("Need a nonempty regional representation matrix")
        if masks.ndim != 2 or masks.shape[1] != regions.shape[0]:
            raise ValueError("Mask/region shape mismatch")
        if not ((masks == 0) | (masks == 1)).all():
            raise ValueError("Masks must be binary")
        values = self.elements(regions)
        keep = masks.to(values)
        kept_count = keep.sum(dim=1, keepdim=True)
        dropped_count = regions.shape[0] - kept_count
        kept = keep @ values / kept_count.clamp_min(1)
        dropped = (1 - keep) @ values / dropped_count.clamp_min(1)
        fraction = kept_count / regions.shape[0]
        return self.readout(torch.cat((kept, dropped, fraction), dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class MaskPredictions:
    direct: torch.Tensor
    correction: torch.Tensor
    total: torch.Tensor


def score_candidate_masks(encoding, masks, correction=None, *, ledger=None):
    """Supporting mask-ranking diagnostic; does not select or rerank deployed masks.

    Reuses one mask-independent encoding for an entire bank. The caller supplies
    the bank; no teacher target enters either head. Default deployment uses Head 1.
    """
    direct = mask_scores(encoding.scores, masks)
    if correction is None:
        delta = torch.zeros_like(direct)
    else:
        with measure(ledger, "selector_head2", encoding.regions):
            delta = correction(encoding.regions, masks)
        if ledger is not None:
            key = "head2_candidate_masks"
            ledger.counters[key] = ledger.counters.get(key, 0) + masks.shape[0]
    return MaskPredictions(direct, delta, direct + delta)


def achievable_budget(costs, budget):
    values = _costs(costs)
    if not isinstance(budget, int) or budget <= 0:
        raise ValueError("Budget must be a positive integer")
    reachable = 1
    limit = (1 << (budget + 1)) - 1
    for cost in values:
        reachable = (reachable | reachable << cost) & limit
    result = reachable.bit_length() - 1
    if result == 0:
        raise ValueError("No nonempty region set fits the budget")
    return result


def _costs(costs):
    raw = (
        costs.detach().cpu().tolist()
        if isinstance(costs, torch.Tensor)
        else list(costs)
    )
    if not raw or any(
        isinstance(c, bool) or not isinstance(c, int) or c <= 0 for c in raw
    ):
        raise ValueError("Region costs must be positive integers")
    return raw


def allocate(scores, costs, budget):
    """Maximize additive utility at B*, with deterministic first-found ties.

    B* depends only on costs/B, never scores, answers, or correctness. O(R B*) time.
    """
    values = _costs(costs)
    utility = scores.detach().double().cpu().tolist()
    if len(utility) != len(values) or not all(math.isfinite(x) for x in utility):
        raise ValueError("Invalid utility vector")
    target = achievable_budget(values, budget)
    best = [-math.inf] * (target + 1)
    best[0] = 0.0
    choices = []
    for cost, value in zip(values, utility):
        take = bytearray(target + 1)
        for b in range(target, cost - 1, -1):
            candidate = best[b - cost] + value
            if candidate > best[b]:
                best[b] = candidate
                take[b] = 1
        choices.append(take)
    keep = [False] * len(values)
    b = target
    for i in range(len(values) - 1, -1, -1):
        if choices[i][b]:
            keep[i] = True
            b -= values[i]
    if b:
        raise RuntimeError("Allocation traceback failed")
    return torch.tensor(keep, device=scores.device), target
