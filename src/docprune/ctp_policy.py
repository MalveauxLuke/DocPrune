"""Truthful corrected-runtime CTP policy descriptors and selections.

This module deliberately keeps native threshold decisions separate from the
Task 3 forced-intervention record.  A descriptor is the single identity used
by factory, decoder, and serialized per-question selection evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

from docprune.ctp_controls import (
    VisualTokenGeometry,
    coverage_matched_identity_shuffle,
    global_uniform_random,
    grid_stratified_random,
    page_stratified_random,
    score_top_m,
)

_SCORE_SEMANTICS = {
    "none",
    "literal-full-key-softmax-mean-head",
    "aggregate-raw-logit-visual-softmax",
    "random",
    "coverage",
}
_FAMILIES = {"no-ctp", "native-threshold", "score-top-m", "random-top-m", "coverage-top-m"}
_RETENTION_SUFFIXES = {
    Fraction(11, 20): "-retain-55",
    Fraction(13, 20): "-retain-65",
    Fraction(4, 5): "-retain-80",
}


@dataclass(frozen=True, slots=True)
class CTPPolicy:
    """One declared runtime policy with enough information to fail closed."""

    name: str
    family: str
    selection_kind: str
    score_semantics: str
    budget_source: str
    fixed_retention: Fraction | None = None

    def __post_init__(self) -> None:
        if not _identity_is_allowed(self):
            raise ValueError("CTP policy identity is inconsistent")

    def resolve_budget(self, population_size: int, *, aggregate_native_budget: int | None) -> int:
        """Resolve this policy's explicit budget without silently substituting M."""

        if not isinstance(population_size, Integral) or isinstance(population_size, bool):
            raise ValueError("visual population must be an integer")
        if population_size < 0:
            raise ValueError("visual population must be non-negative")
        if self.family == "no-ctp" or self.family == "native-threshold":
            return int(population_size)
        if self.fixed_retention is not None:
            numerator, denominator = (
                self.fixed_retention.numerator,
                self.fixed_retention.denominator,
            )
            return min(
                int(population_size),
                max(0, (numerator * int(population_size) + denominator // 2) // denominator),
            )
        if aggregate_native_budget is None:
            raise ValueError("aggregate native threshold budget is required")
        if (
            not isinstance(aggregate_native_budget, Integral)
            or isinstance(aggregate_native_budget, bool)
            or not 0 <= aggregate_native_budget <= population_size
        ):
            raise ValueError("aggregate native threshold budget is outside the visual population")
        return int(aggregate_native_budget)

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "name": self.name,
            "selection_kind": self.selection_kind,
            "score_semantics": self.score_semantics,
            "budget_source": self.budget_source,
            "fixed_retention": None
            if self.fixed_retention is None
            else f"{self.fixed_retention.numerator}/{self.fixed_retention.denominator}",
        }


@dataclass(frozen=True, slots=True)
class CTPSelectionRecord:
    """Durable native/ranking evidence; intentionally not a forced record."""

    policy: CTPPolicy
    boundary: str | None
    native_layer: int | None
    visual_population: int
    requested_budget: int
    achieved_budget: int
    retained_visual_ids: tuple[int, ...]
    aggregate_native_reference_ids: tuple[int, ...] | None
    aggregate_threshold_tied_ids: tuple[int, ...]
    symmetric_difference_ids: tuple[int, ...]
    native_threshold: float | None
    seed_sha256: str | None = None
    geometry_count: int | None = None
    geometry_sha256: str | None = None
    prefill_cache_lengths: tuple[int, ...] = ()
    retained_mrope_position_shape: tuple[int, int, int] | None = None
    retained_mrope_position_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy.to_dict(),
            "boundary": self.boundary,
            "native_layer": self.native_layer,
            "visual_population": self.visual_population,
            "requested_budget": self.requested_budget,
            "achieved_budget": self.achieved_budget,
            "retained_compact_visual_ids": list(self.retained_visual_ids),
            "aggregate_native_reference_ids": None
            if self.aggregate_native_reference_ids is None
            else list(self.aggregate_native_reference_ids),
            "aggregate_threshold_tied_ids": list(self.aggregate_threshold_tied_ids),
            "symmetric_difference_ids": list(self.symmetric_difference_ids),
            "native_threshold": self.native_threshold,
            "seed_sha256": self.seed_sha256,
            "geometry_count": self.geometry_count,
            "geometry_sha256": self.geometry_sha256,
            "prefill_cache_lengths": list(self.prefill_cache_lengths),
            "retained_mrope_position_shape": (
                None
                if self.retained_mrope_position_shape is None
                else list(self.retained_mrope_position_shape)
            ),
            "retained_mrope_position_sha256": self.retained_mrope_position_sha256,
        }


@dataclass(frozen=True, slots=True)
class PolicySelectionContext:
    """Only the Task 4 five-field random-seed identity, carried to a boundary."""

    experiment_version: object
    qid: object
    boundary: str | None
    repetition: object
    geometry: tuple[VisualTokenGeometry, ...] | None = None
    geometry_count: int | None = None
    geometry_sha256: str | None = None


def _retention_suffix(retention: Fraction) -> str:
    try:
        return _RETENTION_SUFFIXES[retention]
    except KeyError as error:
        raise ValueError("only the approved 55%, 65%, and 80% retentions are supported") from error


def _identity_is_allowed(policy: CTPPolicy) -> bool:
    """Accept only the frozen complete policy identity table."""

    if policy.family not in _FAMILIES or policy.score_semantics not in _SCORE_SEMANTICS:
        return False
    observed = (
        policy.family,
        policy.selection_kind,
        policy.score_semantics,
        policy.budget_source,
        policy.fixed_retention,
    )
    primary = {
        "btp-qtp-no-ctp": ("no-ctp", "none", "none", "none", None),
        "literal-native-threshold": (
            "native-threshold",
            "native_threshold",
            "literal-full-key-softmax-mean-head",
            "native-threshold",
            None,
        ),
        "aggregate-native-threshold": (
            "native-threshold",
            "native_threshold",
            "aggregate-raw-logit-visual-softmax",
            "native-threshold",
            None,
        ),
        "literal-score-top-m": (
            "score-top-m",
            "forced_top_m",
            "literal-full-key-softmax-mean-head",
            "aggregate-native-threshold-count",
            None,
        ),
        "aggregate-score-top-m": (
            "score-top-m",
            "forced_top_m",
            "aggregate-raw-logit-visual-softmax",
            "aggregate-native-threshold-count",
            None,
        ),
        "global-uniform-random": (
            "random-top-m",
            "forced_top_m",
            "random",
            "aggregate-native-threshold-count",
            None,
        ),
        "page-stratified-random": (
            "random-top-m",
            "forced_top_m",
            "random",
            "aggregate-native-threshold-count",
            None,
        ),
        "grid-stratified-random": (
            "random-top-m",
            "forced_top_m",
            "random",
            "aggregate-native-threshold-count",
            None,
        ),
        "coverage-matched-identity-shuffle": (
            "coverage-top-m",
            "forced_top_m",
            "coverage",
            "aggregate-native-threshold-count",
            None,
        ),
    }
    if policy.name in primary:
        return observed == primary[policy.name]
    if (
        not isinstance(policy.fixed_retention, Fraction)
        or policy.fixed_retention not in _RETENTION_SUFFIXES
    ):
        return False
    suffix = _retention_suffix(policy.fixed_retention)
    fixed = {
        f"literal-score-top-m{suffix}": (
            "score-top-m",
            "forced_top_m",
            "literal-full-key-softmax-mean-head",
        ),
        f"aggregate-score-top-m{suffix}": (
            "score-top-m",
            "forced_top_m",
            "aggregate-raw-logit-visual-softmax",
        ),
        f"global-uniform-random{suffix}": ("random-top-m", "forced_top_m", "random"),
        f"page-stratified-random{suffix}": ("random-top-m", "forced_top_m", "random"),
        f"grid-stratified-random{suffix}": ("random-top-m", "forced_top_m", "random"),
        f"coverage-matched-identity-shuffle{suffix}": (
            "coverage-top-m",
            "forced_top_m",
            "coverage",
        ),
    }
    expected = fixed.get(policy.name)
    return expected is not None and observed == (
        *expected,
        "fixed-retention",
        policy.fixed_retention,
    )


def btp_qtp_no_ctp_policy() -> CTPPolicy:
    return CTPPolicy("btp-qtp-no-ctp", "no-ctp", "none", "none", "none")


def literal_native_threshold_policy() -> CTPPolicy:
    return CTPPolicy(
        "literal-native-threshold",
        "native-threshold",
        "native_threshold",
        "literal-full-key-softmax-mean-head",
        "native-threshold",
    )


def aggregate_native_threshold_policy() -> CTPPolicy:
    return CTPPolicy(
        "aggregate-native-threshold",
        "native-threshold",
        "native_threshold",
        "aggregate-raw-logit-visual-softmax",
        "native-threshold",
    )


def _score_top_m_policy(score_kind: str, retention: Fraction | None = None) -> CTPPolicy:
    if score_kind not in {"literal", "aggregate"}:
        raise ValueError("score kind must be literal or aggregate")
    score_semantics = (
        "literal-full-key-softmax-mean-head"
        if score_kind == "literal"
        else "aggregate-raw-logit-visual-softmax"
    )
    suffix = "" if retention is None else _retention_suffix(retention)
    return CTPPolicy(
        f"{score_kind}-score-top-m{suffix}",
        "score-top-m",
        "forced_top_m",
        score_semantics,
        "aggregate-native-threshold-count" if retention is None else "fixed-retention",
        retention,
    )


def literal_score_top_m_policy(*, retention: Fraction | None = None) -> CTPPolicy:
    return _score_top_m_policy("literal", retention)


def aggregate_score_top_m_policy(*, retention: Fraction | None = None) -> CTPPolicy:
    return _score_top_m_policy("aggregate", retention)


def random_top_m_policy(selector: str) -> CTPPolicy:
    """Construct one approved primary native-budget random/coverage policy."""

    namespaces = {
        "global-uniform-random": ("random-top-m", "random"),
        "page-stratified-random": ("random-top-m", "random"),
        "grid-stratified-random": ("random-top-m", "random"),
        "coverage-matched-identity-shuffle": ("coverage-top-m", "coverage"),
    }
    try:
        family, semantics = namespaces[selector]
    except KeyError as error:
        raise ValueError("selector must use a Task 4 random or coverage namespace") from error
    return CTPPolicy(
        selector,
        family,
        "forced_top_m",
        semantics,
        "aggregate-native-threshold-count",
    )


def fixed_retention_policies() -> tuple[CTPPolicy, ...]:
    """Return approved developmental score/random/coverage sensitivity arms."""

    retentions = (Fraction(11, 20), Fraction(13, 20), Fraction(4, 5))
    return tuple(
        policy
        for retention in retentions
        for policy in (
            literal_score_top_m_policy(retention=retention),
            aggregate_score_top_m_policy(retention=retention),
            fixed_retention_random_policy("global-uniform-random", str(retention)),
            fixed_retention_random_policy("page-stratified-random", str(retention)),
            fixed_retention_random_policy("grid-stratified-random", str(retention)),
            fixed_retention_random_policy("coverage-matched-identity-shuffle", str(retention)),
        )
    )


def fixed_retention_policy(score_kind: str, retention: str) -> CTPPolicy:
    """Construct one approved fixed-retention score policy from its exact fraction."""

    try:
        fraction = Fraction(retention)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("fixed retention must be one approved exact fraction") from error
    return _score_top_m_policy(score_kind, fraction)


def fixed_retention_random_policy(selector: str, retention: str) -> CTPPolicy:
    """Name a developmental random/coverage arm without changing its Task 4 selector."""

    try:
        fraction = Fraction(retention)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("fixed retention must be one approved exact fraction") from error
    namespaces = {
        "global-uniform-random": ("random-top-m", "random"),
        "page-stratified-random": ("random-top-m", "random"),
        "grid-stratified-random": ("random-top-m", "random"),
        "coverage-matched-identity-shuffle": ("coverage-top-m", "coverage"),
    }
    try:
        family, semantics = namespaces[selector]
    except KeyError as error:
        raise ValueError("selector must use a Task 4 random or coverage namespace") from error
    return CTPPolicy(
        f"{selector}{_retention_suffix(fraction)}",
        family,
        "forced_top_m",
        semantics,
        "fixed-retention",
        fraction,
    )


def _scores(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if any(value != value or value in {float("inf"), float("-inf")} for value in result):
        raise ValueError(f"{name} must contain finite scores")
    return result


def _geometry_identity(
    context: PolicySelectionContext | None,
) -> tuple[int | None, str | None]:
    if context is None or context.geometry is None:
        if context is not None and (
            context.geometry_count is not None or context.geometry_sha256 is not None
        ):
            raise ValueError("geometry identity requires post-QTP token geometry")
        return None, None
    rows = [
        [token.page_index, token.row, token.column, token.height, token.width]
        for token in context.geometry
    ]
    encoded = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    count, digest = len(rows), hashlib.sha256(encoded).hexdigest()
    if context.geometry_count is not None and context.geometry_count != count:
        raise ValueError("post-QTP geometry count mismatch")
    if context.geometry_sha256 is not None and context.geometry_sha256 != digest:
        raise ValueError("post-QTP geometry checksum mismatch")
    return count, digest


def select_boundary_policy(
    policy: CTPPolicy,
    *,
    literal_scores: Sequence[float],
    aggregate_scores: Sequence[float],
    attention_threshold: float,
    boundary: str,
    native_layer: int,
    selection_context: PolicySelectionContext | None = None,
) -> CTPSelectionRecord:
    """Select a native or ranking policy at one already-observed CTP boundary."""

    literal, aggregate = (
        _scores(literal_scores, name="literal_scores"),
        _scores(aggregate_scores, name="aggregate_scores"),
    )
    if len(literal) != len(aggregate):
        raise ValueError("literal and aggregate score vectors must have equal length")
    if not isinstance(attention_threshold, int | float) or not math.isfinite(attention_threshold):
        raise ValueError("attention threshold must be finite")
    if not isinstance(boundary, str) or not boundary.startswith("B_"):
        raise ValueError("boundary must be a canonical B_* name")
    if type(native_layer) is not int or native_layer < 0:
        raise ValueError("native layer must be a non-negative integer")
    population = len(aggregate)
    geometry_count, geometry_sha256 = _geometry_identity(selection_context)
    if geometry_count is not None and geometry_count != population:
        raise ValueError("post-QTP geometry count does not match visual population")
    aggregate_native = tuple(
        index for index, score in enumerate(aggregate) if score >= attention_threshold
    )
    aggregate_tied = tuple(
        index for index, score in enumerate(aggregate) if score == attention_threshold
    )
    native_budget = len(aggregate_native)
    if policy.family == "no-ctp":
        retained = tuple(range(population))
        aggregate_reference: tuple[int, ...] | None = None
        tied: tuple[int, ...] = ()
        threshold: float | None = None
        seed_sha256: str | None = None
    elif policy.family == "native-threshold":
        source = literal if policy.score_semantics.startswith("literal-") else aggregate
        retained = tuple(
            index for index, score in enumerate(source) if score >= attention_threshold
        )
        aggregate_reference = (
            aggregate_native if policy.score_semantics.startswith("aggregate-") else None
        )
        tied = aggregate_tied if aggregate_reference is not None else ()
        threshold = float(attention_threshold)
        seed_sha256 = None
    else:
        budget = policy.resolve_budget(population, aggregate_native_budget=native_budget)
        if policy.family in {"random-top-m", "coverage-top-m"}:
            if selection_context is None:
                raise ValueError("random policy selection requires Task 4 seed context")
            if not isinstance(selection_context.qid, str) or not selection_context.qid:
                raise ValueError("random policy selection requires a source QID")
            if selection_context.experiment_version is None or selection_context.repetition is None:
                raise ValueError(
                    "random policy selection requires experiment version and repetition"
                )
            if selection_context.boundary is not None and selection_context.boundary != boundary:
                raise ValueError("random policy context boundary does not match selection boundary")
            selection_kwargs = {
                "experiment_version": selection_context.experiment_version,
                "qid": selection_context.qid,
                "boundary": boundary,
                "policy": policy.name,
                "repetition": selection_context.repetition,
            }
            if policy.name.startswith("global-uniform-random"):
                random_selection = global_uniform_random(population, budget, **selection_kwargs)
            else:
                geometry = selection_context.geometry
                if geometry is None or len(geometry) != population:
                    raise ValueError(
                        "geometry-aware Task 4 selection requires post-QTP token geometry"
                    )
                reference = score_top_m(
                    aggregate, budget=budget, score_kind="aggregate"
                ).retained_visual_ids
                if policy.name.startswith("page-stratified-random"):
                    random_selection = page_stratified_random(
                        geometry, reference_retained_ids=reference, **selection_kwargs
                    )
                elif policy.name.startswith("grid-stratified-random"):
                    random_selection = grid_stratified_random(geometry, budget, **selection_kwargs)
                elif policy.name.startswith("coverage-matched-identity-shuffle"):
                    random_selection = coverage_matched_identity_shuffle(
                        geometry, reference_retained_ids=reference, **selection_kwargs
                    )
                else:
                    raise ValueError("random policy does not name a Task 4 selector")
            retained, seed_sha256 = (
                random_selection.retained_visual_ids,
                random_selection.seed_sha256,
            )
        else:
            score_kind = "literal" if policy.score_semantics.startswith("literal-") else "aggregate"
            source = literal if score_kind == "literal" else aggregate
            retained = score_top_m(source, budget=budget, score_kind=score_kind).retained_visual_ids
            seed_sha256 = None
        aggregate_reference = aggregate_native
        tied = aggregate_tied
        threshold = float(attention_threshold)
        if policy.name == "aggregate-score-top-m" and not tied and retained != aggregate_native:
            raise AssertionError(
                "aggregate score-top-M disagrees with an untied aggregate native mask"
            )
    difference = (
        ()
        if aggregate_reference is None
        else tuple(sorted(set(retained).symmetric_difference(aggregate_reference)))
    )
    return CTPSelectionRecord(
        policy=policy,
        boundary=boundary,
        native_layer=native_layer,
        visual_population=population,
        requested_budget=len(retained),
        achieved_budget=len(retained),
        retained_visual_ids=tuple(retained),
        aggregate_native_reference_ids=aggregate_reference,
        aggregate_threshold_tied_ids=tied,
        symmetric_difference_ids=difference,
        native_threshold=threshold,
        seed_sha256=seed_sha256,
        geometry_count=geometry_count,
        geometry_sha256=geometry_sha256,
    )


def no_crossing_selection(
    policy: CTPPolicy,
    visual_population: int,
    *,
    selection_context: PolicySelectionContext | None = None,
) -> CTPSelectionRecord:
    """Serialize the no-op state without pretending it was a forced boundary."""

    if not isinstance(visual_population, Integral) or isinstance(visual_population, bool):
        raise ValueError("visual population must be an integer")
    if visual_population < 0:
        raise ValueError("visual population must be non-negative")
    geometry_count, geometry_sha256 = _geometry_identity(selection_context)
    if geometry_count is not None and geometry_count != int(visual_population):
        raise ValueError("post-QTP geometry count does not match visual population")
    retained = tuple(range(int(visual_population)))
    return CTPSelectionRecord(
        policy=policy,
        boundary=None,
        native_layer=None,
        visual_population=int(visual_population),
        requested_budget=int(visual_population),
        achieved_budget=int(visual_population),
        retained_visual_ids=retained,
        aggregate_native_reference_ids=None,
        aggregate_threshold_tied_ids=(),
        symmetric_difference_ids=(),
        native_threshold=None,
        seed_sha256=None,
        geometry_count=geometry_count,
        geometry_sha256=geometry_sha256,
    )
