"""CPU contracts for truthful native and fixed-budget CTP policy identities."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json

import pytest
import torch


def _policies():
    return importlib.import_module("docprune.ctp_policy")


def test_corrected_runtime_policy_factories_keep_native_and_ranking_identities_distinct() -> None:
    """A native policy must never be representable as a forced score-top-M policy."""

    # This assertion deliberately fails before the policy integration exists,
    # rather than hiding the missing public module in collection machinery.
    assert importlib.util.find_spec("docprune.ctp_policy") is not None
    policies = _policies()
    no_ctp = policies.btp_qtp_no_ctp_policy()
    literal_native = policies.literal_native_threshold_policy()
    aggregate_native = policies.aggregate_native_threshold_policy()
    literal_ranked = policies.literal_score_top_m_policy()
    aggregate_ranked = policies.aggregate_score_top_m_policy()

    assert (no_ctp.family, no_ctp.selection_kind, no_ctp.score_semantics, no_ctp.budget_source) == (
        "no-ctp",
        "none",
        "none",
        "none",
    )
    assert (
        literal_native.name,
        literal_native.family,
        literal_native.selection_kind,
        literal_native.score_semantics,
        literal_native.budget_source,
    ) == (
        "literal-native-threshold",
        "native-threshold",
        "native_threshold",
        "literal-full-key-softmax-mean-head",
        "native-threshold",
    )
    assert (
        aggregate_native.name,
        aggregate_native.family,
        aggregate_native.selection_kind,
        aggregate_native.score_semantics,
        aggregate_native.budget_source,
    ) == (
        "aggregate-native-threshold",
        "native-threshold",
        "native_threshold",
        "aggregate-raw-logit-visual-softmax",
        "native-threshold",
    )
    assert (
        literal_ranked.name,
        literal_ranked.family,
        literal_ranked.selection_kind,
        literal_ranked.score_semantics,
        literal_ranked.budget_source,
    ) == (
        "literal-score-top-m",
        "score-top-m",
        "forced_top_m",
        "literal-full-key-softmax-mean-head",
        "aggregate-native-threshold-count",
    )
    assert (
        aggregate_ranked.name,
        aggregate_ranked.family,
        aggregate_ranked.selection_kind,
        aggregate_ranked.score_semantics,
        aggregate_ranked.budget_source,
    ) == (
        "aggregate-score-top-m",
        "score-top-m",
        "forced_top_m",
        "aggregate-raw-logit-visual-softmax",
        "aggregate-native-threshold-count",
    )
    assert literal_native != literal_ranked
    assert aggregate_native != aggregate_ranked


@pytest.mark.parametrize(
    ("policy_factory", "population", "expected_budget"),
    [
        ("literal_score_top_m_policy", 10, None),
        ("aggregate_score_top_m_policy", 10, None),
    ],
)
def test_primary_score_factories_require_aggregate_native_budget(
    policy_factory: str, population: int, expected_budget: None
) -> None:
    """Changing a ranking factory to invent an independent M must break this contract."""

    policy = getattr(_policies(), policy_factory)()

    assert policy.fixed_retention is None
    assert expected_budget is None
    assert policy.resolve_budget(population, aggregate_native_budget=4) == 4
    with pytest.raises(ValueError, match="aggregate native"):
        policy.resolve_budget(population, aggregate_native_budget=None)


def test_boundary_score_vectors_share_raw_logits_but_preserve_the_two_approved_formulas() -> None:
    """Replacing either formula or computing them from different tensors must break this fixture."""

    ctp = importlib.import_module("docprune.ctp")
    assert hasattr(ctp, "boundary_score_vectors_from_logits")

    raw_logits = torch.tensor(
        [[[[0.0, 0.0, 1.0, 2.0]], [[0.0, 0.0, 3.0, 3.0]]]], dtype=torch.float32
    )
    visual_indices = torch.tensor([2, 3])

    literal, aggregate = ctp.boundary_score_vectors_from_logits(raw_logits, visual_indices)

    torch.testing.assert_close(literal, torch.tensor([0.7008, 1.0866]), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(aggregate, torch.tensor([0.7551, 1.2449]), rtol=1e-4, atol=1e-4)


def test_aggregate_score_top_m_matches_native_mask_without_a_threshold_tie() -> None:
    """A non-tied aggregate ranking divergence is an implementation failure, not an arm."""

    policies = _policies()
    assert hasattr(policies, "select_boundary_policy")

    selection = policies.select_boundary_policy(
        policies.aggregate_score_top_m_policy(),
        literal_scores=(0.1, 1.0, 0.2, 0.9),
        aggregate_scores=(0.2, 0.8, 0.3, 0.7),
        attention_threshold=0.5,
        boundary="B_4",
        native_layer=4,
    )

    assert selection.retained_visual_ids == (1, 3)
    assert selection.aggregate_native_reference_ids == (1, 3)
    assert selection.aggregate_threshold_tied_ids == ()
    assert selection.symmetric_difference_ids == ()
    assert selection.requested_budget == selection.achieved_budget == 2


def test_aggregate_threshold_tie_serializes_tied_set_and_ranking_difference() -> None:
    """Ties must be evidence, never an implicit claim that fixed-M matches native threshold."""

    policies = _policies()
    assert hasattr(policies, "select_boundary_policy")

    selection = policies.select_boundary_policy(
        policies.aggregate_score_top_m_policy(),
        literal_scores=(0.1, 0.2, 0.3),
        aggregate_scores=(0.8, 0.5, 0.5),
        attention_threshold=0.5,
        boundary="B_2",
        native_layer=2,
    )

    assert selection.aggregate_native_reference_ids == (0, 1, 2)
    assert selection.retained_visual_ids == (0, 1, 2)
    assert selection.aggregate_threshold_tied_ids == (1, 2)
    assert selection.symmetric_difference_ids == ()


@pytest.mark.parametrize(
    ("retention", "population", "expected"),
    [("11/20", 10, 6), ("13/20", 10, 7), ("4/5", 9, 7)],
)
def test_fixed_retention_budgets_use_exact_half_up_rounding(
    retention: str, population: int, expected: int
) -> None:
    """Changing to banker's rounding or a float approximation must fail this table."""

    policies = _policies()
    assert hasattr(policies, "fixed_retention_policy")

    policy = policies.fixed_retention_policy("aggregate", retention)

    assert policy.name.endswith(
        {"11/20": "-retain-55", "13/20": "-retain-65", "4/5": "-retain-80"}[retention]
    )
    assert policy.resolve_budget(population, aggregate_native_budget=None) == expected


def test_fixed_retention_random_policy_preserves_task_four_namespace_and_budget() -> None:
    """A random sensitivity arm must keep its selector namespace and exact retention budget."""

    policies = _policies()
    assert hasattr(policies, "fixed_retention_random_policy")

    policy = policies.fixed_retention_random_policy("global-uniform-random", "13/20")

    assert policy.name == "global-uniform-random-retain-65"
    assert policy.family == "random-top-m"
    assert policy.selection_kind == "forced_top_m"
    assert policy.score_semantics == "random"
    assert policy.resolve_budget(10, aggregate_native_budget=None) == 7


def test_global_random_selection_uses_the_task_four_five_field_seed_context() -> None:
    """Changing the context tuple or using process/global RNG must alter this exact selection."""

    policies = _policies()
    assert hasattr(policies, "PolicySelectionContext")

    policy = policies.fixed_retention_random_policy("global-uniform-random", "11/20")
    context = policies.PolicySelectionContext("exp-v1", "qid-9", "B_3", 2)
    selection = policies.select_boundary_policy(
        policy,
        literal_scores=(0.1,) * 10,
        aggregate_scores=(0.1,) * 10,
        attention_threshold=0.5,
        boundary="B_3",
        native_layer=3,
        selection_context=context,
    )

    from docprune.ctp_controls import global_uniform_random

    expected = global_uniform_random(
        10,
        6,
        experiment_version="exp-v1",
        qid="qid-9",
        boundary="B_3",
        policy="global-uniform-random-retain-55",
        repetition=2,
    )
    assert selection.retained_visual_ids == expected.retained_visual_ids
    assert selection.seed_sha256 == expected.seed_sha256


def test_random_policy_rejects_missing_source_qid_before_seeding() -> None:
    """A caller bypassing the runner cannot seed a random arm with an invented QID."""

    policies = _policies()
    with pytest.raises(ValueError, match="source QID"):
        policies.select_boundary_policy(
            policies.fixed_retention_random_policy("global-uniform-random", "11/20"),
            literal_scores=(0.1,) * 10,
            aggregate_scores=(0.1,) * 10,
            attention_threshold=0.5,
            boundary="B_3",
            native_layer=3,
            selection_context=policies.PolicySelectionContext("exp-v1", None, "B_3", 2),
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {
            "name": "literal-native-threshold",
            "family": "native-threshold",
            "selection_kind": "native_threshold",
            "score_semantics": "aggregate-raw-logit-visual-softmax",
            "budget_source": "native-threshold",
        },
        {
            "name": "aggregate-score-top-m",
            "family": "score-top-m",
            "selection_kind": "forced_top_m",
            "score_semantics": "literal-full-key-softmax-mean-head",
            "budget_source": "aggregate-native-threshold-count",
        },
        {
            "name": "coverage-matched-identity-shuffle-retain-55",
            "family": "random-top-m",
            "selection_kind": "forced_top_m",
            "score_semantics": "coverage",
            "budget_source": "fixed-retention",
        },
    ),
)
def test_policy_descriptor_rejects_contradictory_cross_field_identity(
    kwargs: dict[str, object],
) -> None:
    """Changing a method label independently of its semantics must fail closed."""

    from docprune.ctp_policy import CTPPolicy

    with pytest.raises(ValueError, match="identity"):
        CTPPolicy(**kwargs)


def test_coverage_policy_uses_task_four_coverage_selector_and_persists_seed() -> None:
    """A coverage arm must not fall through to aggregate score ranking."""

    from docprune.ctp_controls import VisualTokenGeometry, coverage_matched_identity_shuffle

    policies = _policies()
    policy = policies.fixed_retention_random_policy("coverage-matched-identity-shuffle", "11/20")
    geometry = tuple(VisualTokenGeometry(0, row, 0, 10, 1) for row in range(10))
    context = policies.PolicySelectionContext("v1", "q-1", "B_2", 0, geometry)
    selection = policies.select_boundary_policy(
        policy,
        literal_scores=(0.0,) * 10,
        aggregate_scores=tuple(float(index) for index in range(10)),
        attention_threshold=100.0,
        boundary="B_2",
        native_layer=2,
        selection_context=context,
    )
    expected_reference = (4, 5, 6, 7, 8, 9)
    expected = coverage_matched_identity_shuffle(
        geometry,
        reference_retained_ids=expected_reference,
        experiment_version="v1",
        qid="q-1",
        boundary="B_2",
        policy="coverage-matched-identity-shuffle-retain-55",
        repetition=0,
    )

    assert selection.retained_visual_ids == expected.retained_visual_ids
    assert selection.seed_sha256 == expected.seed_sha256


def test_random_selection_persists_exact_post_qtp_geometry_identity() -> None:
    """A Task 6 result must bind the geometry used by its randomized selector."""

    from docprune.ctp_controls import VisualTokenGeometry

    policies = _policies()
    geometry = tuple(VisualTokenGeometry(0, 0, column, 1, 4) for column in range(4))
    geometry_sha256 = hashlib.sha256(
        json.dumps([[0, 0, column, 1, 4] for column in range(4)], separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    selection = policies.select_boundary_policy(
        policies.random_top_m_policy("global-uniform-random"),
        literal_scores=(0.1,) * 4,
        aggregate_scores=(0.9, 0.8, 0.1, 0.0),
        attention_threshold=0.5,
        boundary="B_2",
        native_layer=2,
        selection_context=policies.PolicySelectionContext(
            "task6-v1", "qid", "B_2", 0, geometry, 4, geometry_sha256
        ),
    )

    assert selection.geometry_count == 4
    assert selection.geometry_sha256 == geometry_sha256
    assert selection.to_dict()["geometry_count"] == 4


def test_no_crossing_preserves_authenticated_dynamic_geometry() -> None:
    from docprune.ctp_controls import VisualTokenGeometry

    policies = _policies()
    geometry = (
        VisualTokenGeometry(0, 0, 0, 1, 2),
        VisualTokenGeometry(0, 0, 1, 1, 2),
    )
    geometry_sha256 = hashlib.sha256(b"[[0,0,0,1,2],[0,0,1,1,2]]").hexdigest()
    selection = policies.no_crossing_selection(
        policies.aggregate_native_threshold_policy(),
        2,
        selection_context=policies.PolicySelectionContext(
            None, None, None, None, geometry, 2, geometry_sha256
        ),
    )

    assert selection.geometry_count == 2
    assert selection.geometry_sha256 == geometry_sha256


def test_geometry_aware_policy_fails_closed_before_selection_without_post_qtp_geometry() -> None:
    """Coverage cannot invent geometry before the runtime exposes post-QTP tokens."""

    policies = _policies()
    policy = policies.fixed_retention_random_policy("coverage-matched-identity-shuffle", "11/20")
    with pytest.raises(ValueError, match="post-QTP token geometry"):
        policies.select_boundary_policy(
            policy,
            literal_scores=(0.1,) * 10,
            aggregate_scores=(0.1,) * 10,
            attention_threshold=0.5,
            boundary="B_1",
            native_layer=1,
            selection_context=policies.PolicySelectionContext("v1", "qid", "B_1", 0),
        )


def test_boundary_selection_rejects_infinite_threshold() -> None:
    """Non-finite native thresholds cannot define durable CTP evidence."""

    policies = _policies()
    with pytest.raises(ValueError, match="finite"):
        policies.select_boundary_policy(
            policies.aggregate_native_threshold_policy(),
            literal_scores=(0.1,),
            aggregate_scores=(0.1,),
            attention_threshold=float("inf"),
            boundary="B_0",
            native_layer=0,
        )
