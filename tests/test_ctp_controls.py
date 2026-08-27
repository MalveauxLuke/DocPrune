from collections import Counter

import pytest

from docprune.ctp_controls import (
    VisualTokenGeometry,
    coverage_matched_identity_shuffle,
    coverage_measurements,
    global_uniform_random,
    grid_stratified_random,
    normalized_grid_cells,
    page_stratified_random,
    score_top_m,
)

GEOMETRY = (
    VisualTokenGeometry(page_index=0, row=0, column=0, height=4, width=4),
    VisualTokenGeometry(page_index=0, row=0, column=3, height=4, width=4),
    VisualTokenGeometry(page_index=0, row=3, column=0, height=4, width=4),
    VisualTokenGeometry(page_index=0, row=3, column=3, height=4, width=4),
    VisualTokenGeometry(page_index=1, row=0, column=0, height=2, width=2),
    VisualTokenGeometry(page_index=1, row=1, column=1, height=2, width=2),
)

SEED_ARGUMENTS = {
    "experiment_version": "coverage-v1",
    "qid": "q-7",
    "boundary": "prefill",
    "policy": "global-uniform-random",
    "repetition": 3,
}


def test_score_top_m_has_stable_score_then_visual_id_order_and_score_policy_names() -> None:
    aggregate = score_top_m((0.2, 0.9, 0.9, -0.1), budget=3, score_kind="aggregate")
    literal = score_top_m((0.2, 0.9, 0.9, -0.1), budget=2, score_kind="literal")

    assert aggregate.policy_name == "aggregate-score-top-m"
    assert aggregate.retained_visual_ids == (0, 1, 2)
    assert literal.policy_name == "literal-score-top-m"
    assert literal.retained_visual_ids == (1, 2)
    assert "ctp" not in aggregate.policy_name


def test_global_uniform_random_is_repeatable_sorted_and_emits_literal_seed_digest() -> None:
    selected = global_uniform_random(population_size=4, budget=2, **SEED_ARGUMENTS)

    assert (
        selected.seed_sha256 == "ee58e80c0efe6ba134d6d206e442526defc36b458fe0928871b6930e429c26c0"
    )
    assert global_uniform_random(population_size=4, budget=2, **SEED_ARGUMENTS) == selected
    assert selected.retained_visual_ids == tuple(sorted(selected.retained_visual_ids))


def test_global_uniform_random_has_balanced_enumerable_subset_frequencies() -> None:
    counts = Counter(
        global_uniform_random(
            population_size=4,
            budget=2,
            **(SEED_ARGUMENTS | {"repetition": repetition}),
        ).retained_visual_ids
        for repetition in range(600)
    )

    # Six subsets should each occur 100 times; +/-30 is over three binomial standard deviations.
    assert set(counts) == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    assert all(70 <= count <= 130 for count in counts.values())


def test_page_stratified_random_exactly_matches_reference_page_quotas() -> None:
    selected = page_stratified_random(
        GEOMETRY,
        reference_retained_ids=(0, 2, 4),
        **(SEED_ARGUMENTS | {"policy": "page-stratified-random"}),
    )

    assert selected.policy_name == "page-stratified-random"
    assert len(selected.retained_visual_ids) == 3
    assert sum(identifier < 4 for identifier in selected.retained_visual_ids) == 2
    assert sum(identifier >= 4 for identifier in selected.retained_visual_ids) == 1
    assert selected.retained_visual_ids == tuple(sorted(set(selected.retained_visual_ids)))


def test_normalized_grid_cells_and_grid_stratification_cover_capacity_and_edge_budgets() -> None:
    assert normalized_grid_cells(GEOMETRY) == (
        (0, 0, 0),
        (0, 0, 3),
        (0, 3, 0),
        (0, 3, 3),
        (1, 1, 1),
        (1, 3, 3),
    )

    sparse = grid_stratified_random(
        GEOMETRY,
        budget=3,
        **(SEED_ARGUMENTS | {"policy": "grid-stratified-random"}),
    )
    full = grid_stratified_random(
        GEOMETRY,
        budget=6,
        **(SEED_ARGUMENTS | {"policy": "grid-stratified-random"}),
    )
    empty = grid_stratified_random(
        GEOMETRY,
        budget=0,
        **(SEED_ARGUMENTS | {"policy": "grid-stratified-random"}),
    )

    assert len(sparse.retained_visual_ids) == len(set(sparse.retained_visual_ids)) == 3
    assert (
        len({normalized_grid_cells(GEOMETRY)[index] for index in sparse.retained_visual_ids}) == 3
    )
    assert full.retained_visual_ids == (0, 1, 2, 3, 4, 5)
    assert empty.retained_visual_ids == ()


def test_grid_stratification_uses_largest_remainder_after_one_per_nonempty_cell() -> None:
    geometry = (
        VisualTokenGeometry(page_index=0, row=0, column=0, height=8, width=8),
        VisualTokenGeometry(page_index=0, row=1, column=1, height=8, width=8),
        VisualTokenGeometry(page_index=0, row=0, column=1, height=8, width=8),
        VisualTokenGeometry(page_index=0, row=3, column=3, height=8, width=8),
        VisualTokenGeometry(page_index=0, row=2, column=2, height=8, width=8),
    )
    selected = grid_stratified_random(
        geometry,
        budget=4,
        **(SEED_ARGUMENTS | {"policy": "grid-stratified-random"}),
    )

    cells = normalized_grid_cells(geometry)
    assert len(selected.retained_visual_ids) == 4
    assert Counter(cells[index] for index in selected.retained_visual_ids) == {
        (0, 0, 0): 2,
        (0, 1, 1): 2,
    }


def test_coverage_matched_identity_shuffle_matches_exact_page_cell_quotas() -> None:
    selected = coverage_matched_identity_shuffle(
        GEOMETRY,
        reference_retained_ids=(0, 1, 4),
        **(SEED_ARGUMENTS | {"policy": "coverage-matched-identity-shuffle"}),
    )

    cells = normalized_grid_cells(GEOMETRY)
    assert selected.policy_name == "coverage-matched-identity-shuffle"
    assert len(selected.retained_visual_ids) == 3
    assert {cells[index] for index in selected.retained_visual_ids} == {
        (0, 0, 0),
        (0, 0, 3),
        (1, 1, 1),
    }


def test_coverage_measurements_include_every_page_cell_and_reference_overlap() -> None:
    measurements = coverage_measurements(
        GEOMETRY,
        retained_ids=(0, 3, 4),
        reference_ids=(0, 2, 4),
        total_pages=2,
    )

    assert len(measurements.page_grid_occupancy[0]) == 16
    assert len(measurements.page_grid_occupancy[1]) == 16
    assert measurements.page_grid_occupancy[0][0] == 1
    assert measurements.page_grid_occupancy[0][15] == 1
    assert measurements.page_grid_occupancy[1][5] == 1
    assert measurements.page_centers == ((0.5, 0.5), (0.25, 0.25))
    assert measurements.page_dispersion[0] == pytest.approx(0.5303300858899106)
    assert measurements.page_dispersion[1] == 0.0
    assert measurements.jaccard_overlap == pytest.approx(0.5)


def test_empty_page_measurements_and_empty_jaccard_have_declared_values() -> None:
    measurements = coverage_measurements(GEOMETRY, retained_ids=(), reference_ids=(), total_pages=2)

    assert measurements.page_centers == (None, None)
    assert measurements.page_dispersion == (None, None)
    assert measurements.jaccard_overlap == 1.0


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        ((VisualTokenGeometry(page_index=-1, row=0, column=0, height=1, width=1),), "page_index"),
        ((VisualTokenGeometry(page_index=0, row=1, column=0, height=1, width=1),), "row"),
        ((VisualTokenGeometry(page_index=0, row=0, column=1, height=1, width=1),), "column"),
        ((VisualTokenGeometry(page_index=0, row=0, column=0, height=0, width=1),), "height"),
    ],
)
def test_geometry_must_describe_in_bounds_positive_grid_locations(geometry, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalized_grid_cells(geometry)


@pytest.mark.parametrize("scores", [(0.1, float("nan")), (0.1, float("inf"))])
def test_score_top_m_rejects_nonfinite_scores(scores: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="finite"):
        score_top_m(scores, budget=1, score_kind="aggregate")


@pytest.mark.parametrize("budget", [-1, 7])
def test_policies_reject_budgets_outside_population(budget: int) -> None:
    with pytest.raises(ValueError, match="budget"):
        global_uniform_random(population_size=6, budget=budget, **SEED_ARGUMENTS)


@pytest.mark.parametrize("reference_ids", [(-1,), (6,), (1, 1)])
def test_reference_ids_must_be_unique_population_members(reference_ids: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="reference_retained_ids"):
        page_stratified_random(
            GEOMETRY,
            reference_retained_ids=reference_ids,
            **(SEED_ARGUMENTS | {"policy": "page-stratified-random"}),
        )


def test_seed_identity_changes_when_any_declared_field_changes() -> None:
    seeds = {
        global_uniform_random(
            population_size=4,
            budget=2,
            **(SEED_ARGUMENTS | {field: value}),
        ).seed_sha256
        for field, value in {
            "experiment_version": "coverage-v2",
            "qid": "q-8",
            "boundary": "after-prefill",
            "policy": "global-uniform-random-fixed-budget",
            "repetition": 4,
        }.items()
    }

    assert len(seeds) == 5


def test_random_policy_names_reject_other_algorithm_namespaces() -> None:
    with pytest.raises(ValueError, match="global-uniform-random"):
        global_uniform_random(
            population_size=4,
            budget=2,
            **(SEED_ARGUMENTS | {"policy": "page-stratified-random-fixed-budget"}),
        )
    with pytest.raises(ValueError, match="page-stratified-random"):
        page_stratified_random(
            GEOMETRY,
            reference_retained_ids=(0, 4),
            **(SEED_ARGUMENTS | {"policy": "global-uniform-random"}),
        )
    with pytest.raises(ValueError, match="grid-stratified-random"):
        grid_stratified_random(
            GEOMETRY,
            budget=2,
            **(SEED_ARGUMENTS | {"policy": "coverage-matched-identity-shuffle"}),
        )
    with pytest.raises(ValueError, match="coverage-matched-identity-shuffle"):
        coverage_matched_identity_shuffle(
            GEOMETRY,
            reference_retained_ids=(0, 4),
            **(SEED_ARGUMENTS | {"policy": "grid-stratified-random"}),
        )


def test_sparse_grid_cell_sampling_is_balanced_across_enumerable_cells() -> None:
    geometry = (
        VisualTokenGeometry(page_index=0, row=0, column=0, height=4, width=4),
        VisualTokenGeometry(page_index=0, row=0, column=3, height=4, width=4),
        VisualTokenGeometry(page_index=0, row=3, column=0, height=4, width=4),
    )
    cells = normalized_grid_cells(geometry)
    counts = Counter(
        cells[selected.retained_visual_ids[0]]
        for repetition in range(600)
        for selected in [
            grid_stratified_random(
                geometry,
                budget=1,
                **(SEED_ARGUMENTS | {"policy": "grid-stratified-random", "repetition": repetition}),
            )
        ]
    )

    # Three cells should each occur 200 times; +/-40 is over three standard deviations.
    assert set(counts) == set(cells)
    assert all(160 <= count <= 240 for count in counts.values())


def test_largest_remainder_fractional_ties_are_seed_balanced_not_cell_ordered() -> None:
    geometry = (
        VisualTokenGeometry(page_index=0, row=0, column=0, height=4, width=4),
        VisualTokenGeometry(page_index=0, row=0, column=0, height=4, width=4),
        VisualTokenGeometry(page_index=0, row=0, column=3, height=4, width=4),
        VisualTokenGeometry(page_index=0, row=0, column=3, height=4, width=4),
        VisualTokenGeometry(page_index=0, row=3, column=0, height=4, width=4),
        VisualTokenGeometry(page_index=0, row=3, column=0, height=4, width=4),
    )
    cells = normalized_grid_cells(geometry)
    extra_cell_counts = Counter(
        next(
            cell
            for cell, count in Counter(
                cells[index] for index in selected.retained_visual_ids
            ).items()
            if count == 2
        )
        for repetition in range(600)
        for selected in [
            grid_stratified_random(
                geometry,
                budget=4,
                **(SEED_ARGUMENTS | {"policy": "grid-stratified-random", "repetition": repetition}),
            )
        ]
    )

    # The one residual token has three equal remainders: 200 expected per cell, +/-40 tolerance.
    assert set(extra_cell_counts) == set(cells)
    assert all(160 <= count <= 240 for count in extra_cell_counts.values())


def test_coverage_measurements_require_total_pages_and_preserve_empty_page_positions() -> None:
    geometry = (VisualTokenGeometry(page_index=1, row=0, column=0, height=2, width=2),)
    measurements = coverage_measurements(geometry, retained_ids=(), total_pages=4)
    fully_empty = coverage_measurements((), retained_ids=(), total_pages=3)

    assert measurements.page_grid_occupancy == ((0,) * 16, (0,) * 16, (0,) * 16, (0,) * 16)
    assert measurements.page_centers == (None, None, None, None)
    assert fully_empty.page_centers == (None, None, None)
    with pytest.raises(ValueError, match="total_pages"):
        coverage_measurements(geometry, retained_ids=(), total_pages=1)


def test_quota_controls_ignore_reference_id_order_and_cover_repeated_cell_quotas() -> None:
    ordered_page = tuple(
        page_stratified_random(
            GEOMETRY,
            reference_retained_ids=(0, 2, 4),
            **(SEED_ARGUMENTS | {"policy": "page-stratified-random", "repetition": repetition}),
        ).retained_visual_ids
        for repetition in range(16)
    )
    reordered_page = tuple(
        page_stratified_random(
            GEOMETRY,
            reference_retained_ids=(4, 2, 0),
            **(SEED_ARGUMENTS | {"policy": "page-stratified-random", "repetition": repetition}),
        ).retained_visual_ids
        for repetition in range(16)
    )
    geometry = (
        VisualTokenGeometry(page_index=0, row=0, column=0, height=4, width=4),
        VisualTokenGeometry(page_index=0, row=0, column=0, height=4, width=4),
        VisualTokenGeometry(page_index=0, row=0, column=0, height=4, width=4),
        VisualTokenGeometry(page_index=0, row=0, column=3, height=4, width=4),
        VisualTokenGeometry(page_index=0, row=0, column=3, height=4, width=4),
        VisualTokenGeometry(page_index=0, row=0, column=3, height=4, width=4),
    )
    ordered_cell = tuple(
        coverage_matched_identity_shuffle(
            geometry,
            reference_retained_ids=(0, 1, 3),
            **(
                SEED_ARGUMENTS
                | {"policy": "coverage-matched-identity-shuffle", "repetition": repetition}
            ),
        ).retained_visual_ids
        for repetition in range(16)
    )
    reordered_cell = tuple(
        coverage_matched_identity_shuffle(
            geometry,
            reference_retained_ids=(3, 1, 0),
            **(
                SEED_ARGUMENTS
                | {"policy": "coverage-matched-identity-shuffle", "repetition": repetition}
            ),
        ).retained_visual_ids
        for repetition in range(16)
    )

    assert ordered_page == reordered_page
    assert ordered_cell == reordered_cell
    assert all(len(selection) == 3 for selection in ordered_cell)
