"""Deterministic visual-token ranking, coverage controls, and measurements."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral, Real

GridCell = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class VisualTokenGeometry:
    """Location of one post-BTP+QTP visual token in its merged page grid."""

    page_index: int
    row: int
    column: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class PolicySelection:
    """A policy's retained visual ordinals and, for random policies, seed digest."""

    policy_name: str
    retained_visual_ids: tuple[int, ...]
    seed_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class CoverageMeasurements:
    """Per-page coverage occupancy, spatial summaries, and optional overlap."""

    page_grid_occupancy: tuple[tuple[int, ...], ...]
    page_centers: tuple[tuple[float, float] | None, ...]
    page_dispersion: tuple[float | None, ...]
    jaccard_overlap: float | None


def _require_budget(budget: int, population_size: int) -> None:
    if not isinstance(budget, Integral) or isinstance(budget, bool):
        raise ValueError("budget must be an integer")
    if budget < 0 or budget > population_size:
        raise ValueError("budget must be between zero and the population size")


def _validate_geometry(geometry: Sequence[VisualTokenGeometry]) -> None:
    for token in geometry:
        for field_name in ("page_index", "row", "column", "height", "width"):
            value = getattr(token, field_name)
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise ValueError(f"{field_name} must be an integer")
        if token.page_index < 0:
            raise ValueError("page_index must be non-negative")
        if token.height <= 0:
            raise ValueError("height must be positive")
        if token.width <= 0:
            raise ValueError("width must be positive")
        if not 0 <= token.row < token.height:
            raise ValueError("row must be within height")
        if not 0 <= token.column < token.width:
            raise ValueError("column must be within width")


def _validate_policy_name(policy: str, namespace: str) -> None:
    if policy != namespace and not policy.startswith(f"{namespace}-"):
        raise ValueError(f"policy must use the {namespace} namespace")


def _seed_rng(
    experiment_version: object,
    qid: object,
    boundary: object,
    policy: str,
    repetition: object,
) -> tuple[str, random.Random]:
    payload = json.dumps(
        [experiment_version, qid, boundary, policy, repetition],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = sha256(payload).hexdigest()
    return digest, random.Random(int(digest, 16))


def _validated_ids(ids: Iterable[int], population_size: int, name: str) -> tuple[int, ...]:
    result = tuple(ids)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique population members")
    if any(
        not isinstance(identifier, Integral)
        or isinstance(identifier, bool)
        or identifier < 0
        or identifier >= population_size
        for identifier in result
    ):
        raise ValueError(f"{name} must contain unique population members")
    return result


def _cell_for(token: VisualTokenGeometry) -> GridCell:
    row = min(3, int(math.floor(4 * (token.row + 0.5) / token.height)))
    column = min(3, int(math.floor(4 * (token.column + 0.5) / token.width)))
    return (int(token.page_index), row, column)


def _cells_by_id(geometry: Sequence[VisualTokenGeometry]) -> tuple[GridCell, ...]:
    _validate_geometry(geometry)
    return tuple(_cell_for(token) for token in geometry)


def _groups_by_cell(cells: Sequence[GridCell]) -> dict[GridCell, list[int]]:
    groups: dict[GridCell, list[int]] = defaultdict(list)
    for identifier, cell in enumerate(cells):
        groups[cell].append(identifier)
    return dict(groups)


def _random_selection(
    policy: str,
    selected: Iterable[int],
    digest: str,
) -> PolicySelection:
    return PolicySelection(policy, tuple(sorted(selected)), digest)


def normalized_grid_cells(geometry: Sequence[VisualTokenGeometry]) -> tuple[GridCell, ...]:
    """Return the normalized 4×4 page-grid cell for every visual token."""

    return _cells_by_id(geometry)


def score_top_m(scores: Sequence[float], budget: int, score_kind: str) -> PolicySelection:
    """Select the highest finite aggregate or literal scores with stable ties."""

    _require_budget(budget, len(scores))
    if score_kind not in {"aggregate", "literal"}:
        raise ValueError("score_kind must be 'aggregate' or 'literal'")
    if any(not isinstance(score, Real) or not math.isfinite(score) for score in scores):
        raise ValueError("scores must be finite")
    selected = sorted(
        identifier
        for identifier, _ in sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))[
            :budget
        ]
    )
    return PolicySelection(f"{score_kind}-score-top-m", tuple(selected))


def global_uniform_random(
    population_size: int,
    budget: int,
    *,
    experiment_version: object,
    qid: object,
    boundary: object,
    policy: str,
    repetition: object,
) -> PolicySelection:
    """Uniformly sample visual IDs without replacement from the full population."""

    if not isinstance(population_size, Integral) or isinstance(population_size, bool):
        raise ValueError("population_size must be an integer")
    if population_size < 0:
        raise ValueError("population_size must be non-negative")
    _require_budget(budget, population_size)
    _validate_policy_name(policy, "global-uniform-random")
    digest, rng = _seed_rng(experiment_version, qid, boundary, policy, repetition)
    return _random_selection(policy, rng.sample(range(population_size), budget), digest)


def page_stratified_random(
    geometry: Sequence[VisualTokenGeometry],
    *,
    reference_retained_ids: Iterable[int],
    experiment_version: object,
    qid: object,
    boundary: object,
    policy: str,
    repetition: object,
) -> PolicySelection:
    """Sample visual IDs while exactly matching reference per-page quotas."""

    cells = _cells_by_id(geometry)
    reference_ids = _validated_ids(reference_retained_ids, len(cells), "reference_retained_ids")
    _validate_policy_name(policy, "page-stratified-random")
    digest, rng = _seed_rng(experiment_version, qid, boundary, policy, repetition)
    page_groups: dict[int, list[int]] = defaultdict(list)
    for identifier, (page, _, _) in enumerate(cells):
        page_groups[page].append(identifier)
    quotas = Counter(cells[identifier][0] for identifier in reference_ids)
    selected: list[int] = []
    for page in sorted(quotas):
        selected.extend(rng.sample(page_groups[page], quotas[page]))
    return _random_selection(policy, selected, digest)


def grid_stratified_random(
    geometry: Sequence[VisualTokenGeometry],
    budget: int,
    *,
    experiment_version: object,
    qid: object,
    boundary: object,
    policy: str,
    repetition: object,
) -> PolicySelection:
    """Sample visual IDs using nonempty normalized grid cells as strata."""

    cells = _cells_by_id(geometry)
    _require_budget(budget, len(cells))
    _validate_policy_name(policy, "grid-stratified-random")
    digest, rng = _seed_rng(experiment_version, qid, boundary, policy, repetition)
    groups = _groups_by_cell(cells)
    keys = sorted(groups)
    if budget == 0:
        return _random_selection(policy, (), digest)
    if budget < len(keys):
        selected_cells = rng.sample(keys, budget)
        selected = [rng.choice(groups[cell]) for cell in selected_cells]
        return _random_selection(policy, selected, digest)

    allocation = {cell: 1 for cell in keys}
    residual = budget - len(keys)
    remaining_capacity = {cell: len(groups[cell]) - 1 for cell in keys}
    total_capacity = sum(remaining_capacity.values())
    if residual and total_capacity:
        exact = {cell: residual * remaining_capacity[cell] / total_capacity for cell in keys}
        for cell in keys:
            allocation[cell] += int(math.floor(exact[cell]))
        extra = residual - sum(int(math.floor(value)) for value in exact.values())
        tied_order = list(keys)
        rng.shuffle(tied_order)
        tie_rank = {cell: rank for rank, cell in enumerate(tied_order)}
        for cell in sorted(keys, key=lambda item: (-math.modf(exact[item])[0], tie_rank[item]))[
            :extra
        ]:
            allocation[cell] += 1
    selected = [
        identifier for cell in keys for identifier in rng.sample(groups[cell], allocation[cell])
    ]
    return _random_selection(policy, selected, digest)


def coverage_matched_identity_shuffle(
    geometry: Sequence[VisualTokenGeometry],
    *,
    reference_retained_ids: Iterable[int],
    experiment_version: object,
    qid: object,
    boundary: object,
    policy: str,
    repetition: object,
) -> PolicySelection:
    """Sample IDs while exactly matching reference page-by-cell quotas."""

    cells = _cells_by_id(geometry)
    reference_ids = _validated_ids(reference_retained_ids, len(cells), "reference_retained_ids")
    _validate_policy_name(policy, "coverage-matched-identity-shuffle")
    digest, rng = _seed_rng(experiment_version, qid, boundary, policy, repetition)
    groups = _groups_by_cell(cells)
    quotas = Counter(cells[identifier] for identifier in reference_ids)
    selected = [
        identifier
        for cell in sorted(quotas)
        for identifier in rng.sample(groups[cell], quotas[cell])
    ]
    return _random_selection(policy, selected, digest)


def coverage_measurements(
    geometry: Sequence[VisualTokenGeometry],
    *,
    retained_ids: Iterable[int],
    total_pages: int,
    reference_ids: Iterable[int] | None = None,
) -> CoverageMeasurements:
    """Measure per-page 4×4 occupancy, center, dispersion, and overlap."""

    cells = _cells_by_id(geometry)
    if not isinstance(total_pages, Integral) or isinstance(total_pages, bool) or total_pages < 0:
        raise ValueError("total_pages must be a non-negative integer")
    if any(cell[0] >= total_pages for cell in cells):
        raise ValueError("geometry page_index must be less than total_pages")
    retained = _validated_ids(retained_ids, len(cells), "retained_ids")
    reference = (
        None
        if reference_ids is None
        else _validated_ids(reference_ids, len(cells), "reference_ids")
    )
    occupancy = [[0] * 16 for _ in range(total_pages)]
    positions: list[list[tuple[float, float]]] = [[] for _ in range(total_pages)]
    for identifier in retained:
        token = geometry[identifier]
        page, row, column = cells[identifier]
        occupancy[page][row * 4 + column] += 1
        positions[page].append(
            ((token.row + 0.5) / token.height, (token.column + 0.5) / token.width)
        )
    centers: list[tuple[float, float] | None] = []
    dispersions: list[float | None] = []
    for page_positions in positions:
        if not page_positions:
            centers.append(None)
            dispersions.append(None)
            continue
        center = (
            sum(row for row, _ in page_positions) / len(page_positions),
            sum(column for _, column in page_positions) / len(page_positions),
        )
        centers.append(center)
        dispersions.append(
            math.sqrt(
                sum(
                    (row - center[0]) ** 2 + (column - center[1]) ** 2
                    for row, column in page_positions
                )
                / len(page_positions)
            )
        )
    overlap = None
    if reference is not None:
        retained_set, reference_set = set(retained), set(reference)
        union = retained_set | reference_set
        overlap = 1.0 if not union else len(retained_set & reference_set) / len(union)
    return CoverageMeasurements(
        tuple(tuple(page) for page in occupancy), tuple(centers), tuple(dispersions), overlap
    )
