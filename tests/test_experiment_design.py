"""Real-behavior tests for the sealed experiment-design contracts."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from docprune import experiment_design

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_REGISTRY = (
    ROOT
    / "docs"
    / "experiments"
    / "task9-contextcite"
    / "development-qid-registry.json"
)
TASK9_PRELIMINARY_SEALER = ROOT / "examples" / "seal_task9_preliminary_random48.py"


def _confirmation_row(qid: str, prediction: str = "wrong") -> dict[str, object]:
    return {
        "question_id": qid,
        "question": f"Question {qid}",
        "answers": ["gold"],
        "predicted_answer": prediction,
        "retrieved_pages": [
            {"doc_id": f"retrieved-{qid}-{index}", "page_index": index, "score": 4 - index}
            for index in range(4)
        ],
    }


def test_task9_confirmation_prefers_independent_support_documents() -> None:
    """Catch selecting two confirmation questions that share a support document."""

    rows = [
        _confirmation_row("correct", "gold"),
        *[_confirmation_row(qid) for qid in ("q1", "q2", "q3", "q4", "excluded")],
    ]
    supports = [
        {"qid": "correct", "supporting_document_ids": ["doc-correct"]},
        {"qid": "q1", "supporting_document_ids": ["doc-shared"]},
        {"qid": "q2", "supporting_document_ids": ["doc-shared"]},
        {"qid": "q3", "supporting_document_ids": ["doc-three"]},
        {"qid": "q4", "supporting_document_ids": ["doc-four"]},
        {"qid": "excluded", "supporting_document_ids": ["doc-excluded"]},
    ]

    cohort = experiment_design.build_task9_baseline_wrong_confirmation_cohort(
        rows,
        eligibility_records=supports,
        excluded_qids=["excluded"],
        seed="confirmation-v1",
        sample_size=3,
    )

    assert cohort["eligible_baseline_wrong_count"] == 4
    assert cohort["selection_used_document_fallback"] is False
    assert len(cohort["selected_records"]) == 3
    selected_supports = [
        set(record["supporting_document_ids"]) for record in cohort["selected_records"]
    ]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(selected_supports)
        for right in selected_supports[index + 1 :]
    )
    assert all(record["baseline_em"] == 0.0 for record in cohort["selected_records"])
    assert all(len(record["retrieved_pages"]) == 4 for record in cohort["selected_records"])


def test_task9_confirmation_records_support_cluster_fallback() -> None:
    """Catch silently pretending questions are document-independent after fallback fill."""

    rows = [_confirmation_row(qid) for qid in ("q1", "q2", "q3")]
    supports = [
        {"qid": "q1", "supporting_document_ids": ["doc-shared"]},
        {"qid": "q2", "supporting_document_ids": ["doc-shared"]},
        {"qid": "q3", "supporting_document_ids": ["doc-three"]},
    ]

    cohort = experiment_design.build_task9_baseline_wrong_confirmation_cohort(
        rows,
        eligibility_records=supports,
        excluded_qids=[],
        seed="confirmation-v1",
        sample_size=3,
    )

    assert cohort["selection_used_document_fallback"] is True
    assert cohort["selected_qids"] == ["q3", "q1", "q2"]
    assert len(cohort["support_components"]["components"]) == 2


def test_task9_confirmation_fixture_reuses_exact_cached_top4() -> None:
    """Catch reordering or replacing the four sealed pages during fixture projection."""

    rows = [_confirmation_row(qid) for qid in ("q1", "q2")]
    supports = [
        {"qid": "q1", "supporting_document_ids": ["doc-one"]},
        {"qid": "q2", "supporting_document_ids": ["doc-two"]},
    ]
    cohort = experiment_design.build_task9_baseline_wrong_confirmation_cohort(
        rows,
        eligibility_records=supports,
        excluded_qids=[],
        seed="confirmation-v1",
        sample_size=2,
    )

    reference, eligible = experiment_design.build_task9_confirmation_fixture_inputs(cohort)

    assert reference["question_ids"] == cohort["selected_qids"]
    assert eligible == [
        {
            "qid": record["question_id"],
            "question": record["question"],
        }
        for record in cohort["selected_records"]
    ]
    assert all(
        reference["rows"][record["question_id"]]["retrieved_pages"]
        == record["retrieved_pages"]
        for record in cohort["selected_records"]
    )


def test_experiment_design_is_a_dedicated_module() -> None:
    """Catch coupling Task 2 back into evaluation or model-loading modules."""

    assert importlib.util.find_spec("docprune.experiment_design") is not None


def test_task9_preliminary_cohort_uses_exact_seeded_correct_wrong_strata() -> None:
    """Catch non-random ordering or leakage across the baseline-EM strata."""

    rows = [
        {
            "question_id": qid,
            "question": f"Question {qid}",
            "answers": ["gold"],
            "predicted_answer": prediction,
            "retrieved_pages": [],
        }
        for qid, prediction in (
            ("c1", "gold"),
            ("c2", "gold"),
            ("c3", "gold"),
            ("c4", "gold"),
            ("w1", "wrong"),
            ("w2", "wrong"),
            ("w3", "wrong"),
            ("w4", "wrong"),
        )
    ]

    cohort = experiment_design.build_task9_preliminary_random_cohort(
        rows,
        seed="seed-v1",
        per_stratum=2,
    )

    assert cohort["eligible_counts"] == {"baseline_correct": 4, "baseline_wrong": 4}
    assert cohort["natural_pool_weights"] == {
        "baseline_correct": 0.5,
        "baseline_wrong": 0.5,
    }
    assert cohort["selected_qids"] == {
        "baseline_correct": ["c4", "c3"],
        "baseline_wrong": ["w1", "w2"],
    }
    assert [row["question_id"] for row in cohort["selected_records"]] == [
        "c4",
        "c3",
        "w1",
        "w2",
    ]
    assert [row["baseline_stratum"] for row in cohort["selected_records"]] == [
        "baseline_correct",
        "baseline_correct",
        "baseline_wrong",
        "baseline_wrong",
    ]


def test_task9_preliminary_cohort_rejects_duplicate_or_undersized_pools() -> None:
    """Catch a cohort seal that cannot supply unique members to both strata."""

    row = {
        "question_id": "q1",
        "question": "Question",
        "answers": ["gold"],
        "predicted_answer": "gold",
        "retrieved_pages": [],
    }
    with pytest.raises(ValueError, match="duplicate question_id"):
        experiment_design.build_task9_preliminary_random_cohort(
            [row, row], seed="seed-v1", per_stratum=1
        )

    with pytest.raises(ValueError, match="baseline_wrong.*requires 1"):
        experiment_design.build_task9_preliminary_random_cohort(
            [row], seed="seed-v1", per_stratum=1
        )


def test_task9_preliminary_fixture_inputs_preserve_selected_order_and_pages() -> None:
    """Catch retrieval, reordering, or outcome-label loss while materializing fixed pages."""

    records = []
    selected_qids = {"baseline_correct": [], "baseline_wrong": []}
    for index in range(48):
        stratum = "baseline_correct" if index < 24 else "baseline_wrong"
        qid = f"q{index:02d}"
        selected_qids[stratum].append(qid)
        records.append(
            {
                "question_id": qid,
                "question": f"Question {index}",
                "answers": ["gold"],
                "unpruned_predicted_answer": "gold" if index < 24 else "wrong",
                "retrieved_pages": [
                    {"doc_id": f"d{index:02d}-{rank}", "page_index": rank, "score": 4 - rank}
                    for rank in range(4)
                ],
                "baseline_em": 1.0 if index < 24 else 0.0,
                "baseline_stratum": stratum,
            }
        )
    cohort = {
        "selected_qids": selected_qids,
        "selected_records": records,
    }
    cohort["cohort_sha256"] = experiment_design._canonical_json_sha256(cohort)

    reference, eligible = experiment_design.build_task9_preliminary_fixture_inputs(cohort)

    assert reference["question_ids"] == [row["question_id"] for row in records]
    assert reference["fixed_page_selection_is_outcome_blind"] is True
    assert reference["question_selection"] == "24-baseline-correct/24-baseline-wrong"
    assert reference["rows"]["q00"]["retrieved_pages"] == records[0]["retrieved_pages"]
    assert eligible[0] == {"qid": "q00", "question": "Question 0"}
    assert eligible[-1] == {"qid": "q47", "question": "Question 47"}


def test_task9_preprocessing_batches_cover_remaining_47_questions_four_at_a_time() -> None:
    batches = [
        experiment_design.task9_preprocessing_batch_indices(array_task_id)
        for array_task_id in range(12)
    ]

    assert batches[0] == (1, 2, 3, 4)
    assert batches[-1] == (45, 46, 47)
    assert [index for batch in batches for index in batch] == list(range(1, 48))

    with pytest.raises(ValueError, match="array task"):
        experiment_design.task9_preprocessing_batch_indices(12)


def test_task9_preliminary_sealer_authenticates_sources_and_writes_24_per_stratum(
    tmp_path: Path,
) -> None:
    """Catch sealing from a partial pool or silently replacing an existing cohort."""

    first64 = tmp_path / "first64.jsonl"
    incremental_root = tmp_path / "incremental"
    incremental_results = (
        incremental_root / "eval-shards" / "btp-qtp" / "shard-0004" / "run" / "results.jsonl"
    )
    incremental_results.parent.mkdir(parents=True)
    rows = [
        {
            "question_id": f"q{index:03d}",
            "question": f"Question {index}",
            "answers": ["gold"],
            "predicted_answer": "gold" if index < 90 else "wrong",
            "retrieved_pages": [],
        }
        for index in range(245)
    ]
    first64.write_text(
        "".join(json.dumps(row) + "\n" for row in rows[:64]), encoding="utf-8"
    )
    incremental_results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows[64:]), encoding="utf-8"
    )
    analysis = {
        "schema_version": 1,
        "analysis_sha256": "a" * 64,
        "question_count": 245,
        "question_ids": [row["question_id"] for row in rows],
        "question_ids_sha256": "b" * 64,
        "retrieval_is_identical_across_stages": True,
        "source_roots": {
            "btp_qtp_first64": [str(first64)],
            "incremental": str(incremental_root / "eval-shards"),
        },
    }
    analysis_path = tmp_path / "stage245-analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    output = tmp_path / "cohort.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(TASK9_PRELIMINARY_SEALER),
            "--analysis",
            str(analysis_path),
            "--output",
            str(output),
            "--seed",
            "task9-preliminary-test-seed",
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    cohort = json.loads(output.read_text(encoding="utf-8"))
    assert cohort["eligible_counts"] == {"baseline_correct": 90, "baseline_wrong": 155}
    assert {name: len(qids) for name, qids in cohort["selected_qids"].items()} == {
        "baseline_correct": 24,
        "baseline_wrong": 24,
    }
    assert cohort["source_authority"]["stage_analysis_path"] == str(analysis_path)
    assert len(cohort["source_authority"]["result_files"]) == 2

    repeated = subprocess.run(
        completed.args,
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert repeated.returncode != 0
    assert "already exists" in repeated.stderr


def test_external_file_hashing_streams_multiple_chunks_without_whole_file_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch buffering a large bound feature before computing its digest."""

    content = b"docprune-streaming-fixture\x00" * 120_000
    feature_path = tmp_path / "large-feature.bin"
    feature_path.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    real_os_read = experiment_design.os.read
    real_sha256 = hashlib.sha256
    requested_sizes: list[int] = []
    update_sizes: list[int] = []

    def reject_whole_file_reader(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("streaming hash must not call a whole-file reader")

    def observed_os_read(descriptor: int, size: int) -> bytes:
        requested_sizes.append(size)
        assert size < len(content)
        return real_os_read(descriptor, size)

    class ObservedSha256:
        def __init__(self, initial: bytes = b"") -> None:
            self._digest = real_sha256()
            if initial:
                self.update(initial)

        def update(self, chunk: bytes) -> None:
            update_sizes.append(len(chunk))
            assert len(chunk) < len(content)
            self._digest.update(chunk)

        def hexdigest(self) -> str:
            return self._digest.hexdigest()

    monkeypatch.setattr(experiment_design, "_read_regular_file", reject_whole_file_reader)
    monkeypatch.setattr(Path, "read_bytes", reject_whole_file_reader)
    monkeypatch.setattr(experiment_design.os, "read", observed_os_read)
    monkeypatch.setattr(experiment_design.hashlib, "sha256", ObservedSha256)

    actual = experiment_design._sha256_regular_file(feature_path, label="persisted feature")

    assert actual == expected
    assert len(requested_sizes) >= 4
    assert max(requested_sizes) == 1024 * 1024
    assert len(update_sizes) >= 4
    assert max(update_sizes) == 1024 * 1024


def test_eligibility_rejects_outcome_bearing_input() -> None:
    """Catch candidate scores leaking into method-holdout eligibility."""

    with pytest.raises(ValueError, match="forbidden eligibility key.*f1"):
        experiment_design.build_eligibility_records(
            [
                {
                    "qid": "q1",
                    "metadata": {
                        "type": "single_hop",
                        "supporting_document_ids": ["doc-a"],
                        "source_path": "/immutable/questions.jsonl",
                        "source_sha256": "1" * 64,
                    },
                    "cached_pages": [],
                    "persisted_features": [],
                    "f1": 100.0,
                }
            ],
            development_qids=(),
        )


def test_eligibility_rejects_nested_outcome_keys_even_for_excluded_hop_types() -> None:
    """Catch outcome fields hidden in rows that would otherwise be skipped."""

    row = _eligible_row("multi-hop-qid")
    row["metadata"]["type"] = "multi_hop"
    row["cached_pages"][0]["f1"] = 99.0

    with pytest.raises(ValueError, match="forbidden cached page key.*f1"):
        experiment_design.build_eligibility_records([row], development_qids=())


def _eligible_row(qid: str) -> dict[str, object]:
    pages = [
        {
            "doc_id": f"doc-{index}",
            "page_index": index,
            "source_path": "/immutable/cached-pages.json",
            "source_sha256": "2" * 64,
        }
        for index in range(4)
    ]
    features = [
        {
            "doc_id": f"doc-{index}",
            "page_index": index,
            "feature_path": f"/immutable/features/doc-{index}-{index}.safetensors",
            "feature_sha256": "3" * 64,
            "index_manifest_path": "/immutable/index/manifest.json",
            "index_manifest_sha256": "4" * 64,
        }
        for index in range(4)
    ]
    return {
        "qid": qid,
        "metadata": {
            "type": "single_hop",
            "supporting_document_ids": ["support-a"],
            "source_path": "/immutable/questions.jsonl",
            "source_sha256": "1" * 64,
        },
        "cached_pages": pages,
        "persisted_features": features,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authenticated_eligible_rows(
    root: Path, qids: tuple[str, ...]
) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Create real byte sources for sealing without retrieval or model access."""

    root.mkdir()
    question_source = root / "questions.jsonl"
    page_source = root / "cached-pages.json"
    index_manifest = root / "index-manifest.json"
    question_source.write_bytes(b'{"qid":"fixture"}\n')
    page_source.write_bytes(b'{"pages":[0,1,2,3]}\n')
    index_manifest.write_bytes(b'{"schema_version":1}\n')
    rows: list[dict[str, object]] = []
    for qid in qids:
        row = _eligible_row(qid)
        row["metadata"]["source_path"] = str(question_source)
        row["metadata"]["source_sha256"] = _sha256_file(question_source)
        for index, page in enumerate(row["cached_pages"]):
            page["source_path"] = str(page_source)
            page["source_sha256"] = _sha256_file(page_source)
            feature_path = root / f"{qid}-feature-{index}.bin"
            feature_path.write_bytes(f"{qid}:{index}\n".encode())
            feature = row["persisted_features"][index]
            feature["feature_path"] = str(feature_path)
            feature["feature_sha256"] = _sha256_file(feature_path)
            feature["index_manifest_path"] = str(index_manifest)
            feature["index_manifest_sha256"] = _sha256_file(index_manifest)
        rows.append(row)
    return rows, {str(question_source): _sha256_file(question_source)}


def _write_test_registry(
    root: Path,
    *,
    labels: tuple[str, ...] = ("development",),
    qids: tuple[str, ...] = ("dev-qid",),
) -> Path:
    root.mkdir()
    projections = []
    for label in labels:
        source = root / f"{label}.json"
        source.write_bytes((json.dumps({"qids": list(qids)}) + "\n").encode())
        projections.append(
            experiment_design.development_qid_projection(
                diagnostic_label=label,
                source_path=str(source),
                source_sha256=_sha256_file(source),
                qids=qids,
            )
        )
    registry = experiment_design.build_development_registry(projections, required_labels=labels)
    registry_path = root / "development-qid-registry.json"
    registry_path.write_text(
        json.dumps(registry, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return registry_path


def _seal_prerequisites(
    tmp_path: Path, *, eligible_pool_size: int
) -> tuple[Path, tuple[str, ...], dict[str, object]]:
    labels = ("development",)
    registry_path = _write_test_registry(tmp_path / "registry", labels=labels)
    power = experiment_design.plan_equivalence_power(
        developmental_mean_f1=0.0,
        developmental_sd_f1=2.0,
        cluster_design_effect=1.0,
        eligible_pool_size=eligible_pool_size,
    )
    return registry_path, labels, power


def _resign_manifest(path: Path, payload: dict[str, object]) -> None:
    payload.pop("manifest_sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _seal_valid_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, tuple[str, ...], list[dict[str, object]], dict[str, str]]:
    registry_path, labels, _ = _seal_prerequisites(tmp_path, eligible_pool_size=2)
    power = experiment_design.plan_equivalence_power(
        developmental_mean_f1=0.0,
        developmental_sd_f1=0.01,
        cluster_design_effect=1.0,
        eligible_pool_size=2,
    )
    rows, input_hashes = _authenticated_eligible_rows(
        tmp_path / "valid-inputs", ("holdout-a", "holdout-b")
    )
    rows[0]["metadata"]["supporting_document_ids"] = ["support-a"]
    rows[1]["metadata"]["supporting_document_ids"] = ["support-b"]
    destination = tmp_path / "method-holdout"
    experiment_design.seal_holdout(
        rows,
        development_registry_path=registry_path,
        required_registry_labels=labels,
        power=power,
        calibration=experiment_design.calibrate_random_repetitions({"dev-qid": [1.0] * 10}),
        runtime_pins={"runtime_commit": "d" * 40},
        input_file_hashes=input_hashes,
        destination=destination,
    )
    return destination, registry_path, labels, rows, input_hashes


def test_eligibility_authenticates_top4_and_excludes_development_union() -> None:
    """Catch selection that retains a developmental QID or loses fixed-page proof."""

    records = experiment_design.build_eligibility_records(
        [_eligible_row("development-qid"), _eligible_row("holdout-qid")],
        development_qids=("development-qid",),
    )
    assert records == (
        {
            "qid": "holdout-qid",
            "source": {
                "path": "/immutable/questions.jsonl",
                "sha256": "1" * 64,
                "type": "single_hop",
            },
            "supporting_document_ids": ["support-a"],
            "cached_pages": [
                {
                    "doc_id": f"doc-{index}",
                    "page_index": index,
                    "source_path": "/immutable/cached-pages.json",
                    "source_sha256": "2" * 64,
                }
                for index in range(4)
            ],
            "persisted_features": [
                {
                    "doc_id": f"doc-{index}",
                    "page_index": index,
                    "feature_path": f"/immutable/features/doc-{index}-{index}.safetensors",
                    "feature_sha256": "3" * 64,
                    "index_manifest_path": "/immutable/index/manifest.json",
                    "index_manifest_sha256": "4" * 64,
                }
                for index in range(4)
            ],
        },
    )


def test_holdout_order_uses_exact_version_bytes_and_is_input_order_independent() -> None:
    """Catch a delimiter/prefix mistake or accidental dependence on source order."""

    ordered = experiment_design.holdout_order(["qid-c", "qid-b", "qid-a"])

    assert ordered == (
        {
            "qid": "qid-a",
            "order_sha256": "55dac960ac91723d1fa38360eaf00b5ad0d62d5e8b36fc002586d7542875c530",
        },
        {
            "qid": "qid-b",
            "order_sha256": "9950bbce29cc254f08a3a0ce854583dc0f3925cf94a77ecc1f6800f9f1ce8e7c",
        },
        {
            "qid": "qid-c",
            "order_sha256": "b975f5a760cf3cac8a79754df8cd338a0135c144a85a54b3bc43c1e0973a6991",
        },
    )


def test_support_components_are_transitive_and_report_size_distribution() -> None:
    """Catch IID clustering or failure to close shared-document chains."""

    result = experiment_design.support_document_components(
        [
            {"qid": "q4", "supporting_document_ids": ["doc-z"]},
            {"qid": "q2", "supporting_document_ids": ["doc-a", "doc-b"]},
            {"qid": "q1", "supporting_document_ids": ["doc-a"]},
            {"qid": "q3", "supporting_document_ids": ["doc-b"]},
        ]
    )

    assert result == {
        "component_id_rule": "sha256(newline-joined sorted QIDs)",
        "components": [
            {
                "component_id": "c587492ba28757a9a5bc87417c913c5d998d34849aa2d3bbf2eb6a7cd423bc0a",
                "qids": ["q1", "q2", "q3"],
            },
            {
                "component_id": "112f2dfa31205df3f5f9db109460c8b85067d7dc27f70781c2e4ef903ee9f26a",
                "qids": ["q4"],
            },
        ],
        "size_distribution": {"1": 1, "3": 1},
    }


def test_primary_estimand_averages_random_masks_within_qid_before_qids() -> None:
    """Catch pooling masks across QIDs or failing to pair by QID."""

    estimate = experiment_design.primary_f1_estimand(
        [
            {"qid": "q1", "aggregate_f1": 5.0, "global_random_f1": [1.0, 3.0]},
            {"qid": "q2", "aggregate_f1": 7.0, "global_random_f1": [4.0, 6.0]},
        ]
    )

    assert estimate == pytest.approx(2.5)


def test_nested_bootstrap_resamples_masks_then_support_components_deterministically() -> None:
    """Catch IID-QID resampling, missing inner draws, or nondeterministic seeds."""

    rows = [
        {"qid": "q1", "aggregate_f1": 4.0, "global_random_f1": [0.0, 2.0]},
        {"qid": "q2", "aggregate_f1": 5.0, "global_random_f1": [1.0, 3.0]},
        {"qid": "q3", "aggregate_f1": 6.0, "global_random_f1": [0.0, 4.0]},
    ]
    first = experiment_design.nested_f1_inference(
        rows, components=(("q1", "q2"), ("q3",)), draws=8, seed=7
    )
    second = experiment_design.nested_f1_inference(
        rows, components=(("q1", "q2"), ("q3",)), draws=8, seed=7
    )

    assert first == second
    assert first["point_estimate"] == pytest.approx(10.0 / 3.0)
    # Hand-derived from Random(7): draws 1, 2, and 5 repeat outer component 0,
    # proving one inner mask-resample per QID is reused for each repeated occurrence.
    assert first["bootstrap_draws"] == pytest.approx(
        (4.0, 3.5, 3.5, 4.0, 13.0 / 3.0, 2.0, 8.0 / 3.0, 10.0 / 3.0)
    )


def test_visual_state_removal_uses_simultaneous_cluster_bounds_and_persistence() -> None:
    """Catch pointwise bounds or declaring persistence before a later boundary fails."""

    boundaries = ("B_input", "B_0", "B_6", "B_13", "B_20", "B_23", "B_26")
    first_differences = (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0)
    second_differences = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    rows = [
        {
            "qid": "q1",
            "reference_f1": 10.0,
            "all_drop_f1": {
                boundary: 10.0 + difference
                for boundary, difference in zip(boundaries, first_differences, strict=True)
            },
        },
        {
            "qid": "q2",
            "reference_f1": 10.0,
            "all_drop_f1": {
                boundary: 10.0 + difference
                for boundary, difference in zip(boundaries, second_differences, strict=True)
            },
        },
    ]

    result = experiment_design.visual_state_removal_inference(
        rows,
        components=(("q1",), ("q2",)),
        draws=4,
        seed=3,
    )

    assert result["estimand"] == "mean_qid(all_drop_f1 - btp_qtp_no_ctp_f1)"
    assert result["method"] == (
        "one-sided 95% nonstudentized basic max-statistic support-component bootstrap"
    )
    assert result["confidence_level"] == 0.95
    assert result["boundaries"] == list(boundaries)
    assert result["point_estimates"] == pytest.approx(
        dict(zip(boundaries, (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0), strict=True))
    )
    assert result["max_statistic_critical"] == pytest.approx(1.0)
    assert result["simultaneous_lower_bounds"] == pytest.approx(
        dict(zip(boundaries, first_differences, strict=True))
    )
    assert result["dependence_margin_f1"] == 1.0
    assert result["persistent_boundary"] == "B_23"
    assert result["nonmonotonic_adjacent_pairs"] == []
    assert result["draw_count"] == 4
    assert result["seed"] == 3


def test_visual_state_persistence_rejects_earlier_pass_when_a_later_boundary_fails() -> None:
    """Catch declaring a boundary persistent without checking every later tested point."""

    boundaries = ("B_input", "B_0", "B_6", "B_13", "B_20", "B_23", "B_26")
    differences = (-4.0, -3.0, -2.0, 0.0, -0.5, -2.0, 0.0)
    all_drop = {
        boundary: 10.0 + difference
        for boundary, difference in zip(boundaries, differences, strict=True)
    }

    result = experiment_design.visual_state_removal_inference(
        [
            {"qid": "q1", "reference_f1": 10.0, "all_drop_f1": all_drop},
            {"qid": "q2", "reference_f1": 10.0, "all_drop_f1": all_drop},
        ],
        components=(("q1",), ("q2",)),
        draws=2,
        seed=1,
    )

    assert result["persistent_boundary"] == "B_26"
    assert result["nonmonotonic_adjacent_pairs"] == [
        {"earlier": "B_13", "later": "B_20"},
        {"earlier": "B_20", "later": "B_23"},
    ]


def test_visual_state_basic_lower_bound_uses_bootstrap_minus_point_maximum() -> None:
    """Catch reversing the centered basic-bootstrap statistic on asymmetric clusters."""

    boundaries = ("B_input", "B_0", "B_6", "B_13", "B_20", "B_23", "B_26")
    differences = (-3.0, -3.0, -3.0, 1.0)
    rows = [
        {
            "qid": f"q{index}",
            "reference_f1": 10.0,
            "all_drop_f1": {boundary: 10.0 + difference for boundary in boundaries},
        }
        for index, difference in enumerate(differences)
    ]

    result = experiment_design.visual_state_removal_inference(
        rows,
        components=(("q0",), ("q1",), ("q2",), ("q3",)),
        draws=4,
        seed=0,
    )

    assert result["point_estimates"] == pytest.approx({boundary: -2.0 for boundary in boundaries})
    assert result["max_statistic_critical"] == pytest.approx(1.85)
    assert result["simultaneous_lower_bounds"] == pytest.approx(
        {boundary: -3.85 for boundary in boundaries}
    )


def test_visual_state_opportunity_strata_require_correct_to_incorrect_or_gold_loss() -> None:
    """Catch treating a change between two wrong answers as visual sensitivity."""

    def row(
        qid: str,
        *,
        supports: list[str],
        retrieved: list[str],
        reference_em: bool,
        reference_f1: float,
        input_em: bool,
        gold_drop: float,
    ) -> dict[str, object]:
        return {
            "qid": qid,
            "supporting_document_ids": supports,
            "retrieved_document_ids": retrieved,
            "reference": {
                "name": "btp-qtp-no-ctp",
                "result_sha256": "a" * 64,
                "em_correct": reference_em,
                "f1": reference_f1,
            },
            "input_all_drop": {
                "name": "all-visual-drop-B_input",
                "boundary": "B_input",
                "mode": "physical_delete",
                "retained_visual_ids": [],
                "result_sha256": "b" * 64,
                "em_correct": input_em,
                "best_reference_loglikelihood_drop_per_token": gold_drop,
                "likelihood_target": "best-reference-full-gold-sequence",
                "likelihood_target_sha256": "c" * 64,
            },
        }

    result = experiment_design.visual_state_opportunity_strata(
        [
            row(
                "correct-to-wrong",
                supports=["doc-a"],
                retrieved=["doc-z", "doc-a", "doc-b", "doc-c"],
                reference_em=True,
                reference_f1=100.0,
                input_em=False,
                gold_drop=0.0,
            ),
            row(
                "wrong-to-other-wrong",
                supports=["doc-missing"],
                retrieved=["doc-z", "doc-a", "doc-b", "doc-c"],
                reference_em=False,
                reference_f1=0.0,
                input_em=False,
                gold_drop=0.09,
            ),
            row(
                "gold-loss",
                supports=["doc-b", "doc-c"],
                retrieved=["doc-z", "doc-c", "doc-b", "doc-a"],
                reference_em=False,
                reference_f1=25.0,
                input_em=False,
                gold_drop=0.1,
            ),
        ]
    )

    assert result == {
        "full": ["correct-to-wrong", "wrong-to-other-wrong", "gold-loss"],
        "support_document_retrieved": ["correct-to-wrong", "gold-loss"],
        "no_ctp_em_correct": ["correct-to-wrong"],
        "no_ctp_f1_positive": ["correct-to-wrong", "gold-loss"],
        "input_visually_sensitive": ["correct-to-wrong", "gold-loss"],
        "input_visual_sensitivity_rule": (
            "(reference_em_correct and not input_all_drop_em_correct) or "
            "best_reference_loglikelihood_drop_per_token >= 0.1"
        ),
    }


def test_task7_report_derives_components_curve_and_strata_together() -> None:
    """Catch reporting the artifact-bound curve as an information horizon."""

    boundaries = ("B_input", "B_0", "B_6", "B_13", "B_20", "B_23", "B_26")
    curve_rows = [
        {
            "qid": qid,
            "reference_f1": 50.0,
            "all_drop_f1": {boundary: score for boundary in boundaries},
        }
        for qid, score in (("q1", 45.0), ("q2", 50.0))
    ]

    def opportunity(qid: str, support: str, sensitive: bool) -> dict[str, object]:
        return {
            "qid": qid,
            "supporting_document_ids": [support],
            "retrieved_document_ids": [support, "d2", "d3", "d4"],
            "reference": {
                "name": "btp-qtp-no-ctp",
                "result_sha256": "a" * 64,
                "em_correct": True,
                "f1": 50.0,
            },
            "input_all_drop": {
                "name": "all-visual-drop-B_input",
                "boundary": "B_input",
                "mode": "physical_delete",
                "retained_visual_ids": [],
                "result_sha256": "b" * 64,
                "em_correct": not sensitive,
                "best_reference_loglikelihood_drop_per_token": 0.1 if sensitive else 0.0,
                "likelihood_target": "best-reference-full-gold-sequence",
                "likelihood_target_sha256": "c" * 64,
            },
        }

    def bundle(
        curve_row: dict[str, object], opportunity_row: dict[str, object], marker: str
    ) -> dict[str, object]:
        provenance = {
            "fixture_sha256": marker * 64,
            "run_manifest_file_sha256": "d" * 64,
            "run_manifest_sha256": "e" * 64,
            "results_file_sha256": "f" * 64,
            "likelihood_file_sha256": "1" * 64,
            "reference_result_sha256": opportunity_row["reference"]["result_sha256"],
            "input_all_drop_result_sha256": opportunity_row["input_all_drop"]["result_sha256"],
        }
        value = {
            "schema_version": 1,
            "qid": curve_row["qid"],
            "curve": {"provenance": dict(provenance), "row": curve_row},
            "opportunity": {"provenance": dict(provenance), "row": opportunity_row},
        }
        value["analysis_bundle_sha256"] = experiment_design._canonical_json_sha256(value)
        return value

    bundles = [
        bundle(curve_rows[0], opportunity("q1", "doc-a", True), "2"),
        bundle(curve_rows[1], opportunity("q2", "doc-b", False), "3"),
    ]
    report = experiment_design.compile_task7_visual_state_report(bundles, draws=4, seed=3)

    assert report["analysis_name"] == "explicit-visual-state-removal-dependence-curve"
    assert report["claim_boundary"] == (
        "not-an-information-horizon-without-separate-standalone-token-information"
    )
    assert report["support_components"]["size_distribution"] == {"1": 2}
    assert report["curve"]["draw_count"] == 4
    assert report["opportunity_strata"]["input_visually_sensitive"] == ["q1"]
    assert len(report["curve_rows_sha256"]) == 64
    assert len(report["opportunity_rows_sha256"]) == 64

    different_artifact = deepcopy(bundles[0])
    different_artifact["opportunity"]["provenance"]["results_file_sha256"] = "4" * 64
    different_artifact.pop("analysis_bundle_sha256")
    different_artifact["analysis_bundle_sha256"] = experiment_design._canonical_json_sha256(
        different_artifact
    )
    with pytest.raises(ValueError, match="same exact artifact provenance"):
        experiment_design.compile_task7_visual_state_report(
            [different_artifact, bundles[1]], draws=4, seed=3
        )

    different_reference_f1 = deepcopy(bundles[0])
    different_reference_f1["opportunity"]["row"]["reference"]["f1"] = 75.0
    different_reference_f1.pop("analysis_bundle_sha256")
    different_reference_f1["analysis_bundle_sha256"] = experiment_design._canonical_json_sha256(
        different_reference_f1
    )
    with pytest.raises(ValueError, match="same reference F1"):
        experiment_design.compile_task7_visual_state_report(
            [different_reference_f1, bundles[1]], draws=4, seed=3
        )


@pytest.mark.parametrize(
    ("tost_90", "superiority_95", "label", "flags"),
    [
        ((-0.4, 0.5), (-0.6, 0.7), "equivalent", (True, False, False)),
        ((0.2, 0.8), (0.1, 0.9), "superior", (True, True, False)),
        ((-2.0, -1.1), (-2.2, -0.1), "inferior", (False, False, True)),
        ((-1.2, 1.3), (-1.5, 1.5), "unresolved", (False, False, False)),
    ],
)
def test_result_terminology_uses_tost_and_directional_interval_precedence(
    tost_90: tuple[float, float],
    superiority_95: tuple[float, float],
    label: str,
    flags: tuple[bool, bool, bool],
) -> None:
    """Catch treating any zero-containing interval as unresolved or equivalent."""

    result = experiment_design.classify_f1_result(
        tost_90_interval=tost_90,
        superiority_95_interval=superiority_95,
    )

    assert result == {
        "equivalent": flags[0],
        "superior": flags[1],
        "inferior": flags[2],
        "label": label,
        "equivalence_margin_f1": 1.0,
    }


def test_holm_adjustment_is_step_down_monotone_and_label_preserving() -> None:
    """Catch independent Bonferroni adjustment or rank-order leakage."""

    adjusted = experiment_design.holm_adjust({"layer": 0.04, "coverage": 0.01, "budget": 0.03})

    assert adjusted == pytest.approx({"layer": 0.06, "coverage": 0.03, "budget": 0.06})


@pytest.mark.parametrize(
    ("values", "expected_mcse", "frozen_repetitions"),
    [
        ([0.0, 2.0] * 5, 0.23570226039551584, 10),
        ([0.0, 3.0] * 5, 0.3535533905932738, 20),
    ],
)
def test_random_calibration_freezes_twenty_only_above_mcse_threshold(
    values: list[float], expected_mcse: float, frozen_repetitions: int
) -> None:
    """Catch population variance, pooled-mask variance, or a >= threshold branch."""

    result = experiment_design.calibrate_random_repetitions({"q1": values, "q2": values})

    assert result["status"] == "passed"
    assert result["initial_repetitions"] == 10
    assert result["mcse_f1"] == pytest.approx(expected_mcse)
    assert result["threshold_f1"] == 0.25
    assert result["frozen_repetitions"] == frozen_repetitions


@pytest.mark.parametrize(
    ("eligible_pool_size", "status", "achieved_power", "launch_admissible"),
    [
        (100, "passed", 0.8060789470312404, True),
        (40, "underpowered_full_pool", 0.6513110193138418, False),
    ],
)
def test_equivalence_power_selects_smallest_n_and_fails_closed_when_pool_is_short(
    eligible_pool_size: int,
    status: str,
    achieved_power: float,
    launch_admissible: bool,
) -> None:
    """Catch IID variance, a superiority power formula, or admitting an undersized pool."""

    result = experiment_design.plan_equivalence_power(
        developmental_mean_f1=0.0,
        developmental_sd_f1=2.0,
        cluster_design_effect=1.5,
        eligible_pool_size=eligible_pool_size,
    )

    assert result["required_n"] == 52
    assert result["selected_n"] == min(52, eligible_pool_size)
    assert result["status"] == status
    assert result["achieved_power"] == pytest.approx(achieved_power)
    assert result["target_power"] == 0.8
    assert result["launch_admissible"] is launch_admissible


def test_maximum_available_amendment_preserves_failed_power_plan_and_admits_full_pool() -> None:
    """Catch rewriting the planned N or silently calling the reduced cohort powered."""

    original = experiment_design.plan_equivalence_power(
        developmental_mean_f1=0.4945312500000001,
        developmental_sd_f1=9.532151596701555,
        cluster_design_effect=1.0,
        eligible_pool_size=1213,
    )

    amended = experiment_design.approve_maximum_available_power(original)

    assert original["status"] == "underpowered_full_pool"
    assert original["required_n"] == 2199
    assert amended["status"] == "approved-maximum-available"
    assert amended["planned_required_n"] == 2199
    assert amended["required_n"] == amended["selected_n"] == 1213
    assert amended["achieved_power"] == pytest.approx(0.5799759173834684)
    assert amended["launch_admissible"] is True
    assert amended["original_power"] == original
    assert amended["claim_rule"] == (
        "passing the frozen TOST establishes equivalence; failing it is unresolved"
    )


def test_maximum_available_amendment_rejects_resigned_tampered_original() -> None:
    """Catch prospective approval being attached to a changed margin or required N."""

    original = experiment_design.plan_equivalence_power(
        developmental_mean_f1=0.4945312500000001,
        developmental_sd_f1=9.532151596701555,
        cluster_design_effect=1.0,
        eligible_pool_size=1213,
    )
    original["equivalence_margin_f1"] = 2.0

    with pytest.raises(ValueError, match="canonical digest"):
        experiment_design.approve_maximum_available_power(original)


def test_development_registry_records_relationships_and_union_digest() -> None:
    """Catch incomplete union exclusion or undocumented overlap between diagnostics."""

    stage64 = experiment_design.development_qid_projection(
        diagnostic_label="stage-64",
        source_path="/immutable/stage64.json",
        source_sha256="a" * 64,
        qids=("q1", "q2"),
    )
    stage245 = experiment_design.development_qid_projection(
        diagnostic_label="stage-245",
        source_path="/immutable/stage245.json",
        source_sha256="b" * 64,
        qids=("q1", "q2", "q3"),
    )

    registry = experiment_design.build_development_registry(
        [stage245, stage64], required_labels=("stage-64", "stage-245")
    )

    assert registry["diagnostic_labels"] == ["stage-245", "stage-64"]
    assert registry["relationships"] == [
        {
            "left": "stage-245",
            "right": "stage-64",
            "relationship": "superset",
            "intersection_count": 2,
        }
    ]
    assert registry["union_qids"] == ["q1", "q2", "q3"]
    assert registry["union_sha256"] == (
        "c587492ba28757a9a5bc87417c913c5d998d34849aa2d3bbf2eb6a7cd423bc0a"
    )


def test_development_registry_fails_closed_when_required_label_is_missing() -> None:
    """Catch holdout construction from a partial diagnostic inventory."""

    projection = experiment_design.development_qid_projection(
        diagnostic_label="stage-245",
        source_path="/immutable/stage245.json",
        source_sha256="b" * 64,
        qids=("q1",),
    )
    with pytest.raises(ValueError, match="missing required diagnostic labels.*threshold"):
        experiment_design.build_development_registry(
            [projection], required_labels=("stage-245", "threshold")
        )


def test_holdout_seal_publishes_nothing_when_power_is_underpowered(tmp_path: Path) -> None:
    """Catch partial holdout publication before the developmental power gate passes."""

    registry_path, labels, power = _seal_prerequisites(tmp_path, eligible_pool_size=2)
    calibration = experiment_design.calibrate_random_repetitions({"dev-qid": [1.0] * 10})
    destination = tmp_path / "method-holdout"
    rows, input_hashes = _authenticated_eligible_rows(
        tmp_path / "underpowered-inputs", ("holdout-a", "holdout-b")
    )

    with pytest.raises(ValueError, match="power.*not launch-admissible"):
        experiment_design.seal_holdout(
            rows,
            development_registry_path=registry_path,
            required_registry_labels=labels,
            power=power,
            calibration=calibration,
            runtime_pins={"runtime_commit": "d" * 40, "python": "3.10.0"},
            input_file_hashes=input_hashes,
            destination=destination,
        )

    assert not destination.exists()


def test_holdout_seal_accepts_authenticated_maximum_available_amendment(tmp_path: Path) -> None:
    """Catch approving the plan on paper while the launch validator still rejects it."""

    registry_path, labels, original = _seal_prerequisites(tmp_path, eligible_pool_size=2)
    power = experiment_design.approve_maximum_available_power(original)
    calibration = experiment_design.calibrate_random_repetitions({"dev-qid": [1.0] * 10})
    rows, input_hashes = _authenticated_eligible_rows(
        tmp_path / "amended-inputs", ("holdout-a", "holdout-b")
    )
    rows[0]["metadata"]["supporting_document_ids"] = ["support-a"]
    rows[1]["metadata"]["supporting_document_ids"] = ["support-b"]
    destination = tmp_path / "method-holdout"

    sealed = experiment_design.seal_holdout(
        rows,
        development_registry_path=registry_path,
        required_registry_labels=labels,
        power=power,
        calibration=calibration,
        runtime_pins={"runtime_commit": "d" * 40},
        input_file_hashes=input_hashes,
        destination=destination,
    )

    validated = experiment_design.validate_holdout_for_launch(
        destination,
        required_registry_path=registry_path,
        required_registry_labels=labels,
    )
    assert validated == sealed
    assert validated["power"]["planned_required_n"] > validated["required_n"] == 2


def test_holdout_seal_uses_manifest_last_commit_on_filesystem_without_renameat2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch BeeGFS/NFS EINVAL leaving a valid holdout impossible or partially admitted."""

    registry_path, labels, _ = _seal_prerequisites(tmp_path, eligible_pool_size=2)
    power = experiment_design.plan_equivalence_power(
        developmental_mean_f1=0.0,
        developmental_sd_f1=0.01,
        cluster_design_effect=1.0,
        eligible_pool_size=2,
    )
    rows, input_hashes = _authenticated_eligible_rows(
        tmp_path / "fallback-inputs", ("holdout-a", "holdout-b")
    )
    rows[0]["metadata"]["supporting_document_ids"] = ["support-a"]
    rows[1]["metadata"]["supporting_document_ids"] = ["support-b"]
    destination = tmp_path / "method-holdout"

    def unsupported(*args: object) -> None:
        raise OSError(errno.EINVAL, "renameat2 unsupported")

    monkeypatch.setattr(experiment_design, "_renameat2_noreplace", unsupported)
    sealed = experiment_design.seal_holdout(
        rows,
        development_registry_path=registry_path,
        required_registry_labels=labels,
        power=power,
        calibration=experiment_design.calibrate_random_repetitions(
            {"dev-qid": [1.0] * 10}
        ),
        runtime_pins={"runtime_commit": "d" * 40},
        input_file_hashes=input_hashes,
        destination=destination,
    )

    assert experiment_design.validate_holdout_for_launch(
        destination,
        required_registry_path=registry_path,
        required_registry_labels=labels,
    ) == sealed
    assert not list(tmp_path.glob(".method-holdout-*"))


def test_holdout_seal_is_atomic_no_replace_and_launch_validated(tmp_path: Path) -> None:
    """Catch unvalidated launch admission or replacement of an immutable holdout."""

    registry_path, labels, _ = _seal_prerequisites(tmp_path, eligible_pool_size=2)
    power = experiment_design.plan_equivalence_power(
        developmental_mean_f1=0.0,
        developmental_sd_f1=0.01,
        cluster_design_effect=1.0,
        eligible_pool_size=2,
    )
    calibration = experiment_design.calibrate_random_repetitions({"dev-qid": [1.0] * 10})
    rows, input_hashes = _authenticated_eligible_rows(
        tmp_path / "seal-inputs", ("holdout-a", "holdout-b")
    )
    rows[0]["metadata"]["supporting_document_ids"] = ["support-a"]
    rows[1]["metadata"]["supporting_document_ids"] = ["support-b"]
    destination = tmp_path / "method-holdout"

    sealed = experiment_design.seal_holdout(
        rows,
        development_registry_path=registry_path,
        required_registry_labels=labels,
        power=power,
        calibration=calibration,
        runtime_pins={"runtime_commit": "d" * 40, "python": "3.10.0"},
        input_file_hashes=input_hashes,
        destination=destination,
    )

    assert sealed["status"] == "sealed"
    assert sealed["required_n"] == 2
    assert len(sealed["selected_qids"]) == 2
    assert (
        experiment_design.validate_holdout_for_launch(
            destination,
            required_registry_path=registry_path,
            required_registry_labels=labels,
        )
        == sealed
    )
    with pytest.raises(FileExistsError, match="already exists"):
        experiment_design.seal_holdout(
            rows,
            development_registry_path=registry_path,
            required_registry_labels=labels,
            power=power,
            calibration=calibration,
            runtime_pins={"runtime_commit": "d" * 40, "python": "3.10.0"},
            input_file_hashes=input_hashes,
            destination=destination,
        )


def test_holdout_seal_rejects_dangling_symlink_destination(tmp_path: Path) -> None:
    """Catch following a dangling target and publishing outside the requested identity."""

    registry_path, labels, _ = _seal_prerequisites(tmp_path, eligible_pool_size=2)
    power = experiment_design.plan_equivalence_power(
        developmental_mean_f1=0.0,
        developmental_sd_f1=0.01,
        cluster_design_effect=1.0,
        eligible_pool_size=2,
    )
    calibration = experiment_design.calibrate_random_repetitions({"dev-qid": [1.0] * 10})
    rows, input_hashes = _authenticated_eligible_rows(
        tmp_path / "dangling-inputs", ("holdout-a", "holdout-b")
    )
    rows[0]["metadata"]["supporting_document_ids"] = ["support-a"]
    rows[1]["metadata"]["supporting_document_ids"] = ["support-b"]
    destination = tmp_path / "method-holdout"
    symlink_target = tmp_path / "unexpected-target"
    destination.symlink_to(symlink_target)

    with pytest.raises(FileExistsError, match="already exists"):
        experiment_design.seal_holdout(
            rows,
            development_registry_path=registry_path,
            required_registry_labels=labels,
            power=power,
            calibration=calibration,
            runtime_pins={"runtime_commit": "d" * 40},
            input_file_hashes=input_hashes,
            destination=destination,
        )

    assert destination.is_symlink()
    assert not symlink_target.exists()


def test_atomic_publish_does_not_replace_destination_created_after_precheck(
    tmp_path: Path,
) -> None:
    """Catch an empty destination winning the publication race."""

    source = tmp_path / ".method-holdout-temporary"
    destination = tmp_path / "method-holdout"
    source.mkdir()
    (source / "manifest.json").write_text("source\n", encoding="utf-8")
    destination.mkdir()

    with pytest.raises(FileExistsError):
        experiment_design._atomic_rename_noreplace(source, destination)

    assert (source / "manifest.json").read_text(encoding="utf-8") == "source\n"
    assert list(destination.iterdir()) == []


def test_regular_file_publish_falls_back_when_filesystem_rejects_renameat2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch Lustre returning EINVAL for RENAME_NOREPLACE on a regular file."""

    class UnsupportedRename:
        argtypes: object = None
        restype: object = None

        def __call__(self, *args: object) -> int:
            ctypes.set_errno(errno.EINVAL)
            return -1

    class UnsupportedLibc:
        renameat2 = UnsupportedRename()

    monkeypatch.setattr(experiment_design.ctypes, "CDLL", lambda *args, **kwargs: UnsupportedLibc())
    source = tmp_path / ".artifact-temporary"
    destination = tmp_path / "artifact.jsonl"
    source.write_bytes(b"sealed artifact\n")
    parent_fd = experiment_design._open_directory_nofollow(tmp_path)
    try:
        experiment_design._renameat2_noreplace(
            parent_fd,
            source.name,
            parent_fd,
            destination.name,
        )
    finally:
        experiment_design.os.close(parent_fd)

    assert not source.exists()
    assert destination.read_bytes() == b"sealed artifact\n"


def test_holdout_seal_rejects_symlinked_parent_component(tmp_path: Path) -> None:
    """Catch publication through a parent path redirected by a symlink."""

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    registry_path, labels, _ = _seal_prerequisites(tmp_path, eligible_pool_size=2)
    power = experiment_design.plan_equivalence_power(
        developmental_mean_f1=0.0,
        developmental_sd_f1=0.01,
        cluster_design_effect=1.0,
        eligible_pool_size=2,
    )
    rows, input_hashes = _authenticated_eligible_rows(
        tmp_path / "symlink-parent-inputs", ("holdout-a", "holdout-b")
    )
    rows[0]["metadata"]["supporting_document_ids"] = ["support-a"]
    rows[1]["metadata"]["supporting_document_ids"] = ["support-b"]

    with pytest.raises(ValueError, match="symlink|non-directory"):
        experiment_design.seal_holdout(
            rows,
            development_registry_path=registry_path,
            required_registry_labels=labels,
            power=power,
            calibration=experiment_design.calibrate_random_repetitions({"dev-qid": [1.0] * 10}),
            runtime_pins={"runtime_commit": "d" * 40},
            input_file_hashes=input_hashes,
            destination=linked_parent / "method-holdout",
        )

    assert not (real_parent / "method-holdout").exists()


def test_launch_validator_rejects_self_consistent_manifest_missing_runtime_pins(
    tmp_path: Path,
) -> None:
    """Catch treating a checksum as a substitute for required-field validation."""

    registry_path, labels, _ = _seal_prerequisites(tmp_path, eligible_pool_size=2)
    power = experiment_design.plan_equivalence_power(
        developmental_mean_f1=0.0,
        developmental_sd_f1=0.01,
        cluster_design_effect=1.0,
        eligible_pool_size=2,
    )
    calibration = experiment_design.calibrate_random_repetitions({"dev-qid": [1.0] * 10})
    rows, input_hashes = _authenticated_eligible_rows(
        tmp_path / "runtime-pin-inputs", ("holdout-a", "holdout-b")
    )
    rows[0]["metadata"]["supporting_document_ids"] = ["support-a"]
    rows[1]["metadata"]["supporting_document_ids"] = ["support-b"]
    destination = tmp_path / "method-holdout"
    experiment_design.seal_holdout(
        rows,
        development_registry_path=registry_path,
        required_registry_labels=labels,
        power=power,
        calibration=calibration,
        runtime_pins={"runtime_commit": "d" * 40},
        input_file_hashes=input_hashes,
        destination=destination,
    )
    manifest_path = destination / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("runtime_pins")
    payload.pop("manifest_sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="required fields"):
        experiment_design.validate_holdout_for_launch(
            destination,
            required_registry_path=registry_path,
            required_registry_labels=labels,
        )


def test_holdout_seal_rejects_partial_signed_development_registry(tmp_path: Path) -> None:
    """Catch a caller substituting a valid but incomplete development registry."""

    registry_path = _write_test_registry(tmp_path / "partial-registry")
    required_labels = ("development", "threshold")
    rows, input_hashes = _authenticated_eligible_rows(
        tmp_path / "partial-inputs", ("holdout-a", "holdout-b")
    )
    power = experiment_design.plan_equivalence_power(
        developmental_mean_f1=0.0,
        developmental_sd_f1=0.01,
        cluster_design_effect=1.0,
        eligible_pool_size=2,
    )

    with pytest.raises(ValueError, match="missing required diagnostic labels.*threshold"):
        experiment_design.seal_holdout(
            rows,
            development_registry_path=registry_path,
            required_registry_labels=required_labels,
            power=power,
            calibration=experiment_design.calibrate_random_repetitions({"dev-qid": [1.0] * 10}),
            runtime_pins={"runtime_commit": "d" * 40},
            input_file_hashes=input_hashes,
            destination=tmp_path / "method-holdout",
        )


def test_holdout_seal_rejects_nonexistent_bound_external_file(tmp_path: Path) -> None:
    """Catch digest-shaped declarations that are not authenticated against bytes."""

    registry_path, labels, _ = _seal_prerequisites(tmp_path, eligible_pool_size=2)
    rows, input_hashes = _authenticated_eligible_rows(
        tmp_path / "missing-inputs", ("holdout-a", "holdout-b")
    )
    missing_feature = Path(rows[0]["persisted_features"][0]["feature_path"])
    missing_feature.unlink()
    power = experiment_design.plan_equivalence_power(
        developmental_mean_f1=0.0,
        developmental_sd_f1=0.01,
        cluster_design_effect=1.0,
        eligible_pool_size=2,
    )

    with pytest.raises(ValueError, match="persisted feature.*missing|regular file"):
        experiment_design.seal_holdout(
            rows,
            development_registry_path=registry_path,
            required_registry_labels=labels,
            power=power,
            calibration=experiment_design.calibrate_random_repetitions({"dev-qid": [1.0] * 10}),
            runtime_pins={"runtime_commit": "d" * 40},
            input_file_hashes=input_hashes,
            destination=tmp_path / "method-holdout",
        )


def test_launch_rejects_bound_external_file_tampered_after_seal(tmp_path: Path) -> None:
    """Catch launch admission after cached-page bytes drift from the seal."""

    destination, registry_path, labels, rows, _ = _seal_valid_fixture(tmp_path)
    Path(rows[0]["cached_pages"][0]["source_path"]).write_bytes(b"tampered\n")

    with pytest.raises(ValueError, match="cached page.*SHA-256 mismatch"):
        experiment_design.validate_holdout_for_launch(
            destination,
            required_registry_path=registry_path,
            required_registry_labels=labels,
        )


def test_launch_rejects_self_consistent_selected_development_qid(tmp_path: Path) -> None:
    """Catch signed-manifest substitution that leaks a development QID."""

    destination, registry_path, labels, _, _ = _seal_valid_fixture(tmp_path)
    manifest_path = destination / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    attacked_records = deepcopy(payload["eligible_records"])
    attacked_records[0]["qid"] = "dev-qid"
    record_by_qid = {record["qid"]: record for record in attacked_records}
    attacked_order = list(experiment_design.holdout_order(list(record_by_qid)))
    ordered_qids = [row["qid"] for row in attacked_order]
    attacked_records = [record_by_qid[qid] for qid in ordered_qids]
    payload["eligible_order"] = attacked_order
    payload["eligible_records"] = attacked_records
    payload["selected_qids"] = ordered_qids
    payload["selected_records"] = attacked_records
    payload["eligibility_projection_sha256"] = experiment_design._canonical_json_sha256(
        attacked_records
    )
    eligible_bytes = ("\n".join(ordered_qids) + "\n").encode()
    payload["eligible_order_sha256"] = hashlib.sha256(eligible_bytes.rstrip(b"\n")).hexdigest()
    payload["selection_sha256"] = payload["eligible_order_sha256"]
    component_input = [
        {"qid": record["qid"], "supporting_document_ids": record["supporting_document_ids"]}
        for record in attacked_records
    ]
    payload["support_components"] = experiment_design.support_document_components(component_input)
    for member_name in ("eligible-order.qids", "selected.qids"):
        member_path = destination / member_name
        member_path.write_bytes(eligible_bytes)
        payload["member_files"][member_name] = hashlib.sha256(eligible_bytes).hexdigest()
    _resign_manifest(manifest_path, payload)

    with pytest.raises(ValueError, match="development|eligibility"):
        experiment_design.validate_holdout_for_launch(
            destination,
            required_registry_path=registry_path,
            required_registry_labels=labels,
        )


def test_launch_rejects_resigned_semantically_invalid_embedded_registry(tmp_path: Path) -> None:
    """Catch trusting embedded registry semantics over the authenticated source."""

    destination, registry_path, labels, _, _ = _seal_valid_fixture(tmp_path)
    manifest_path = destination / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedded = payload["development_registry"]
    embedded["union_qids"] = []
    embedded["union_count"] = 0
    embedded["union_sha256"] = hashlib.sha256(b"").hexdigest()
    embedded.pop("registry_sha256")
    embedded["registry_sha256"] = experiment_design._canonical_json_sha256(embedded)
    _resign_manifest(manifest_path, payload)

    with pytest.raises(ValueError, match="embedded development registry identity mismatch"):
        experiment_design.validate_holdout_for_launch(
            destination,
            required_registry_path=registry_path,
            required_registry_labels=labels,
        )


def test_launch_rejects_self_consistent_invalid_embedded_eligibility_record(
    tmp_path: Path,
) -> None:
    """Catch a signed embedded record whose page and feature identities disagree."""

    destination, registry_path, labels, _, _ = _seal_valid_fixture(tmp_path)
    manifest_path = destination / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["eligible_records"][0]["persisted_features"][0]["page_index"] = 99
    payload["selected_records"] = deepcopy(payload["eligible_records"])
    payload["eligibility_projection_sha256"] = experiment_design._canonical_json_sha256(
        payload["eligible_records"]
    )
    _resign_manifest(manifest_path, payload)

    with pytest.raises(ValueError, match="page/feature identity mismatch"):
        experiment_design.validate_holdout_for_launch(
            destination,
            required_registry_path=registry_path,
            required_registry_labels=labels,
        )


def test_launch_rejects_self_consistent_eligible_qid_member_mismatch(tmp_path: Path) -> None:
    """Catch a re-signed member digest that is not bound to eligible-order semantics."""

    destination, registry_path, labels, _, _ = _seal_valid_fixture(tmp_path)
    manifest_path = destination / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    attacked_bytes = b"different-qid-a\ndifferent-qid-b\n"
    (destination / "eligible-order.qids").write_bytes(attacked_bytes)
    payload["member_files"]["eligible-order.qids"] = hashlib.sha256(attacked_bytes).hexdigest()
    _resign_manifest(manifest_path, payload)

    with pytest.raises(ValueError, match="eligible QID member does not match manifest"):
        experiment_design.validate_holdout_for_launch(
            destination,
            required_registry_path=registry_path,
            required_registry_labels=labels,
        )


def test_intervention_artifact_binds_policy_budget_indices_cache_and_runtime() -> None:
    """Catch an artifact that cannot distinguish or replay the exact intervention."""

    artifact = experiment_design.build_intervention_artifact(
        qid="holdout-a",
        cohort_classification="method-holdout",
        page_hashes=["1" * 64, "2" * 64, "3" * 64, "4" * 64],
        feature_hashes=["5" * 64, "6" * 64, "7" * 64, "8" * 64],
        policy_family="score-top-m",
        policy_name="aggregate-score-top-m",
        mode="physical_delete",
        selection_kind="forced",
        boundary="B_13",
        native_layer=13,
        visual_population=10,
        requested_budget=4,
        achieved_budget=4,
        retained_original_indices=[1, 3, 5, 8],
        seed=7,
        per_layer_cache_lengths={"0": 100, "13": 100, "14": 94},
        runtime_pins={"runtime_commit": "d" * 40, "torch": "2.4.0"},
    )

    assert artifact["schema_version"] == 1
    assert artifact["policy"] == {
        "family": "score-top-m",
        "name": "aggregate-score-top-m",
    }
    assert artifact["mode"] == "physical_delete"
    assert artifact["selection_kind"] == "forced"
    assert artifact["retained_original_indices"] == [1, 3, 5, 8]
    assert len(artifact["artifact_sha256"]) == 64


def test_intervention_artifact_rejects_native_name_in_score_top_m_family() -> None:
    """Catch collapsing native-threshold and matched-budget score policies."""

    with pytest.raises(ValueError, match="score-top-m policy name"):
        experiment_design.build_intervention_artifact(
            qid="q1",
            cohort_classification="synthetic",
            page_hashes=["1" * 64] * 4,
            feature_hashes=["2" * 64] * 4,
            policy_family="score-top-m",
            policy_name="aggregate-native-threshold",
            mode="physical_delete",
            selection_kind="forced",
            boundary="B_13",
            native_layer=13,
            visual_population=2,
            requested_budget=1,
            achieved_budget=1,
            retained_original_indices=[0],
            seed=0,
            per_layer_cache_lengths={"13": 10, "14": 9},
            runtime_pins={"runtime_commit": "d" * 40},
        )


def test_intervention_artifact_requires_explicit_valid_mode_and_selection_kind() -> None:
    """Catch silently inferring forced/native or deletion/zero-mask identity."""

    kwargs = {
        "qid": "q1",
        "cohort_classification": "synthetic",
        "page_hashes": ["1" * 64] * 4,
        "feature_hashes": ["2" * 64] * 4,
        "policy_family": "visual-state-removal",
        "policy_name": "all-visual-removal",
        "boundary": "B_input",
        "native_layer": None,
        "visual_population": 2,
        "requested_budget": 0,
        "achieved_budget": 0,
        "retained_original_indices": [],
        "seed": 0,
        "per_layer_cache_lengths": {"0": 4},
        "runtime_pins": {"runtime_commit": "d" * 40},
    }
    for mode, selection_kind, message in (
        ("unknown", "forced", "mode"),
        ("physical_delete", "unknown", "selection kind"),
    ):
        with pytest.raises(ValueError, match=message):
            experiment_design.build_intervention_artifact(
                **kwargs,
                mode=mode,
                selection_kind=selection_kind,
            )


def test_intervention_artifact_rejects_native_identity_overloaded_as_forced_or_zero_mask() -> None:
    """Catch a forced/zero diagnostic being serialized as the native threshold policy."""

    kwargs = {
        "qid": "q1",
        "cohort_classification": "synthetic",
        "page_hashes": ["1" * 64] * 4,
        "feature_hashes": ["2" * 64] * 4,
        "policy_family": "native-threshold",
        "policy_name": "aggregate-native-threshold",
        "boundary": "B_13",
        "native_layer": 13,
        "visual_population": 2,
        "requested_budget": 1,
        "achieved_budget": 1,
        "retained_original_indices": [0],
        "seed": 0,
        "per_layer_cache_lengths": {"13": 10, "14": 9},
        "runtime_pins": {"runtime_commit": "d" * 40},
    }
    for mode, selection_kind in (("zero_mask", "native_threshold"), ("physical_delete", "forced")):
        with pytest.raises(ValueError, match="native-threshold"):
            experiment_design.build_intervention_artifact(
                **kwargs,
                mode=mode,
                selection_kind=selection_kind,
            )


@pytest.mark.parametrize(
    ("boundary", "native_layer", "message"),
    [
        ("B_input", 0, "native boundary"),
        ("B_13", None, "native_layer"),
        ("B_13", 12, "native boundary"),
    ],
)
def test_intervention_artifact_rejects_impossible_native_boundary_identity(
    boundary: str, native_layer: int | None, message: str
) -> None:
    """Catch a native CTP artifact claiming an input or different-block decision."""

    with pytest.raises(ValueError, match=message):
        experiment_design.build_intervention_artifact(
            qid="q1",
            cohort_classification="synthetic",
            page_hashes=["1" * 64] * 4,
            feature_hashes=["2" * 64] * 4,
            policy_family="native-threshold",
            policy_name="literal-native-threshold",
            mode="physical_delete",
            selection_kind="native_threshold",
            boundary=boundary,
            native_layer=native_layer,
            visual_population=2,
            requested_budget=1,
            achieved_budget=1,
            retained_original_indices=[0],
            seed=0,
            per_layer_cache_lengths={"0": 4},
            runtime_pins={"runtime_commit": "d" * 40},
        )


def test_intervention_artifact_forced_identity_does_not_use_native_layer_as_boundary() -> None:
    """Catch forced selection being relabeled as a native CTP decision when observed l* exists."""

    artifact = experiment_design.build_intervention_artifact(
        qid="q1",
        cohort_classification="synthetic",
        page_hashes=["1" * 64] * 4,
        feature_hashes=["2" * 64] * 4,
        policy_family="score-top-m",
        policy_name="attention-score-top-m",
        mode="physical_delete",
        selection_kind="forced",
        boundary="B_1",
        native_layer=12,
        visual_population=2,
        requested_budget=1,
        achieved_budget=1,
        retained_original_indices=[0],
        seed=0,
        per_layer_cache_lengths={"0": 4},
        runtime_pins={"runtime_commit": "d" * 40},
    )

    assert artifact["selection_kind"] == "forced"
    assert artifact["boundary"] == "B_1"
    assert artifact["native_layer"] == 12


def test_durable_development_registry_authenticates_sources_and_union() -> None:
    """Catch registry drift, a missing diagnostic label, or changed immutable QID sources."""

    registry = experiment_design.load_development_registry(
        DEVELOPMENT_REGISTRY,
        required_labels=(
            "normalization-16",
            "aggregate-logit-threshold-16",
            "grouped-query-head-16",
            "generated-query-capture-timing-16",
            "generated-query-qa-16",
            "pruning-semantics-16",
            "original-token-scaling-16",
            "stage-64",
            "aggregate-logit-stage64",
            "stage-245",
            "aggregate-logit-stage245",
            "qtp-mean-stage245",
        ),
        authenticate_sources=True,
    )

    assert registry["status"] == "complete"
    assert registry["union_count"] == 254
    assert registry["union_sha256"] == (
        "dd8d39df9e92335f1c130703ab2fd9341ea8257a18bad7453201a58eb72dc93e"
    )
