from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_dynamic_geometry_uses_page_major_merged_grid_and_exact_combined_mask() -> None:
    from docprune.task6_runtime import derive_post_qtp_geometry

    identity = derive_post_qtp_geometry(
        torch.tensor([[1, 4, 6], [1, 2, 4]]),
        torch.tensor([True, False, True, False, True, False, True, False]),
        merge_size=2,
    )

    assert [
        (row.page_index, row.row, row.column, row.height, row.width) for row in identity.geometry
    ] == [
        (0, 0, 0, 2, 3),
        (0, 0, 2, 2, 3),
        (0, 1, 1, 2, 3),
        (1, 0, 0, 1, 2),
    ]
    assert identity.count == 4
    assert (
        identity.sha256
        == hashlib.sha256(b"[[0,0,0,2,3],[0,0,2,2,3],[0,1,1,2,3],[1,0,0,1,2]]").hexdigest()
    )


@pytest.mark.parametrize(
    ("grid", "mask", "message"),
    (
        ([[2, 2, 2]], [True], "temporal-one"),
        ([[1, 3, 2]], [True], "merge divisible"),
        ([[1, 2, 2]], [True, False], "mask length"),
    ),
)
def test_dynamic_geometry_fails_closed_on_grid_or_population_drift(
    grid: list[list[int]], mask: list[bool], message: str
) -> None:
    from docprune.task6_runtime import derive_post_qtp_geometry

    with pytest.raises(ValueError, match=message):
        derive_post_qtp_geometry(grid, mask, merge_size=2)


def test_task6_policy_matrix_closes_names_and_uses_ten_native_three_fixed_repetitions() -> None:
    from docprune.task6_runtime import task6_policy, task6_policy_matrix

    native = task6_policy_matrix("native")
    extension = task6_policy_matrix("native-extension")
    fixed = task6_policy_matrix("fixed")
    assert len(native) == 45
    assert len(extension) == 40
    assert len(fixed) == 42
    assert [
        cell.repetition for cell in native if cell.policy.name == "global-uniform-random"
    ] == list(range(10))
    assert [
        cell.repetition for cell in fixed if cell.policy.name == "global-uniform-random-retain-55"
    ] == [0, 1, 2]
    assert [
        cell.repetition for cell in extension if cell.policy.name == "global-uniform-random"
    ] == list(range(10, 20))
    assert task6_policy("coverage-matched-identity-shuffle").budget_source == (
        "aggregate-native-threshold-count"
    )
    assert task6_policy("grid-stratified-random-retain-80").fixed_retention is not None
    with pytest.raises(ValueError, match="approved Task 6 policy"):
        task6_policy("random-ish")


def _fixture(tmp_path: Path):
    from docprune.task6_runtime import (
        TASK6_RENDERER_CONTRACT,
        FixedPageFixture,
        FixedPageQuestion,
        FixedPageRecord,
    )

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"sealed-pdf")
    feature = tmp_path / "000000.safetensors"
    save_file(
        {
            "embeddings": torch.arange(384, dtype=torch.float32).reshape(3, 128),
            "page_offsets": torch.tensor([0, 1, 3], dtype=torch.int64),
            "raster_indices": torch.tensor([0, 2, 5], dtype=torch.int64),
        },
        feature,
    )
    feature_manifest = tmp_path / "embeddings.json"
    feature_manifest.write_text('{"schema_version":5}\n', encoding="utf-8")
    ledger = tmp_path / "completion-ledger.json"
    ledger.write_text(
        json.dumps(
            [
                {
                    "schema_version": 5,
                    "doc_id": "doc",
                    "document_path": str(feature.resolve()),
                    "sha256": _sha256(feature),
                    "page_offsets": [0, 1, 3],
                    "pages": [
                        {"doc_id": "doc", "page_index": 0, "source_hw": [32, 32]},
                        {"doc_id": "doc", "page_index": 1, "source_hw": [32, 32]},
                    ],
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reference = tmp_path / "reference.json"
    reference.write_text('{"question_ids":["q1"]}\n', encoding="utf-8")
    eligible = tmp_path / "eligible.jsonl"
    eligible.write_text(
        json.dumps(
            {
                "qid": "q1",
                "question": "question",
                "answers": [{"answer": "gold"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rgb = Image.new("RGB", (2, 1), (1, 2, 3))
    rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    page = FixedPageRecord(
        rank=0,
        doc_id="doc",
        page_index=1,
        score=0.75,
        source_pdf_path=pdf.resolve(),
        source_pdf_sha256=_sha256(pdf),
        rendered_rgb_width=2,
        rendered_rgb_height=1,
        rendered_rgb_sha256=rgb_sha,
        renderer_contract=TASK6_RENDERER_CONTRACT,
        feature_shard_path=feature.resolve(),
        feature_shard_sha256=_sha256(feature),
        feature_page_index=1,
    )
    fixture = FixedPageFixture(
        fixture_version="task6-test-v1",
        reference_path=reference.resolve(),
        reference_sha256=_sha256(reference),
        eligible_questions_path=eligible.resolve(),
        eligible_questions_sha256=_sha256(eligible),
        feature_manifest_path=feature_manifest.resolve(),
        feature_manifest_sha256=_sha256(feature_manifest),
        completion_ledger_path=ledger.resolve(),
        completion_ledger_sha256=_sha256(ledger),
        questions=(FixedPageQuestion("q1", hashlib.sha256(b"question").hexdigest(), (page,)),),
    )
    return fixture, rgb


def test_fixed_fixture_reads_selected_samples_only_from_bound_eligible_jsonl(
    tmp_path: Path,
) -> None:
    fixture, _ = _fixture(tmp_path)

    samples = fixture.selected_samples(("q1",))

    assert [(sample.question_id, sample.question, sample.answers) for sample in samples] == [
        ("q1", "question", ("gold",))
    ]
    with pytest.raises(ValueError, match="duplicate"):
        fixture.selected_samples(("q1", "q1"))
    with pytest.raises(ValueError, match="absent"):
        fixture.selected_samples(("unknown",))


def test_fixed_fixture_selected_validation_does_not_hash_unselected_page_sources(
    tmp_path: Path,
) -> None:
    fixture, _ = _fixture(tmp_path)
    page = fixture.questions[0].pages[0]
    second_pdf = tmp_path / "second.pdf"
    second_pdf.write_bytes(b"second-pdf")
    second_feature = tmp_path / "second.safetensors"
    second_feature.write_bytes(b"second-feature")
    second_page = replace(
        page,
        source_pdf_path=second_pdf.resolve(),
        source_pdf_sha256=_sha256(second_pdf),
        feature_shard_path=second_feature.resolve(),
        feature_shard_sha256=_sha256(second_feature),
    )
    fixture = replace(
        fixture,
        questions=fixture.questions
        + (
            replace(
                fixture.questions[0],
                qid="q2",
                question_sha256=hashlib.sha256(b"question two").hexdigest(),
                pages=(second_page,),
            ),
        ),
    )
    second_pdf.write_bytes(b"corrupted")
    second_feature.write_bytes(b"corrupted")

    fixture.validate_external_bytes(selected_qids=("q1",))
    with pytest.raises(ValueError, match="source PDF checksum"):
        fixture.validate_external_bytes(selected_qids=("q2",))


def test_fixed_fixture_validates_bytes_loads_only_bound_shard_and_authenticates_rendered_rgb(
    tmp_path: Path,
) -> None:
    from docprune.task6_runtime import AuthenticatedFixedPageLoader, FixedPageFeatureStore

    fixture, rgb = _fixture(tmp_path)
    fixture.validate_external_bytes()
    store = FixedPageFeatureStore(fixture)
    loaded = store.load("q1", 0)
    assert loaded.doc_id == "doc"
    assert loaded.page_index == 1
    assert loaded.visual_embeddings.shape == (2, 128)
    assert loaded.raster_indices.tolist() == [2, 5]

    render_calls: list[tuple[Path, int]] = []

    def render(path: Path, index: int) -> Image.Image:
        render_calls.append((path, index))
        return rgb.copy()

    loader = AuthenticatedFixedPageLoader(fixture, render)
    assert loader.load_page_for_question("q1", 0).tobytes() == rgb.tobytes()
    assert loader.load_page("doc", 1).tobytes() == rgb.tobytes()
    assert render_calls == [(fixture.questions[0].pages[0].source_pdf_path, 1)]
    with pytest.raises(ValueError, match="rendered RGB"):
        AuthenticatedFixedPageLoader(
            fixture, lambda path, index: Image.new("RGB", (2, 1), (9, 9, 9))
        ).load_page_for_question("q1", 0)


def test_fixed_retriever_encodes_query_but_never_searches_or_reorders_pages(
    tmp_path: Path,
) -> None:
    from docprune.task6_runtime import AuthenticatedFixedPageRetriever

    fixture, _ = _fixture(tmp_path)

    class QueryEncoder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def encode_queries(self, questions: list[str]) -> list[torch.Tensor]:
            self.calls.append(questions)
            return [torch.ones((2, 128))]

    encoder = QueryEncoder()
    retriever = AuthenticatedFixedPageRetriever(fixture, encoder)
    output = retriever.retrieve("question", 1)
    replay = retriever.retrieve("question", 1)

    assert encoder.calls == [["question"]]
    assert [(page.doc_id, page.page_index, page.score) for page in output.pages] == [
        ("doc", 1, 0.75)
    ]
    assert output.page_features[0].raster_indices.tolist() == [2, 5]
    assert replay.pages == output.pages
    retriever.release_query_encoder()
    assert retriever.retrieve("question", 1).pages == output.pages
    with pytest.raises(ValueError, match="sealed page count"):
        retriever.retrieve("question", 2)
    with pytest.raises(KeyError, match="question"):
        retriever.retrieve("different", 1)


def test_fixed_fixture_rejects_hash_order_and_global_index_claim_drift(tmp_path: Path) -> None:
    fixture, _ = _fixture(tmp_path)
    bad_hash = replace(
        fixture,
        questions=(
            replace(
                fixture.questions[0],
                pages=(replace(fixture.questions[0].pages[0], feature_shard_sha256="b" * 64),),
            ),
        ),
    )
    with pytest.raises(ValueError, match="feature shard checksum"):
        bad_hash.validate_external_bytes()

    payload = fixture.to_dict()
    assert payload["fixed_page_provenance"] is True
    assert payload["global_index_loaded"] is False
    payload["global_index_loaded"] = True
    from docprune.task6_runtime import FixedPageFixture

    with pytest.raises(ValueError, match="global index"):
        FixedPageFixture.from_dict(payload)


def test_fixture_publish_and_load_are_no_replace_and_digest_bound(tmp_path: Path) -> None:
    from docprune.task6_runtime import load_fixed_page_fixture, publish_fixed_page_fixture

    fixture, _ = _fixture(tmp_path)
    target = tmp_path / "fixture.json"
    digest = publish_fixed_page_fixture(fixture, target)
    loaded = load_fixed_page_fixture(target, expected_sha256=digest)
    assert loaded.to_dict() == fixture.to_dict()
    with pytest.raises(FileExistsError):
        publish_fixed_page_fixture(fixture, target)
    with pytest.raises(ValueError, match="fixture checksum"):
        load_fixed_page_fixture(target, expected_sha256="0" * 64)


def test_task6_run_evidence_rejects_geometry_fixture_policy_and_resume_drift(
    tmp_path: Path,
) -> None:
    from docprune.task6_runtime import Task6ResultIdentity, validate_task6_result_record

    expected = Task6ResultIdentity(
        fixture_sha256="a" * 64,
        policy_name="global-uniform-random",
        experiment_version="task6-v1",
        repetition=2,
    )
    record = {
        "question_id": "q1",
        "fixed_page_fixture_sha256": "a" * 64,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "policy_selection": {
            "policy": {"name": "global-uniform-random"},
            "boundary": "B_0",
            "geometry_count": 3,
            "geometry_sha256": "b" * 64,
            "visual_population": 3,
            "seed_sha256": "c" * 64,
            "prefill_cache_lengths": [10, 7],
            "retained_mrope_position_shape": [3, 1, 7],
            "retained_mrope_position_sha256": "d" * 64,
        },
        "policy_context": {"experiment_version": "task6-v1", "repetition": 2},
    }
    validate_task6_result_record(record, expected)
    record["global_index_loaded"] = True
    with pytest.raises(ValueError, match="global index"):
        validate_task6_result_record(record, expected)


def test_task6_random_no_crossing_is_a_valid_authenticated_noop() -> None:
    from docprune.task6_runtime import Task6ResultIdentity, validate_task6_result_record

    expected = Task6ResultIdentity("a" * 64, "global-uniform-random", "task6-v1", 0)
    record = {
        "fixed_page_fixture_sha256": "a" * 64,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "policy_selection": {
            "policy": {"name": "global-uniform-random"},
            "boundary": None,
            "geometry_count": 2,
            "geometry_sha256": "b" * 64,
            "visual_population": 2,
            "seed_sha256": None,
            "prefill_cache_lengths": [10, 10],
            "retained_mrope_position_shape": [3, 1, 10],
            "retained_mrope_position_sha256": "d" * 64,
        },
        "policy_context": {"experiment_version": "task6-v1", "repetition": 0},
    }

    validate_task6_result_record(record, expected)


def test_task6_factory_resolves_only_closed_environment_identity_before_model_load(
    tmp_path: Path, monkeypatch
) -> None:
    import docprune.m3docvqa_factory as factory

    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("DOCPRUNE_TASK6_FIXED_PAGE_FIXTURE", str(fixture_path.resolve()))
    monkeypatch.setenv("DOCPRUNE_TASK6_FIXED_PAGE_FIXTURE_SHA256", "a" * 64)
    monkeypatch.setenv("DOCPRUNE_TASK6_POLICY", "global-uniform-random")
    monkeypatch.setenv("DOCPRUNE_TASK6_EXPERIMENT_VERSION", "task6-v1")
    monkeypatch.setenv("DOCPRUNE_TASK6_REPETITION", "7")
    monkeypatch.setenv("DOCPRUNE_TASK6_RUNTIME_COMMIT", "b" * 40)
    observed: dict[str, object] = {}

    def fake_build_workload(**kwargs: object) -> str:
        observed.update(kwargs)
        return "workload"

    monkeypatch.setattr(factory, "build_workload", fake_build_workload)

    assert factory.build_task6_fixed_page_workload(operation="evaluate", mode="docprune") == (
        "workload"
    )
    assert observed["fixed_page_fixture"] == fixture_path.resolve()
    assert observed["fixed_page_fixture_sha256"] == "a" * 64
    assert observed["ctp_policy"].name == "global-uniform-random"
    assert observed["policy_experiment_version"] == "task6-v1"
    assert observed["policy_repetition"] == 7
    assert observed["execution_runtime_commit"] == "b" * 40
    assert (
        factory.controlled_policy_identity(
            "docprune.m3docvqa_factory:build_task6_fixed_page_workload"
        )["name"]
        == "global-uniform-random"
    )

    monkeypatch.setenv("DOCPRUNE_TASK6_POLICY", "invented")
    with pytest.raises(ValueError, match="approved Task 6 policy"):
        factory.controlled_policy_identity(
            "docprune.m3docvqa_factory:build_task6_fixed_page_workload"
        )


def test_task6_execution_identity_separates_feature_build_and_live_runtime_commits() -> None:
    import docprune.m3docvqa_factory as factory

    identity = factory._task6_execution_identity(
        {"runtime_commit": "a" * 40, "mode": "docprune"},
        "b" * 40,
    )

    assert identity["feature_build_runtime_commit"] == "a" * 40
    assert identity["runtime_commit"] == "b" * 40
    with pytest.raises(ValueError, match="execution runtime commit"):
        factory._task6_execution_identity({"runtime_commit": "a" * 40}, "not-a-commit")


def test_task6_factory_components_use_only_sealed_fixture_without_global_index(
    tmp_path: Path, monkeypatch
) -> None:
    import docprune.m3docvqa_factory as factory
    from docprune.m3docrag import SampleInput
    from docprune.task6_runtime import publish_fixed_page_fixture

    fixture, rgb = _fixture(tmp_path)
    fixture_path = tmp_path / "sealed-fixture.json"
    fixture_sha256 = publish_fixed_page_fixture(fixture, fixture_path)

    class QueryEncoder:
        def encode_queries(self, questions: list[str]) -> list[torch.Tensor]:
            return [torch.ones((2, 128))]

    class ForbiddenGlobalIndex:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"global index access is forbidden: {name}")

    monkeypatch.setitem(__import__("sys").modules, "faiss", ForbiddenGlobalIndex())
    monkeypatch.setattr(factory, "render_task6_pdf_page", lambda path, page_index: rgb.copy())

    loaded, retriever, loader = factory._build_task6_fixed_page_components(
        fixture_path,
        fixture_sha256,
        QueryEncoder(),
        samples=(SampleInput("q1", "question"),),
        page_count=1,
    )

    assert loaded.to_dict() == fixture.to_dict()
    assert retriever.retrieve("question", 1).pages[0].doc_id == "doc"
    assert loader.load_page("doc", 1).tobytes() == rgb.tobytes()
    with pytest.raises(ValueError, match="question text"):
        factory._build_task6_fixed_page_components(
            fixture_path,
            fixture_sha256,
            QueryEncoder(),
            samples=(SampleInput("q1", "changed"),),
            page_count=1,
        )


def test_task6_factory_sample_selection_never_constructs_global_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    import docprune.m3docvqa_factory as factory
    from docprune.task6_runtime import publish_fixed_page_fixture

    fixture, _ = _fixture(tmp_path)
    fixture_path = tmp_path / "selected-sample-fixture.json"
    fixture_sha256 = publish_fixed_page_fixture(fixture, fixture_path)
    monkeypatch.setattr(
        factory,
        "_make_dataset",
        lambda run_config: (_ for _ in ()).throw(AssertionError("global dataset constructed")),
    )

    loaded, samples = factory._load_task6_fixed_page_samples(
        fixture_path,
        fixture_sha256,
        sample_ids=("q1",),
        limit=None,
        page_count=1,
    )

    assert loaded.to_dict() == fixture.to_dict()
    assert [(sample.question_id, sample.answers) for sample in samples] == [("q1", ("gold",))]


def test_task6_cli_authority_uses_the_same_fixture_bound_sample_path() -> None:
    from docprune.cli import _resolve_evaluate_authority

    source = inspect.getsource(_resolve_evaluate_authority)
    assert "_load_task6_fixed_page_samples" in source
    assert "feature_build_runtime_commit" in source


def test_cli_binds_and_validates_task6_result_evidence_from_immutable_manifest() -> None:
    from docprune.cli import _bind_task6_result_evidence, _task6_result_identity
    from docprune.ctp_controls import VisualTokenGeometry
    from docprune.ctp_policy import (
        PolicySelectionContext,
        random_top_m_policy,
        select_boundary_policy,
    )
    from docprune.task6_runtime import validate_task6_result_record

    geometry_sha256 = hashlib.sha256(b"[[0,0,0,1,1]]").hexdigest()
    policy = random_top_m_policy("global-uniform-random")
    selection = select_boundary_policy(
        policy,
        literal_scores=(0.1,),
        aggregate_scores=(0.9,),
        attention_threshold=0.5,
        boundary="B_0",
        native_layer=0,
        selection_context=PolicySelectionContext(
            "task6-v1",
            "q1",
            "B_0",
            2,
            (VisualTokenGeometry(0, 0, 0, 1, 1),),
            1,
            geometry_sha256,
        ),
    )
    manifest = {
        "fixed_page_fixture_sha256": "a" * 64,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "ctp_policy": policy.to_dict(),
        "ctp_policy_context": {
            "experiment_version": "task6-v1",
            "repetition": 2,
            "geometry": None,
        },
    }
    selection_payload = selection.to_dict()
    selection_payload.update(
        {
            "prefill_cache_lengths": [10, 8],
            "retained_mrope_position_shape": [3, 1, 8],
            "retained_mrope_position_sha256": "d" * 64,
        }
    )
    record = {"question_id": "q1", "policy_selection": selection_payload}

    _bind_task6_result_evidence(record, manifest)
    validate_task6_result_record(record, _task6_result_identity(manifest))
    assert record["policy_context"] == {
        "experiment_version": "task6-v1",
        "repetition": 2,
    }


def test_cpu_sealer_builds_fixture_from_reference_ledger_and_existing_bytes(
    tmp_path: Path,
) -> None:
    from docprune.task6_runtime import build_fixed_page_fixture

    fixture, rgb = _fixture(tmp_path)
    page = fixture.questions[0].pages[0]
    reference = {
        "schema_version": 1,
        "selection_is_outcome_blind": True,
        "question_ids": ["q1"],
        "rows": {
            "q1": {
                "retrieved_pages": [
                    {
                        "doc_id": page.doc_id,
                        "page_index": page.page_index,
                        "score": page.score,
                    }
                ]
            }
        },
    }
    fixture.reference_path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    eligible = tmp_path / "eligible.jsonl"
    eligible.write_text(
        json.dumps({"qid": "q1", "question": "question", "answers": []}) + "\n",
        encoding="utf-8",
    )
    fixture.feature_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "mode": "docprune",
                "page_count": 1,
                "completion_ledger_path": str(fixture.completion_ledger_path),
                "completion_ledger_sha256": _sha256(fixture.completion_ledger_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sealed = build_fixed_page_fixture(
        reference_path=fixture.reference_path,
        eligible_questions_path=eligible,
        feature_manifest_path=fixture.feature_manifest_path,
        pdf_dir=tmp_path,
        fixture_version="task6-test-seal-v1",
        page_count=1,
        render_page=lambda path, index: rgb.copy(),
    )

    assert [question.qid for question in sealed.questions] == ["q1"]
    assert sealed.questions[0].question_sha256 == hashlib.sha256(b"question").hexdigest()
    assert sealed.questions[0].pages[0].feature_shard_path == page.feature_shard_path
    assert (
        sealed.questions[0].pages[0].rendered_rgb_sha256
        == hashlib.sha256(rgb.tobytes()).hexdigest()
    )


def test_gate_manifest_seals_qid_shards_policy_matrices_and_gpu_roles(tmp_path: Path) -> None:
    from docprune.task6_runtime import build_task6_gate_manifest

    fixture, _ = _fixture(tmp_path)
    manifest = build_task6_gate_manifest(
        fixture,
        fixture_path=(tmp_path / "fixture.json").resolve(),
        fixture_sha256="a" * 64,
        smoke_qid="q1",
    )

    assert manifest["qid_shards"] == [{"shard": 0, "qid": "q1"}]
    assert len(manifest["native_cells"]) == 45
    assert len(manifest["fixed_cells"]) == 42
    assert len(manifest["native_extension_cells"]) == 40
    assert len(manifest["smoke_cells"]) == 9
    assert manifest["canonical_gpu_family"] == "L40S"
    assert manifest["portability_gpu_families"] == ["A30", "A100-40GB", "H100", "L40S"]
    assert manifest["generation_counts"] == {
        "smoke_per_qid": 9,
        "native_per_qid": 45,
        "fixed_per_qid": 42,
        "native_total": 45,
        "native_extension_total": 40,
        "fixed_total": 42,
    }


def test_task6_smoke_launcher_binds_execution_checkout_and_exact_a100_40_memory() -> None:
    matrix = Path("examples/run_task6_matrix.py").read_text(encoding="utf-8")
    launcher = Path("examples/sbatch/34_docprune_task6_smoke.sbatch").read_text(encoding="utf-8")

    assert 'parser.add_argument("--runtime-dir", type=Path, required=True)' in matrix
    assert 'parser.add_argument("--runtime-commit", required=True)' in matrix
    assert '"feature_build_runtime_commit": identity["runtime_commit"]' in matrix
    assert '"feature_build_source_order_sha256": feature_manifest.source_order_sha256' in matrix
    assert '"runtime_commit": args.runtime_commit' in matrix
    assert "execution_runtime_commit=args.runtime_commit" in matrix
    assert '--runtime-dir "$RUNTIME_DIR"' in launcher
    assert '--runtime-commit "$RUNTIME_COMMIT"' in launcher
    assert 'test "$GPU_MEMORY_MIB" -lt 50000' in launcher


def test_task6_l40s_launcher_rejects_existing_shard_before_runtime_access(
    tmp_path: Path,
) -> None:
    matrix_root = tmp_path / "matrix"
    (matrix_root / "shard-0000").mkdir(parents=True)
    launcher = Path("examples/sbatch/35_docprune_task6_l40s_matrix.sbatch").resolve()
    environment = os.environ.copy()
    environment.update(
        {
            "RUNTIME_DIR": str(tmp_path / "missing-runtime"),
            "RUNTIME_COMMIT": "0" * 40,
            "MATRIX_ROOT": str(matrix_root),
            "MATRIX_KIND": "native",
            "SLURM_ARRAY_TASK_ID": "0",
        }
    )

    completed = subprocess.run(
        ["bash", str(launcher)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "Task 6 shard output already exists" in completed.stderr
    assert "missing-runtime" not in completed.stderr


def test_task6_l40s_launcher_pins_python_and_all_mutable_input_bytes() -> None:
    launcher = Path("examples/sbatch/35_docprune_task6_l40s_matrix.sbatch").read_text(
        encoding="utf-8"
    )

    assert "${PYTHON:=" not in launcher
    assert "#SBATCH --no-requeue" in launcher
    assert "readonly PYTHON=/home/lmalveau/mamba-envs/docprune-sol/bin/python" in launcher
    assert "readonly M3DOCRAG_DIR=/home/lmalveau/src/m3docrag-task6-clean-20260828" in launcher
    assert (
        'test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain --untracked-files=all)"'
    ) in launcher
    assert (
        "readonly RUN_CONFIG_SHA256="
        "b9a6668aaf70c6059b175d18e29bc0082f76231d233a4d993b93fcb55ebbbb8f"
    ) in launcher
    assert (
        "readonly INDEX_MANIFEST_SHA256="
        "ffa5979b3bf157adefcc132b0438af295ddafb2243377db4cdb8fcb8eaafe5da"
    ) in launcher
    assert (
        "readonly CONFIG_SHA256=82463d2ef3296a199521f3f637b256341aad7938eb199cb55c6249debfea44aa"
    ) in launcher
    assert 'test "$(sha256sum "$RUN_CONFIG" | cut -d\' \' -f1)" = "$RUN_CONFIG_SHA256"' in launcher
    assert (
        'test "$(sha256sum "$INDEX_MANIFEST" | cut -d\' \' -f1)" = "$INDEX_MANIFEST_SHA256"'
    ) in launcher
    assert 'test "$(sha256sum "$CONFIG" | cut -d\' \' -f1)" = "$CONFIG_SHA256"' in launcher
