from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from docprune.artifacts import IndexManifest
from docprune.benchmark_config import (
    COLPALI_BACKBONE_MODEL,
    COLPALI_BACKBONE_REVISION,
    COLPALI_MODEL,
    COLPALI_REVISION,
    QWEN_MODEL,
)
from docprune.m3docrag import SampleInput, SampleTiming
from docprune.m3docvqa_factory import (
    _is_cli_placeholder,
    _load_index_manifest,
    _validate_result_record,
    _write_run_manifest,
    build_workload,
    filter_samples,
    load_completed_qids,
    validate_processor_contract_file,
)


def result_record(qid: str, *, question: str = "question", answers: list[str] | None = None):
    return {
        "question_id": qid,
        "question": question,
        "answers": [] if answers is None else answers,
        "predicted_answer": "answer",
        "retrieved_pages": [{"doc_id": "doc", "page_index": 0, "score": 1.0}],
        "trace": {
            "original_visual_tokens": 4,
            "post_btp_visual_tokens": 3,
            "post_qtp_visual_tokens": 2,
            "post_ctp_visual_tokens": 1,
            "ctp_layer": None,
        },
        "timing": {
            "retrieval_seconds": 0.1,
            "qa_seconds": 0.2,
            "profiler_enabled": False,
        },
    }


def test_filter_samples_is_deterministic_and_rejects_duplicate_requested_ids() -> None:
    samples = tuple(SampleInput(f"q-{i}", f"question-{i}") for i in range(4))

    assert [sample.question_id for sample in filter_samples(samples, limit=2)] == ["q-0", "q-1"]
    assert [
        sample.question_id for sample in filter_samples(samples, sample_ids=("q-3", "q-1"))
    ] == [
        "q-1",
        "q-3",
    ]
    with pytest.raises(ValueError, match="duplicate"):
        filter_samples(samples, sample_ids=("q-1", "q-1"))


def test_validate_result_record_enforces_mode_aware_trace_contract() -> None:
    all_kept = result_record("q-1")
    all_kept["trace"]["ctp_layer"] = 3
    with pytest.raises(ValueError, match="all-kept"):
        _validate_result_record(all_kept, line_number=1, mode="all-kept")

    docprune = result_record("q-1")
    docprune["trace"]["ctp_layer"] = 28
    with pytest.raises(ValueError, match="ctp_layer"):
        _validate_result_record(docprune, line_number=1, mode="docprune")

    for mode in ("all-kept", "docprune"):
        zero_trace = result_record("q-1")
        zero_trace["trace"].update(
            {
                "original_visual_tokens": 0,
                "post_btp_visual_tokens": 0,
                "post_qtp_visual_tokens": 0,
                "post_ctp_visual_tokens": 0,
            }
        )
        with pytest.raises(ValueError, match="positive"):
            _validate_result_record(zero_trace, line_number=1, mode=mode)


@pytest.mark.parametrize("value", [None, "false", 0])
def test_validate_result_record_requires_boolean_profiler_state(value) -> None:
    record = result_record("q-1")
    if value is None:
        del record["timing"]["profiler_enabled"]
    else:
        record["timing"]["profiler_enabled"] = value

    with pytest.raises(ValueError, match="profiler"):
        _validate_result_record(record, line_number=1)


@pytest.mark.parametrize("field", [
    "retrieval_seconds",
    "page_load_seconds",
    "qa_seconds",
    "total_sample_seconds",
    "encoder_seconds",
    "decoder_seconds",
])
def test_production_result_record_requires_positive_exact_stage_timings(field: str) -> None:
    record = result_record("q-1")
    record["timing"].update(
        {
            "page_load_seconds": 0.1,
            "total_sample_seconds": 0.4,
            "encoder_seconds": 0.1,
            "decoder_seconds": 0.1,
        }
    )
    del record["timing"][field]
    with pytest.raises(ValueError, match="(production timing|invalid timing schema)"):
        _validate_result_record(record, line_number=1, production=True)

    record = result_record("q-1")
    record["timing"].update(
        {
            "page_load_seconds": 0.1,
            "total_sample_seconds": 0.4,
            "encoder_seconds": 0.1,
            "decoder_seconds": 0.1,
        }
    )
    record["timing"][field] = 0.0
    with pytest.raises(ValueError, match="invalid timings"):
        _validate_result_record(record, line_number=1, production=True)

    record["timing"][field] = float("nan")
    with pytest.raises(ValueError, match="invalid timings"):
        _validate_result_record(record, line_number=1, production=True)


def test_production_resume_validation_requires_raw_total_sample_seconds(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(result_record("q-1")) + "\n")

    with pytest.raises(ValueError, match="production timing"):
        load_completed_qids(path, expected_qids=("q-1",), production=True)


def test_production_record_rejects_synthesized_total_but_accepts_explicit_total() -> None:
    record = result_record("q-1")
    record["timing"] = SampleTiming(
        retrieval_seconds=0.1,
        qa_seconds=0.2,
        page_load_seconds=0.1,
        encoder_seconds=0.1,
        decoder_seconds=0.1,
    ).to_dict()
    with pytest.raises(ValueError, match="stage boundaries"):
        _validate_result_record(record, line_number=1, production=True)

    record["timing"]["total_sample_seconds"] = 0.5
    _validate_result_record(record, line_number=1, production=True)


def test_model_loaders_explicitly_place_production_models_on_cuda(monkeypatch) -> None:
    import torch

    import docprune.m3docvqa_factory as factory

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(factory, "_cached_snapshot", lambda repo, revision: Path("/cached"))
    monkeypatch.setattr(factory, "assert_supported_colpali", lambda model, processor: None)

    class ColPaliModel:
        def __init__(self):
            self.devices = []

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def load_adapter(self, path):
            self.adapter_path = path

        def eval(self):
            return self

        def to(self, device):
            self.devices.append(device)
            return self

    colpali_model = ColPaliModel()
    colpali_models = SimpleNamespace(
        ColPali=SimpleNamespace(from_pretrained=lambda *args, **kwargs: colpali_model),
        ColPaliProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
    )
    monkeypatch.setitem(sys.modules, "colpali_engine", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "colpali_engine.models", colpali_models)
    model, _ = factory._load_colpali(SimpleNamespace())
    assert model.devices == [torch.device("cuda")]

    class QwenModel(ColPaliModel):
        pass

    qwen_model = QwenModel()
    qwen_calls = []
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
            Qwen2VLForConditionalGeneration=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: (qwen_calls.append(kwargs) or qwen_model)
            ),
        ),
    )
    model, _ = factory._load_qwen(SimpleNamespace())
    assert model.devices == [torch.device("cuda")]
    assert qwen_calls[0]["attn_implementation"] == "flash_attention_2"
    assert qwen_calls[0]["torch_dtype"] is torch.bfloat16


def test_model_loaders_fail_before_import_when_cuda_is_unavailable(monkeypatch) -> None:
    import torch

    import docprune.m3docvqa_factory as factory

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        factory._load_qwen(SimpleNamespace())


def test_load_index_manifest_rejects_schema_four(tmp_path: Path) -> None:
    path = tmp_path / "schema-4.json"
    path.write_text(json.dumps({"schema_version": 4, "manifest_sha256": "0" * 64}))

    with pytest.raises(ValueError, match="schema_version.*5"):
        _load_index_manifest(path)


def test_load_completed_qids_rejects_duplicate_and_unknown_records(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps(result_record("q-1")) + "\n" + json.dumps(result_record("q-2")) + "\n"
    )
    assert load_completed_qids(path, expected_qids=("q-1", "q-2", "q-3")) == {"q-1", "q-2"}
    path.write_text(
        json.dumps(result_record("q-1")) + "\n" + json.dumps(result_record("q-1")) + "\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_completed_qids(path, expected_qids=("q-1", "q-2"))
    path.write_text(json.dumps(result_record("q-x")) + "\n")
    with pytest.raises(ValueError, match="unexpected"):
        load_completed_qids(path, expected_qids=("q-1", "q-2"))

    path.write_text(json.dumps(result_record("q-1", question="changed")) + "\n")
    with pytest.raises(ValueError, match="question drift"):
        load_completed_qids(
            path,
            expected_samples=(SampleInput("q-1", "original", ("answer",)),),
        )


def test_validate_processor_contract_file_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "mapping_checks": {"raster_order_verified": True},
            }
        )
    )
    with pytest.raises(ValueError, match="structural"):
        validate_processor_contract_file(path)


def test_resume_manifest_requires_complete_identity_and_valid_digest(tmp_path: Path) -> None:
    payload = {
        "schema_version": 2,
        "status": "configured",
        "operation": "evaluate",
        "output": str(tmp_path.resolve()),
        "mode": "all-kept",
        "page_count": 1,
        "runtime_commit": "a" * 40,
        "m3docrag_commit": "b" * 40,
        "resources": {"qwen": {"revision": "c" * 40}},
        "processor_contract_path": str(tmp_path / "contract.json"),
        "processor_contract_sha256": "d" * 64,
        "corpus": {"integrity_sha256": "e" * 64},
        "generation": {"max_new_tokens": 128},
        "index_manifest": {"manifest_sha256": "f" * 64},
    }
    _write_run_manifest(tmp_path, payload, resume=False)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["resources"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing an immutable"):
        _write_run_manifest(tmp_path, payload, resume=True)

    with pytest.raises(FileExistsError, match="complete run"):
        _write_run_manifest(tmp_path, payload, resume=False)
    manifest_path.unlink()
    _write_run_manifest(tmp_path, payload, resume=False)
    manifest = json.loads(manifest_path.read_text())
    manifest["run_manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest"):
        _write_run_manifest(tmp_path, payload, resume=True)


def test_resume_requires_existing_output_and_incomplete_direct_manifests_collide(
    tmp_path: Path,
) -> None:
    payload = {"mode": "all-kept", "page_count": 1}
    with pytest.raises(FileNotFoundError, match="existing"):
        _write_run_manifest(tmp_path / "missing", payload, resume=True)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps({"status": "configured", "command": "evaluate", "factory": "x"})
    )
    with pytest.raises(FileExistsError, match="unrelated"):
        _write_run_manifest(output, payload, resume=False)


def test_direct_resume_binds_raw_run_and_index_source_bytes(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    output = tmp_path / "run"
    run_config = sources / "run-config.json"
    index_manifest = sources / "index-manifest.json"
    run_config.write_bytes(b"run-v1")
    index_manifest.write_bytes(b"index-v1")
    payload = {
        "schema_version": 2,
        "status": "configured",
        "operation": "evaluate",
        "output": str(output.resolve()),
        "mode": "all-kept",
        "page_count": 1,
        "runtime_commit": "a" * 40,
        "m3docrag_commit": "b" * 40,
        "resources": {},
        "processor_contract_path": "contract.json",
        "processor_contract_sha256": "c" * 64,
        "processor_contract": {},
        "corpus": {},
        "generation": {},
        "pruning_config": {},
        "selection": {},
        "index_manifest": {"manifest_sha256": "d" * 64},
        "run_config_source_path": str(run_config.resolve()),
        "run_config_source_sha256": __import__("hashlib").sha256(b"run-v1").hexdigest(),
        "index_manifest_source_path": str(index_manifest.resolve()),
        "index_manifest_source_sha256": __import__("hashlib").sha256(b"index-v1").hexdigest(),
    }
    _write_run_manifest(output, payload, resume=False)
    run_config.write_bytes(b"run-v2")
    with pytest.raises(ValueError, match="source bytes changed"):
        _write_run_manifest(output, payload, resume=True)
    run_config.write_bytes(b"run-v1")
    changed = dict(payload)
    changed["run_config_source_sha256"] = __import__("hashlib").sha256(b"run-v2").hexdigest()
    with pytest.raises(ValueError, match="exactly match"):
        _write_run_manifest(output, changed, resume=True)


def test_cli_placeholder_requires_exact_invocation_values(tmp_path: Path) -> None:
    from docprune.cli import _manifest
    from docprune.config import load_config

    config_path = Path("configs/docprune-m3docvqa.toml")
    config = load_config(config_path)
    placeholder = _manifest(
        "evaluate",
        config_path,
        config,
        1,
        "fake:factory",
        output=tmp_path,
        mode="all-kept",
        run_config=None,
        index_manifest=None,
        limit=2,
        sample_ids=("q-1",),
    )
    payload = {
        "operation": "evaluate",
        "output": str(tmp_path.resolve()),
        "mode": "all-kept",
        "page_count": 1,
    }
    assert _is_cli_placeholder(placeholder, payload, invocation=placeholder)
    for field, value in (
        ("config", "/tmp/stale.toml"),
        ("factory", "stale:factory"),
        ("limit", 3),
        ("sample_ids", ["q-2"]),
        ("output", str((tmp_path / "other").resolve())),
    ):
        stale = dict(placeholder)
        stale[field] = value
        assert not _is_cli_placeholder(stale, payload, invocation=placeholder)


def test_json_run_config_is_authoritative_and_complete(tmp_path: Path, monkeypatch) -> None:
    contract = tmp_path / "contract.json"
    corpus_root = tmp_path / "corpus"
    path_values = {
        "root": str(corpus_root),
        "questions_path": str(corpus_root / "questions.jsonl"),
        "document_ids_path": str(corpus_root / "doc_ids.json"),
        "pdf_dir": str(corpus_root / "pdfs"),
        "integrity_report_path": str(corpus_root / "integrity.json"),
        "archive_checksum_manifest_path": str(corpus_root / "archives.sha256"),
        "integrity_sha256": "a" * 64,
        "archive_checksum_manifest_sha256": "b" * 64,
        "questions_sha256": "c" * 64,
        "document_ids_sha256": "d" * 64,
        "expected_question_count": 1,
        "expected_pdf_count": 1,
        "expected_page_count": 1,
        "archive_hashes": {},
        "is_fixture": True,
    }
    payload = {
        "mode": "all-kept",
        "page_count": 1,
        "runtime_commit": "e" * 40,
        "m3docrag_commit": "f" * 40,
        "resources": {
            "qwen": {"model": "qwen", "revision": "1" * 40},
            "colpali": {"model": "colpali", "revision": "2" * 40},
            "colpali_backbone": {"model": "backbone", "revision": "3" * 40},
        },
        "processor_contract_path": str(contract),
        "corpus": path_values,
        "generation": {
            "max_new_tokens": 128,
            "do_sample": False,
            "num_beams": 1,
            "prompt": "question: $question\noutput only answer.",
        },
    }
    contract.write_text("{}")
    path = tmp_path / "run-config.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(
        "docprune.m3docvqa_factory.BenchmarkRunConfig.from_env",
        lambda *args, **kwargs: pytest.fail("authoritative JSON must not consult the environment"),
    )
    from docprune.m3docvqa_factory import _resolve_run_config

    resolved = _resolve_run_config(path, mode="all-kept", page_count=1)
    assert resolved.runtime_commit == "e" * 40
    assert resolved.corpus.root == corpus_root


@pytest.mark.parametrize("value", [1, 0, "true", "false"])
def test_json_run_config_rejects_non_boolean_corpus_fixture_flag(value) -> None:
    from docprune.m3docvqa_factory import _normalise_run_config_mapping

    corpus = {
        "root": "/corpus",
        "questions_path": "/corpus/questions.jsonl",
        "document_ids_path": "/corpus/doc_ids.json",
        "pdf_dir": "/corpus/pdfs",
        "integrity_report_path": "/corpus/integrity.json",
        "archive_checksum_manifest_path": "/corpus/archives.sha256",
        "integrity_sha256": "a" * 64,
        "archive_checksum_manifest_sha256": "b" * 64,
        "questions_sha256": "c" * 64,
        "document_ids_sha256": "d" * 64,
        "expected_question_count": 1,
        "expected_pdf_count": 1,
        "expected_page_count": 1,
        "archive_hashes": {},
        "is_fixture": value,
    }
    payload = {
        "mode": "all-kept",
        "page_count": 1,
        "runtime_commit": "e" * 40,
        "m3docrag_commit": "f" * 40,
        "resources": {
            "qwen": {"model": "qwen", "revision": "1" * 40},
            "colpali": {"model": "colpali", "revision": "2" * 40},
            "colpali_backbone": {"model": "backbone", "revision": "3" * 40},
        },
        "processor_contract_path": "/contract.json",
        "corpus": corpus,
        "generation": {
            "max_new_tokens": 128,
            "do_sample": False,
            "num_beams": 1,
            "prompt": "question: $question\noutput only answer.",
        },
    }

    with pytest.raises(ValueError, match="is_fixture"):
        _normalise_run_config_mapping(payload)


def test_index_manifest_schema_and_digest_are_checked_before_construction(tmp_path: Path) -> None:
    wrong_schema = tmp_path / "wrong-schema.json"
    wrong_schema.write_text(json.dumps({"schema_version": 3, "manifest_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="schema_version"):
        _load_index_manifest(wrong_schema)
    wrong_digest = tmp_path / "wrong-digest.json"
    wrong_digest.write_text(json.dumps({"schema_version": 5, "manifest_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="canonical"):
        _load_index_manifest(wrong_digest)


def test_index_manifest_rejects_fake_objects_instead_of_trusting_validate_files() -> None:
    class FakeManifest:
        def validate_files(self):
            return None

    with pytest.raises(TypeError, match="IndexManifest"):
        _load_index_manifest(FakeManifest())


def test_index_manifest_rejects_a_malicious_index_manifest_subclass() -> None:
    class FakeManifest(IndexManifest):
        def validate_files(self):
            return None

    fake = object.__new__(FakeManifest)
    with pytest.raises(TypeError, match="IndexManifest"):
        _load_index_manifest(fake)


def test_index_manifest_reconstructs_an_exact_but_uninitialized_instance() -> None:
    fake = object.__new__(IndexManifest)
    object.__setattr__(
        fake,
        "to_dict",
        lambda: {
            "manifest_sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        },
    )
    object.__setattr__(fake, "validate_files", lambda: None)

    with pytest.raises(ValueError, match="required immutable fields"):
        _load_index_manifest(fake)


def test_completed_results_require_full_schema_and_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"question_id": "q-1"}) + "\n")
    with pytest.raises(ValueError, match="missing"):
        load_completed_qids(path, expected_qids=("q-1",))
    target = tmp_path / "target.jsonl"
    target.write_text(json.dumps(result_record("q-1")) + "\n")
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError, match="regular"):
        load_completed_qids(path, expected_qids=("q-1",))


def test_completed_results_require_exact_requested_page_count(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(result_record("q-1")) + "\n")
    with pytest.raises(ValueError, match="page count"):
        load_completed_qids(path, expected_qids=("q-1",), expected_page_count=2)


def test_completed_results_canonicalize_legacy_qid_and_reject_conflicts(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    legacy = result_record("q-1")
    legacy["qid"] = legacy.pop("question_id")
    path.write_text(json.dumps(legacy) + "\n")
    assert load_completed_qids(path, expected_qids=("q-1",)) == {"q-1"}

    conflict = result_record("q-1")
    conflict["qid"] = "q-other"
    path.write_text(json.dumps(conflict) + "\n")
    with pytest.raises(ValueError, match="conflicting"):
        load_completed_qids(path, expected_qids=("q-1",))


def test_processor_contract_resources_are_exactly_pinned(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "resources": {
                    "qwen": {"model": QWEN_MODEL, "revision": "wrong"},
                    "colpali": {"model": COLPALI_MODEL, "revision": COLPALI_REVISION},
                    "colpali_backbone": {
                        "model": COLPALI_BACKBONE_MODEL,
                        "revision": COLPALI_BACKBONE_REVISION,
                    },
                },
                "mapping_checks": {
                    "colpali_visual_grid_inferred": True,
                    "qwen_merge_groups_valid": True,
                    "raster_order_verified": True,
                },
            }
        )
    )
    with pytest.raises(ValueError, match="resources"):
        validate_processor_contract_file(path)


def test_build_workload_rejects_unknown_operation_without_loading_models() -> None:
    with pytest.raises(ValueError, match="operation"):
        build_workload(
            operation="unknown",
            config=SimpleNamespace(),
            page_count=1,
            output=Path("/tmp/unused"),
        )
