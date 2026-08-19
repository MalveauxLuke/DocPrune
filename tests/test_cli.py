import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import docprune.processor_probe
from docprune.cli import (
    EvaluationWorkload,
    _manifest,
    _manifest_digest,
    _prepare_output,
    _validate_embed_result,
    main,
)
from docprune.config import load_config
from docprune.indexing import IndexBuildResult
from docprune.m3docrag import RetrievedPage, SampleInput, SampleResult, SampleTiming
from docprune.qwen2vl.model import PruningTrace


def test_inspect_validates_config_without_loading_models(capsys) -> None:
    exit_code = main(["inspect", "--config", "configs/docprune-m3docvqa.toml", "--pages", "4"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["page_count"] == 4
    assert payload["paper_values"]["question_threshold"] == 0.4
    assert payload["upstream"]["m3docrag_commit"].startswith("29e6ac")


def test_summarize_prints_aggregate_json(tmp_path, capsys) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(
            {
                "trace": {
                    "original_visual_tokens": 10,
                    "post_btp_visual_tokens": 8,
                    "post_qtp_visual_tokens": 6,
                    "post_ctp_visual_tokens": 4,
                    "ctp_layer": 3,
                },
                "timing": {"retrieval_seconds": 1.0, "qa_seconds": 1.0},
            }
        )
        + "\n"
    )

    exit_code = main(["summarize", "--results", str(results)])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["samples"] == 1


def test_evaluate_dry_run_refuses_existing_output(tmp_path, capsys) -> None:
    output = tmp_path / "run"
    output.mkdir()

    exit_code = main(
        [
            "evaluate",
            "--config",
            "configs/docprune-m3docvqa.toml",
            "--pages",
            "1",
            "--output",
            str(output),
            "--factory",
            "fake.module:build",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "already exists" in capsys.readouterr().err


def test_evaluate_dry_run_emits_manifest_for_new_output(tmp_path, capsys) -> None:
    output = tmp_path / "run"

    exit_code = main(
        [
            "evaluate",
            "--config",
            "configs/docprune-m3docvqa.toml",
            "--pages",
            "2",
            "--output",
            str(output),
            "--factory",
            "bridge:build",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "dry_run"
    assert payload["factory"] == "bridge:build"
    assert payload["page_count"] == 2
    assert not output.exists()


def test_evaluate_resume_requires_existing_output(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "evaluate",
            "--config",
            "configs/docprune-m3docvqa.toml",
            "--pages",
            "1",
            "--output",
            str(tmp_path / "missing"),
            "--factory",
            "fake:factory",
            "--resume",
        ]
    )
    assert exit_code == 2
    assert "existing output" in capsys.readouterr().err


def test_evaluate_resume_rejects_incomplete_manifest_before_custom_factory(
    tmp_path, capsys, monkeypatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    config_path = Path("configs/docprune-m3docvqa.toml")
    config = load_config(config_path)
    (output / "run_manifest.json").write_text(
        json.dumps(
            _manifest(
                "evaluate",
                config_path,
                config,
                1,
                "fake:factory",
                output=output,
                mode="all-kept",
                run_config=None,
                index_manifest=None,
                limit=None,
                sample_ids=None,
            )
        )
    )
    calls = []
    monkeypatch.setattr("docprune.cli._load_factory", lambda spec: calls.append(spec))

    exit_code = main(
        [
            "evaluate",
            "--config",
            str(config_path),
            "--pages",
            "1",
            "--output",
            str(output),
            "--factory",
            "fake:factory",
            "--resume",
        ]
    )

    assert exit_code == 2
    assert calls == []
    assert "complete" in capsys.readouterr().err


def test_evaluate_rejects_workload_without_manifest_before_results(tmp_path, monkeypatch) -> None:
    output = tmp_path / "run"

    class Runner:
        def run_sample(self, sample):
            raise AssertionError("runner must not execute without a workload manifest")

    def factory(**kwargs):
        return EvaluationWorkload(Runner(), (SimpleNamespace(question_id="q-1", question="q"),))

    monkeypatch.setattr("docprune.cli._load_factory", lambda spec: factory)
    exit_code = main(
        [
            "evaluate",
            "--config",
            "configs/docprune-m3docvqa.toml",
            "--pages",
            "1",
            "--output",
            str(output),
            "--factory",
            "fake:factory",
        ]
    )

    assert exit_code == 2
    assert not (output / "results.jsonl").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operation", "embed"),
        ("output", "/tmp/wrong-output"),
        ("mode", "docprune"),
        ("page_count", 2),
        ("selection", {"count": 999}),
        ("schema_version", 99),
        ("status", "tampered"),
    ],
)
def test_evaluate_factory_manifest_cannot_change_invocation_identity(
    tmp_path, monkeypatch, field, value
) -> None:
    output = tmp_path / "run"
    config = load_config(Path("configs/docprune-m3docvqa.toml"))
    from docprune.m3docvqa_factory import _expected_pruning_identity

    baseline = {
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
        "run_config_source_path": None,
        "run_config_source_sha256": None,
        "index_manifest_source_path": None,
        "index_manifest_source_sha256": None,
        "corpus": {},
        "generation": {},
        "pruning_config": _expected_pruning_identity(config, mode="all-kept", page_count=1),
        "selection": {
            "requested_sample_ids": None,
            "limit": None,
            "resolved_question_ids": [],
            "count": 0,
        },
        "index_manifest": {},
    }
    baseline["run_manifest_sha256"] = _manifest_digest(baseline)
    monkeypatch.setattr(
        "docprune.cli._resolve_evaluate_authority",
        lambda *args, **kwargs: dict(baseline),
    )

    def factory(**kwargs):
        manifest = dict(baseline)
        manifest[field] = value
        return EvaluationWorkload(SimpleNamespace(run_sample=lambda sample: None), (), manifest)

    monkeypatch.setattr("docprune.cli._load_factory", lambda spec: factory)
    assert (
        main(
            [
                "evaluate",
                "--config",
                "configs/docprune-m3docvqa.toml",
                "--pages",
                "1",
                "--output",
                str(output),
                "--factory",
                "fake:factory",
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runtime_commit", "f" * 40),
        ("resources", {"pinned": False}),
        ("corpus", {"integrity_sha256": "0" * 64}),
        ("index_manifest", {"manifest_sha256": "0" * 64}),
    ],
)
def test_fresh_evaluate_factory_cannot_change_authoritative_identity(
    tmp_path, monkeypatch, field, value
) -> None:
    output = tmp_path / "run"
    config = load_config(Path("configs/docprune-m3docvqa.toml"))
    from docprune.m3docvqa_factory import _expected_pruning_identity

    authoritative = {
        "schema_version": 2,
        "status": "configured",
        "operation": "evaluate",
        "output": str(output.resolve()),
        "mode": "all-kept",
        "page_count": 1,
        "runtime_commit": "a" * 40,
        "m3docrag_commit": "b" * 40,
        "resources": {"pinned": True},
        "processor_contract_path": "contract.json",
        "processor_contract_sha256": "c" * 64,
        "processor_contract": {"mapping": "pinned"},
        "run_config_source_path": None,
        "run_config_source_sha256": None,
        "index_manifest_source_path": None,
        "index_manifest_source_sha256": None,
        "corpus": {"integrity_sha256": "d" * 64},
        "generation": {"do_sample": False},
        "pruning_config": _expected_pruning_identity(config, mode="all-kept", page_count=1),
        "selection": {
            "requested_sample_ids": None,
            "limit": None,
            "resolved_question_ids": [],
            "count": 0,
        },
        "index_manifest": {"manifest_sha256": "e" * 64},
    }
    authoritative["run_manifest_sha256"] = _manifest_digest(authoritative)
    monkeypatch.setattr(
        "docprune.cli._resolve_evaluate_authority",
        lambda *args, **kwargs: dict(authoritative),
    )

    def factory(**kwargs):
        wrong = dict(authoritative)
        wrong[field] = value
        return EvaluationWorkload(SimpleNamespace(run_sample=lambda sample: None), (), wrong)

    monkeypatch.setattr("docprune.cli._load_factory", lambda spec: factory)
    assert (
        main(
            [
                "evaluate",
                "--config",
                "configs/docprune-m3docvqa.toml",
                "--pages",
                "1",
                "--output",
                str(output),
                "--factory",
                "fake:factory",
            ]
        )
        == 2
    )
    assert not (output / "results.jsonl").exists()


def test_fresh_evaluate_factory_samples_bind_to_authoritative_rows(tmp_path, monkeypatch) -> None:
    output = tmp_path / "run"
    config = load_config(Path("configs/docprune-m3docvqa.toml"))
    from docprune.m3docvqa_factory import _expected_pruning_identity

    authoritative_sample = SampleInput("q-1", "authoritative question", ("gold",))
    authoritative = {
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
        "run_config_source_path": None,
        "run_config_source_sha256": None,
        "index_manifest_source_path": None,
        "index_manifest_source_sha256": None,
        "corpus": {},
        "generation": {},
        "pruning_config": _expected_pruning_identity(config, mode="all-kept", page_count=1),
        "selection": {
            "requested_sample_ids": None,
            "limit": None,
            "resolved_question_ids": ["q-1"],
            "count": 1,
        },
        "index_manifest": {},
    }
    authoritative["run_manifest_sha256"] = _manifest_digest(authoritative)
    authority = dict(authoritative)
    authority["_authoritative_samples"] = (authoritative_sample,)
    monkeypatch.setattr(
        "docprune.cli._resolve_evaluate_authority",
        lambda *args, **kwargs: authority,
    )

    class Runner:
        def run_sample(self, sample):
            raise AssertionError("runner must not run for an untrusted sample")

    def factory(**kwargs):
        wrong_sample = SampleInput("q-1", "factory-controlled question", ("wrong",))
        return EvaluationWorkload(Runner(), (wrong_sample,), dict(authoritative))

    monkeypatch.setattr("docprune.cli._load_factory", lambda spec: factory)
    assert (
        main(
            [
                "evaluate",
                "--config",
                "configs/docprune-m3docvqa.toml",
                "--pages",
                "1",
                "--output",
                str(output),
                "--factory",
                "fake:factory",
            ]
        )
        == 2
    )
    assert not (output / "results.jsonl").exists()


def test_resume_rejects_changed_run_config_content_at_same_path(tmp_path) -> None:
    output = tmp_path / "run"
    run_config = tmp_path / "run-config.json"
    run_config.write_text('{"generation": {"max_new_tokens": 128}}')
    index_manifest = tmp_path / "index-manifest.json"
    index_manifest.write_text('{"manifest_sha256": "old"}')
    config_path = Path("configs/docprune-m3docvqa.toml")
    config = load_config(config_path)
    manifest = _manifest(
        "evaluate",
        config_path,
        config,
        1,
        "fake:factory",
        output=output,
        mode="all-kept",
        run_config=run_config,
        index_manifest=index_manifest,
        limit=None,
        sample_ids=None,
    )
    _prepare_output(output, manifest, resume=False)
    complete = json.loads((output / "run_manifest.json").read_text())
    complete.update(
        {
            "operation": "evaluate",
            "runtime_commit": "a" * 40,
            "m3docrag_commit": "b" * 40,
            "resources": {},
            "processor_contract_path": "contract.json",
            "processor_contract_sha256": "c" * 64,
            "processor_contract": {},
            "run_config_source_path": None,
            "run_config_source_sha256": None,
            "index_manifest_source_path": None,
            "index_manifest_source_sha256": None,
            "corpus": {},
            "generation": {},
            "pruning_config": {},
            "selection": {},
            "index_manifest": {},
        }
    )
    complete["run_manifest_sha256"] = _manifest_digest(
        {key: value for key, value in complete.items() if key != "run_manifest_sha256"}
    )
    (output / "run_manifest.json").write_text(json.dumps(complete))
    run_config.write_text('{"generation": {"max_new_tokens": 64}}')
    index_manifest.write_text('{"manifest_sha256": "new"}')
    changed = _manifest(
        "evaluate",
        config_path,
        config,
        1,
        "fake:factory",
        output=output,
        mode="all-kept",
        run_config=run_config,
        index_manifest=index_manifest,
        limit=None,
        sample_ids=None,
    )
    with pytest.raises(ValueError, match="exactly match"):
        _prepare_output(output, changed, resume=True)


def test_embed_requires_authoritative_default_run_config_before_custom_factory(
    tmp_path, monkeypatch
) -> None:
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr("docprune.cli._load_factory", lambda spec: factory)
    monkeypatch.delenv("DOCPRUNE_CORPUS_ROOT", raising=False)
    monkeypatch.delenv("DOCPRUNE_RUNTIME_COMMIT", raising=False)

    exit_code = main(
        [
            "embed",
            "--config",
            "configs/docprune-m3docvqa.toml",
            "--pages",
            "1",
            "--output",
            str(tmp_path / "embed"),
            "--factory",
            "fake:factory",
        ]
    )

    assert exit_code == 2
    assert calls == []


def test_probe_processors_forwards_exact_inputs(tmp_path, capsys, monkeypatch) -> None:
    page = tmp_path / "page.png"
    page.write_bytes(b"fake")
    output = tmp_path / "processor-contract.json"
    calls = []

    def fake_probe(**kwargs):
        calls.append(kwargs)
        output.write_text('{"schema_version": 1}\n')
        return {"schema_version": 1}

    monkeypatch.setattr(docprune.processor_probe, "run_processor_probe", fake_probe)
    exit_code = main(
        [
            "probe-processors",
            "--page-image",
            str(page),
            "--qwen-model",
            "Qwen/Qwen2-VL-7B-Instruct",
            "--qwen-revision",
            "eed13092ef92e448dd6875b2a00151bd3f7db0ac",
            "--colpali-model",
            "vidore/colpali-v1.2",
            "--colpali-revision",
            "961b51745de3e9adb3468ac5c9ccca0ac626c217",
            "--colpali-backbone-model",
            "vidore/colpaligemma-3b-pt-448-base",
            "--colpali-backbone-revision",
            "30ab955d073de4a91dc5a288e8c97226647e3e5a",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"schema_version": 1}
    assert calls == [
        {
            "page_image": page,
            "qwen_model": "Qwen/Qwen2-VL-7B-Instruct",
            "qwen_revision": "eed13092ef92e448dd6875b2a00151bd3f7db0ac",
            "colpali_model": "vidore/colpali-v1.2",
            "colpali_revision": "961b51745de3e9adb3468ac5c9ccca0ac626c217",
            "colpali_backbone_model": "vidore/colpaligemma-3b-pt-448-base",
            "colpali_backbone_revision": "30ab955d073de4a91dc5a288e8c97226647e3e5a",
            "output": output,
        }
    ]


def test_probe_processors_rejects_symbolic_revision(tmp_path, capsys) -> None:
    page = tmp_path / "page.png"
    page.write_bytes(b"fake")

    exit_code = main(
        [
            "probe-processors",
            "--page-image",
            str(page),
            "--qwen-model",
            "Qwen/Qwen2-VL-7B-Instruct",
            "--qwen-revision",
            "main",
            "--colpali-model",
            "vidore/colpali-v1.2",
            "--colpali-revision",
            "961b51745de3e9adb3468ac5c9ccca0ac626c217",
            "--colpali-backbone-model",
            "vidore/colpaligemma-3b-pt-448-base",
            "--colpali-backbone-revision",
            "30ab955d073de4a91dc5a288e8c97226647e3e5a",
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert exit_code == 2
    assert "must equal the pinned" in capsys.readouterr().err


def test_evaluate_resume_uses_qids_without_duplicate_records(tmp_path, monkeypatch) -> None:
    output = tmp_path / "run"
    calls = {"runs": 0, "answers": []}
    config = load_config(Path("configs/docprune-m3docvqa.toml"))
    from docprune.m3docvqa_factory import _expected_pruning_identity

    workload_manifest = {
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
        "run_config_source_path": None,
        "run_config_source_sha256": None,
        "index_manifest_source_path": None,
        "index_manifest_source_sha256": None,
        "corpus": {},
        "generation": {},
        "pruning_config": _expected_pruning_identity(config, mode="all-kept", page_count=1),
        "selection": {
            "requested_sample_ids": None,
            "limit": None,
            "resolved_question_ids": ["q-1", "q-2", "q-3"],
            "count": 3,
        },
        "index_manifest": {},
    }
    workload_manifest["run_manifest_sha256"] = _manifest_digest(workload_manifest)
    monkeypatch.setattr(
        "docprune.cli._resolve_evaluate_authority",
        lambda *args, **kwargs: dict(workload_manifest),
    )

    class Runner:
        def run_sample(self, sample):
            calls["answers"].append(sample.question_id)
            if calls["runs"] == 1 and sample.question_id == "q-2":
                raise RuntimeError("interrupted")
            return SampleResult(
                question_id=sample.question_id,
                question=sample.question,
                answers=sample.answers,
                predicted_answer="answer",
                retrieved_pages=(RetrievedPage("doc", 0, 1.0),),
                trace=PruningTrace(4, 3, 2, 1, None),
                timing=SampleTiming(0.1, 0.2),
            )

    def factory(**kwargs):
        calls["runs"] += 1
        from docprune.m3docrag import SampleInput

        return EvaluationWorkload(
            Runner(),
            (
                SampleInput("q-1", "one"),
                SampleInput("q-2", "two"),
                SampleInput("q-3", "three"),
            ),
            manifest=dict(workload_manifest),
        )

    monkeypatch.setattr("docprune.cli._load_factory", lambda spec: factory)
    args = [
        "evaluate",
        "--config",
        "configs/docprune-m3docvqa.toml",
        "--pages",
        "1",
        "--output",
        str(output),
        "--factory",
        "fake:factory",
    ]
    with pytest.raises(RuntimeError, match="interrupted"):
        main(args)
    assert calls["answers"] == ["q-1", "q-2"]

    calls["answers"] = []
    assert main(args + ["--resume"]) == 0
    assert calls["answers"] == ["q-2", "q-3"]
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert [record["question_id"] for record in records] == ["q-1", "q-2", "q-3"]
    manifest = json.loads((output / "run_manifest.json").read_text())
    unsigned = dict(manifest)
    digest = unsigned.pop("run_manifest_sha256")
    import hashlib

    assert (
        digest
        == hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert not any(path.name.startswith(".run_manifest.") for path in output.iterdir())


def test_cli_embed_validates_manifest_identity_before_publication(tmp_path) -> None:
    config = load_config(Path("configs/docprune-m3docvqa.toml"))
    output = tmp_path / "output"
    artifact_root = output / "all-kept"
    artifact_root.mkdir(parents=True)
    contract = tmp_path / "contract.json"
    contract.write_text("contract")
    manifest = SimpleNamespace(
        mode="all-kept",
        page_count=1,
        m3docrag_commit=config.m3docrag_commit,
        qwen_model="Qwen/Qwen2-VL-7B-Instruct",
        qwen_revision="eed13092ef92e448dd6875b2a00151bd3f7db0ac",
        colpali_model="vidore/colpali-v1.2",
        colpali_revision="961b51745de3e9adb3468ac5c9ccca0ac626c217",
        colpali_backbone_model="vidore/colpaligemma-3b-pt-448-base",
        colpali_backbone_revision="30ab955d073de4a91dc5a288e8c97226647e3e5a",
        pruning_config={
            "enabled": False,
            "page_settings": {
                "retrieval_background_threshold": 0.9,
                "qa_background_threshold": 0.9,
                "background_error_tolerance": 1.0,
                "question_threshold": 0.3,
                "comprehension_threshold": 65.0,
                "attention_threshold": 0.5,
            },
            "reconstruction_defaults": {
                "source": "reconstruction_default",
                "grayscale": "bt601_uint8",
                "gaussian_sigma": 1.0,
                "group_retention": "any",
                "attention_head_aggregation": "mean",
                "ctp_timing": "prefill_last_prompt_token",
            },
            "siglip_patch_size": 14,
        },
        artifact_root=artifact_root,
        processor_contract_path=contract,
        processor_contract_sha256="0" * 64,
        corpus_integrity_sha256="1" * 64,
        source_order_sha256="2" * 64,
        validate_files=lambda: None,
    )
    result = IndexBuildResult(
        manifest=manifest,
        manifest_path=artifact_root / "manifest.json",
        mode_root=artifact_root,
        token2pageuid_path=artifact_root / "token2pageuid.json",
        completion_ledger_path=artifact_root / "completion-ledger.json",
        documents_built=0,
        documents_resumed=0,
    )
    with pytest.raises(ValueError, match="processor contract checksum"):
        _validate_embed_result(result, config=config, pages=1, mode="all-kept", output=output)
    manifest.processor_contract_sha256 = __import__("hashlib").sha256(b"contract").hexdigest()
    manifest.mode = "docprune"
    with pytest.raises(ValueError, match="mode/page_count"):
        _validate_embed_result(result, config=config, pages=1, mode="all-kept", output=output)
