"""Tests for sealed Task 8 MinerU smoke inputs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from docprune.task6_runtime import (
    TASK6_RENDERER_CONTRACT,
    FixedPageFixture,
    FixedPageQuestion,
    FixedPageRecord,
)
from docprune.task8_runtime import (
    finalize_task8_mineru_smoke,
    load_task8_mineru_completion,
    load_task8_smoke_inputs,
    prepare_task8_mineru_smoke,
    seal_task8_smoke_inputs,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, str, tuple[Image.Image, ...]]:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"fixed-source")
    feature = tmp_path / "features.safetensors"
    feature.write_bytes(b"fixed-features")
    dependencies = []
    for name in ("reference.json", "eligible.jsonl", "features.json", "ledger.jsonl"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        dependencies.append(path)
    images = (
        Image.new("RGB", (8, 6), color=(1, 2, 3)),
        Image.new("RGB", (8, 6), color=(4, 5, 6)),
    )
    pages = tuple(
        FixedPageRecord(
            rank=rank,
            doc_id="doc-a",
            page_index=rank + 3,
            score=float(2 - rank),
            source_pdf_path=source,
            source_pdf_sha256=_sha(source),
            rendered_rgb_width=image.width,
            rendered_rgb_height=image.height,
            rendered_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            renderer_contract=TASK6_RENDERER_CONTRACT,
            feature_shard_path=feature,
            feature_shard_sha256=_sha(feature),
            feature_page_index=rank + 3,
        )
        for rank, image in enumerate(images)
    )
    fixture = FixedPageFixture(
        fixture_version="task8-smoke-test-v1",
        reference_path=dependencies[0],
        reference_sha256=_sha(dependencies[0]),
        eligible_questions_path=dependencies[1],
        eligible_questions_sha256=_sha(dependencies[1]),
        feature_manifest_path=dependencies[2],
        feature_manifest_sha256=_sha(dependencies[2]),
        completion_ledger_path=dependencies[3],
        completion_ledger_sha256=_sha(dependencies[3]),
        questions=(FixedPageQuestion("q-1", "9" * 64, pages),),
    )
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture.to_dict(), sort_keys=True, separators=(",", ":")))
    return path, _sha(path), images


def test_smoke_input_seal_is_atomic_authenticated_and_no_replace(tmp_path: Path) -> None:
    fixture_path, fixture_sha, images = _fixture(tmp_path)
    output = tmp_path / "sealed"

    manifest = seal_task8_smoke_inputs(
        fixture_path=fixture_path,
        fixture_sha256=fixture_sha,
        qid="q-1",
        output_root=output,
        runtime_commit="a" * 40,
        render_page=lambda _path, page: images[page - 3].copy(),
    )

    assert manifest["global_index_loaded"] is False
    assert manifest["qid"] == "q-1"
    assert len(manifest["pages"]) == 2
    assert [page["input_page_index"] for page in manifest["pages"]] == [0, 1]
    assert [page["mineru_page_index"] for page in manifest["pages"]] == [0, 0]
    assert load_task8_smoke_inputs(output / "smoke-input-manifest.json") == manifest
    manifest_path = output / "smoke-input-manifest.json"
    canonical_manifest_bytes = manifest_path.read_bytes()
    manifest_path.write_bytes(canonical_manifest_bytes + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        load_task8_smoke_inputs(manifest_path)
    manifest_path.write_bytes(canonical_manifest_bytes)
    manifest_link = tmp_path / "manifest-link.json"
    manifest_link.symlink_to(output / "smoke-input-manifest.json")
    with pytest.raises(ValueError, match="regular file"):
        load_task8_smoke_inputs(manifest_link)
    with pytest.raises(FileExistsError):
        seal_task8_smoke_inputs(
            fixture_path=fixture_path,
            fixture_sha256=fixture_sha,
            qid="q-1",
            output_root=output,
            runtime_commit="a" * 40,
            render_page=lambda _path, page: images[page - 3].copy(),
        )

    first_png = Path(manifest["pages"][0]["mineru_input_path"])
    first_png.write_bytes(first_png.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="MinerU input"):
        load_task8_smoke_inputs(output / "smoke-input-manifest.json")


def test_smoke_input_seal_rejects_render_drift_without_output(tmp_path: Path) -> None:
    fixture_path, fixture_sha, _images = _fixture(tmp_path)
    output = tmp_path / "rejected"

    with pytest.raises(ValueError, match="rendered RGB"):
        seal_task8_smoke_inputs(
            fixture_path=fixture_path,
            fixture_sha256=fixture_sha,
            qid="q-1",
            output_root=output,
            runtime_commit="a" * 40,
            render_page=lambda _path, _page: Image.new("RGB", (8, 6), color="white"),
        )
    assert not output.exists()

    dangling_output = tmp_path / "dangling-output"
    dangling_output.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    with pytest.raises(FileExistsError):
        seal_task8_smoke_inputs(
            fixture_path=fixture_path,
            fixture_sha256=fixture_sha,
            qid="q-1",
            output_root=dangling_output,
            runtime_commit="a" * 40,
            render_page=lambda _path, _page: Image.new("RGB", (8, 6), color="white"),
        )

    parent_target = tmp_path / "parent-target"
    parent_target.mkdir()
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(parent_target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        seal_task8_smoke_inputs(
            fixture_path=fixture_path,
            fixture_sha256=fixture_sha,
            qid="q-1",
            output_root=parent_link / "sealed",
            runtime_commit="a" * 40,
            render_page=lambda _path, _page: Image.new("RGB", (8, 6), color="white"),
        )


def test_smoke_input_seal_does_not_replace_destination_created_during_staging(
    tmp_path: Path,
) -> None:
    fixture_path, fixture_sha, images = _fixture(tmp_path)
    output = tmp_path / "raced"
    created = False

    def render(_path: Path, page: int) -> Image.Image:
        nonlocal created
        if not created:
            output.mkdir()
            (output / "owner-marker").write_text("preexisting")
            created = True
        return images[page - 3].copy()

    with pytest.raises(FileExistsError):
        seal_task8_smoke_inputs(
            fixture_path=fixture_path,
            fixture_sha256=fixture_sha,
            qid="q-1",
            output_root=output,
            runtime_commit="a" * 40,
            render_page=render,
        )
    assert (output / "owner-marker").read_text() == "preexisting"
    assert not (output / "smoke-input-manifest.json").exists()


def _write_mineru_contract(
    tmp_path: Path,
) -> tuple[Path, str, Path, str, Path, Path, str]:
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"pinned model")
    model_config = model / "config.json"
    model_config.write_bytes(b"pinned config")
    config = tmp_path / "mineru-config.json"
    config.write_text(
        json.dumps({"models-dir": {"vlm": str(model)}}, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    config_sha = _sha(config)
    tool = tmp_path / "mineru-tool.json"
    tool.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pinned-mineru-tool",
                "backend": "vlm",
                "version": "3.0.9",
                "repository_id": "https://github.com/opendatalab/MinerU",
                "repository_revision": "d9cd58add047c2364c1198eefcb1ee9cd63a971a",
                "model_repository_id": "opendatalab/MinerU2.5-Pro-2604-1.2B",
                "model_revision": "d3f5e08d073c21466bbabe21c71bb1e9c2e595da",
                "configuration_path": str(config),
                "configuration_sha256": config_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    inventory = tmp_path / "model-sha256.txt"
    inventory.write_text(
        "".join(f"{_sha(path)}  {path}\n" for path in sorted((model_config, weights))),
        encoding="utf-8",
    )
    return config, config_sha, tool, _sha(tool), weights, inventory, _sha(inventory)


def _write_middle(path: Path, *, version: str = "3.0.9") -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "_backend": "vlm",
                "_version_name": version,
                "pdf_info": [
                    {
                        "page_idx": 0,
                        "page_size": [8, 6],
                        "para_blocks": [{"type": "text", "bbox": [0, 0, 8, 6]}],
                    }
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def test_mineru_smoke_prepare_and_finalize_authenticate_exact_stage_contract(
    tmp_path: Path,
) -> None:
    fixture_path, fixture_sha, images = _fixture(tmp_path)
    sealed = tmp_path / "sealed"
    smoke = seal_task8_smoke_inputs(
        fixture_path=fixture_path,
        fixture_sha256=fixture_sha,
        qid="q-1",
        output_root=sealed,
        runtime_commit="a" * 40,
        render_page=lambda _path, page: images[page - 3].copy(),
    )
    smoke_path = sealed / "smoke-input-manifest.json"
    config, config_sha, tool, tool_sha, weights, inventory, inventory_sha = _write_mineru_contract(
        tmp_path
    )
    model_config = weights.parent / "config.json"
    original_model_config = model_config.read_bytes()
    model_config.write_bytes(b"drifted config")
    drift_job = tmp_path / "drift-job"
    drift_job.mkdir()
    with pytest.raises(ValueError, match="snapshot file SHA-256 mismatch"):
        prepare_task8_mineru_smoke(
            smoke_input_manifest_path=smoke_path,
            smoke_input_manifest_sha256=_sha(smoke_path),
            configuration_path=config,
            configuration_sha256=config_sha,
            tool_manifest_path=tool,
            tool_manifest_sha256=tool_sha,
            model_weights_path=weights,
            model_weights_sha256=_sha(weights),
            model_inventory_path=inventory,
            model_inventory_sha256=inventory_sha,
            job_root=drift_job,
            output_dir=drift_job / "mineru-output",
            runtime_commit="f" * 40,
        )
    assert not (drift_job / "run-manifest.json").exists()
    model_config.write_bytes(original_model_config)
    job_root = tmp_path / "job"
    job_root.mkdir()
    output = job_root / "mineru-output"
    gpu = job_root / "gpu.json"
    gpu.write_text(
        json.dumps(
            {"name": "NVIDIA L40S", "memory_total_mib": 46068, "driver_version": "1"},
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    prepared = prepare_task8_mineru_smoke(
        smoke_input_manifest_path=smoke_path,
        smoke_input_manifest_sha256=_sha(smoke_path),
        configuration_path=config,
        configuration_sha256=config_sha,
        tool_manifest_path=tool,
        tool_manifest_sha256=tool_sha,
        model_weights_path=weights,
        model_weights_sha256=_sha(weights),
        model_inventory_path=inventory,
        model_inventory_sha256=inventory_sha,
        job_root=job_root,
        output_dir=output,
        runtime_commit="f" * 40,
    )

    assert prepared["status"] == "prepared"
    assert prepared["engine_expected"] == "transformers"
    assert prepared["model_inventory_count"] == 2
    assert prepared["global_index_loaded"] is False
    assert prepared["retrieval_run"] is False
    assert [Path(row["path"]).name for row in prepared["inputs"]] == [
        "page-00.png",
        "page-01.png",
    ]
    assert not output.exists()
    assert (job_root / "run-manifest.json").is_file()
    with pytest.raises(FileExistsError):
        prepare_task8_mineru_smoke(
            smoke_input_manifest_path=smoke_path,
            smoke_input_manifest_sha256=_sha(smoke_path),
            configuration_path=config,
            configuration_sha256=config_sha,
            tool_manifest_path=tool,
            tool_manifest_sha256=tool_sha,
            model_weights_path=weights,
            model_weights_sha256=_sha(weights),
            model_inventory_path=inventory,
            model_inventory_sha256=inventory_sha,
            job_root=job_root,
            output_dir=output,
            runtime_commit="f" * 40,
        )

    for page in smoke["pages"]:
        stem = Path(page["mineru_input_path"]).stem
        _write_middle(output / stem / "vlm" / f"{stem}_middle.json")
    (output / "page-00" / "vlm" / "page-00.md").write_text("parsed", encoding="utf-8")

    completed = finalize_task8_mineru_smoke(
        job_root=job_root,
        output_dir=output,
        gpu_manifest_path=gpu,
        gpu_manifest_sha256=_sha(gpu),
    )

    assert completed["status"] == "complete"
    assert completed["raw_middle_json_count"] == 2
    assert completed["global_index_loaded"] is False
    assert completed["retrieval_run"] is False
    assert [row["region_count"] for row in completed["artifacts"]] == [1, 1]
    assert (job_root / "completion-manifest.json").is_file()
    completion_path = job_root / "completion-manifest.json"
    assert (
        load_task8_mineru_completion(completion_path, expected_sha256=_sha(completion_path))
        == completed
    )
    run_path = job_root / "run-manifest.json"
    canonical_run = run_path.read_bytes()
    contradicted_run = json.loads(canonical_run)
    contradicted_run["qid"] = "different-qid"
    unsigned_run = {
        key: value for key, value in contradicted_run.items() if key != "manifest_sha256"
    }
    contradicted_run["manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned_run, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    run_path.write_text(json.dumps(contradicted_run, sort_keys=True, separators=(",", ":")))
    contradicted_completion = dict(completed)
    contradicted_completion["run_manifest_sha256"] = _sha(run_path)
    unsigned_completion = {
        key: value for key, value in contradicted_completion.items() if key != "manifest_sha256"
    }
    contradicted_completion["manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned_completion, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    completion_path.write_text(
        json.dumps(contradicted_completion, sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(ValueError, match="identity"):
        load_task8_mineru_completion(completion_path, expected_sha256=_sha(completion_path))
    run_path.write_bytes(canonical_run)
    completion_path.write_text(json.dumps(completed, sort_keys=True, separators=(",", ":")))
    with pytest.raises(FileExistsError):
        finalize_task8_mineru_smoke(
            job_root=job_root,
            output_dir=output,
            gpu_manifest_path=gpu,
            gpu_manifest_sha256=_sha(gpu),
        )
    canonical = completion_path.read_bytes()
    completion_path.write_bytes(canonical + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        load_task8_mineru_completion(completion_path, expected_sha256=_sha(completion_path))


def test_mineru_smoke_finalize_rejects_missing_or_wrong_output_without_completion(
    tmp_path: Path,
) -> None:
    fixture_path, fixture_sha, images = _fixture(tmp_path)
    sealed = tmp_path / "sealed"
    seal_task8_smoke_inputs(
        fixture_path=fixture_path,
        fixture_sha256=fixture_sha,
        qid="q-1",
        output_root=sealed,
        runtime_commit="a" * 40,
        render_page=lambda _path, page: images[page - 3].copy(),
    )
    smoke_path = sealed / "smoke-input-manifest.json"
    config, config_sha, tool, tool_sha, weights, inventory, inventory_sha = _write_mineru_contract(
        tmp_path
    )
    job_root = tmp_path / "job"
    job_root.mkdir()
    output = job_root / "mineru-output"
    gpu = job_root / "gpu.json"
    gpu.write_text(
        json.dumps(
            {"name": "GPU", "memory_total_mib": 24000, "driver_version": "1"},
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    prepare_task8_mineru_smoke(
        smoke_input_manifest_path=smoke_path,
        smoke_input_manifest_sha256=_sha(smoke_path),
        configuration_path=config,
        configuration_sha256=config_sha,
        tool_manifest_path=tool,
        tool_manifest_sha256=tool_sha,
        model_weights_path=weights,
        model_weights_sha256=_sha(weights),
        model_inventory_path=inventory,
        model_inventory_sha256=inventory_sha,
        job_root=job_root,
        output_dir=output,
        runtime_commit="f" * 40,
    )
    run_path = job_root / "run-manifest.json"
    original_run = run_path.read_bytes()
    tampered_run = json.loads(original_run)
    tampered_run["qid"] = "wrong-qid"
    unsigned = {key: value for key, value in tampered_run.items() if key != "manifest_sha256"}
    tampered_run["manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    run_path.write_text(
        json.dumps(tampered_run, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="identity does not replay"):
        finalize_task8_mineru_smoke(
            job_root=job_root,
            output_dir=output,
            gpu_manifest_path=gpu,
            gpu_manifest_sha256=_sha(gpu),
        )
    assert not (job_root / "completion-manifest.json").exists()
    run_path.write_bytes(original_run)
    _write_middle(output / "page-00" / "vlm" / "page-00_middle.json", version="3.1.0")

    with pytest.raises(ValueError, match="exact expected middle JSON set|backend or version"):
        finalize_task8_mineru_smoke(
            job_root=job_root,
            output_dir=output,
            gpu_manifest_path=gpu,
            gpu_manifest_sha256=_sha(gpu),
        )
    assert not (job_root / "completion-manifest.json").exists()


def test_task8_mineru_launcher_is_short_offline_fixed_page_and_no_requeue() -> None:
    launcher = Path("examples/sbatch/36_docprune_task8_mineru_smoke.sbatch").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --time=00:20:00" in launcher
    assert "#SBATCH --no-requeue" in launcher
    assert "#SBATCH --array" not in launcher
    assert "--backend vlm-auto-engine" in launcher
    assert "MINERU_MODEL_SOURCE=local" in launcher
    assert "MINERU_API_MAX_CONCURRENT_REQUESTS=1" in launcher
    assert 'find_spec("vllm") is None' in launcher
    assert 'find_spec("lmdeploy") is None' in launcher
    assert 'PYTHONPATH="$MINERU_SOURCE:$RUNTIME_DIR/src"' in launcher
    assert "print(mineru.__file__)" in launcher
    assert 'sha256sum --quiet -c "$MODEL_INVENTORY"' in launcher
    assert "MODEL_INVENTORY_SHA256=5ec9100c" in launcher
    assert "HF_HUB_OFFLINE=1" in launcher
    assert "TRANSFORMERS_OFFLINE=1" in launcher
    assert "smoke-input-manifest.json" in launcher
    assert "global index" not in launcher.lower()
    assert "retrieve" not in launcher.lower()
    assert launcher.index("prepare \\\n") < launcher.index('mkdir "$OUTPUT_DIR"')
    assert launcher.index('mkdir "$OUTPUT_DIR"') < launcher.index('"$MINERU" \\\n')


def test_task9_preliminary_mineru_launcher_is_question_sharded_and_retrieval_free() -> None:
    launcher = Path(
        "examples/sbatch/42_docprune_task9_preliminary_mineru.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --array=0-47%8" in launcher
    assert "#SBATCH --constraint=l40s" in launcher
    assert "#SBATCH --time=00:20:00" in launcher
    assert "preprocessing-manifest.json" in launcher
    assert 'SLURM_ARRAY_TASK_ID' in launcher
    assert 'row["input_manifest"]' in launcher
    assert 'row["input_root"]' in launcher
    assert "--backend vlm-auto-engine" in launcher
    assert "HF_HUB_OFFLINE=1" in launcher
    assert "TRANSFORMERS_OFFLINE=1" in launcher
    assert "#SBATCH --no-requeue" in launcher


def test_task8_gpu_probe_uses_the_single_cuda_visible_device_without_sigpipe(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' '595.71.05' '595.71.05'\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    (fake_modules / "torch.py").write_text(
        "class Properties:\n"
        "    name = 'NVIDIA L40S'\n"
        "    total_memory = 46068 * 1024 * 1024\n"
        "class Cuda:\n"
        "    @staticmethod\n"
        "    def is_available(): return True\n"
        "    @staticmethod\n"
        "    def device_count(): return 1\n"
        "    @staticmethod\n"
        "    def current_device(): return 0\n"
        "    @staticmethod\n"
        "    def get_device_properties(index): return Properties()\n"
        "cuda = Cuda()\n",
        encoding="utf-8",
    )
    output = tmp_path / "gpu.json"
    probe = Path("examples/probe_task8_gpu.sh").resolve()
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["PYTHONPATH"] = f"{fake_modules}:{environment.get('PYTHONPATH', '')}"

    subprocess.run(
        [str(probe), "/home/lmalveau/mamba-envs/docprune-sol/bin/python", str(output)],
        check=True,
        env=environment,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "driver_version": "595.71.05",
        "memory_total_mib": 46068,
        "name": "NVIDIA L40S",
    }
    launcher = Path("examples/sbatch/36_docprune_task8_mineru_smoke.sbatch").read_text(
        encoding="utf-8"
    )
    assert "probe_task8_gpu.sh" in launcher
    assert "nvidia-smi" not in launcher
    assert "head -n 1" not in launcher
