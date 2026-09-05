"""Tests for the H200 Task 9 one-page MinerU smoke preparation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

from docprune.task6_runtime import (
    TASK6_RENDERER_CONTRACT,
    FixedPageFixture,
    FixedPageQuestion,
    FixedPageRecord,
)
from docprune.task8_runtime import load_task8_smoke_inputs, seal_task8_smoke_inputs
from docprune.task9_h200_mineru import (
    _foreign_gpu_pids,
    prepare_task9_one_page_mineru_smoke,
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
        fixture_version="task9-h200-test-v1",
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


def test_prepare_task9_one_page_smoke_is_authenticated_and_deterministic(
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
    cache = tmp_path / "cache"
    blobs = tmp_path / "blobs"
    cache.mkdir()
    blobs.mkdir()
    for name, raw in (("config.json", b"config"), ("model.safetensors", b"weights")):
        blob = blobs / name
        blob.write_bytes(raw)
        (cache / name).symlink_to(blob)
    model_dir = tmp_path / "model"
    job_root = tmp_path / "job"

    result = prepare_task9_one_page_mineru_smoke(
        source_fixture_path=fixture_path,
        source_fixture_sha256=fixture_sha,
        source_smoke_manifest_path=sealed / "smoke-input-manifest.json",
        source_model_snapshot=cache,
        model_dir=model_dir,
        job_root=job_root,
        runtime_commit="b" * 40,
    )

    assert result["status"] == "prepared"
    assert result["source_fixture_sha256"] == fixture_sha
    assert result["selected_qid"] == "q-1"
    assert result["selected_fixture_rank"] == 0
    assert result["global_index_loaded"] is False
    assert result["retrieval_run"] is False
    assert all(path.is_file() and not path.is_symlink() for path in model_dir.iterdir())
    smoke = load_task8_smoke_inputs(Path(result["smoke_input_manifest_path"]))
    assert len(smoke["pages"]) == 1
    assert smoke["pages"][0]["fixture_rank"] == 0
    run = json.loads((job_root / "run-manifest.json").read_text())
    assert run["inputs"] == [
        {
            "path": smoke["pages"][0]["mineru_input_path"],
            "sha256": smoke["pages"][0]["mineru_input_sha256"],
        }
    ]
    assert run["backend_cli"] == "vlm-auto-engine"
    preparation = json.loads((job_root / "preparation-manifest.json").read_text())
    supplied = preparation.pop("manifest_sha256")
    assert (
        supplied
        == hashlib.sha256(
            json.dumps(preparation, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_gpu_idle_gate_rejects_every_sign_of_use() -> None:
    from docprune.task9_h200_mineru import require_completely_idle_gpu

    require_completely_idle_gpu(
        "5, GPU-five, NVIDIA H200, 143771, 0, 0, 0, 570.172.08",
        "",
        expected_gpu_id=5,
        expected_gpu_uuid="GPU-five",
    )
    for gpu_row, process_rows in (
        ("5, GPU-five, NVIDIA H200, 143771, 1, 0, 0, 570.172.08", ""),
        ("5, GPU-five, NVIDIA H200, 143771, 0, 1, 0, 570.172.08", ""),
        ("5, GPU-five, NVIDIA H200, 143771, 0, 0, 1, 570.172.08", ""),
        (
            "5, GPU-five, NVIDIA H200, 143771, 0, 0, 0, 570.172.08",
            "GPU-five, 123, python, 1",
        ),
    ):
        with __import__("pytest").raises(RuntimeError, match="not completely idle"):
            require_completely_idle_gpu(
                gpu_row,
                process_rows,
                expected_gpu_id=5,
                expected_gpu_uuid="GPU-five",
            )


def test_gpu_monitor_accepts_owned_descendant_in_a_new_session() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        assert os.getsid(child.pid) != os.getsid(0)
        rows = f"GPU-five, {child.pid}, python, 1"
        assert _foreign_gpu_pids(rows, "GPU-five", os.getpid()) == []
    finally:
        child.terminate()
        child.wait()


def test_gpu_monitor_rejects_process_outside_owned_tree() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        rows = f"GPU-five, {os.getpid()}, python, 1"
        assert _foreign_gpu_pids(rows, "GPU-five", child.pid) == [os.getpid()]
    finally:
        child.terminate()
        child.wait()


def test_prepare_retry_reuses_identical_materialized_model(tmp_path: Path) -> None:
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
    cache = tmp_path / "cache"
    cache.mkdir()
    for name, raw in (("config.json", b"config"), ("model.safetensors", b"weights")):
        (cache / name).write_bytes(raw)
    model_dir = tmp_path / "model"
    first = prepare_task9_one_page_mineru_smoke(
        source_fixture_path=fixture_path,
        source_fixture_sha256=fixture_sha,
        source_smoke_manifest_path=sealed / "smoke-input-manifest.json",
        source_model_snapshot=cache,
        model_dir=model_dir,
        job_root=tmp_path / "attempt-1",
        runtime_commit="b" * 40,
    )

    second = prepare_task9_one_page_mineru_smoke(
        source_fixture_path=fixture_path,
        source_fixture_sha256=fixture_sha,
        source_smoke_manifest_path=sealed / "smoke-input-manifest.json",
        source_model_snapshot=cache,
        model_dir=model_dir,
        job_root=tmp_path / "attempt-2",
        runtime_commit="c" * 40,
    )

    assert second["model_inventory_sha256"] == first["model_inventory_sha256"]
