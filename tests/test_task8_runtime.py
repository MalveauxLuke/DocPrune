"""Tests for sealed Task 8 MinerU smoke inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from docprune.task6_runtime import (
    TASK6_RENDERER_CONTRACT,
    FixedPageFixture,
    FixedPageQuestion,
    FixedPageRecord,
)
from docprune.task8_runtime import load_task8_smoke_inputs, seal_task8_smoke_inputs


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
