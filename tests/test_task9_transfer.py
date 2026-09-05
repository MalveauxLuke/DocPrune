"""Tests for relocating sealed Task 9 inputs onto H200 storage."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

from docprune.task6_runtime import (
    TASK6_RENDERER_CONTRACT,
    FixedPageFixture,
    FixedPageQuestion,
    FixedPageRecord,
)
from docprune.task8_runtime import load_task8_smoke_inputs, seal_task8_smoke_inputs
from docprune.task9_transfer import relocate_task9_fixed_inputs


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mirror(source: Path, raw_root: Path) -> Path:
    destination = raw_root / source.relative_to("/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def test_relocate_task9_fixed_inputs_rewrites_paths_without_changing_source_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    fixed = source / "fixed"
    qid_root = fixed / "qid-inputs" / "000-q-1"
    fixed.mkdir(parents=True)
    pdf = source / "doc.pdf"
    feature = source / "features.safetensors"
    pdf.write_bytes(b"pdf")
    feature.write_bytes(b"features")
    dependencies: list[Path] = []
    for name in ("reference.json", "eligible.jsonl", "feature-manifest.json", "ledger.json"):
        path = fixed / name
        path.write_bytes(name.encode())
        dependencies.append(path)
    image = Image.new("RGB", (8, 6), color=(1, 2, 3))
    page = FixedPageRecord(
        rank=0,
        doc_id="doc-1",
        page_index=0,
        score=1.0,
        source_pdf_path=pdf,
        source_pdf_sha256=_sha(pdf),
        rendered_rgb_width=image.width,
        rendered_rgb_height=image.height,
        rendered_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        renderer_contract=TASK6_RENDERER_CONTRACT,
        feature_shard_path=feature,
        feature_shard_sha256=_sha(feature),
        feature_page_index=0,
    )
    fixture = FixedPageFixture(
        fixture_version="task9-transfer-test-v1",
        reference_path=dependencies[0],
        reference_sha256=_sha(dependencies[0]),
        eligible_questions_path=dependencies[1],
        eligible_questions_sha256=_sha(dependencies[1]),
        feature_manifest_path=dependencies[2],
        feature_manifest_sha256=_sha(dependencies[2]),
        completion_ledger_path=dependencies[3],
        completion_ledger_sha256=_sha(dependencies[3]),
        questions=(FixedPageQuestion("q-1", "9" * 64, (page,)),),
    )
    fixture_path = fixed / "fixture.json"
    fixture_path.write_text(
        json.dumps(fixture.to_dict(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    seal_task8_smoke_inputs(
        fixture_path=fixture_path,
        fixture_sha256=_sha(fixture_path),
        qid="q-1",
        output_root=qid_root,
        runtime_commit="a" * 40,
        render_page=lambda _path, _page: image.copy(),
    )
    selected = fixed / "selected-source-results.jsonl"
    selected.write_text('{"question_id":"q-1"}\n', encoding="utf-8")
    cohort = source / "cohort.json"
    cohort.write_bytes(b"cohort")
    preprocessing = fixed / "preprocessing-manifest.json"
    preprocessing.write_text(
        json.dumps(
            {
                "schema_version": "docprune-task9-confirmation-inputs-v1",
                "status": "sealed",
                "runtime_commit": "a" * 40,
                "cohort_path": str(cohort),
                "cohort_file_sha256": _sha(cohort),
                "cohort_sha256": "c" * 64,
                "fixture_path": str(fixture_path),
                "fixture_sha256": _sha(fixture_path),
                "selected_source_results_path": str(selected),
                "selected_source_results_sha256": _sha(selected),
                "question_count": 1,
                "page_count": 1,
                "qid_inputs": [
                    {
                        "array_index": 0,
                        "qid": "q-1",
                        "baseline_stratum": "baseline_wrong",
                        "supporting_document_ids": ["doc-1"],
                        "input_root": str(qid_root),
                        "input_manifest": str(qid_root / "smoke-input-manifest.json"),
                        "input_manifest_sha256": _sha(qid_root / "smoke-input-manifest.json"),
                        "expected_post_qtp_visual_tokens": 1,
                        "page_count": 1,
                    }
                ],
                "global_index_loaded": False,
                "retrieval_run": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"
    raw = bundle / "raw"
    original_files = tuple(path for path in source.rglob("*") if path.is_file())
    original_hashes = {path.relative_to(source): _sha(path) for path in original_files}
    for path in original_files:
        _mirror(path, raw)
    mirrored_fixed = raw / fixed.relative_to("/")
    shutil.rmtree(source)

    output = tmp_path / "h200-fixed"
    result = relocate_task9_fixed_inputs(
        bundle_root=bundle,
        source_fixed_root=mirrored_fixed,
        output_root=output,
    )

    relocated_fixture = json.loads((output / "fixture.json").read_text(encoding="utf-8"))
    relocated_page = relocated_fixture["questions"][0]["pages"][0]
    assert relocated_page["source_pdf_path"].startswith(str(raw))
    assert relocated_page["feature_shard_path"].startswith(str(raw))
    assert result["question_count"] == 1
    assert result["page_count"] == 1
    assert result["retrieval_run"] is False
    smoke = load_task8_smoke_inputs(output / "qid-inputs/000-q-1/smoke-input-manifest.json")
    assert smoke["fixed_page_fixture_path"] == str(output / "fixture.json")
    assert all(_sha(raw / path.relative_to("/")) == digest for path, digest in (
        (fixed / relative, original_hashes[Path("fixed") / relative])
        for relative in (
            Path("fixture.json"),
            Path("qid-inputs/000-q-1/page-00.png"),
            Path("qid-inputs/000-q-1/smoke-input-manifest.json"),
        )
    ))
