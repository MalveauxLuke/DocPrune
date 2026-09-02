"""Authenticate and relocate sealed Task 9 inputs onto a new filesystem."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from docprune.task6_runtime import FixedPageFixture


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a regular file: {path}")
    return path


def _mirrored_path(raw_root: Path, original: str | Path, label: str) -> Path:
    original_path = Path(original)
    if not original_path.is_absolute():
        raise ValueError(f"{label} path is not absolute")
    return _regular_file(raw_root / original_path.relative_to("/"), label)


def _require_hash(path: Path, expected: object, label: str) -> None:
    if not isinstance(expected, str) or _sha256(path) != expected:
        raise ValueError(f"{label} checksum mismatch")


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def relocate_task9_fixed_inputs(
    *, bundle_root: Path, source_fixed_root: Path, output_root: Path
) -> dict[str, object]:
    """Create H200-local fixed inputs while preserving all transferred source bytes."""

    bundle = Path(bundle_root).resolve()
    raw = (bundle / "raw").resolve()
    source_fixed = Path(source_fixed_root).resolve()
    output = Path(output_root).absolute()
    if raw not in source_fixed.parents:
        raise ValueError("source fixed-input root must be inside bundle/raw")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Task 9 relocated input root already exists: {output}")

    preprocessing_path = _regular_file(
        source_fixed / "preprocessing-manifest.json", "source preprocessing manifest"
    )
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    if not isinstance(preprocessing, Mapping):
        raise ValueError("source preprocessing manifest is invalid")
    if (
        preprocessing.get("status") != "sealed"
        or preprocessing.get("retrieval_run") is not False
        or preprocessing.get("global_index_loaded") is not False
    ):
        raise ValueError("source preprocessing manifest permits retrieval")

    fixture_path = _regular_file(source_fixed / "fixture.json", "source fixture")
    _require_hash(fixture_path, preprocessing.get("fixture_sha256"), "source fixture")
    fixture = FixedPageFixture.from_dict(json.loads(fixture_path.read_text(encoding="utf-8")))

    provenance = (
        (fixture.reference_path, fixture.reference_sha256, "reference"),
        (fixture.eligible_questions_path, fixture.eligible_questions_sha256, "eligible questions"),
        (fixture.feature_manifest_path, fixture.feature_manifest_sha256, "feature manifest"),
        (fixture.completion_ledger_path, fixture.completion_ledger_sha256, "completion ledger"),
    )
    mirrored_provenance: dict[str, Path] = {}
    for original, expected, label in provenance:
        mirrored = _mirrored_path(raw, original, label)
        _require_hash(mirrored, expected, label)
        mirrored_provenance[label] = mirrored
    for question in fixture.questions:
        for page in question.pages:
            for original, expected, label in (
                (page.source_pdf_path, page.source_pdf_sha256, "source PDF"),
                (page.feature_shard_path, page.feature_shard_sha256, "feature shard"),
            ):
                mirrored = _mirrored_path(raw, original, label)
                _require_hash(mirrored, expected, label)

    qid_inputs = preprocessing.get("qid_inputs")
    if not isinstance(qid_inputs, list) or len(qid_inputs) != len(fixture.questions):
        raise ValueError("source preprocessing QID inventory is invalid")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for source_name, destination_name in (
            ("reference", "reference.json"),
            ("eligible questions", "eligible.jsonl"),
        ):
            _link_or_copy(mirrored_provenance[source_name], stage / destination_name)
        selected_source = _mirrored_path(
            raw, preprocessing["selected_source_results_path"], "selected source results"
        )
        _require_hash(
            selected_source,
            preprocessing.get("selected_source_results_sha256"),
            "selected source results",
        )
        _link_or_copy(selected_source, stage / "selected-source-results.jsonl")
        cohort_source = _mirrored_path(raw, preprocessing["cohort_path"], "cohort")
        _require_hash(cohort_source, preprocessing.get("cohort_file_sha256"), "cohort")
        _link_or_copy(cohort_source, stage / "cohort.json")

        relocated_fixture = fixture.to_dict()
        relocated_fixture["reference_path"] = str(output / "reference.json")
        relocated_fixture["eligible_questions_path"] = str(output / "eligible.jsonl")
        relocated_fixture["feature_manifest_path"] = str(
            mirrored_provenance["feature manifest"]
        )
        relocated_fixture["completion_ledger_path"] = str(
            mirrored_provenance["completion ledger"]
        )
        for question in relocated_fixture["questions"]:
            for page in question["pages"]:
                page["source_pdf_path"] = str(
                    _mirrored_path(raw, page["source_pdf_path"], "source PDF")
                )
                page["feature_shard_path"] = str(
                    _mirrored_path(raw, page["feature_shard_path"], "feature shard")
                )
        fixture_bytes = _canonical_bytes(relocated_fixture)
        (stage / "fixture.json").write_bytes(fixture_bytes)
        relocated_fixture_sha = hashlib.sha256(fixture_bytes).hexdigest()

        relocated_qids: list[dict[str, object]] = []
        fixture_by_qid = {
            question["qid"]: question for question in relocated_fixture["questions"]
        }
        page_count = 0
        for row in qid_inputs:
            if not isinstance(row, Mapping):
                raise ValueError("source preprocessing QID row is invalid")
            index = row["array_index"]
            qid = row["qid"]
            directory_name = f"{index:03d}-{qid}"
            source_manifest = _mirrored_path(raw, row["input_manifest"], "QID input manifest")
            _require_hash(source_manifest, row.get("input_manifest_sha256"), "QID input manifest")
            smoke = json.loads(source_manifest.read_text(encoding="utf-8"))
            pages = smoke.get("pages")
            if not isinstance(pages, list) or qid not in fixture_by_qid:
                raise ValueError(f"source QID input manifest is invalid: {qid}")
            relocated_pages = fixture_by_qid[qid]["pages"]
            if len(pages) != len(relocated_pages):
                raise ValueError(f"source QID page count differs from fixture: {qid}")
            final_qid_root = output / "qid-inputs" / directory_name
            staged_qid_root = stage / "qid-inputs" / directory_name
            rewritten_pages: list[dict[str, object]] = []
            for rank, (page_row, fixed_page) in enumerate(zip(pages, relocated_pages, strict=True)):
                source_png = _mirrored_path(raw, page_row["mineru_input_path"], "MinerU input")
                _require_hash(source_png, page_row.get("mineru_input_sha256"), "MinerU input")
                destination_png = staged_qid_root / f"page-{rank:02d}.png"
                _link_or_copy(source_png, destination_png)
                rewritten = dict(page_row)
                rewritten["fixed_page_record"] = fixed_page
                rewritten["fixed_page_record_sha256"] = hashlib.sha256(
                    _canonical_bytes(fixed_page)
                ).hexdigest()
                rewritten["mineru_input_path"] = str(final_qid_root / destination_png.name)
                rewritten_pages.append(rewritten)
            unsigned_smoke = {
                key: value for key, value in smoke.items() if key != "manifest_sha256"
            }
            unsigned_smoke["fixed_page_fixture_path"] = str(output / "fixture.json")
            unsigned_smoke["fixed_page_fixture_sha256"] = relocated_fixture_sha
            unsigned_smoke["pages"] = rewritten_pages
            rewritten_smoke = {
                **unsigned_smoke,
                "manifest_sha256": hashlib.sha256(_canonical_bytes(unsigned_smoke)).hexdigest(),
            }
            staged_manifest = staged_qid_root / "smoke-input-manifest.json"
            staged_manifest.write_bytes(_canonical_bytes(rewritten_smoke))
            relocated_row = dict(row)
            relocated_row["input_root"] = str(final_qid_root)
            relocated_row["input_manifest"] = str(final_qid_root / staged_manifest.name)
            relocated_row["input_manifest_sha256"] = _sha256(staged_manifest)
            relocated_qids.append(relocated_row)
            page_count += len(rewritten_pages)

        relocated_preprocessing = dict(preprocessing)
        relocated_preprocessing["cohort_path"] = str(output / "cohort.json")
        relocated_preprocessing["fixture_path"] = str(output / "fixture.json")
        relocated_preprocessing["fixture_sha256"] = relocated_fixture_sha
        relocated_preprocessing["selected_source_results_path"] = str(
            output / "selected-source-results.jsonl"
        )
        relocated_preprocessing["qid_inputs"] = relocated_qids
        (stage / "preprocessing-manifest.json").write_text(
            json.dumps(relocated_preprocessing, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        relocation = {
            "schema_version": "docprune-task9-h200-relocation-v1",
            "status": "complete",
            "source_fixture_path": str(fixture_path),
            "source_fixture_sha256": preprocessing["fixture_sha256"],
            "relocated_fixture_path": str(output / "fixture.json"),
            "relocated_fixture_sha256": relocated_fixture_sha,
            "question_count": len(relocated_qids),
            "page_count": page_count,
            "retrieval_run": False,
            "global_index_loaded": False,
        }
        (stage / "relocation-manifest.json").write_bytes(_canonical_bytes(relocation))
        stage.rename(output)
        return relocation
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = ["relocate_task9_fixed_inputs"]
