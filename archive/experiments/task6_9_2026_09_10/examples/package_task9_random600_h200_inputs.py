#!/usr/bin/env python3
"""Package exact Task 9 random-600 inputs using the proven H200 raw mirror layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from docprune.task6_runtime import load_fixed_page_fixture


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"regular input file required: {path}")
    return path


def _mirror(source: Path, raw: Path) -> Path:
    destination = raw / source.relative_to("/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination) != _sha256(source):
            raise ValueError(f"mirrored path collision: {source}")
        return destination
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--splits-file-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    fixed = args.fixed_root.resolve()
    splits = _require_file(args.splits)
    output = args.output_root.absolute()
    if not fixed.is_dir() or fixed.is_symlink():
        raise ValueError("fixed input root must be a directory")
    if _sha256(splits) != args.splits_file_sha256:
        raise ValueError("split manifest checksum mismatch")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"transfer root already exists: {output}")

    preprocessing_path = _require_file(fixed / "preprocessing-manifest.json")
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    if preprocessing.get("question_count") != 600 or preprocessing.get("page_count") != 2400:
        raise ValueError("preprocessing manifest is not the random-600 input set")
    if preprocessing.get("splits_file_sha256") != args.splits_file_sha256:
        raise ValueError("preprocessing and split identities differ")
    fixture = load_fixed_page_fixture(
        Path(preprocessing["fixture_path"]), expected_sha256=preprocessing["fixture_sha256"]
    )

    sources = {_require_file(path) for path in fixed.rglob("*") if path.is_file()}
    sources.add(_require_file(Path(preprocessing["cohort_path"])))
    sources.add(splits)
    sources.update({
        _require_file(fixture.feature_manifest_path),
        _require_file(fixture.completion_ledger_path),
        _require_file(fixture.reference_path),
        _require_file(fixture.eligible_questions_path),
    })
    for question in fixture.questions:
        for page in question.pages:
            sources.add(_require_file(page.source_pdf_path))
            sources.add(_require_file(page.feature_shard_path))

    output.mkdir(parents=True)
    try:
        raw = output / "raw"
        raw.mkdir()
        for source in sorted(sources, key=str):
            _mirror(source, raw)
        mirrored = sorted(path for path in raw.rglob("*") if path.is_file())
        lines = [f"{_sha256(path)}  {path.relative_to(output)}" for path in mirrored]
        checksum = output / "MANIFEST.sha256"
        checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")
        checksum_digest = _sha256(checksum)
        (output / "MANIFEST.sha256.sha256").write_text(
            f"{checksum_digest}  MANIFEST.sha256\n", encoding="utf-8"
        )
        bundle = {
            "schema_version": "docprune-task9-shared-probe-transfer-v1",
            "status": "sealed",
            "source_fixed_root": str(fixed),
            "source_preprocessing_manifest_sha256": _sha256(preprocessing_path),
            "cohort_file_sha256": preprocessing["cohort_file_sha256"],
            "splits_file_sha256": args.splits_file_sha256,
            "question_count": 600,
            "page_count": 2400,
            "raw_file_count": len(mirrored),
            "raw_byte_count": sum(path.stat().st_size for path in mirrored),
            "manifest_sha256": checksum_digest,
            "global_index_included": False,
        }
        (output / "bundle.json").write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(output)
        raise
    print(json.dumps(bundle, sort_keys=True))


if __name__ == "__main__":
    main()
