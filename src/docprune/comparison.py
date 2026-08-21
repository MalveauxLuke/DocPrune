"""Fail-closed validation and reporting for the six-cell DocPrune matrix."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from docprune.evaluation import ValidationReport, _load_result_rows, validate_benchmark_run

_MODES = ("all-kept", "docprune")
_PAGE_COUNTS = (1, 2, 4)
_EXPECTED_CELLS = frozenset((mode, pages) for pages in _PAGE_COUNTS for mode in _MODES)
_IDENTITY_FIELDS = (
    "corpus",
    "runtime_commit",
    "m3docrag_commit",
    "resources",
    "processor_contract_path",
    "processor_contract_sha256",
    "processor_contract",
    "generation",
    "measurement",
)


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _digest(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("comparison_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _json_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _manifest_for(run_dir: Path) -> dict[str, object]:
    return _json_object(run_dir / "run_manifest.json", "run manifest")


def _key_from_text(value: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"(all-kept|docprune)(?:[-_@](?:top[-_])?)?(1|2|4)", value)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _normalise_inputs(
    runs: Mapping[object, object] | Sequence[object], errors: list[str]
) -> dict[tuple[str, int], Path]:
    items: list[tuple[object | None, object]] = []
    if isinstance(runs, Mapping):
        items = list(runs.items())
    elif isinstance(runs, Sequence) and not isinstance(runs, str | bytes):
        items = [(None, value) for value in runs]
    else:
        errors.append("comparison runs must be a sequence or mapping")
        return {}
    result: dict[tuple[str, int], Path] = {}
    for supplied_key, raw_path in items:
        try:
            run_dir = Path(raw_path)  # type: ignore[arg-type]
            manifest = _manifest_for(run_dir)
            mode = manifest.get("mode")
            page_count = manifest.get("page_count")
            manifest_key = (
                (mode, page_count)
                if mode in _MODES and isinstance(page_count, int) and page_count in _PAGE_COUNTS
                else None
            )
            requested_key: tuple[str, int] | None = None
            if supplied_key is not None:
                if isinstance(supplied_key, tuple) and len(supplied_key) == 2:
                    supplied_mode, supplied_pages = supplied_key
                    if supplied_mode in _MODES and isinstance(supplied_pages, int):
                        requested_key = (str(supplied_mode), supplied_pages)
                elif isinstance(supplied_key, str):
                    requested_key = _key_from_text(supplied_key)
                if requested_key is None:
                    errors.append(f"invalid comparison cell key: {supplied_key!r}")
                    continue
                if manifest_key is not None and requested_key != manifest_key:
                    errors.append(
                        f"comparison cell key {requested_key!r} disagrees with run manifest {manifest_key!r}"
                    )
            cell = requested_key or manifest_key
            if cell is None:
                errors.append(f"run manifest does not identify a supported cell: {run_dir}")
                continue
            if cell in result:
                errors.append(f"duplicate comparison cell: {cell!r}")
                continue
            result[cell] = run_dir
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"cannot inspect comparison run {raw_path!r}: {error}")
    missing = sorted(_EXPECTED_CELLS - set(result))
    extra = sorted(set(result) - _EXPECTED_CELLS)
    if missing:
        errors.append(f"comparison matrix must contain exactly six cells; missing {missing!r}")
    if extra:
        errors.append(f"comparison matrix contains unsupported cells: {extra!r}")
    if len(result) != 6:
        errors.append(f"comparison matrix must contain exactly six cells; got {len(result)}")
    return result


def _corpus_paths(corpus_root: object) -> tuple[Path, Path, Path]:
    if corpus_root is None:
        raise ValueError("pinned corpus root is required")
    if isinstance(corpus_root, Mapping):
        root = Path(str(corpus_root.get("root", "")))
        document_ids = Path(str(corpus_root.get("document_ids_path", root / "dev_doc_ids.json")))
        pdf_dir = Path(str(corpus_root.get("pdf_dir", root / "pdfs_dev")))
    else:
        root_value = (
            getattr(corpus_root, "root")
            if not isinstance(corpus_root, str | bytes | Path) and hasattr(corpus_root, "root")
            else corpus_root
        )
        root = Path(root_value)
        document_ids = Path(
            str(getattr(corpus_root, "document_ids_path", root / "dev_doc_ids.json"))
        )
        pdf_dir = Path(str(getattr(corpus_root, "pdf_dir", root / "pdfs_dev")))
    if not document_ids.is_absolute():
        document_ids = root / document_ids
    if not pdf_dir.is_absolute():
        pdf_dir = root / pdf_dir
    if not document_ids.is_file():
        raise ValueError(f"pinned corpus document IDs are missing: {document_ids}")
    if not pdf_dir.is_dir():
        raise ValueError(f"pinned corpus PDF directory is missing: {pdf_dir}")
    value = json.loads(document_ids.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError("pinned corpus document IDs must be a JSON list of nonempty strings")
    if len(value) != len(set(value)):
        raise ValueError("pinned corpus document IDs contain duplicates")
    return root, document_ids, pdf_dir


def _actual_pdf_page_count(path: Path) -> int:
    """Count pages from PDF bytes, with pdfinfo as a fallback for unusual PDFs."""

    try:
        completed = subprocess.run(
            ["pdfinfo", str(path)], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        completed = None
    if completed is not None and completed.returncode == 0:
        match = re.search(r"^Pages:\s*(\d+)\s*$", completed.stdout, re.MULTILINE)
        if match:
            return int(match.group(1))
    data = path.read_bytes()
    count = len(re.findall(rb"/Type\s*/Page(?:\b)", data))
    if count:
        return count
    raise ValueError(f"could not determine actual PDF page count: {path}")


def _validate_retrieved_pages(
    run_dir: Path,
    pdf_dir: Path,
    document_ids: set[str],
    errors: list[str],
    page_counts: dict[str, int],
) -> None:
    try:
        records = _load_result_rows(run_dir / "results.jsonl")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"cannot independently read results for {run_dir}: {error}")
        return
    for record in records:
        qid = record.get("question_id")
        pages = record.get("retrieved_pages", ())
        if not isinstance(pages, Sequence) or isinstance(pages, str | bytes):
            errors.append(f"{run_dir} result {qid!r} has invalid retrieved pages")
            continue
        for page in pages:
            if not isinstance(page, Mapping):
                errors.append(f"{run_dir} result {qid!r} has an invalid retrieved page")
                continue
            doc_id = page.get("doc_id")
            page_index = page.get("page_index")
            if not isinstance(doc_id, str) or not doc_id:
                errors.append(f"{run_dir} result {qid!r} has an invalid retrieved document ID")
                continue
            if doc_id not in document_ids:
                errors.append(f"{run_dir} result {qid!r} retrieved document is not in pinned corpus: {doc_id}")
                continue
            if not isinstance(page_index, int) or isinstance(page_index, bool) or page_index < 0:
                errors.append(f"{run_dir} result {qid!r} has an invalid page index for {doc_id}")
                continue
            if doc_id not in page_counts:
                pdf_path = pdf_dir / f"{doc_id}.pdf"
                if pdf_path.is_symlink() or not pdf_path.is_file():
                    errors.append(f"{run_dir} retrieved document PDF is missing: {doc_id}")
                    continue
                try:
                    page_counts[doc_id] = _actual_pdf_page_count(pdf_path)
                except (OSError, ValueError) as error:
                    errors.append(str(error))
                    continue
            if page_index >= page_counts[doc_id]:
                errors.append(
                    f"{run_dir} result {qid!r} page index {page_index} is outside {doc_id} "
                    f"actual page count {page_counts[doc_id]}"
                )


def _validate_corpus_binding(
    manifest: Mapping[str, object],
    run_dir: Path,
    pinned_root: Path,
    pinned_document_ids: Path,
    pinned_pdf_dir: Path,
    errors: list[str],
) -> None:
    corpus = manifest.get("corpus")
    if not isinstance(corpus, Mapping):
        errors.append(f"{run_dir}: run manifest has no corpus identity")
        return
    for field, expected in (
        ("root", pinned_root),
        ("document_ids_path", pinned_document_ids),
        ("pdf_dir", pinned_pdf_dir),
    ):
        raw = corpus.get(field)
        if not isinstance(raw, str) or not raw:
            errors.append(f"{run_dir}: corpus identity is missing {field}")
            continue
        actual = Path(raw)
        if not actual.is_absolute():
            actual = run_dir / actual
        if actual.resolve() != expected.resolve():
            errors.append(
                f"{run_dir}: corpus identity {field} does not match the pinned corpus root"
            )


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def _numeric_delta(left: object, right: object) -> object | None:
    if _number(left) and _number(right):
        return float(right) - float(left)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result = {
            str(key): delta
            for key in sorted(set(left) & set(right), key=str)
            if (delta := _numeric_delta(left[key], right[key])) is not None
        }
        return result or None
    return None


def _cell_metrics(summary: Mapping[str, object]) -> dict[str, object]:
    quality = summary.get("quality")
    efficiency = summary.get("efficiency")
    if not isinstance(quality, Mapping) or not isinstance(efficiency, Mapping):
        raise ValueError("run summary must contain quality and efficiency metrics")
    raw_drop_rates = efficiency.get("drop_rates", {})
    retention_rates = {
        str(stage): 1.0 - float(value)
        for stage, value in raw_drop_rates.items()
        if _number(value)
    }
    return {
        "quality": {
            "overall": quality.get("overall", {}),
            "modalities": quality.get("modalities", {}),
            "hop_types": quality.get("hop_types", {}),
            "retrieval": quality.get("document_recall", quality.get("average_recall_at_k", {})),
        },
        "efficiency": {
            "visual_tokens": efficiency.get("visual_tokens", {}),
            "drop_rates": raw_drop_rates,
            "retention_rates": retention_rates,
            "token_weighted_drop_rates": efficiency.get("token_weighted_drop_rates", {}),
            "timing_seconds": efficiency.get("timing_seconds", {}),
            "encoder_samples_per_second": efficiency.get("encoder_samples_per_second"),
            "decoder_samples_per_second": efficiency.get("decoder_samples_per_second"),
            "measurement": efficiency.get("measurement", {}),
        },
    }


def _profile_payload(cells: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    derived_tflops: list[float] = []
    definitions: list[str] = []
    profiler_states: list[bool] = []
    for cell in cells:
        metrics = cell.get("metrics")
        efficiency = metrics.get("efficiency") if isinstance(metrics, Mapping) else None
        measurement = efficiency.get("measurement") if isinstance(efficiency, Mapping) else None
        if not isinstance(measurement, Mapping):
            profiler_states.append(False)
            continue
        profiler_states.append(measurement.get("profiler_enabled") is True)
    if not any(profiler_states):
        return None
    if not all(profiler_states):
        raise ValueError("partial profiling identity across comparison cells")
    for cell in cells:
        metrics = cell.get("metrics")
        efficiency = metrics.get("efficiency") if isinstance(metrics, Mapping) else None
        measurement = efficiency.get("measurement") if isinstance(efficiency, Mapping) else None
        if not isinstance(measurement, Mapping):
            raise ValueError("profiler measurement identity is missing")
        flops = measurement.get("flops")
        definition = measurement.get("profiler_definition")
        total = efficiency.get("timing_seconds", {}).get("total") if isinstance(efficiency, Mapping) else None
        if not isinstance(definition, str) or not definition:
            raise ValueError("profiler definition identity is missing")
        definitions.append(definition)
        if not _number(flops) or float(flops) <= 0:
            raise ValueError("profiler FLOPs must be finite and positive")
        if not _number(total) or float(total) <= 0:
            raise ValueError("profiler timing must be finite and positive")
        flops_value = float(flops)
        total_value = float(total)
        derived = flops_value / total_value / 1e12
        if not math.isfinite(derived) or derived <= 0:
            raise ValueError("derived TFLOPs must be finite and positive")
        derived_tflops.append(derived)
    if len(set(definitions)) != 1:
        raise ValueError("profiler definition identity mismatched across comparison cells")
    return {
        "profiler_definition": definitions[0],
        "tflops": derived_tflops,
    }


@dataclass(frozen=True)
class ComparisonValidationReport:
    valid: bool
    errors: tuple[str, ...]
    cells: tuple[dict[str, object], ...] = ()
    deltas: tuple[dict[str, object], ...] = ()
    payload: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "cells": list(self.cells),
            "deltas": list(self.deltas),
            "payload": self.payload,
        }


def validate_comparison_matrix(
    runs: Mapping[object, object] | Sequence[object],
    corpus_root: object,
    *,
    expected_questions: int = 2441,
    allow_fixture: bool = False,
) -> ComparisonValidationReport:
    """Independently validate and compare exactly six benchmark run directories."""

    errors: list[str] = []
    if expected_questions < 1:
        raise ValueError("expected_questions must be positive")
    paths = _normalise_inputs(runs, errors)
    manifests: dict[tuple[str, int], dict[str, object]] = {}
    reports: dict[tuple[str, int], ValidationReport] = {}
    for cell, run_dir in sorted(paths.items(), key=lambda item: (item[0][1], item[0][0])):
        try:
            manifest = _manifest_for(run_dir)
            manifests[cell] = manifest
            report = validate_benchmark_run(
                run_dir, expected_questions, allow_fixture=allow_fixture
            )
            reports[cell] = report
            if not report.valid:
                errors.extend(f"{run_dir}: {error}" for error in report.errors)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{run_dir}: independent run validation failed: {error}")
    try:
        pinned_root, document_ids_path, pdf_dir = _corpus_paths(corpus_root)
        document_ids = set(json.loads(document_ids_path.read_text(encoding="utf-8")))
        page_counts: dict[str, int] = {}
        for cell, run_dir in paths.items():
            if cell in manifests:
                _validate_corpus_binding(
                    manifests[cell],
                    run_dir,
                    pinned_root,
                    document_ids_path,
                    pdf_dir,
                    errors,
                )
            _validate_retrieved_pages(run_dir, pdf_dir, document_ids, errors, page_counts)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"pinned corpus validation failed: {error}")

    for pages in _PAGE_COUNTS:
        baseline = ("all-kept", pages)
        candidate = ("docprune", pages)
        if baseline not in manifests or candidate not in manifests:
            continue
        left = {field: manifests[baseline].get(field) for field in _IDENTITY_FIELDS}
        right = {field: manifests[candidate].get(field) for field in _IDENTITY_FIELDS}
        if left != right:
            mismatched = [field for field in _IDENTITY_FIELDS if left[field] != right[field]]
            errors.append(f"page-count pair {pages} has mismatched shared identity: {mismatched!r}")
        if baseline in reports and candidate in reports and reports[baseline].qids != reports[candidate].qids:
            errors.append(f"page-count pair {pages} does not preserve exact source-order QIDs")

    cells: list[dict[str, object]] = []
    if not errors:
        for cell, run_dir in sorted(paths.items(), key=lambda item: (item[0][1], item[0][0])):
            report = reports[cell]
            if not isinstance(report.reproduced_summary, Mapping):
                errors.append(f"{run_dir}: independently reproduced summary is missing")
                continue
            metrics = _cell_metrics(report.reproduced_summary)
            cells.append(
                {
                    "mode": cell[0],
                    "page_count": cell[1],
                    "run_dir": str(run_dir.resolve()),
                    "qids": list(report.qids),
                    "metrics": metrics,
                    "quality": metrics["quality"],
                    "efficiency": metrics["efficiency"],
                    "measurement": manifests[cell].get("measurement"),
                }
            )
    deltas: list[dict[str, object]] = []
    if not errors and len(cells) == 6:
        by_cell = {(cell["mode"], cell["page_count"]): cell for cell in cells}
        for pages in _PAGE_COUNTS:
            left = by_cell[("all-kept", pages)]
            right = by_cell[("docprune", pages)]
            deltas.append(
                {
                    "page_count": pages,
                    "baseline": "all-kept",
                    "comparison": "docprune",
                    "direction": "docprune_minus_all-kept",
                    "metrics": _numeric_delta(left["metrics"], right["metrics"]),
                }
            )
    payload: dict[str, object] | None = None
    if not errors and len(cells) == 6 and len(deltas) == 3:
        payload = {
            "schema_version": 1,
            "contract": "docprune-six-cell-comparison-v1",
            "corpus_root": str(_corpus_paths(corpus_root)[0].resolve()),
            "hardware_note": (
                "A100 measurements are reconstruction results, not RTX A6000 parity; "
                "paired relative comparisons use matched runtime identity."
            ),
            "cells": cells,
            "deltas": deltas,
        }
        try:
            profile = _profile_payload(cells)
        except ValueError as error:
            errors.append(str(error))
            profile = None
        if profile is not None:
            payload["profiling"] = profile
    if errors:
        payload = None
    elif payload is not None:
        payload["comparison_sha256"] = _digest(payload)
    return ComparisonValidationReport(not errors, tuple(errors), tuple(cells), tuple(deltas), payload)


def comparison_markdown(payload: Mapping[str, object]) -> str:
    """Render a deterministic human-readable six-cell report."""

    cells = payload.get("cells", ())
    deltas = payload.get("deltas", ())
    lines = [
        "# DocPrune six-cell comparison",
        "",
        f"Comparison SHA-256: `{payload.get('comparison_sha256', '')}`",
        "",
        str(payload.get("hardware_note", "")),
        "",
        "## Absolute cells",
        "",
        "| Pages | Mode | EM | F1 | Modality F1 | Hop F1 | Recall@k | BTP drop | QTP drop | CTP drop | BTP retain | QTP retain | CTP retain | Retrieval s | Page load s | QA s | Total s | Encoder samples/s | Decoder samples/s | Peak allocated GPU bytes |",
        "| ---: | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in cells if isinstance(cells, Sequence) else ():
        metrics = cell.get("metrics", {}) if isinstance(cell, Mapping) else {}
        quality = metrics.get("quality", {}) if isinstance(metrics, Mapping) else {}
        overall = quality.get("overall", {}) if isinstance(quality, Mapping) else {}
        recall = quality.get("retrieval", {}) if isinstance(quality, Mapping) else {}
        modalities = quality.get("modalities", {}) if isinstance(quality, Mapping) else {}
        hop_types = quality.get("hop_types", {}) if isinstance(quality, Mapping) else {}
        efficiency = metrics.get("efficiency", {}) if isinstance(metrics, Mapping) else {}
        drops = efficiency.get("drop_rates", {}) if isinstance(efficiency, Mapping) else {}
        retention = efficiency.get("retention_rates", {}) if isinstance(efficiency, Mapping) else {}
        timing = efficiency.get("timing_seconds", {}) if isinstance(efficiency, Mapping) else {}
        measurement = efficiency.get("measurement", {}) if isinstance(efficiency, Mapping) else {}
        lines.append(
            "| {page_count} | {mode} | {em} | {f1} | {modalities} | {hop_types} | {recall} | {btp} | {qtp} | {ctp} | {btp_retain} | {qtp_retain} | {ctp_retain} | {retrieval} | {page_load} | {qa} | {total} | {encoder} | {decoder} | {peak} |".format(
                page_count=cell.get("page_count", ""),
                mode=cell.get("mode", ""),
                em=overall.get("list_em", ""),
                f1=overall.get("list_f1", ""),
                modalities=json.dumps(modalities, sort_keys=True, separators=(",", ":")),
                hop_types=json.dumps(hop_types, sort_keys=True, separators=(",", ":")),
                recall=json.dumps(recall, sort_keys=True, separators=(",", ":")),
                btp=drops.get("btp", "") if isinstance(drops, Mapping) else "",
                qtp=drops.get("qtp", "") if isinstance(drops, Mapping) else "",
                ctp=drops.get("ctp", "") if isinstance(drops, Mapping) else "",
                btp_retain=retention.get("btp", "") if isinstance(retention, Mapping) else "",
                qtp_retain=retention.get("qtp", "") if isinstance(retention, Mapping) else "",
                ctp_retain=retention.get("ctp", "") if isinstance(retention, Mapping) else "",
                retrieval=timing.get("retrieval", "") if isinstance(timing, Mapping) else "",
                page_load=timing.get("page_load", "") if isinstance(timing, Mapping) else "",
                qa=timing.get("qa", "") if isinstance(timing, Mapping) else "",
                total=timing.get("total", "") if isinstance(timing, Mapping) else "",
                encoder=efficiency.get("encoder_samples_per_second", "") if isinstance(efficiency, Mapping) else "",
                decoder=efficiency.get("decoder_samples_per_second", "") if isinstance(efficiency, Mapping) else "",
                peak=measurement.get("peak_allocated_gpu_bytes", "") if isinstance(measurement, Mapping) else "",
            )
        )
    lines.extend(
        [
            "",
            "## Paired deltas (DocPrune − all-kept)",
            "",
            "| Pages | Quality and efficiency delta object |",
            "| ---: | --- |",
        ]
    )
    for delta in deltas if isinstance(deltas, Sequence) else ():
        lines.append(f"| {delta.get('page_count', '')} | `{json.dumps(delta.get('metrics', {}), sort_keys=True, separators=(',', ':'))}` |")
    if "profiling" in payload:
        lines.extend(["", "## Profiling", "", f"`{json.dumps(payload['profiling'], sort_keys=True, separators=(',', ':'))}`"])
    return "\n".join(lines) + "\n"


def _publish_noreplace(source: Path, destination: Path) -> None:
    """Publish one staged file atomically without replacing a destination."""

    os.link(source, destination)
    source.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_owned_file(path: Path, identity: tuple[int, int]) -> None:
    try:
        observed = path.stat()
    except FileNotFoundError:
        return
    if (observed.st_dev, observed.st_ino) == identity:
        path.unlink(missing_ok=True)


def write_comparison_report(
    runs: Mapping[object, object] | Sequence[object],
    corpus_root: object,
    *,
    json_path: Path,
    markdown_path: Path,
    expected_questions: int = 2441,
    allow_fixture: bool = False,
) -> ComparisonValidationReport:
    """Validate and atomically publish signed JSON/Markdown reports."""

    report = validate_comparison_matrix(
        runs,
        corpus_root=corpus_root,
        expected_questions=expected_questions,
        allow_fixture=allow_fixture,
    )
    if not report.valid or report.payload is None:
        raise ValueError("comparison matrix is invalid: " + "; ".join(report.errors))
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    if json_path.resolve() == markdown_path.resolve():
        raise ValueError("JSON and Markdown output paths must differ")
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("comparison output already exists; refusing to overwrite")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_bytes = (json.dumps(report.payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode()
    markdown_bytes = comparison_markdown(report.payload).encode()
    temporary: list[tuple[Path, Path]] = []
    staged_publications: list[tuple[Path, Path, tuple[int, int]]] = []
    locks: list[tuple[Path, tuple[int, int]]] = []
    try:
        for destination, data in ((json_path, json_bytes), (markdown_path, markdown_bytes)):
            descriptor, raw_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
            temporary_path = Path(raw_name)
            temporary.append((temporary_path, destination))
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        for _, destination in temporary:
            lock_path = Path(f"{destination}.lock")
            descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                lock_stat = os.fstat(descriptor)
                locks.append((lock_path, (lock_stat.st_dev, lock_stat.st_ino)))
            finally:
                os.close(descriptor)
        for temporary_path, destination in temporary:
            staged_stat = temporary_path.stat()
            staged_publications.append(
                (temporary_path, destination, (staged_stat.st_dev, staged_stat.st_ino))
            )
        for temporary_path, destination, _ in staged_publications:
            _publish_noreplace(temporary_path, destination)
        _fsync_directory(json_path.parent)
        if markdown_path.parent != json_path.parent:
            _fsync_directory(markdown_path.parent)
    except BaseException:
        for _, destination, identity in reversed(staged_publications):
            _remove_owned_file(destination, identity)
        for temporary_path, _ in temporary:
            temporary_path.unlink(missing_ok=True)
        raise
    finally:
        for lock_path, identity in reversed(locks):
            _remove_owned_file(lock_path, identity)
        for temporary_path, _ in temporary:
            temporary_path.unlink(missing_ok=True)
    return report


# Public aliases used by callers that describe this boundary as comparison/report validation.
validate_comparison = validate_comparison_matrix
generate_comparison_report = write_comparison_report
compare_runs = validate_comparison_matrix
render_comparison_markdown = comparison_markdown
ComparisonReport = ComparisonValidationReport


__all__ = [
    "ComparisonValidationReport",
    "ComparisonReport",
    "compare_runs",
    "comparison_markdown",
    "generate_comparison_report",
    "render_comparison_markdown",
    "validate_comparison",
    "validate_comparison_matrix",
    "write_comparison_report",
]
