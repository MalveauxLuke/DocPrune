"""Official M3DocVQA scoring and independent benchmark-run validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
import string
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from docprune.m3docrag import official_answer_text
from docprune.metrics import MEASUREMENT_DEFINITION, summarize_jsonl

try:
    from word2number.w2n import word_to_num
except ImportError as error:  # pragma: no cover - dependency is declared and pinned by the env
    raise RuntimeError(
        "word2number is required for official M3DocVQA answer normalization"
    ) from error


_MULTI_HOP_QUESTION_TYPES = frozenset(
    {
        "Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ))",
        "Compare(Compose(TableQ,ImageQ),TableQ)",
        "Compare(TableQ,Compose(TableQ,TextQ))",
        "Compose(ImageQ,TableQ)",
        "Compose(ImageQ,TextQ)",
        "Compose(TableQ,ImageListQ)",
        "Compose(TableQ,TextQ)",
        "Compose(TextQ,ImageListQ)",
        "Compose(TextQ,TableQ)",
        "Intersect(ImageListQ,TableQ)",
        "Intersect(ImageListQ,TextQ)",
        "Intersect(TableQ,TextQ)",
    }
)
_RECALL_LEVELS = (1, 2, 4, 5, 10)
_PUNCTUATION = set(string.punctuation)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    if word_to_num is None:  # defensive for a corrupted/monkeypatched runtime
        raise RuntimeError("word2number is required for official M3DocVQA answer normalization")
    try:
        return str(float(word_to_num(text)))
    except (TypeError, ValueError):
        pass
    return text


def _normalize_answer(text: object) -> str:
    value = str(text).lower()
    tokens = re.split(" |-", value)
    normalized: list[str] = []
    for token in tokens:
        if not token:
            continue
        if not _is_number(token):
            token = "".join(char for char in token if char not in _PUNCTUATION)
        token = " ".join(token.split())
        token = " ".join(re.sub(r"\b(a|an|the)\b", " ", token).split())
        token = _normalize_number(token)
        if token.strip():
            normalized.append(token.strip())
    return " ".join(normalized).strip()


def _answer_to_bags(answer: object) -> tuple[list[str], list[set[str]]]:
    raw_spans = answer if isinstance(answer, list | tuple) else [answer]
    normalized = [_normalize_answer(span) for span in raw_spans]
    return normalized, [set(span.split()) for span in normalized]


def _match_numbers(gold: set[str], predicted: set[str]) -> bool:
    gold_numbers = {token for token in gold if _is_number(token)}
    predicted_numbers = {token for token in predicted if _is_number(token)}
    return not gold_numbers or bool(gold_numbers.intersection(predicted_numbers))


def _bag_f1(predicted: set[str], gold: set[str]) -> float:
    intersection = len(predicted.intersection(gold))
    precision = 1.0 if not predicted else intersection / len(predicted)
    recall = 1.0 if not gold else intersection / len(gold)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _aligned_f1(predicted: list[set[str]], gold: list[set[str]]) -> float:
    """Match answer bags with the same maximum-weight alignment as the reference."""

    if not gold and not predicted:
        return 0.0
    scores = [
        [_bag_f1(pred, ref) if _match_numbers(ref, pred) else 0.0 for pred in predicted]
        for ref in gold
    ]
    # The official implementation uses scipy's Hungarian algorithm.  Answer
    # lists are short; this exact bitmask DP has the same maximum assignment
    # value without adding scipy as a production dependency.
    if len(predicted) > 20:
        raise ValueError("answer lists longer than 20 spans are unsupported")

    @cache
    def best(row: int, used: int) -> float:
        if row == len(gold):
            return 0.0
        value = best(row + 1, used)  # leave this gold span unmatched
        for column, score in enumerate(scores[row]):
            if not used & (1 << column):
                value = max(value, score + best(row + 1, used | (1 << column)))
        return value

    return round(best(0, 0) / max(len(gold), len(predicted)), 2)


def list_em(predicted: object, gold: object) -> float:
    predicted_spans, _ = _answer_to_bags(predicted)
    gold_spans, _ = _answer_to_bags(gold)
    return float(
        set(predicted_spans) == set(gold_spans) and len(predicted_spans) == len(gold_spans)
    )


def list_f1(predicted: object, gold: object) -> float:
    _, predicted_bags = _answer_to_bags(predicted)
    _, gold_bags = _answer_to_bags(gold)
    return _aligned_f1(predicted_bags, gold_bags)


def _result_rows(results: object) -> tuple[dict[str, object], ...]:
    if isinstance(results, Mapping):
        if any(key in results for key in ("question_id", "qid")):
            values: Iterable[object] = (results,)
        else:
            values = (
                dict(value, question_id=str(key)) if isinstance(value, Mapping) else value
                for key, value in results.items()
            )
    else:
        values = results  # type: ignore[assignment]
    rows: list[dict[str, object]] = []
    for value in values:
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        if not isinstance(value, Mapping):
            raise ValueError("results must contain mapping rows")
        row = dict(value)
        if "question_id" not in row and "qid" in row:
            row["question_id"] = row.pop("qid")
        if not isinstance(row.get("question_id"), str) or not row["question_id"]:
            raise ValueError("result row is missing question_id")
        rows.append(row)
    return tuple(rows)


def _source_rows(source_rows: object) -> tuple[dict[str, object], ...]:
    if isinstance(source_rows, Mapping):
        if "qid" in source_rows or "question_id" in source_rows:
            values: Iterable[object] = (source_rows,)
        else:
            values = (
                dict(value, qid=str(key)) if isinstance(value, Mapping) else value
                for key, value in source_rows.items()
            )
    else:
        values = source_rows  # type: ignore[assignment]
    rows: list[dict[str, object]] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("source_rows must contain mapping rows")
        row = dict(value)
        if "qid" not in row and "question_id" in row:
            row["qid"] = row["question_id"]
        if not isinstance(row.get("qid"), str) or not row["qid"]:
            raise ValueError("source row is missing qid")
        rows.append(row)
    return tuple(rows)


def _gold_answers(row: Mapping[str, object]) -> list[str]:
    answers = row.get("answers", ())
    if not isinstance(answers, Sequence) or isinstance(answers, str | bytes):
        raise ValueError(f"source row {row.get('qid')} has invalid answers")
    values: list[str] = []
    for answer in answers:
        try:
            values.append(official_answer_text(answer, require_mapping=True))
        except ValueError as error:
            raise ValueError(f"source row {row.get('qid')} has an invalid answer: {error}") from error
    return values


def _group_label(row: Mapping[str, object], key: str, default: str) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get(key) is not None:
        return str(metadata[key])
    return str(row.get(key, default))


def _modality(row: Mapping[str, object]) -> str:
    values = row.get("answers", ())
    modalities = {
        str(answer.get("modality"))
        for answer in values
        if isinstance(answer, Mapping) and answer.get("modality") is not None
    }
    if len(modalities) > 1:
        raise ValueError(f"source row {row.get('qid')} has multiple answer modalities")
    return next(iter(modalities), "unknown")


def _slice_scores(
    qids: Sequence[str], scores: Mapping[str, Mapping[str, float]], labels: Mapping[str, str]
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for qid in qids:
        label = labels[qid]
        grouped.setdefault(label, {"list_em": [], "list_f1": []})
        grouped[label]["list_em"].append(scores[qid]["list_em"])
        grouped[label]["list_f1"].append(scores[qid]["list_f1"])
    return {
        label: {metric: sum(values) / len(values) * 100 for metric, values in metrics.items()}
        for label, metrics in sorted(grouped.items())
    }


def _retrieved_doc_ids(row: Mapping[str, object]) -> list[str]:
    values = row.get("retrieved_pages", row.get("page_retrieval_results", ()))
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        raise ValueError(f"result row {row.get('question_id')} has invalid retrieved pages")
    docs: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            doc_id = value.get("doc_id")
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            doc_id = value[0] if value else None
        else:
            doc_id = None
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"result row {row.get('question_id')} has invalid retrieved page")
        docs.append(doc_id)
    return docs


def _supporting_doc_ids(row: Mapping[str, object]) -> set[str]:
    contexts = row.get("supporting_context", ())
    if not isinstance(contexts, Sequence) or isinstance(contexts, str | bytes):
        raise ValueError(f"source row {row.get('qid')} has invalid supporting_context")
    docs: set[str] = set()
    for value in contexts:
        if isinstance(value, Mapping):
            doc_id = value.get("doc_id")
        else:
            doc_id = value
        if isinstance(doc_id, str) and doc_id:
            docs.add(doc_id)
    return docs


@dataclass(frozen=True)
class M3DocVQAMetrics:
    count: int
    overall: dict[str, float]
    modalities: dict[str, dict[str, float]]
    hop_types: dict[str, dict[str, float]]
    question_types: dict[str, dict[str, float]]
    document_recall: dict[int, float]
    per_question: dict[str, dict[str, float]]
    max_retrieval_depth: int = 0

    @property
    def list_em(self) -> float:
        return self.overall["list_em"]

    @property
    def list_f1(self) -> float:
        return self.overall["list_f1"]

    @property
    def average_recall_at_k(self) -> dict[int, float]:
        return self.document_recall

    @property
    def retrieval(self) -> dict[int, float]:
        return self.document_recall

    @property
    def observed_retrieval_depth(self) -> int:
        return self.max_retrieval_depth

    @property
    def modalities_by_type(self) -> dict[str, dict[str, float]]:
        return self.modalities

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "overall": self.overall,
            "modalities": self.modalities,
            "hop_types": self.hop_types,
            "question_types": self.question_types,
            "average_recall_at_k": {
                str(level): self.document_recall[level] for level in _RECALL_LEVELS
            },
            "document_recall": {
                str(level): self.document_recall[level] for level in _RECALL_LEVELS
            },
            "observed_retrieval_depth": self.max_retrieval_depth,
            "per_question": self.per_question,
        }


def evaluate_m3docvqa(results: object, source_rows: object) -> M3DocVQAMetrics:
    """Evaluate predictions with the pinned M3DocRAG/M3DocVQA formulas."""

    if word_to_num is None:
        raise RuntimeError("word2number is required for official M3DocVQA answer normalization")
    sources = _source_rows(source_rows)
    source_by_qid = {str(row["qid"]): row for row in sources}
    if len(source_by_qid) != len(sources):
        raise ValueError("source rows contain duplicate qids")
    prediction_rows = _result_rows(results)
    predictions: dict[str, dict[str, object]] = {}
    for row in prediction_rows:
        qid = str(row["question_id"])
        if qid in predictions:
            raise ValueError(f"results contain duplicate qid: {qid}")
        if qid not in source_by_qid:
            raise ValueError(f"results contain unexpected qid: {qid}")
        predictions[qid] = row

    qids = tuple(source_by_qid)
    per_question: dict[str, dict[str, float]] = {}
    modalities: dict[str, str] = {}
    hops: dict[str, str] = {}
    question_types: dict[str, str] = {}
    retrieval: dict[int, list[float]] = {level: [] for level in _RECALL_LEVELS}
    max_retrieval_depth = 0
    for qid in qids:
        source = source_by_qid[qid]
        prediction = predictions.get(qid)
        gold = _gold_answers(source)
        predicted = (
            ""
            if prediction is None
            else prediction.get("predicted_answer", prediction.get("pred_answer", ""))
        )
        scores = {"list_em": list_em(predicted, gold), "list_f1": list_f1(predicted, gold)}
        per_question[qid] = scores
        modality = _modality(source)
        qtype = _group_label(source, "type", "unknown")
        modalities[qid] = modality
        question_types[qid] = qtype
        hops[qid] = "Multi-hop" if qtype in _MULTI_HOP_QUESTION_TYPES else "Single-hop"
        relevant = _supporting_doc_ids(source)
        retrieved = [] if prediction is None else _retrieved_doc_ids(prediction)
        max_retrieval_depth = max(max_retrieval_depth, len(retrieved))
        for level in _RECALL_LEVELS:
            top_docs = set(retrieved[:level])
            retrieval[level].append(
                len(top_docs.intersection(relevant)) / len(relevant) if relevant else 0.0
            )

    overall = {
        metric: sum(score[metric] for score in per_question.values()) / len(qids) * 100
        for metric in ("list_em", "list_f1")
    }
    return M3DocVQAMetrics(
        count=len(qids),
        overall=overall,
        modalities=_slice_scores(qids, per_question, modalities),
        hop_types=_slice_scores(qids, per_question, hops),
        question_types=_slice_scores(qids, per_question, question_types),
        document_recall={
            level: sum(values) / len(values) if values else 0.0
            for level, values in retrieval.items()
        },
        per_question=per_question,
        max_retrieval_depth=max_retrieval_depth,
    )


def load_source_rows(path: Path) -> tuple[dict[str, object], ...]:
    """Load source JSONL rows without applying a dataset-specific transformation."""

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"source questions must be a regular file: {path}")
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"source questions contains a blank line at {line_number}")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"source question line {line_number} is not an object")
            rows.append(dict(value))
    return tuple(rows)


def source_rows_from_manifest(
    manifest: Mapping[str, object], run_dir: Path
) -> tuple[dict[str, object], ...] | None:
    corpus = manifest.get("corpus")
    if not isinstance(corpus, Mapping) or corpus.get("questions_path") is None:
        return None
    path = Path(str(corpus["questions_path"]))
    if not path.is_absolute():
        path = Path(run_dir) / path
    return load_source_rows(path)


def _load_result_rows(path: Path) -> tuple[dict[str, object], ...]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"results JSONL must be a regular file: {path}")
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"results JSONL contains a blank line at {line_number}")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"results JSONL line {line_number} is not an object")
            rows.append(dict(value))
    return tuple(rows)


def summarize_benchmark_run(
    results_path: Path,
    source_rows: object | None = None,
    *,
    require_positive: bool = False,
) -> dict[str, object]:
    """Produce the deterministic quality-plus-efficiency run summary."""

    efficiency = summarize_jsonl(Path(results_path), require_positive=require_positive)
    manifest_path = Path(results_path).parent / "run_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
        if isinstance(manifest, Mapping):
            measurement = manifest.get("measurement")
            if isinstance(measurement, Mapping) and "result_classification" in measurement:
                summary_measurement = efficiency.get("measurement")
                if isinstance(summary_measurement, dict):
                    summary_measurement["result_classification"] = measurement[
                        "result_classification"
                    ]
    if source_rows is None:
        return efficiency
    records = _load_result_rows(Path(results_path))
    source = _source_rows(source_rows)
    result_qids = {str(row.get("question_id", row.get("qid", ""))) for row in records}
    selected = tuple(row for row in source if str(row["qid"]) in result_qids)
    quality = evaluate_m3docvqa(records, selected)
    return {"quality": quality.to_dict(), "efficiency": efficiency}


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...]
    question_count: int
    qids: tuple[str, ...]
    summary: dict[str, object] | None = None
    reproduced_summary: dict[str, object] | None = None

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "question_count": self.question_count,
            "qids": list(self.qids),
            "summary": self.summary,
            "reproduced_summary": self.reproduced_summary,
        }


def _canonical_digest(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("run_manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _regular_file(path: Path, label: str, errors: list[str]) -> bool:
    if path.is_symlink() or not path.is_file():
        errors.append(f"{label} is missing or not a regular file: {path}")
        return False
    return True


def _finite_json(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_json(key) and _finite_json(item) for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return all(_finite_json(item) for item in value)
    return True


def _validate_source_questions(
    manifest: Mapping[str, object],
    run_dir: Path,
    expected_qids: Sequence[str],
    records: Sequence[Mapping[str, object]],
    errors: list[str],
    *,
    required: bool,
) -> None:
    corpus = manifest.get("corpus")
    if not isinstance(corpus, Mapping):
        if required:
            errors.append("production run manifest is missing corpus source identity")
        return
    raw_path = corpus.get("questions_path")
    declared_sha = corpus.get("questions_sha256")
    if raw_path is None:
        if required or declared_sha is not None:
            errors.append("source question digest is present without a questions path")
        return
    if not isinstance(declared_sha, str) or not declared_sha:
        errors.append("source questions SHA-256 is missing")
        if required:
            return
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = run_dir / path
    if not _regular_file(path, "source questions", errors):
        return
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if declared_sha is not None and actual_sha != declared_sha:
        errors.append("source questions digest mismatch")
    source_rows: list[dict[str, object]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    errors.append(f"source questions contains a blank line at {line_number}")
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"source question line {line_number} is not an object")
                source_rows.append(dict(value))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        errors.append(f"source question content is invalid: {error}")
        return
    source_qids = tuple(str(row.get("qid", row.get("question_id", ""))) for row in source_rows)
    if any(not qid for qid in source_qids) or len(source_qids) != len(set(source_qids)):
        errors.append("source question content has missing or duplicate qids")
        return
    expected_count = corpus.get("expected_question_count")
    if isinstance(expected_count, int) and len(source_rows) != expected_count:
        errors.append(
            "source question content count does not match corpus identity: "
            f"expected {expected_count}, got {len(source_rows)}"
        )
    positions: list[int] = []
    source_by_qid = {qid: row for qid, row in zip(source_qids, source_rows)}
    for qid in expected_qids:
        if qid not in source_by_qid:
            errors.append(f"source question order/content is missing {qid}")
            continue
        positions.append(source_qids.index(qid))
    if positions != sorted(positions) or len(positions) != len(expected_qids):
        errors.append("source question order/content does not match the run selection")
        return
    if len(source_rows) < len(records):
        errors.append("source question content count does not match results")
        return
    for record, qid in zip(records, expected_qids):
        source = source_by_qid.get(qid)
        if source is None:
            continue
        if record.get("question") != source.get("question"):
            errors.append(f"source question content mismatch for {record.get('question_id')}")
        raw_answers = source.get("answers", ())
        if not isinstance(raw_answers, Sequence) or isinstance(raw_answers, str | bytes):
            errors.append(
                f"source question content has invalid answers for {record.get('question_id')}"
            )
            continue
        expected_answers: list[str] = []
        for answer in raw_answers:
            try:
                expected_answers.append(official_answer_text(answer, require_mapping=True))
            except ValueError as error:
                errors.append(
                    "source question content has an invalid answer for "
                    f"{record.get('question_id')}: {error}"
                )
                expected_answers = []
                break
        if record.get("answers") != expected_answers:
            errors.append(f"source answer content mismatch for {record.get('question_id')}")


def _manifest_is_fixture(manifest: Mapping[str, object]) -> bool:
    corpus = manifest.get("corpus")
    return (
        type(manifest.get("fixture_mode")) is bool
        and manifest.get("fixture_mode") is True
        and isinstance(corpus, Mapping)
        and type(corpus.get("is_fixture")) is bool
        and corpus.get("is_fixture") is True
    )


def _manifest_declares_fixture(manifest: Mapping[str, object]) -> bool:
    corpus = manifest.get("corpus")
    if type(manifest.get("fixture_mode")) is bool and manifest.get("fixture_mode") is True:
        return True
    if not isinstance(corpus, Mapping) or "is_fixture" not in corpus:
        return False
    value = corpus["is_fixture"]
    return type(value) is not bool or value is True


def _fixture_identity_complete(manifest: Mapping[str, object]) -> bool:
    corpus = manifest.get("corpus")
    required = {
        "root",
        "questions_path",
        "document_ids_path",
        "pdf_dir",
        "integrity_report_path",
        "integrity_sha256",
        "archive_checksum_manifest_path",
        "archive_checksum_manifest_sha256",
        "questions_sha256",
        "document_ids_sha256",
        "expected_question_count",
        "expected_pdf_count",
        "expected_page_count",
        "is_fixture",
        "archive_hashes",
    }
    return (
        manifest.get("operation") == "evaluate"
        and _manifest_is_fixture(manifest)
        and isinstance(corpus, Mapping)
        and required <= set(corpus)
    )


def _manifest_path(value: object, run_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def _validate_production_corpus(
    manifest: Mapping[str, object], errors: list[str], *, production: bool = True
) -> str | None:
    corpus = manifest.get("corpus")
    if not isinstance(corpus, Mapping):
        errors.append("production run manifest is missing complete corpus identity")
        return None
    required = {
        "root",
        "questions_path",
        "document_ids_path",
        "pdf_dir",
        "integrity_report_path",
        "integrity_sha256",
        "archive_checksum_manifest_path",
        "archive_checksum_manifest_sha256",
        "questions_sha256",
        "document_ids_sha256",
        "expected_question_count",
        "expected_pdf_count",
        "expected_page_count",
        "is_fixture",
        "archive_hashes",
    }
    missing = sorted(required - set(corpus))
    if missing:
        errors.append(f"production corpus identity is missing fields: {missing!r}")
        return None
    is_fixture = corpus["is_fixture"]
    if type(is_fixture) is not bool:
        errors.append("corpus is_fixture must be a boolean")
        return None
    if production and is_fixture is not False:
        errors.append("production corpus is_fixture must be exactly false")
        return None
    if not production and is_fixture is not True:
        errors.append("fixture corpus is_fixture must be exactly true")
        return None
    try:
        from docprune.benchmark_config import CorpusIdentity

        identity = CorpusIdentity(
            **{key: corpus[key] for key in required if key != "is_fixture"},
            is_fixture=is_fixture,
        )
        identity.validate()
        from docprune.m3docvqa_dataset import M3DocVQADevDataset

        dataset = M3DocVQADevDataset(
            identity, expected_question_count=identity.expected_question_count
        )
        from docprune.m3docvqa_factory import _corpus_identity_payload

        if dict(corpus) != _corpus_identity_payload(identity):
            errors.append("manifest corpus identity does not match its complete validated payload")
        return dataset.source_order_sha256
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        errors.append(f"corpus identity validation failed: {error}")
        return None


def _validate_production_run_config(
    manifest: Mapping[str, object], run_dir: Path, errors: list[str]
) -> None:
    path = _manifest_path(manifest.get("run_config_source_path"), run_dir)
    digest = manifest.get("run_config_source_sha256")
    if path is None or not isinstance(digest, str) or not digest:
        errors.append("production run requires run configuration path and SHA-256")
        return
    if not _regular_file(path, "run configuration", errors):
        return
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        errors.append("run configuration digest mismatch")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("run configuration is not an object")
        from docprune.m3docvqa_factory import _normalise_run_config_mapping, _validate_run_identity

        resolved = _normalise_run_config_mapping(payload, base=path.parent)
        raw_mode = getattr(resolved, "mode", None)
        raw_page_count = getattr(resolved, "page_count", None)
        if raw_mode != manifest.get("mode"):
            errors.append("run configuration mode does not match the run manifest")
        if raw_page_count != manifest.get("page_count"):
            errors.append("run configuration page_count does not match the run manifest")
        if (
            not isinstance(raw_mode, str)
            or isinstance(raw_page_count, bool)
            or not isinstance(raw_page_count, int)
            or raw_mode not in {"all-kept", "docprune"}
            or raw_page_count not in {1, 2, 4}
            or raw_mode != manifest.get("mode")
            or raw_page_count != manifest.get("page_count")
        ):
            return
        identity = _validate_run_identity(
            resolved,
            mode=raw_mode,
            page_count=raw_page_count,
        )
        for key in (
            "runtime_commit",
            "m3docrag_commit",
            "resources",
            "processor_contract_path",
            "processor_contract_sha256",
            "processor_contract",
            "corpus",
            "generation",
        ):
            if manifest.get(key) != identity.get(key):
                errors.append(f"manifest identity does not match run configuration: {key}")
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError) as error:
        errors.append(f"run configuration validation failed: {error}")


def _validate_production_index(
    manifest: Mapping[str, object],
    run_dir: Path,
    errors: list[str],
    source_order_sha256: str | None,
) -> None:
    path = _manifest_path(manifest.get("index_manifest_source_path"), run_dir)
    digest = manifest.get("index_manifest_source_sha256")
    nested = manifest.get("index_manifest")
    if path is None or not isinstance(digest, str) or not digest:
        errors.append("production run requires index manifest path and SHA-256")
        return
    if not isinstance(nested, Mapping):
        errors.append("production run requires nested index manifest identity")
    if not _regular_file(path, "index manifest", errors):
        return
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        errors.append("index manifest digest mismatch")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("index manifest is not an object")
        if payload.get("schema_version") != 5:
            raise ValueError("index manifest schema_version must equal 5")
        supplied = payload.get("manifest_sha256")
        unsigned = dict(payload)
        unsigned.pop("manifest_sha256", None)
        if supplied != _canonical_digest_index(unsigned):
            raise ValueError("index manifest canonical digest is invalid")
        if isinstance(nested, Mapping) and dict(payload) != dict(nested):
            errors.append("index manifest identity does not match its source file")
        from docprune.m3docvqa_factory import _load_index_manifest

        loaded = _load_index_manifest(path)
        if loaded.to_dict() != dict(payload):
            errors.append("index manifest loader payload does not match source bytes")
        resources = payload.get("resources")
        expected_resources = manifest.get("resources")
        if resources != expected_resources:
            errors.append("index manifest resources do not match the run")
        for key in ("mode", "page_count", "runtime_commit", "m3docrag_commit", "pruning_config"):
            if payload.get(key) != manifest.get(key):
                errors.append(f"index manifest {key} does not match the run")
        corpus = manifest.get("corpus")
        if isinstance(corpus, Mapping) and payload.get("corpus_integrity_sha256") != corpus.get(
            "integrity_sha256"
        ):
            errors.append("index manifest corpus identity does not match the run")
        if (
            source_order_sha256 is not None
            and payload.get("source_order_sha256") != source_order_sha256
        ):
            errors.append("index manifest source order does not match the corpus")
        if payload.get("processor_contract_path") != manifest.get("processor_contract_path"):
            errors.append("index manifest processor contract path does not match the run")
        if payload.get("processor_contract_sha256") != manifest.get("processor_contract_sha256"):
            errors.append("index manifest processor contract digest does not match the run")
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError) as error:
        errors.append(f"index artifact validation failed: {error}")


def _canonical_digest_index(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def validate_benchmark_run(
    run_dir: Path, expected_questions: int = 2441, *, allow_fixture: bool = False
) -> ValidationReport:
    """Independently validate immutable results, measurements, and summaries."""

    run_dir = Path(run_dir)
    errors: list[str] = []
    if expected_questions < 1:
        raise ValueError("expected_questions must be positive")
    if not isinstance(allow_fixture, bool):
        raise TypeError("allow_fixture must be a boolean")
    manifest_path = run_dir / "run_manifest.json"
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    manifest: dict[str, object] = {}
    if _regular_file(manifest_path, "run manifest", errors):
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("not an object")
            manifest = loaded
        except (OSError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"run manifest is invalid: {error}")
    if manifest:
        supplied = manifest.get("run_manifest_sha256")
        if not isinstance(supplied, str) or supplied != _canonical_digest(manifest):
            errors.append("run manifest digest mismatch")
    fixture_relaxation = (
        allow_fixture and expected_questions != 2441 and _fixture_identity_complete(manifest)
    )
    production_run = (
        manifest.get("operation") == "evaluate" or "corpus" in manifest
    ) and not fixture_relaxation
    if _manifest_declares_fixture(manifest) and not fixture_relaxation:
        errors.append(
            "fixture identity is not valid for production validation; "
            "use the explicit allow_fixture API only for tiny internal fixtures"
        )
    if production_run:
        if manifest.get("operation") != "evaluate":
            errors.append("production benchmark validation requires an evaluate manifest")
        if manifest.get("schema_version") != 2 or manifest.get("status") != "configured":
            errors.append("production run manifest schema/status is invalid")
        required_manifest_fields = {
            "schema_version",
            "status",
            "operation",
            "output",
            "mode",
            "page_count",
            "runtime_commit",
            "m3docrag_commit",
            "resources",
            "processor_contract_path",
            "processor_contract_sha256",
            "processor_contract",
            "run_config_source_path",
            "run_config_source_sha256",
            "index_manifest_source_path",
            "index_manifest_source_sha256",
            "corpus",
            "generation",
            "pruning_config",
            "selection",
            "index_manifest",
            "measurement",
            "run_manifest_sha256",
        }
        missing_manifest = sorted(required_manifest_fields - set(manifest))
        if missing_manifest:
            errors.append(
                f"production run manifest is missing immutable fields: {missing_manifest!r}"
            )
        output = _manifest_path(manifest.get("output"), run_dir)
        if output is None or output.resolve() != run_dir.resolve():
            errors.append(
                "production run manifest output does not match the validated run directory"
            )
    measurement = manifest.get("measurement")
    declared_profiler = False
    required_measurement = {
        "definition",
        "warmup_required",
        "profiler_enabled",
        "hardware",
        "result_classification",
        "software",
        "precision",
        "attention_backend",
        "allocator",
        "timer_boundaries",
        "warmup",
    }
    if production_run and not isinstance(measurement, Mapping):
        errors.append("production run manifest is missing measurement definition")
    if isinstance(measurement, Mapping):
        if production_run and set(measurement) != required_measurement:
            errors.append("measurement manifest contains unexpected or missing fields")
        if production_run and not required_measurement <= set(measurement):
            errors.append("measurement manifest is missing required fields")
        if measurement.get("definition") != MEASUREMENT_DEFINITION:
            errors.append("measurement definition does not match the benchmark contract")
        if measurement.get("warmup_required") is not True:
            errors.append("measurement manifest must require an explicit warmup")
        profiler_value = measurement.get("profiler_enabled")
        if not isinstance(profiler_value, bool):
            errors.append("measurement profiler_enabled must be boolean")
        else:
            declared_profiler = profiler_value
        if production_run:
            hardware = measurement.get("hardware")
            software = measurement.get("software")
            precision = measurement.get("precision")
            allocator = measurement.get("allocator")
            boundaries = measurement.get("timer_boundaries")
            if not isinstance(hardware, Mapping) or not isinstance(
                hardware.get("gpu_model"), str
            ) or not hardware.get("gpu_model", "").strip():
                errors.append("measurement hardware identity is invalid")
            capability = hardware.get("compute_capability") if isinstance(hardware, Mapping) else None
            if capability is None or (
                not isinstance(capability, list)
                or len(capability) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) for value in capability)
            ):
                errors.append("measurement compute capability is invalid")
            from docprune.metrics import hardware_result_classification

            try:
                expected_classification = hardware_result_classification(
                    str(hardware.get("gpu_model")) if isinstance(hardware, Mapping) else ""
                )
            except ValueError:
                expected_classification = None
            if measurement.get("result_classification") != expected_classification:
                errors.append("measurement result classification is invalid")
            if not isinstance(software, Mapping) or any(
                not isinstance(software.get(name), str) or not software.get(name).strip()
                for name in ("python", "cuda", "pytorch", "transformers")
            ):
                errors.append("measurement software identity is invalid")
            if not isinstance(precision, Mapping) or any(
                precision.get(name) != expected
                for name, expected in (
                    ("weights", "bfloat16"),
                    ("vision", "bfloat16"),
                    ("attention_accumulation", "float32"),
                )
            ):
                errors.append("measurement precision identity is invalid")
            if not isinstance(allocator, Mapping) or allocator != {
                "peak_memory": "torch.cuda.max_memory_allocated",
                "reset": "torch.cuda.reset_peak_memory_stats",
            }:
                errors.append("measurement allocator identity is invalid")
            if measurement.get("attention_backend") != "flash_attention_2":
                errors.append("measurement attention backend identity is invalid")
            expected_boundary_values = {
                "synchronization": "torch.cuda.synchronize before and after each stage",
                "encoder": "Qwen visual encoder execution only",
                "decoder": "language-model prefill, first logits, and greedy decode",
                "qa": "complete page preparation, BTP/QTP, encoder, decoder, and answer decode",
                "sample": "retrieval, page loading, and complete QA wall time",
            }
            if not isinstance(boundaries, Mapping) or dict(boundaries) != expected_boundary_values:
                errors.append("measurement timer boundaries are incomplete")
            warmup = measurement.get("warmup")
            if not isinstance(warmup, Mapping):
                errors.append("measurement manifest is missing warmup identity")
            else:
                if (
                    not isinstance(warmup.get("count"), int)
                    or isinstance(warmup.get("count"), bool)
                    or warmup.get("count") != 1
                ):
                    errors.append("measurement warmup count must equal one")
                sample_ids = (
                    manifest.get("selection", {}).get("resolved_question_ids", [])
                    if isinstance(manifest.get("selection"), Mapping)
                    else []
                )
                if (
                    not isinstance(warmup.get("sample_id"), str)
                    or warmup.get("sample_id") not in sample_ids
                ):
                    errors.append("measurement warmup sample identity is not selected")
                expected_sample_hash = hashlib.sha256(
                    "\n".join(str(value) for value in sample_ids).encode("utf-8")
                ).hexdigest()
                if warmup.get("sample_identity_sha256") != expected_sample_hash:
                    errors.append("measurement sample identity digest is invalid")
    if production_run:
        source_order_sha256 = _validate_production_corpus(manifest, errors)
        _validate_production_run_config(manifest, run_dir, errors)
        _validate_production_index(manifest, run_dir, errors, source_order_sha256)
    elif fixture_relaxation:
        _validate_production_corpus(manifest, errors, production=False)

    expected_qids: tuple[str, ...] = ()
    selection = manifest.get("selection")
    if isinstance(selection, Mapping):
        raw_qids = selection.get("resolved_question_ids")
        if isinstance(raw_qids, list) and all(isinstance(qid, str) and qid for qid in raw_qids):
            expected_qids = tuple(raw_qids)
            if len(expected_qids) != len(set(expected_qids)):
                errors.append("manifest source question IDs contain duplicates")
            if selection.get("count") != len(expected_qids):
                errors.append("manifest selection count does not match resolved question IDs")
        else:
            errors.append("manifest is missing source question order")
    else:
        errors.append("manifest is missing source question order")
    if len(expected_qids) != expected_questions:
        errors.append(
            f"wrong question count in manifest: expected {expected_questions}, got {len(expected_qids)}"
        )

    records: list[dict[str, object]] = []
    if _regular_file(results_path, "results JSONL", errors):
        from docprune.m3docvqa_factory import _canonicalize_result_record, _validate_result_record

        with results_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    errors.append(f"results JSONL contains a blank line at {line_number}")
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise ValueError("record is not an object")
                    record = _canonicalize_result_record(value, line_number=line_number)
                    expected_pages = manifest.get("page_count")
                    _validate_result_record(
                        record,
                        line_number=line_number,
                        expected_page_count=expected_pages
                        if isinstance(expected_pages, int)
                        else None,
                        mode=manifest.get("mode")
                        if isinstance(manifest.get("mode"), str)
                        else None,
                        # Keep independent record-shape validation separate from
                        # the production measurement checks below so memory and
                        # stage failures remain independently diagnosable.
                        production=False,
                    )
                    timing = record["timing"]
                    if not isinstance(timing, Mapping):
                        raise ValueError("timing is not an object")
                    required_measurements = {
                        "peak_allocated_gpu_bytes",
                        "warmup_excluded",
                    }
                    if not required_measurements <= set(timing):
                        raise ValueError(
                            "measurement fields require peak GPU bytes and warmup flag"
                        )
                    peak = timing["peak_allocated_gpu_bytes"]
                    if not isinstance(peak, int) or isinstance(peak, bool) or peak < 0:
                        raise ValueError("peak allocated GPU bytes must be nonnegative")
                    if production_run and peak == 0:
                        raise ValueError("production peak allocated GPU bytes must be positive")
                    if timing["warmup_excluded"] is not True:
                        raise ValueError("warmup must be explicitly excluded")
                    if production_run:
                        required_stages = {
                            "retrieval_seconds",
                            "qa_seconds",
                            "encoder_seconds",
                            "decoder_seconds",
                            "page_load_seconds",
                            "total_sample_seconds",
                        }
                        if not required_stages <= set(timing):
                            raise ValueError("production timing is missing stage boundaries")
                        for stage in required_stages:
                            value = timing[stage]
                            if (
                                not isinstance(value, int | float)
                                or isinstance(value, bool)
                                or not math.isfinite(float(value))
                                or float(value) <= 0
                            ):
                                raise ValueError(
                                    f"production {stage} must be finite and positive"
                                )
                    if "profiler_enabled" not in timing or not isinstance(
                        timing["profiler_enabled"], bool
                    ):
                        raise ValueError("per-record profiler_enabled must be a boolean")
                    if timing["profiler_enabled"] is not declared_profiler:
                        raise ValueError(
                            "per-record profiler state does not match the run manifest"
                        )
                    records.append(record)
                except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                    errors.append(f"invalid result record {line_number}: {error}")

    qids = tuple(str(record.get("question_id")) for record in records)
    counts = Counter(qids)
    duplicates = sorted(qid for qid, count in counts.items() if count > 1)
    if duplicates:
        errors.append(f"duplicate question IDs: {duplicates!r}")
    if len(records) != expected_questions:
        errors.append(f"wrong result count: expected {expected_questions}, got {len(records)}")
    expected_set = set(expected_qids)
    actual_set = set(qids)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing:
        errors.append(f"missing question IDs: {missing!r}")
    if unexpected:
        errors.append(f"unexpected question IDs: {unexpected!r}")
    if expected_qids and qids != expected_qids:
        errors.append("results do not preserve source order")
    _validate_source_questions(
        manifest,
        run_dir,
        expected_qids,
        records,
        errors,
        required=production_run,
    )

    index_path_value = manifest.get("index_manifest_source_path")
    index_digest = manifest.get("index_manifest_source_sha256")
    nested_index = manifest.get("index_manifest")
    if nested_index is not None and not isinstance(nested_index, Mapping):
        errors.append("index manifest identity is not an object")
        nested_index = None
    if isinstance(nested_index, Mapping):
        supplied_nested = nested_index.get("manifest_sha256")
        unsigned_nested = dict(nested_index)
        unsigned_nested.pop("manifest_sha256", None)
        if (
            not isinstance(supplied_nested, str)
            or supplied_nested
            != hashlib.sha256(
                json.dumps(unsigned_nested, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        ):
            errors.append("index manifest nested canonical digest mismatch")
        for key in ("mode", "page_count"):
            if key in manifest and key in nested_index and manifest[key] != nested_index[key]:
                errors.append(f"index manifest {key} does not match the run")
    if index_path_value is not None:
        index_path = Path(str(index_path_value))
        if not index_path.is_absolute():
            index_path = run_dir / index_path
        if _regular_file(index_path, "index manifest", errors):
            actual = hashlib.sha256(index_path.read_bytes()).hexdigest()
            if actual != index_digest:
                errors.append("index manifest digest mismatch")
            try:
                index_payload = json.loads(index_path.read_text(encoding="utf-8"))
                if isinstance(index_payload, Mapping):
                    supplied_index = index_payload.get("manifest_sha256")
                    unsigned = dict(index_payload)
                    unsigned.pop("manifest_sha256", None)
                    if (
                        supplied_index
                        != hashlib.sha256(
                            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
                        ).hexdigest()
                    ):
                        errors.append("index manifest canonical digest mismatch")
                    if isinstance(nested_index, Mapping) and dict(index_payload) != dict(
                        nested_index
                    ):
                        errors.append("index manifest identity does not match its source file")
                    if index_payload.get("schema_version") != 5:
                        errors.append("index manifest schema_version must equal 5")
                    else:
                        try:
                            from docprune.m3docvqa_factory import _load_index_manifest

                            _load_index_manifest(index_path)
                        except (FileNotFoundError, TypeError, ValueError, OSError) as error:
                            errors.append(f"index artifact validation failed: {error}")
            except (OSError, json.JSONDecodeError):
                errors.append("index manifest is invalid JSON")
    elif index_digest is not None:
        errors.append("index manifest digest is present without a source path")

    summary: dict[str, object] | None = None
    reproduced: dict[str, object] | None = None
    if _regular_file(summary_path, "summary", errors):
        try:
            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_summary, dict) or not _finite_json(loaded_summary):
                raise ValueError("summary must be a finite JSON object")
            summary = loaded_summary
            source_for_summary = source_rows_from_manifest(manifest, run_dir)
            if production_run and source_for_summary is None:
                errors.append("summary cannot include official quality without source questions")
            reproduced = summarize_benchmark_run(
                results_path,
                source_for_summary,
                require_positive=production_run,
            )
            if production_run and "quality" not in reproduced:
                errors.append("summary is missing official quality metrics")
            if not _finite_json(reproduced):
                raise ValueError("reproduced summary contains non-finite values")
            if summary != reproduced:
                errors.append("summary does not reproduce from results JSONL")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            errors.append(f"summary reproduction failed: {error}")

    return ValidationReport(
        valid=not errors,
        errors=tuple(errors),
        question_count=len(records),
        qids=qids,
        summary=summary,
        reproduced_summary=reproduced,
    )
