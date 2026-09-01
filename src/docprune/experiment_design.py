"""Outcome-blind cohort and statistical contracts for controlled experiments."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import random
import re
import secrets
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist

_ELIGIBILITY_KEYS = {"qid", "metadata", "cached_pages", "persisted_features"}
_METADATA_KEYS = {"type", "supporting_document_ids", "source_path", "source_sha256"}
_PAGE_KEYS = {"doc_id", "page_index", "source_path", "source_sha256"}
_FEATURE_KEYS = {
    "doc_id",
    "page_index",
    "feature_path",
    "feature_sha256",
    "index_manifest_path",
    "index_manifest_sha256",
}
_HOLDOUT_ORDER_PREFIX = b"docprune-random-coverage-v2"
_HOLDOUT_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "selection_method",
    "development_registry",
    "development_registry_path",
    "development_registry_file_sha256",
    "required_registry_labels",
    "power",
    "calibration",
    "required_n",
    "eligible_count",
    "eligible_order",
    "eligible_order_sha256",
    "selected_qids",
    "selection_sha256",
    "eligible_records",
    "eligibility_projection_sha256",
    "selected_records",
    "support_components",
    "runtime_pins",
    "input_file_hashes",
    "member_files",
    "manifest_sha256",
}


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def build_task9_preliminary_random_cohort(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: str,
    per_stratum: int = 24,
) -> dict[str, object]:
    """Build the seeded baseline-correct/wrong Task 9 development cohort."""

    from docprune.evaluation import list_em

    if not isinstance(seed, str) or not seed:
        raise ValueError("selection seed must be a nonempty string")
    if type(per_stratum) is not int or per_stratum <= 0:
        raise ValueError("per_stratum must be a positive integer")

    projected: dict[str, dict[str, object]] = {}
    strata: dict[str, list[str]] = {"baseline_correct": [], "baseline_wrong": []}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("cohort input row must be a mapping")
        qid = row.get("question_id")
        question = row.get("question")
        answers = row.get("answers")
        prediction = row.get("predicted_answer")
        retrieved_pages = row.get("retrieved_pages")
        if not isinstance(qid, str) or not qid:
            raise ValueError("question_id must be a nonempty string")
        if qid in projected:
            raise ValueError(f"duplicate question_id: {qid}")
        if not isinstance(question, str) or not question:
            raise ValueError(f"question must be a nonempty string: {qid}")
        if (
            not isinstance(answers, list)
            or not answers
            or any(not isinstance(answer, str) or not answer for answer in answers)
        ):
            raise ValueError(f"answers must be nonempty strings: {qid}")
        if not isinstance(prediction, str):
            raise ValueError(f"predicted_answer must be a string: {qid}")
        if not isinstance(retrieved_pages, list):
            raise ValueError(f"retrieved_pages must be a list: {qid}")

        baseline_correct = list_em(prediction, answers) == 1.0
        stratum = "baseline_correct" if baseline_correct else "baseline_wrong"
        projected[qid] = {
            "question_id": qid,
            "question": question,
            "answers": list(answers),
            "unpruned_predicted_answer": prediction,
            "retrieved_pages": list(retrieved_pages),
            "baseline_em": 1.0 if baseline_correct else 0.0,
            "baseline_stratum": stratum,
        }
        strata[stratum].append(qid)

    eligible_counts = {name: len(qids) for name, qids in strata.items()}
    for name, count in eligible_counts.items():
        if count < per_stratum:
            raise ValueError(f"{name} pool has {count} questions but requires {per_stratum}")

    def selection_key(qid: str, stratum: str) -> tuple[str, str]:
        encoded = f"{seed}\0{stratum}\0{qid}".encode()
        return hashlib.sha256(encoded).hexdigest(), qid

    eligible_order = {
        name: sorted(qids, key=lambda qid, name=name: selection_key(qid, name))
        for name, qids in strata.items()
    }
    selected_qids = {
        name: qids[:per_stratum] for name, qids in eligible_order.items()
    }
    selected_records = [
        projected[qid]
        for name in ("baseline_correct", "baseline_wrong")
        for qid in selected_qids[name]
    ]
    total_eligible = sum(eligible_counts.values())
    cohort: dict[str, object] = {
        "schema_version": 1,
        "status": "selected",
        "purpose": "Task 9 preliminary stratified-random development pilot",
        "selection_method": (
            "within each canonical list-EM stratum, sort ascending by "
            "sha256(seed + NUL + stratum + NUL + question_id), then question_id"
        ),
        "seed": seed,
        "per_stratum": per_stratum,
        "eligible_count": total_eligible,
        "eligible_counts": eligible_counts,
        "natural_pool_weights": {
            name: count / total_eligible for name, count in eligible_counts.items()
        },
        "eligible_order_sha256": {
            name: hashlib.sha256("\n".join(qids).encode("utf-8")).hexdigest()
            for name, qids in eligible_order.items()
        },
        "selected_qids": selected_qids,
        "selected_records": selected_records,
    }
    cohort["cohort_sha256"] = _canonical_json_sha256(cohort)
    return cohort


def build_task9_preliminary_fixture_inputs(
    cohort: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, str]]]:
    """Project the sealed 48-question cohort into retrieval-free fixed-page inputs."""

    supplied_sha = cohort.get("cohort_sha256")
    unsigned = dict(cohort)
    unsigned.pop("cohort_sha256", None)
    if supplied_sha != _canonical_json_sha256(unsigned):
        raise ValueError("Task 9 preliminary cohort checksum is invalid")
    selected = cohort.get("selected_qids")
    records = cohort.get("selected_records")
    if (
        not isinstance(selected, Mapping)
        or set(selected) != {"baseline_correct", "baseline_wrong"}
        or any(not isinstance(selected[name], list) or len(selected[name]) != 24 for name in selected)
        or not isinstance(records, list)
        or len(records) != 48
    ):
        raise ValueError("Task 9 preliminary cohort must contain the sealed 24/24 selection")
    expected_qids = [*selected["baseline_correct"], *selected["baseline_wrong"]]
    rows: dict[str, dict[str, object]] = {}
    eligible: list[dict[str, str]] = []
    observed_qids: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("Task 9 preliminary selected record is invalid")
        qid = record.get("question_id")
        question = record.get("question")
        pages = record.get("retrieved_pages")
        stratum = "baseline_correct" if index < 24 else "baseline_wrong"
        if (
            not isinstance(qid, str)
            or not qid
            or not isinstance(question, str)
            or not question
            or record.get("baseline_stratum") != stratum
            or not isinstance(pages, list)
            or len(pages) != 4
        ):
            raise ValueError("Task 9 preliminary selected record identity is invalid")
        checked_pages: list[dict[str, object]] = []
        for page in pages:
            if not isinstance(page, Mapping) or set(page) != {"doc_id", "page_index", "score"}:
                raise ValueError("Task 9 preliminary fixed page schema is invalid")
            checked_pages.append(dict(page))
        observed_qids.append(qid)
        rows[qid] = {"retrieved_pages": checked_pages}
        eligible.append({"qid": qid, "question": question})
    if observed_qids != expected_qids or len(set(observed_qids)) != 48:
        raise ValueError("Task 9 preliminary selected record order is invalid")
    reference: dict[str, object] = {
        "schema_version": 1,
        "selection_is_outcome_blind": True,
        "fixed_page_selection_is_outcome_blind": True,
        "question_selection": "24-baseline-correct/24-baseline-wrong",
        "cohort_sha256": supplied_sha,
        "question_ids": observed_qids,
        "rows": rows,
    }
    return reference, eligible


def task9_preprocessing_batch_indices(
    array_task_id: int,
    *,
    question_count: int = 48,
    completed_prefix_count: int = 1,
    batch_size: int = 4,
) -> tuple[int, ...]:
    """Return one array task's contiguous remaining-question batch."""

    if type(array_task_id) is not int or array_task_id < 0:
        raise ValueError("array task ID must be a nonnegative integer")
    if (
        type(question_count) is not int
        or type(completed_prefix_count) is not int
        or type(batch_size) is not int
        or question_count <= 0
        or not 0 <= completed_prefix_count < question_count
        or batch_size <= 0
    ):
        raise ValueError("Task 9 preprocessing batch dimensions are invalid")
    start = completed_prefix_count + array_task_id * batch_size
    if start >= question_count:
        raise ValueError("array task ID exceeds the remaining-question batch count")
    return tuple(range(start, min(start + batch_size, question_count)))


def _open_directory_nofollow(path: Path) -> int:
    """Open an absolute directory by walking every component without symlinks."""

    absolute = Path(os.path.abspath(path))
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as error:
        os.close(descriptor)
        raise ValueError(
            f"publication parent contains a symlink, missing, or non-directory component: {absolute}"
        ) from error
    return descriptor


def _renameat2_noreplace(
    old_directory_fd: int,
    old_name: str,
    new_directory_fd: int,
    new_name: str,
) -> None:
    """Move without replacement, using a hard-link fallback for regular files."""

    def link_regular_fallback(error_number: int) -> None:
        source = os.stat(old_name, dir_fd=old_directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(source.st_mode):
            raise OSError(error_number, os.strerror(error_number), new_name)
        try:
            os.link(
                old_name,
                new_name,
                src_dir_fd=old_directory_fd,
                dst_dir_fd=new_directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), new_name) from error
        try:
            published = os.stat(new_name, dir_fd=new_directory_fd, follow_symlinks=False)
            current_source = os.stat(old_name, dir_fd=old_directory_fd, follow_symlinks=False)
            if (published.st_dev, published.st_ino) != (
                current_source.st_dev,
                current_source.st_ino,
            ):
                raise OSError(errno.ESTALE, "regular-file source changed during publication")
            os.unlink(old_name, dir_fd=old_directory_fd)
        except BaseException:
            try:
                os.unlink(new_name, dir_fd=new_directory_fd)
            except FileNotFoundError:
                pass
            raise

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        link_regular_fallback(errno.ENOSYS)
        return
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        old_directory_fd,
        os.fsencode(old_name),
        new_directory_fd,
        os.fsencode(new_name),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), new_name)
    if error_number in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
        link_regular_fallback(error_number)
        return
    raise OSError(error_number, os.strerror(error_number), new_name)


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename one sibling without replacing any destination."""

    source = Path(os.path.abspath(source))
    destination = Path(os.path.abspath(destination))
    if source.parent != destination.parent:
        raise ValueError("atomic no-replace rename requires sibling paths")
    parent_fd = _open_directory_nofollow(source.parent)
    try:
        source_stat = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(source_stat.st_mode):
            raise ValueError("atomic no-replace source must be a real directory")
        _renameat2_noreplace(parent_fd, source.name, parent_fd, destination.name)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _link_directory_manifest_last_noreplace(
    *,
    parent_fd: int,
    temporary_fd: int,
    temporary_name: str,
    destination_name: str,
    member_names: Sequence[str],
) -> None:
    """Commit a directory on filesystems lacking renameat2, with manifest.json last."""

    os.mkdir(destination_name, mode=0o700, dir_fd=parent_fd)
    destination_fd: int | None = None
    linked: list[str] = []
    try:
        destination_fd = os.open(
            destination_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        order = [name for name in member_names if name != "manifest.json"]
        if "manifest.json" not in member_names:
            raise ValueError("directory commit requires manifest.json as its commit marker")
        order.append("manifest.json")
        for name in order:
            os.link(
                name,
                name,
                src_dir_fd=temporary_fd,
                dst_dir_fd=destination_fd,
                follow_symlinks=False,
            )
            linked.append(name)
        os.fsync(destination_fd)
        os.fsync(parent_fd)
    except BaseException:
        if destination_fd is not None:
            for name in reversed(linked):
                try:
                    os.unlink(name, dir_fd=destination_fd)
                except FileNotFoundError:
                    pass
        try:
            os.rmdir(destination_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
    for name in member_names:
        os.unlink(name, dir_fd=temporary_fd)
    os.rmdir(temporary_name, dir_fd=parent_fd)


def _require_mapping_keys(value: object, allowed: set[str], *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    forbidden = sorted(set(value) - allowed)
    if forbidden:
        raise ValueError(f"forbidden {label} key: {forbidden[0]}")
    if set(value) != allowed:
        missing = sorted(allowed - set(value))
        raise ValueError(f"{label} is missing key: {missing[0]}")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _read_regular_file(path: Path, *, label: str) -> bytes:
    """Read one non-symlink regular file through an O_NOFOLLOW descriptor."""

    file_path = Path(path)
    if not file_path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    descriptor: int | None = None
    try:
        descriptor = os.open(file_path, os.O_RDONLY | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} must be a regular file: {file_path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    except OSError as error:
        raise ValueError(f"{label} is missing or not a regular file: {file_path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return b"".join(chunks)


def _sha256_regular_file(path: Path, *, label: str, expected: str | None = None) -> str:
    """Stream-hash one non-symlink regular file and optionally match its digest."""

    file_path = Path(path)
    if not file_path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    descriptor: int | None = None
    try:
        descriptor = os.open(file_path, os.O_RDONLY | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} must be a regular file: {file_path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    except OSError as error:
        raise ValueError(f"{label} is missing or not a regular file: {file_path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    actual = digest.hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return actual


def _authenticate_eligibility_records(records: Sequence[Mapping[str, object]]) -> None:
    """Authenticate all external byte identities bound by eligible records."""

    authenticated: dict[tuple[str, str], str] = {}

    def authenticate(path_value: object, digest_value: object, label: str) -> None:
        if not isinstance(path_value, str):
            raise ValueError(f"{label} path is invalid")
        expected = _require_sha256(digest_value, label=f"{label} digest")
        cache_key = (path_value, expected)
        if cache_key not in authenticated:
            authenticated[cache_key] = _sha256_regular_file(
                Path(path_value), label=label, expected=expected
            )

    for record in records:
        source = record["source"]
        authenticate(source["path"], source["sha256"], "question source")
        for page in record["cached_pages"]:
            authenticate(page["source_path"], page["source_sha256"], "cached page source")
        for feature in record["persisted_features"]:
            authenticate(feature["feature_path"], feature["feature_sha256"], "persisted feature")
            authenticate(
                feature["index_manifest_path"],
                feature["index_manifest_sha256"],
                "index manifest",
            )


def _eligibility_inputs_from_records(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Project sealed records back through the outcome-blind eligibility boundary."""

    raw_rows: list[dict[str, object]] = []
    record_keys = {
        "qid",
        "source",
        "supporting_document_ids",
        "cached_pages",
        "persisted_features",
    }
    source_keys = {"path", "sha256", "type"}
    for record in records:
        checked = _require_mapping_keys(record, record_keys, label="eligible record")
        source = _require_mapping_keys(checked["source"], source_keys, label="eligible source")
        raw_rows.append(
            {
                "qid": checked["qid"],
                "metadata": {
                    "type": source["type"],
                    "supporting_document_ids": checked["supporting_document_ids"],
                    "source_path": source["path"],
                    "source_sha256": source["sha256"],
                },
                "cached_pages": checked["cached_pages"],
                "persisted_features": checked["persisted_features"],
            }
        )
    return raw_rows


def build_eligibility_records(
    rows: Sequence[Mapping[str, object]], *, development_qids: Sequence[str]
) -> tuple[dict[str, object], ...]:
    """Validate the outcome-blind eligibility input boundary."""

    development = set(development_qids)
    if len(development) != len(development_qids):
        raise ValueError("development_qids must be unique")
    records: list[dict[str, object]] = []
    observed: set[str] = set()
    for row in rows:
        forbidden = sorted(set(row) - _ELIGIBILITY_KEYS)
        if forbidden:
            raise ValueError(f"forbidden eligibility key: {forbidden[0]}")
        if set(row) != _ELIGIBILITY_KEYS:
            missing = sorted(_ELIGIBILITY_KEYS - set(row))
            raise ValueError(f"eligibility row is missing key: {missing[0]}")
        qid = row["qid"]
        if not isinstance(qid, str) or not qid:
            raise ValueError("eligibility qid must be a nonempty string")
        if qid in observed:
            raise ValueError(f"duplicate eligibility qid: {qid}")
        observed.add(qid)
        metadata = _require_mapping_keys(row["metadata"], _METADATA_KEYS, label="metadata")
        supports = metadata["supporting_document_ids"]
        if (
            not isinstance(supports, Sequence)
            or isinstance(supports, str | bytes)
            or not supports
            or any(not isinstance(doc_id, str) or not doc_id for doc_id in supports)
            or len(set(supports)) != len(supports)
        ):
            raise ValueError(f"eligibility qid {qid} has invalid support-document IDs")
        source_path = metadata["source_path"]
        if not isinstance(source_path, str) or not source_path.startswith("/"):
            raise ValueError("source_path must be absolute")
        source_sha256 = _require_sha256(metadata["source_sha256"], label="source_sha256")
        pages_value = row["cached_pages"]
        features_value = row["persisted_features"]
        if (
            not isinstance(pages_value, Sequence)
            or isinstance(pages_value, str | bytes)
            or len(pages_value) != 4
        ):
            raise ValueError(f"eligibility qid {qid} must have cached DocPrune top-4 pages")
        if (
            not isinstance(features_value, Sequence)
            or isinstance(features_value, str | bytes)
            or len(features_value) != 4
        ):
            raise ValueError(f"eligibility qid {qid} must have four persisted features")
        pages: list[dict[str, object]] = []
        features: list[dict[str, object]] = []
        identities: set[tuple[str, int]] = set()
        for position, (page_value, feature_value) in enumerate(
            zip(pages_value, features_value, strict=True)
        ):
            page = _require_mapping_keys(page_value, _PAGE_KEYS, label="cached page")
            feature = _require_mapping_keys(feature_value, _FEATURE_KEYS, label="persisted feature")
            doc_id = page["doc_id"]
            page_index = page["page_index"]
            if not isinstance(doc_id, str) or not doc_id or type(page_index) is not int:
                raise ValueError(f"eligibility qid {qid} has invalid cached page {position}")
            identity = (doc_id, page_index)
            if identity in identities:
                raise ValueError(f"eligibility qid {qid} has duplicate cached pages")
            identities.add(identity)
            if (feature["doc_id"], feature["page_index"]) != identity:
                raise ValueError(f"eligibility qid {qid} page/feature identity mismatch")
            for path_key in ("source_path",):
                if not isinstance(page[path_key], str) or not page[path_key].startswith("/"):
                    raise ValueError(f"cached page {path_key} must be absolute")
            _require_sha256(page["source_sha256"], label="cached page source_sha256")
            for path_key in ("feature_path", "index_manifest_path"):
                if not isinstance(feature[path_key], str) or not feature[path_key].startswith("/"):
                    raise ValueError(f"persisted feature {path_key} must be absolute")
            _require_sha256(feature["feature_sha256"], label="feature_sha256")
            _require_sha256(feature["index_manifest_sha256"], label="index_manifest_sha256")
            pages.append(dict(page))
            features.append(dict(feature))
        if metadata["type"] != "single_hop" or qid in development:
            continue
        records.append(
            {
                "qid": qid,
                "source": {
                    "path": source_path,
                    "sha256": source_sha256,
                    "type": "single_hop",
                },
                "supporting_document_ids": list(supports),
                "cached_pages": pages,
                "persisted_features": features,
            }
        )
    return tuple(records)


def holdout_order(qids: Sequence[str]) -> tuple[dict[str, str], ...]:
    """Order unique QIDs by the frozen versioned SHA-256 contract."""

    values = tuple(qids)
    if any(not isinstance(qid, str) or not qid for qid in values):
        raise ValueError("holdout QIDs must be nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError("holdout QIDs must be unique")
    rows = [
        {
            "qid": qid,
            "order_sha256": hashlib.sha256(_HOLDOUT_ORDER_PREFIX + qid.encode("utf-8")).hexdigest(),
        }
        for qid in values
    ]
    return tuple(sorted(rows, key=lambda row: (row["order_sha256"], row["qid"])))


def support_document_components(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build deterministic QID components linked by any support document."""

    parent: dict[str, str] = {}

    def find(qid: str) -> str:
        while parent[qid] != qid:
            parent[qid] = parent[parent[qid]]
            qid = parent[qid]
        return qid

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    owner_by_document: dict[str, str] = {}
    for record in records:
        if set(record) != {"qid", "supporting_document_ids"}:
            raise ValueError("component records expose only qid and supporting_document_ids")
        qid = record["qid"]
        supports = record["supporting_document_ids"]
        if not isinstance(qid, str) or not qid or qid in parent:
            raise ValueError("component QIDs must be unique nonempty strings")
        if (
            not isinstance(supports, Sequence)
            or isinstance(supports, str | bytes)
            or not supports
            or any(not isinstance(doc_id, str) or not doc_id for doc_id in supports)
        ):
            raise ValueError(f"component qid {qid} has empty support identity")
        parent[qid] = qid
        for doc_id in set(supports):
            owner = owner_by_document.setdefault(doc_id, qid)
            union(qid, owner)
    members_by_root: dict[str, list[str]] = {}
    for qid in sorted(parent):
        members_by_root.setdefault(find(qid), []).append(qid)
    member_lists = sorted(members_by_root.values(), key=lambda members: tuple(members))
    components = [
        {
            "component_id": hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest(),
            "qids": members,
        }
        for members in member_lists
    ]
    distribution = Counter(len(members) for members in member_lists)
    return {
        "component_id_rule": "sha256(newline-joined sorted QIDs)",
        "components": components,
        "size_distribution": {str(size): count for size, count in sorted(distribution.items())},
    }


def _paired_differences(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, tuple[float, tuple[float, ...]]]:
    pairs: dict[str, tuple[float, tuple[float, ...]]] = {}
    for row in rows:
        if set(row) != {"qid", "aggregate_f1", "global_random_f1"}:
            raise ValueError("statistical rows have an invalid schema")
        qid = row["qid"]
        random_values = row["global_random_f1"]
        if not isinstance(qid, str) or not qid or qid in pairs:
            raise ValueError("statistical QIDs must be unique nonempty strings")
        if (
            not isinstance(random_values, Sequence)
            or isinstance(random_values, str | bytes)
            or not random_values
        ):
            raise ValueError(f"qid {qid} must have random-mask repetitions")
        try:
            aggregate = float(row["aggregate_f1"])
            repetitions = tuple(float(value) for value in random_values)
        except (TypeError, ValueError) as error:
            raise ValueError(f"qid {qid} has nonnumeric F1") from error
        if not math.isfinite(aggregate) or any(not math.isfinite(value) for value in repetitions):
            raise ValueError(f"qid {qid} has nonfinite F1")
        pairs[qid] = (aggregate, repetitions)
    if not pairs:
        raise ValueError("statistical rows must not be empty")
    return pairs


def primary_f1_estimand(rows: Sequence[Mapping[str, object]]) -> float:
    """Mean paired aggregate-score minus within-QID global-random F1."""

    pairs = _paired_differences(rows)
    return sum(
        aggregate - sum(repetitions) / len(repetitions) for aggregate, repetitions in pairs.values()
    ) / len(pairs)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def nested_f1_inference(
    rows: Sequence[Mapping[str, object]],
    *,
    components: Sequence[Sequence[str]],
    draws: int = 100_000,
    seed: int = 20_260_827,
) -> dict[str, object]:
    """Run the frozen nested mask/component bootstrap for paired F1."""

    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    pairs = _paired_differences(rows)
    normalized_components = tuple(tuple(component) for component in components)
    if len(normalized_components) < 2:
        raise ValueError("cluster inference requires at least two support components")
    flattened = [qid for component in normalized_components for qid in component]
    if (
        any(not component for component in normalized_components)
        or len(flattened) != len(set(flattened))
        or set(flattened) != set(pairs)
    ):
        raise ValueError("support components must partition statistical QIDs exactly once")
    generator = random.Random(seed)
    bootstrap: list[float] = []
    for _ in range(draws):
        inner_differences: dict[str, float] = {}
        for qid, (aggregate, repetitions) in pairs.items():
            random_mean = sum(
                repetitions[generator.randrange(len(repetitions))] for _ in range(len(repetitions))
            ) / len(repetitions)
            inner_differences[qid] = aggregate - random_mean
        selected_components = [
            normalized_components[generator.randrange(len(normalized_components))]
            for _ in range(len(normalized_components))
        ]
        selected_qids = [qid for component in selected_components for qid in component]
        bootstrap.append(sum(inner_differences[qid] for qid in selected_qids) / len(selected_qids))
    return {
        "estimand": "mean_qid(aggregate_score_f1 - mean_mask(global_random_f1))",
        "point_estimate": primary_f1_estimand(rows),
        "draw_count": draws,
        "seed": seed,
        "tost_90_interval": [_quantile(bootstrap, 0.05), _quantile(bootstrap, 0.95)],
        "superiority_95_interval": [
            _quantile(bootstrap, 0.025),
            _quantile(bootstrap, 0.975),
        ],
        "bootstrap_draws": tuple(bootstrap),
    }


_VISUAL_STATE_BOUNDARIES = ("B_input", "B_0", "B_6", "B_13", "B_20", "B_23", "B_26")


def visual_state_removal_inference(
    rows: Sequence[Mapping[str, object]],
    *,
    components: Sequence[Sequence[str]],
    draws: int = 100_000,
    seed: int = 20_260_827,
    dependence_margin_f1: float = 1.0,
) -> dict[str, object]:
    """Simultaneous lower bounds for the fixed all-visual-state removal curve."""

    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    margin = float(dependence_margin_f1)
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("dependence_margin_f1 must be positive and finite")

    differences: dict[str, dict[str, float]] = {}
    for row in rows:
        if set(row) != {"qid", "reference_f1", "all_drop_f1"}:
            raise ValueError("visual-state removal rows have an invalid schema")
        qid = row["qid"]
        all_drop = row["all_drop_f1"]
        if not isinstance(qid, str) or not qid or qid in differences:
            raise ValueError("visual-state removal QIDs must be unique nonempty strings")
        if not isinstance(all_drop, Mapping) or tuple(all_drop) != _VISUAL_STATE_BOUNDARIES:
            raise ValueError("all-drop F1 must contain the exact ordered fixed boundaries")
        try:
            reference = float(row["reference_f1"])
            values = {boundary: float(all_drop[boundary]) for boundary in _VISUAL_STATE_BOUNDARIES}
        except (TypeError, ValueError) as error:
            raise ValueError(f"qid {qid} has nonnumeric F1") from error
        if not math.isfinite(reference) or any(
            not math.isfinite(value) for value in values.values()
        ):
            raise ValueError(f"qid {qid} has nonfinite F1")
        differences[qid] = {
            boundary: values[boundary] - reference for boundary in _VISUAL_STATE_BOUNDARIES
        }
    if not differences:
        raise ValueError("visual-state removal rows must not be empty")

    normalized_components = tuple(tuple(component) for component in components)
    if len(normalized_components) < 2:
        raise ValueError("cluster inference requires at least two support components")
    flattened = [qid for component in normalized_components for qid in component]
    if (
        any(not component for component in normalized_components)
        or len(flattened) != len(set(flattened))
        or set(flattened) != set(differences)
    ):
        raise ValueError("support components must partition visual-state QIDs exactly once")

    point_estimates = {
        boundary: sum(row[boundary] for row in differences.values()) / len(differences)
        for boundary in _VISUAL_STATE_BOUNDARIES
    }
    generator = random.Random(seed)
    max_statistics: list[float] = []
    for _ in range(draws):
        selected_components = [
            normalized_components[generator.randrange(len(normalized_components))]
            for _ in range(len(normalized_components))
        ]
        selected_qids = [qid for component in selected_components for qid in component]
        bootstrap_estimates = {
            boundary: sum(differences[qid][boundary] for qid in selected_qids) / len(selected_qids)
            for boundary in _VISUAL_STATE_BOUNDARIES
        }
        max_statistics.append(
            max(
                bootstrap_estimates[boundary] - point_estimates[boundary]
                for boundary in _VISUAL_STATE_BOUNDARIES
            )
        )
    critical = _quantile(max_statistics, 0.95)
    lower_bounds = {
        boundary: point_estimates[boundary] - critical for boundary in _VISUAL_STATE_BOUNDARIES
    }
    persistent_boundary = next(
        (
            boundary
            for index, boundary in enumerate(_VISUAL_STATE_BOUNDARIES)
            if all(lower_bounds[later] > -margin for later in _VISUAL_STATE_BOUNDARIES[index:])
        ),
        None,
    )
    nonmonotonic = [
        {"earlier": earlier, "later": later}
        for earlier, later in zip(_VISUAL_STATE_BOUNDARIES, _VISUAL_STATE_BOUNDARIES[1:])
        if point_estimates[later] < point_estimates[earlier]
    ]
    return {
        "estimand": "mean_qid(all_drop_f1 - btp_qtp_no_ctp_f1)",
        "method": ("one-sided 95% nonstudentized basic max-statistic support-component bootstrap"),
        "confidence_level": 0.95,
        "boundaries": list(_VISUAL_STATE_BOUNDARIES),
        "point_estimates": point_estimates,
        "simultaneous_lower_bounds": lower_bounds,
        "max_statistic_critical": critical,
        "dependence_margin_f1": margin,
        "persistent_boundary": persistent_boundary,
        "nonmonotonic_adjacent_pairs": nonmonotonic,
        "draw_count": draws,
        "seed": seed,
    }


def visual_state_opportunity_strata(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build the prespecified descriptive evidence-opportunity strata."""

    keys = {
        "qid",
        "supporting_document_ids",
        "retrieved_document_ids",
        "reference",
        "input_all_drop",
    }
    reference_keys = {"name", "result_sha256", "em_correct", "f1"}
    input_keys = {
        "name",
        "boundary",
        "mode",
        "retained_visual_ids",
        "result_sha256",
        "em_correct",
        "best_reference_loglikelihood_drop_per_token",
        "likelihood_target",
        "likelihood_target_sha256",
    }
    full: list[str] = []
    support: list[str] = []
    em_correct: list[str] = []
    f1_positive: list[str] = []
    visually_sensitive: list[str] = []
    observed: set[str] = set()
    for row in rows:
        if set(row) != keys:
            raise ValueError("visual-state opportunity rows have an invalid schema")
        qid = row["qid"]
        if not isinstance(qid, str) or not qid or qid in observed:
            raise ValueError("visual-state opportunity QIDs must be unique nonempty strings")
        observed.add(qid)
        supports = row["supporting_document_ids"]
        retrieved = row["retrieved_document_ids"]
        if (
            not isinstance(supports, Sequence)
            or isinstance(supports, str | bytes)
            or not supports
            or any(not isinstance(doc_id, str) or not doc_id for doc_id in supports)
            or len(set(supports)) != len(supports)
        ):
            raise ValueError(f"qid {qid} has invalid support-document IDs")
        if (
            not isinstance(retrieved, Sequence)
            or isinstance(retrieved, str | bytes)
            or len(retrieved) != 4
            or any(not isinstance(doc_id, str) or not doc_id for doc_id in retrieved)
        ):
            raise ValueError(f"qid {qid} must have four cached retrieved document IDs")
        reference = _require_mapping_keys(
            row["reference"], reference_keys, label="visual-state reference"
        )
        input_all_drop = _require_mapping_keys(
            row["input_all_drop"], input_keys, label="B_input all-drop result"
        )
        if reference["name"] != "btp-qtp-no-ctp":
            raise ValueError(f"qid {qid} has the wrong visual-state reference")
        if (
            input_all_drop["name"] != "all-visual-drop-B_input"
            or input_all_drop["boundary"] != "B_input"
            or input_all_drop["mode"] != "physical_delete"
            or input_all_drop["retained_visual_ids"] != []
        ):
            raise ValueError(f"qid {qid} has the wrong B_input all-drop identity")
        _require_sha256(reference["result_sha256"], label="visual-state reference result")
        _require_sha256(input_all_drop["result_sha256"], label="B_input all-drop result")
        if input_all_drop["likelihood_target"] != "best-reference-full-gold-sequence":
            raise ValueError(f"qid {qid} has the wrong likelihood target")
        _require_sha256(input_all_drop["likelihood_target_sha256"], label="gold likelihood target")
        if (
            type(reference["em_correct"]) is not bool
            or type(input_all_drop["em_correct"]) is not bool
        ):
            raise ValueError(f"qid {qid} has a non-boolean opportunity indicator")
        try:
            reference_f1 = float(reference["f1"])
            gold_drop = float(input_all_drop["best_reference_loglikelihood_drop_per_token"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"qid {qid} has a nonnumeric opportunity value") from error
        if not 0.0 <= reference_f1 <= 100.0 or not math.isfinite(gold_drop):
            raise ValueError(f"qid {qid} has a nonfinite opportunity value")
        full.append(qid)
        if set(supports).issubset(set(retrieved)):
            support.append(qid)
        if reference["em_correct"]:
            em_correct.append(qid)
        if reference_f1 > 0:
            f1_positive.append(qid)
        if (reference["em_correct"] and not input_all_drop["em_correct"]) or gold_drop >= 0.1:
            visually_sensitive.append(qid)
    if not full:
        raise ValueError("visual-state opportunity rows must not be empty")
    return {
        "full": full,
        "support_document_retrieved": support,
        "no_ctp_em_correct": em_correct,
        "no_ctp_f1_positive": f1_positive,
        "input_visually_sensitive": visually_sensitive,
        "input_visual_sensitivity_rule": (
            "(reference_em_correct and not input_all_drop_em_correct) or "
            "best_reference_loglikelihood_drop_per_token >= 0.1"
        ),
    }


def compile_task7_visual_state_report(
    analysis_bundles: Sequence[Mapping[str, object]],
    *,
    draws: int = 100_000,
    seed: int = 20_260_827,
) -> dict[str, object]:
    """Compile only curve/opportunity rows bound to the same exact artifacts."""

    bundle_keys = {
        "schema_version",
        "qid",
        "curve",
        "opportunity",
        "analysis_bundle_sha256",
    }
    arm_keys = {"provenance", "row"}
    provenance_keys = {
        "fixture_sha256",
        "run_manifest_file_sha256",
        "run_manifest_sha256",
        "results_file_sha256",
        "likelihood_file_sha256",
        "reference_result_sha256",
        "input_all_drop_result_sha256",
    }
    curves: list[dict[str, object]] = []
    opportunities: list[dict[str, object]] = []
    bundle_sha256s: list[str] = []
    for raw_bundle in analysis_bundles:
        bundle = dict(_require_mapping_keys(raw_bundle, bundle_keys, label="Task 7 bundle"))
        if bundle["schema_version"] != 1:
            raise ValueError("Task 7 analysis bundle has an unsupported schema")
        bundle_sha256 = _require_sha256(
            bundle["analysis_bundle_sha256"], label="Task 7 analysis bundle"
        )
        unsigned_bundle = dict(bundle)
        unsigned_bundle.pop("analysis_bundle_sha256")
        if bundle_sha256 != _canonical_json_sha256(unsigned_bundle):
            raise ValueError("Task 7 analysis bundle identity is invalid")
        curve_arm = _require_mapping_keys(bundle["curve"], arm_keys, label="Task 7 curve arm")
        opportunity_arm = _require_mapping_keys(
            bundle["opportunity"], arm_keys, label="Task 7 opportunity arm"
        )
        curve_provenance = dict(
            _require_mapping_keys(
                curve_arm["provenance"], provenance_keys, label="Task 7 curve provenance"
            )
        )
        opportunity_provenance = dict(
            _require_mapping_keys(
                opportunity_arm["provenance"],
                provenance_keys,
                label="Task 7 opportunity provenance",
            )
        )
        for key in provenance_keys:
            _require_sha256(curve_provenance[key], label=f"Task 7 provenance {key}")
            _require_sha256(opportunity_provenance[key], label=f"Task 7 provenance {key}")
        if curve_provenance != opportunity_provenance:
            raise ValueError("Task 7 rows must have the same exact artifact provenance")
        curve_row = dict(
            _require_mapping_keys(
                curve_arm["row"],
                {"qid", "reference_f1", "all_drop_f1"},
                label="Task 7 curve row",
            )
        )
        opportunity_row = dict(
            _require_mapping_keys(
                opportunity_arm["row"],
                {
                    "qid",
                    "supporting_document_ids",
                    "retrieved_document_ids",
                    "reference",
                    "input_all_drop",
                },
                label="Task 7 opportunity row",
            )
        )
        qid = bundle["qid"]
        if (
            not isinstance(qid, str)
            or not qid
            or curve_row["qid"] != qid
            or opportunity_row["qid"] != qid
        ):
            raise ValueError("Task 7 bundle and rows must contain the same nonempty QID")
        reference = opportunity_row["reference"]
        input_all_drop = opportunity_row["input_all_drop"]
        if not isinstance(reference, Mapping) or not isinstance(input_all_drop, Mapping):
            raise ValueError("Task 7 opportunity result identities are invalid")
        if (
            reference.get("result_sha256") != curve_provenance["reference_result_sha256"]
            or input_all_drop.get("result_sha256")
            != curve_provenance["input_all_drop_result_sha256"]
        ):
            raise ValueError("Task 7 result hashes do not match artifact provenance")
        try:
            reference_f1 = float(reference["f1"])
            curve_reference_f1 = float(curve_row["reference_f1"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Task 7 rows have invalid reference F1") from error
        if curve_reference_f1 != reference_f1:
            raise ValueError("Task 7 rows must have the same reference F1")
        curves.append(curve_row)
        opportunities.append(opportunity_row)
        bundle_sha256s.append(bundle_sha256)
    if not curves:
        raise ValueError("Task 7 analysis bundles must not be empty")
    curve_qids = tuple(row["qid"] for row in curves)
    component_records = [
        {
            "qid": row["qid"],
            "supporting_document_ids": row.get("supporting_document_ids"),
        }
        for row in opportunities
    ]
    support_components = support_document_components(component_records)
    components = tuple(tuple(component["qids"]) for component in support_components["components"])
    curve = visual_state_removal_inference(
        tuple(curves),
        components=components,
        draws=draws,
        seed=seed,
    )
    strata = visual_state_opportunity_strata(tuple(opportunities))
    report: dict[str, object] = {
        "schema_version": 1,
        "analysis_name": "explicit-visual-state-removal-dependence-curve",
        "claim_boundary": (
            "not-an-information-horizon-without-separate-standalone-token-information"
        ),
        "qids": list(curve_qids),
        "analysis_bundle_sha256s": bundle_sha256s,
        "support_components": support_components,
        "curve": curve,
        "opportunity_strata": strata,
        "curve_rows_sha256": _canonical_json_sha256(tuple(curves)),
        "opportunity_rows_sha256": _canonical_json_sha256(tuple(opportunities)),
    }
    report["report_sha256"] = _canonical_json_sha256(report)
    return report


def classify_f1_result(
    *,
    tost_90_interval: Sequence[float],
    superiority_95_interval: Sequence[float],
    equivalence_margin_f1: float = 1.0,
) -> dict[str, object]:
    """Return orthogonal inference flags and the frozen precedence label."""

    if not math.isfinite(equivalence_margin_f1) or equivalence_margin_f1 <= 0:
        raise ValueError("equivalence_margin_f1 must be positive and finite")

    def interval(value: Sequence[float], label: str) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError(f"{label} must contain two bounds")
        lower, upper = float(value[0]), float(value[1])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(f"{label} bounds are invalid")
        return lower, upper

    tost_lower, tost_upper = interval(tost_90_interval, "TOST 90% interval")
    superiority_lower, superiority_upper = interval(
        superiority_95_interval, "superiority 95% interval"
    )
    equivalent = tost_lower >= -equivalence_margin_f1 and tost_upper <= equivalence_margin_f1
    superior = superiority_lower > 0.0
    inferior = superiority_upper < 0.0
    if superior:
        label = "superior"
    elif inferior:
        label = "inferior"
    elif equivalent:
        label = "equivalent"
    else:
        label = "unresolved"
    return {
        "equivalent": equivalent,
        "superior": superior,
        "inferior": inferior,
        "label": label,
        "equivalence_margin_f1": equivalence_margin_f1,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Apply deterministic Holm adjustment to one declared secondary family."""

    if not p_values:
        raise ValueError("Holm family must not be empty")
    validated: list[tuple[str, float]] = []
    for label, value in p_values.items():
        if not isinstance(label, str) or not label:
            raise ValueError("Holm labels must be nonempty strings")
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Holm p-value for {label} must be between zero and one")
        validated.append((label, probability))
    ranked = sorted(validated, key=lambda item: (item[1], item[0]))
    adjusted_by_label: dict[str, float] = {}
    running_max = 0.0
    family_size = len(ranked)
    for rank, (label, probability) in enumerate(ranked):
        running_max = max(running_max, min(1.0, (family_size - rank) * probability))
        adjusted_by_label[label] = running_max
    return {label: adjusted_by_label[label] for label in p_values}


def calibrate_random_repetitions(
    random_f1_by_qid: Mapping[str, Sequence[float]],
    *,
    threshold_f1: float = 0.25,
) -> dict[str, object]:
    """Freeze 10 or 20 masks from the prescribed developmental MCSE."""

    if not random_f1_by_qid:
        raise ValueError("random calibration requires developmental QIDs")
    if not math.isfinite(threshold_f1) or threshold_f1 <= 0:
        raise ValueError("threshold_f1 must be positive and finite")
    variance_terms: list[float] = []
    qids = sorted(random_f1_by_qid)
    for qid in qids:
        if not isinstance(qid, str) or not qid:
            raise ValueError("calibration QIDs must be nonempty strings")
        raw_values = random_f1_by_qid[qid]
        if len(raw_values) != 10:
            raise ValueError(f"calibration qid {qid} must have exactly 10 initial masks")
        values = tuple(float(value) for value in raw_values)
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"calibration qid {qid} has nonfinite F1")
        mean = sum(values) / len(values)
        sample_variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        variance_terms.append(sample_variance / len(values))
    mcse = math.sqrt(sum(variance_terms)) / len(qids)
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "method": "sqrt(sum_i(sample_variance_i / R_i)) / N",
        "qid_count": len(qids),
        "qid_sha256": hashlib.sha256("\n".join(qids).encode("utf-8")).hexdigest(),
        "initial_repetitions": 10,
        "mcse_f1": mcse,
        "threshold_f1": threshold_f1,
        "decision_rule": "freeze 20 iff MCSE is strictly greater than threshold; else 10",
        "frozen_repetitions": 20 if mcse > threshold_f1 else 10,
    }
    payload["calibration_sha256"] = _canonical_json_sha256(payload)
    return payload


def plan_equivalence_power(
    *,
    developmental_mean_f1: float,
    developmental_sd_f1: float,
    cluster_design_effect: float,
    eligible_pool_size: int,
    equivalence_margin_f1: float = 1.0,
    target_power: float = 0.80,
) -> dict[str, object]:
    """Choose the smallest N under the documented clustered normal TOST approximation."""

    mean = float(developmental_mean_f1)
    standard_deviation = float(developmental_sd_f1)
    design_effect = float(cluster_design_effect)
    if not all(math.isfinite(value) for value in (mean, standard_deviation, design_effect)):
        raise ValueError("power inputs must be finite")
    if standard_deviation <= 0 or design_effect < 1:
        raise ValueError("developmental SD must be positive and design effect at least one")
    if type(eligible_pool_size) is not int or eligible_pool_size < 2:
        raise ValueError("eligible_pool_size must be an integer of at least two")
    if not 0 < target_power < 1 or equivalence_margin_f1 <= 0:
        raise ValueError("power target and equivalence margin are invalid")
    normal = NormalDist()
    critical = normal.inv_cdf(0.95)

    def power(sample_size: int) -> float:
        standard_error = standard_deviation * math.sqrt(design_effect / sample_size)
        acceptance_lower = -equivalence_margin_f1 + critical * standard_error
        acceptance_upper = equivalence_margin_f1 - critical * standard_error
        if acceptance_lower >= acceptance_upper:
            return 0.0
        return max(
            0.0,
            normal.cdf((acceptance_upper - mean) / standard_error)
            - normal.cdf((acceptance_lower - mean) / standard_error),
        )

    required_n = next(
        (sample_size for sample_size in range(2, 1_000_001) if power(sample_size) >= target_power),
        None,
    )
    if required_n is None:
        raise ValueError("developmental mean cannot reach target equivalence power")
    selected_n = min(required_n, eligible_pool_size)
    achieved_power = power(selected_n)
    admissible = eligible_pool_size >= required_n
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "passed" if admissible else "underpowered_full_pool",
        "method": (
            "normal TOST approximation centered on developmental paired mean; "
            "SE=developmental_sd*sqrt(cluster_design_effect/N); 90% CI critical z_0.95"
        ),
        "developmental_mean_f1": mean,
        "developmental_sd_f1": standard_deviation,
        "cluster_design_effect": design_effect,
        "equivalence_margin_f1": equivalence_margin_f1,
        "target_power": target_power,
        "eligible_pool_size": eligible_pool_size,
        "required_n": required_n,
        "selected_n": selected_n,
        "achieved_power": achieved_power,
        "launch_admissible": admissible,
    }
    payload["power_sha256"] = _canonical_json_sha256(payload)
    return payload


def approve_maximum_available_power(power: Mapping[str, object]) -> dict[str, object]:
    """Prospectively admit the complete eligible pool without erasing the failed power gate."""

    _validate_signed_record(power, "power_sha256", "power")
    if power.get("status") != "underpowered_full_pool":
        raise ValueError("maximum-available approval requires an underpowered full-pool plan")
    if power.get("launch_admissible") is not False:
        raise ValueError("original underpowered power plan must fail launch admission")
    eligible = power.get("eligible_pool_size")
    selected = power.get("selected_n")
    required = power.get("required_n")
    if (
        type(eligible) is not int
        or type(selected) is not int
        or type(required) is not int
        or selected != eligible
        or not 1 <= selected < required
    ):
        raise ValueError("original power plan is not a maximum-available shortfall")
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "approved-maximum-available",
        "method": (
            "prospective maximum-available amendment preserving the original signed "
            "normal-TOST power plan"
        ),
        "planned_required_n": required,
        "required_n": selected,
        "selected_n": selected,
        "eligible_pool_size": eligible,
        "achieved_power": power.get("achieved_power"),
        "target_power": power.get("target_power"),
        "equivalence_margin_f1": power.get("equivalence_margin_f1"),
        "launch_admissible": True,
        "claim_rule": (
            "passing the frozen TOST establishes equivalence; failing it is unresolved"
        ),
        "original_power": dict(power),
    }
    payload["power_sha256"] = _canonical_json_sha256(payload)
    return payload


def _validate_launch_power(power: Mapping[str, object]) -> int:
    """Authenticate either the original powered plan or its prospective full-pool amendment."""

    _validate_signed_record(power, "power_sha256", "power")
    status = power.get("status")
    if status == "passed" and power.get("launch_admissible") is True:
        required_n = power.get("required_n")
        if type(required_n) is not int or required_n < 1:
            raise ValueError("power record required N is invalid")
        return required_n
    if status == "approved-maximum-available" and power.get("launch_admissible") is True:
        original = power.get("original_power")
        if not isinstance(original, Mapping):
            raise ValueError("maximum-available power amendment lacks its original plan")
        expected = approve_maximum_available_power(original)
        if dict(power) != expected:
            raise ValueError("maximum-available power amendment semantic mismatch")
        return int(expected["required_n"])
    raise ValueError("power record is not launch-admissible")


def development_qid_projection(
    *,
    diagnostic_label: str,
    source_path: str,
    source_sha256: str,
    qids: Sequence[str],
) -> dict[str, object]:
    """Construct a QID-only diagnostic projection without an outcome input surface."""

    if not isinstance(diagnostic_label, str) or not diagnostic_label:
        raise ValueError("diagnostic_label must be a nonempty string")
    if not isinstance(source_path, str) or not source_path.startswith("/"):
        raise ValueError("development source_path must be absolute")
    _require_sha256(source_sha256, label="development source_sha256")
    ordered_qids = list(qids)
    if (
        not ordered_qids
        or any(not isinstance(qid, str) or not qid for qid in ordered_qids)
        or len(set(ordered_qids)) != len(ordered_qids)
    ):
        raise ValueError("development projection QIDs must be unique nonempty strings")
    payload: dict[str, object] = {
        "diagnostic_label": diagnostic_label,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "projection_fields": ["qid"],
        "qids": ordered_qids,
        "qid_count": len(ordered_qids),
    }
    payload["projection_sha256"] = _canonical_json_sha256(
        {"projection_fields": ["qid"], "qids": ordered_qids}
    )
    return payload


def build_development_registry(
    projections: Sequence[Mapping[str, object]], *, required_labels: Sequence[str]
) -> dict[str, object]:
    """Validate complete diagnostic projections and compute their exclusion union."""

    required = tuple(required_labels)
    if any(not isinstance(label, str) or not label for label in required) or len(
        set(required)
    ) != len(required):
        raise ValueError("required diagnostic labels must be unique nonempty strings")
    entries: dict[str, dict[str, object]] = {}
    projection_keys = {
        "diagnostic_label",
        "source_path",
        "source_sha256",
        "projection_fields",
        "qids",
        "qid_count",
        "projection_sha256",
    }
    for projection in projections:
        if set(projection) != projection_keys:
            raise ValueError("development projection schema is invalid")
        digest = projection["projection_sha256"]
        if digest != _canonical_json_sha256(
            {"projection_fields": projection["projection_fields"], "qids": projection["qids"]}
        ):
            raise ValueError("development projection canonical digest mismatch")
        label = projection["diagnostic_label"]
        if not isinstance(label, str) or label in entries:
            raise ValueError("development diagnostic labels must be unique strings")
        qids = projection["qids"]
        if (
            projection["projection_fields"] != ["qid"]
            or not isinstance(qids, list)
            or projection["qid_count"] != len(qids)
            or not qids
            or any(not isinstance(qid, str) or not qid for qid in qids)
            or len(set(qids)) != len(qids)
        ):
            raise ValueError(f"development projection {label} has invalid QIDs")
        entries[label] = dict(projection)
    missing = sorted(set(required) - set(entries))
    if missing:
        raise ValueError(f"missing required diagnostic labels: {', '.join(missing)}")
    labels = sorted(entries)
    relationships: list[dict[str, object]] = []
    for left_index, left in enumerate(labels):
        left_qids = set(entries[left]["qids"])
        for right in labels[left_index + 1 :]:
            right_qids = set(entries[right]["qids"])
            intersection = left_qids & right_qids
            if left_qids == right_qids:
                relationship = "equal"
            elif left_qids < right_qids:
                relationship = "subset"
            elif left_qids > right_qids:
                relationship = "superset"
            elif intersection:
                relationship = "overlap"
            else:
                relationship = "disjoint"
            relationships.append(
                {
                    "left": left,
                    "right": right,
                    "relationship": relationship,
                    "intersection_count": len(intersection),
                }
            )
    union_qids = sorted({qid for entry in entries.values() for qid in entry["qids"]})
    registry: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "projection_policy": "explicit QID fields only; outcome fields forbidden",
        "required_labels": list(required),
        "diagnostic_labels": labels,
        "entries": [entries[label] for label in labels],
        "relationships": relationships,
        "union_qids": union_qids,
        "union_count": len(union_qids),
        "union_sha256": hashlib.sha256("\n".join(union_qids).encode("utf-8")).hexdigest(),
    }
    registry["registry_sha256"] = _canonical_json_sha256(registry)
    return registry


def _load_development_registry_and_sha256(
    path: Path,
    *,
    required_labels: Sequence[str],
    authenticate_sources: bool = True,
) -> tuple[dict[str, object], str]:
    """Load and hash the exact canonical registry bytes through one descriptor."""

    registry_path = Path(path)
    registry_bytes = _read_regular_file(registry_path, label="development registry")
    try:
        value = json.loads(registry_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("development registry is not JSON") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("development registry schema is invalid")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise ValueError("development registry entries are invalid")
    rebuilt = build_development_registry(entries, required_labels=required_labels)
    if rebuilt != value:
        raise ValueError("development registry canonical content mismatch")
    if set(value["diagnostic_labels"]) != set(required_labels) or len(
        value["diagnostic_labels"]
    ) != len(required_labels):
        raise ValueError("development registry label set is not exact")
    if authenticate_sources:
        authenticated: dict[str, str] = {}
        for entry in entries:
            source_path = Path(entry["source_path"])
            source_key = str(source_path)
            actual = authenticated.get(source_key)
            if actual is None:
                actual = _sha256_regular_file(source_path, label="development QID source")
                authenticated[source_key] = actual
            if actual != entry["source_sha256"]:
                raise ValueError(f"development QID source SHA-256 mismatch: {source_path}")
    return value, hashlib.sha256(registry_bytes).hexdigest()


def load_development_registry(
    path: Path,
    *,
    required_labels: Sequence[str],
    authenticate_sources: bool = True,
) -> dict[str, object]:
    """Load a canonical QID-only registry and optionally hash its immutable sources."""

    value, _ = _load_development_registry_and_sha256(
        path,
        required_labels=required_labels,
        authenticate_sources=authenticate_sources,
    )
    return value


def seal_holdout(
    eligibility_records: Sequence[Mapping[str, object]],
    *,
    development_registry_path: Path,
    required_registry_labels: Sequence[str],
    power: Mapping[str, object],
    calibration: Mapping[str, object],
    runtime_pins: Mapping[str, str],
    input_file_hashes: Mapping[str, str],
    destination: Path,
) -> dict[str, object]:
    """Atomically publish a fully admitted method-holdout manifest."""

    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"holdout destination already exists: {destination}")
    required_n = _validate_launch_power(power)
    _validate_signed_record(calibration, "calibration_sha256", "calibration")
    registry_path = Path(development_registry_path)
    if not registry_path.is_absolute():
        raise ValueError("development registry path must be absolute")
    registry_path = Path(os.path.abspath(registry_path))
    required_labels = tuple(required_registry_labels)
    development_registry, registry_file_sha256 = _load_development_registry_and_sha256(
        registry_path,
        required_labels=required_labels,
        authenticate_sources=True,
    )
    if calibration.get("status") != "passed" or calibration.get("frozen_repetitions") not in {
        10,
        20,
    }:
        raise ValueError("random calibration record is invalid")
    development_qids = development_registry.get("union_qids")
    if not isinstance(development_qids, list):
        raise ValueError("development registry union is invalid")
    eligible = build_eligibility_records(eligibility_records, development_qids=development_qids)
    _authenticate_eligibility_records(eligible)
    if len(eligible) < required_n:
        raise ValueError("eligible cohort does not satisfy the sealed required N")
    order = holdout_order([str(record["qid"]) for record in eligible])
    record_by_qid = {str(record["qid"]): record for record in eligible}
    ordered_qids = [row["qid"] for row in order]
    selected_qids = ordered_qids[:required_n]
    ordered_records = [record_by_qid[qid] for qid in ordered_qids]
    selected_records = [record_by_qid[qid] for qid in selected_qids]
    component_input = [
        {
            "qid": record["qid"],
            "supporting_document_ids": record["supporting_document_ids"],
        }
        for record in selected_records
    ]
    components = support_document_components(component_input)
    if len(components["components"]) < 2:
        raise ValueError("sealed cohort has only one support component; inference is unidentified")
    if not runtime_pins or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in runtime_pins.items()
    ):
        raise ValueError("runtime/version pins must be nonempty strings")
    runtime_commit = runtime_pins.get("runtime_commit")
    if (
        not isinstance(runtime_commit, str)
        or len(runtime_commit) != 40
        or any(character not in "0123456789abcdef" for character in runtime_commit)
    ):
        raise ValueError("runtime_commit pin must be an exact lowercase 40-hex revision")
    checked_input_hashes: dict[str, str] = {}
    for path, digest in input_file_hashes.items():
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError("input file hash paths must be absolute")
        checked_input_hashes[path] = _require_sha256(digest, label=f"input hash for {path}")
    if not checked_input_hashes:
        raise ValueError("input file hashes must not be empty")
    for path, digest in checked_input_hashes.items():
        _sha256_regular_file(Path(path), label="holdout input file", expected=digest)
    eligible_qid_bytes = ("\n".join(ordered_qids) + "\n").encode("utf-8")
    selected_qid_bytes = ("\n".join(selected_qids) + "\n").encode("utf-8")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "sealed",
        "selection_method": (
            "sort sha256(b'docprune-random-coverage-v2' + qid.encode('utf-8')), then QID; "
            "take required-N prefix after development-union exclusion"
        ),
        "development_registry": dict(development_registry),
        "development_registry_path": str(registry_path),
        "development_registry_file_sha256": registry_file_sha256,
        "required_registry_labels": list(required_labels),
        "power": dict(power),
        "calibration": dict(calibration),
        "required_n": required_n,
        "eligible_count": len(ordered_qids),
        "eligible_order": list(order),
        "eligible_order_sha256": hashlib.sha256(eligible_qid_bytes.rstrip(b"\n")).hexdigest(),
        "selected_qids": selected_qids,
        "selection_sha256": hashlib.sha256(selected_qid_bytes.rstrip(b"\n")).hexdigest(),
        "eligible_records": ordered_records,
        "eligibility_projection_sha256": _canonical_json_sha256(ordered_records),
        "selected_records": selected_records,
        "support_components": components,
        "runtime_pins": dict(sorted(runtime_pins.items())),
        "input_file_hashes": dict(sorted(checked_input_hashes.items())),
        "member_files": {
            "eligible-order.qids": hashlib.sha256(eligible_qid_bytes).hexdigest(),
            "selected.qids": hashlib.sha256(selected_qid_bytes).hexdigest(),
        },
    }
    manifest["manifest_sha256"] = _canonical_json_sha256(manifest)
    parent_fd = _open_directory_nofollow(destination.parent)
    temporary_name = f".{destination.name}-{secrets.token_hex(16)}"
    temporary_fd: int | None = None
    temporary_created = False
    member_names: list[str] = []
    try:
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"holdout destination already exists: {destination}")
        os.mkdir(temporary_name, mode=0o700, dir_fd=parent_fd)
        temporary_created = True
        temporary_fd = os.open(
            temporary_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        members = {
            "eligible-order.qids": eligible_qid_bytes,
            "selected.qids": selected_qid_bytes,
            "manifest.json": (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(
                "utf-8"
            ),
        }
        for name, content in members.items():
            member_fd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=temporary_fd,
            )
            member_names.append(name)
            try:
                remaining = memoryview(content)
                while remaining:
                    written = os.write(member_fd, remaining)
                    remaining = remaining[written:]
                os.fsync(member_fd)
            finally:
                os.close(member_fd)
        os.fsync(temporary_fd)
        try:
            _renameat2_noreplace(parent_fd, temporary_name, parent_fd, destination.name)
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
                raise
            _link_directory_manifest_last_noreplace(
                parent_fd=parent_fd,
                temporary_fd=temporary_fd,
                temporary_name=temporary_name,
                destination_name=destination.name,
                member_names=member_names,
            )
        temporary_created = False
        os.fsync(parent_fd)
    except BaseException:
        if temporary_created:
            cleanup_fd = temporary_fd
            close_cleanup_fd = False
            if cleanup_fd is None:
                cleanup_fd = os.open(
                    temporary_name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
                close_cleanup_fd = True
            for name in member_names:
                try:
                    os.unlink(name, dir_fd=cleanup_fd)
                except FileNotFoundError:
                    pass
            if close_cleanup_fd:
                os.close(cleanup_fd)
            try:
                os.rmdir(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        os.close(parent_fd)
    return manifest


def _validate_signed_record(value: Mapping[str, object], digest_field: str, label: str) -> None:
    unsigned = dict(value)
    digest = unsigned.pop(digest_field, None)
    if digest != _canonical_json_sha256(unsigned):
        raise ValueError(f"{label} canonical digest mismatch")


def validate_holdout_for_launch(
    path: Path,
    *,
    required_registry_path: Path,
    required_registry_labels: Sequence[str],
) -> dict[str, object]:
    """Fail closed unless an immutable sealed holdout passes every admission record."""

    requested_root = Path(path)
    if requested_root.is_symlink() or not requested_root.is_dir():
        raise ValueError(f"sealed holdout directory is missing: {requested_root}")
    root = requested_root.resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("sealed holdout manifest is missing")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("sealed holdout manifest is not JSON") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("sealed holdout manifest schema is invalid")
    if set(value) != _HOLDOUT_MANIFEST_KEYS:
        raise ValueError("sealed holdout manifest required fields are invalid")
    _validate_signed_record(value, "manifest_sha256", "holdout manifest")
    if value.get("status") != "sealed":
        raise ValueError("holdout manifest is not sealed")
    power = value.get("power")
    calibration = value.get("calibration")
    registry = value.get("development_registry")
    if not isinstance(power, Mapping) or not isinstance(calibration, Mapping):
        raise ValueError("holdout power/calibration records are missing")
    if not isinstance(registry, Mapping):
        raise ValueError("holdout development registry is missing")
    required_n = _validate_launch_power(power)
    _validate_signed_record(calibration, "calibration_sha256", "calibration")
    if calibration.get("status") != "passed":
        raise ValueError("holdout calibration is not admitted")
    registry_path = Path(required_registry_path)
    if not registry_path.is_absolute():
        raise ValueError("required development registry path must be absolute")
    registry_path = Path(os.path.abspath(registry_path))
    labels = tuple(required_registry_labels)
    if value["development_registry_path"] != str(registry_path):
        raise ValueError("holdout development registry path identity mismatch")
    if value["required_registry_labels"] != list(labels):
        raise ValueError("holdout required registry label identity mismatch")
    actual_registry, registry_file_sha256 = _load_development_registry_and_sha256(
        registry_path,
        required_labels=labels,
        authenticate_sources=True,
    )
    if value["development_registry_file_sha256"] != registry_file_sha256:
        raise ValueError("holdout development registry file SHA-256 mismatch")
    if registry != actual_registry:
        raise ValueError("holdout embedded development registry identity mismatch")
    runtime_pins = value["runtime_pins"]
    input_file_hashes = value["input_file_hashes"]
    if not isinstance(runtime_pins, Mapping) or not runtime_pins:
        raise ValueError("holdout runtime/version pins are missing")
    runtime_commit = runtime_pins.get("runtime_commit")
    if not isinstance(runtime_commit, str) or re.fullmatch(r"[0-9a-f]{40}", runtime_commit) is None:
        raise ValueError("holdout runtime commit pin is invalid")
    if not isinstance(input_file_hashes, Mapping) or not input_file_hashes:
        raise ValueError("holdout input file hashes are missing")
    for source_path, digest in input_file_hashes.items():
        if not isinstance(source_path, str) or not source_path.startswith("/"):
            raise ValueError("holdout input file hash path is invalid")
        checked_digest = _require_sha256(digest, label=f"holdout input hash for {source_path}")
        _sha256_regular_file(Path(source_path), label="holdout input file", expected=checked_digest)
    member_files = value.get("member_files")
    if not isinstance(member_files, Mapping) or set(member_files) != {
        "eligible-order.qids",
        "selected.qids",
    }:
        raise ValueError("holdout member-file inventory is invalid")
    member_bytes: dict[str, bytes] = {}
    for name, expected_digest in member_files.items():
        member_path = root / name
        if member_path.is_symlink() or not member_path.is_file():
            raise ValueError(f"holdout member is missing: {name}")
        content = member_path.read_bytes()
        member_bytes[name] = content
        if hashlib.sha256(content).hexdigest() != expected_digest:
            raise ValueError(f"holdout member digest mismatch: {name}")
    selected_qids = value.get("selected_qids")
    eligible_order = value.get("eligible_order")
    eligible_records = value.get("eligible_records")
    selected_records = value.get("selected_records")
    if (
        not isinstance(selected_qids, list)
        or not isinstance(eligible_order, list)
        or not isinstance(eligible_records, list)
        or not isinstance(selected_records, list)
        or value.get("required_n") != len(selected_qids)
        or required_n != len(selected_qids)
    ):
        raise ValueError("holdout selected prefix does not equal required N")
    record_qids = [record.get("qid") for record in eligible_records if isinstance(record, Mapping)]
    if len(record_qids) != len(eligible_records) or value.get("eligible_count") != len(record_qids):
        raise ValueError("holdout eligible record inventory is invalid")
    if eligible_order != list(holdout_order(record_qids)):
        raise ValueError("holdout eligible order is invalid")
    ordered_qids = [row["qid"] for row in eligible_order]
    rebuilt_records = build_eligibility_records(
        _eligibility_inputs_from_records(eligible_records),
        development_qids=actual_registry["union_qids"],
    )
    if list(rebuilt_records) != eligible_records:
        raise ValueError("holdout eligibility semantic revalidation mismatch")
    _authenticate_eligibility_records(rebuilt_records)
    if selected_qids != ordered_qids[: len(selected_qids)]:
        raise ValueError("holdout selection is not the required ordered prefix")
    if selected_records != eligible_records[: len(selected_qids)]:
        raise ValueError("holdout selected records do not match the ordered prefix")
    expected_components = support_document_components(
        [
            {
                "qid": record["qid"],
                "supporting_document_ids": record["supporting_document_ids"],
            }
            for record in selected_records
        ]
    )
    if value["support_components"] != expected_components:
        raise ValueError("holdout support components do not match selected records")
    if value["eligibility_projection_sha256"] != _canonical_json_sha256(eligible_records):
        raise ValueError("holdout eligibility projection digest mismatch")
    if (
        value["eligible_order_sha256"]
        != hashlib.sha256("\n".join(ordered_qids).encode("utf-8")).hexdigest()
    ):
        raise ValueError("holdout eligible-order digest mismatch")
    if (
        value["selection_sha256"]
        != hashlib.sha256("\n".join(selected_qids).encode("utf-8")).hexdigest()
    ):
        raise ValueError("holdout selection digest mismatch")
    expected_eligible_bytes = ("\n".join(ordered_qids) + "\n").encode("utf-8")
    expected_selected_bytes = ("\n".join(selected_qids) + "\n").encode("utf-8")
    if member_bytes["eligible-order.qids"] != expected_eligible_bytes:
        raise ValueError("holdout eligible QID member does not match manifest")
    if member_bytes["selected.qids"] != expected_selected_bytes:
        raise ValueError("holdout selected QID member does not match manifest")
    return value


def build_intervention_artifact(
    *,
    qid: str,
    cohort_classification: str,
    page_hashes: Sequence[str],
    feature_hashes: Sequence[str],
    policy_family: str,
    policy_name: str,
    mode: str,
    selection_kind: str,
    boundary: str,
    native_layer: int | None,
    visual_population: int,
    requested_budget: int,
    achieved_budget: int,
    retained_original_indices: Sequence[int],
    seed: int,
    per_layer_cache_lengths: Mapping[str, int],
    runtime_pins: Mapping[str, str],
) -> dict[str, object]:
    """Build the canonical replay contract for one Task 3 intervention."""

    if not isinstance(qid, str) or not qid:
        raise ValueError("intervention qid must be a nonempty string")
    if cohort_classification not in {"development", "method-holdout", "synthetic"}:
        raise ValueError("intervention cohort classification is invalid")
    if len(page_hashes) != 4 or len(feature_hashes) != 4:
        raise ValueError("intervention must bind four page and feature hashes")
    checked_pages = [
        _require_sha256(value, label="intervention page hash") for value in page_hashes
    ]
    checked_features = [
        _require_sha256(value, label="intervention feature hash") for value in feature_hashes
    ]
    allowed_families = {
        "native-threshold",
        "score-top-m",
        "random-top-m",
        "visual-state-removal",
        "regional-attribution",
        "standalone-diagnostic",
    }
    if policy_family not in allowed_families or not isinstance(policy_name, str):
        raise ValueError("intervention policy family/name is invalid")
    if policy_family == "native-threshold" and not policy_name.endswith("-native-threshold"):
        raise ValueError("native-threshold policy name must preserve native-threshold naming")
    if policy_family == "score-top-m" and not policy_name.endswith("-score-top-m"):
        raise ValueError("score-top-m policy name must preserve score-top-M naming")
    if mode not in {"physical_delete", "zero_mask"}:
        raise ValueError("intervention mode is invalid")
    if selection_kind not in {"native_threshold", "forced"}:
        raise ValueError("intervention selection kind is invalid")
    if policy_family == "native-threshold" and (
        selection_kind != "native_threshold" or mode != "physical_delete"
    ):
        raise ValueError("native-threshold artifact must retain native-threshold physical identity")
    if selection_kind == "native_threshold" and policy_family != "native-threshold":
        raise ValueError("native-threshold selection kind requires native-threshold policy family")
    if boundary != "B_input" and re.fullmatch(r"B_[0-9]+", boundary) is None:
        raise ValueError("intervention boundary must be B_input or B_K")
    if native_layer is not None and (type(native_layer) is not int or native_layer < 0):
        raise ValueError("native_layer must be a nonnegative integer or null")
    if policy_family == "native-threshold":
        if native_layer is None:
            raise ValueError("native-threshold artifact requires a native_layer")
        if boundary == "B_input" or boundary != f"B_{native_layer}":
            raise ValueError("native-threshold artifact has an invalid native boundary")
    if any(
        type(value) is not int or value < 0
        for value in (visual_population, requested_budget, achieved_budget)
    ):
        raise ValueError("intervention populations and budgets must be nonnegative integers")
    if achieved_budget > requested_budget or requested_budget > visual_population:
        raise ValueError("intervention budgets exceed their population or request")
    indices = list(retained_original_indices)
    if (
        len(indices) != achieved_budget
        or any(type(index) is not int or not 0 <= index < visual_population for index in indices)
        or indices != sorted(set(indices))
    ):
        raise ValueError("retained original indices do not match achieved budget")
    if type(seed) is not int:
        raise ValueError("intervention seed must be an integer")
    if not per_layer_cache_lengths or any(
        not isinstance(layer, str)
        or re.fullmatch(r"[0-9]+", layer) is None
        or type(length) is not int
        or length < 0
        for layer, length in per_layer_cache_lengths.items()
    ):
        raise ValueError("per-layer cache lengths are invalid")
    if not runtime_pins or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in runtime_pins.items()
    ):
        raise ValueError("intervention runtime pins are invalid")
    runtime_commit = runtime_pins.get("runtime_commit")
    if not isinstance(runtime_commit, str) or re.fullmatch(r"[0-9a-f]{40}", runtime_commit) is None:
        raise ValueError("intervention runtime_commit must be exact lowercase 40-hex")
    payload: dict[str, object] = {
        "schema_version": 1,
        "qid": qid,
        "cohort_classification": cohort_classification,
        "page_hashes": checked_pages,
        "feature_hashes": checked_features,
        "policy": {"family": policy_family, "name": policy_name},
        "mode": mode,
        "selection_kind": selection_kind,
        "boundary": boundary,
        "native_layer": native_layer,
        "visual_population": visual_population,
        "requested_budget": requested_budget,
        "achieved_budget": achieved_budget,
        "retained_original_indices": indices,
        "seed": seed,
        "per_layer_cache_lengths": dict(sorted(per_layer_cache_lengths.items())),
        "runtime_pins": dict(sorted(runtime_pins.items())),
    }
    payload["artifact_sha256"] = _canonical_json_sha256(payload)
    return payload
