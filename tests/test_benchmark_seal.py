"""Executable checks for the immutable benchmark launch/evidence seal."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from docprune.benchmark_seal import (
    align_retrieved_page_context,
    validate_attempt_root,
    validate_gate_evidence_payload,
    validate_mini_index_manifest,
    validate_runtime_identity,
)

QIDS = ("q1", "q2", "q3", "q4", "q5")
COMMIT = "a" * 40


def _valid_semantic() -> dict[str, object]:
    orders = {
        qid: {
            str(top_k): [
                {"doc_id": f"doc-{index}", "page_index": index}
                for index in range(top_k)
            ]
            for top_k in (1, 2, 4)
        }
        for qid in QIDS
    }
    timings = {
        qid: {
            field: 0.01
            for field in (
                "retrieval_seconds",
                "page_load_seconds",
                "qa_seconds",
                "total_sample_seconds",
                "encoder_seconds",
                "decoder_seconds",
            )
        }
        for qid in QIDS
    }
    return {
        "status": "passed",
        "baseline_equivalence": True,
        "equivalence_qids": list(QIDS),
        "supporting_documents": {qid: f"doc-{index}" for index, qid in enumerate(QIDS)},
        "complete_colpali_equivalence": True,
        "schema5_mini_index": {"status": "passed", "schema_version": 5, "fixture": True},
        "flash_attention_2": {
            "cuda": "12.4",
            "torch": "2.6.0",
            "transformers": "4.49.0",
            "attn_implementation": "flash_attention_2",
        },
        "ctp_policy": {
            "head_aggregation": "arithmetic_mean",
            "score_scale": "current_visual_token_count",
            "raw_attention_semantics": "mean_head_attention_scores_before_visual_count_scaling",
            "transformed_attention_semantics": "raw_mean_times_current_visual_count",
            "prefill_query_token": "last_prompt_token",
        },
        "upstream_retrieval_orders": orders,
        "runtime_retrieval_orders": orders,
        "retrieval_orders": orders,
        "colpali_counters": {
            "before_retrieval": {"processor_queries": 0, "processor_images": 20, "model_forwards": 20},
            "after_retrieval": {"processor_queries": 5, "processor_images": 20, "model_forwards": 25},
            "after_qa": {"processor_queries": 5, "processor_images": 20, "model_forwards": 25},
        },
        "timings": timings,
        "docprune_traces": {str(page): [100, 80, 60, 40] for page in (1, 2, 4)},
    }


def test_runtime_identity_accepts_exact_clean_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "runtime"
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
    (checkout / "runtime.txt").write_text("sealed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "runtime.txt"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "--quiet", "-m", "seal"], check=True)
    expected = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert validate_runtime_identity(expected, checkout) == expected


def test_runtime_identity_rejects_wrong_sha_and_dirty_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "runtime"
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
    (checkout / "runtime.txt").write_text("sealed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "runtime.txt"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "--quiet", "-m", "seal"], check=True)

    with pytest.raises(ValueError, match="40-hex|revision mismatch"):
        validate_runtime_identity("b" * 40, checkout)
    (checkout / "runtime.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        validate_runtime_identity(
            subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            checkout,
        )


def test_attempt_root_rejects_historical_or_arbitrary_root() -> None:
    expected = "/scratch/lmalveau/docprune/benchmark-aaaaaaa/attempt-1"
    assert validate_attempt_root(expected, expected) == expected
    for candidate in (
        "/scratch/lmalveau/docprune/benchmark-runtime7/attempt-2",
        "/scratch/lmalveau/docprune/benchmark-old/attempt-1",
        "/tmp/arbitrary",
    ):
        with pytest.raises(ValueError, match="attempt root"):
            validate_attempt_root(candidate, expected)


def test_schema_four_mini_index_is_rejected_and_sealed_fixture_is_accepted() -> None:
    with pytest.raises(ValueError, match="schema_version.*5"):
        validate_mini_index_manifest({"schema_version": 4, "fixture": True})
    assert validate_mini_index_manifest({"schema_version": 5, "fixture": True}) is None
    with pytest.raises(ValueError, match="fixture"):
        validate_mini_index_manifest({"schema_version": 5, "fixture": False})


def test_retrieved_images_and_features_follow_ordered_page_identity() -> None:
    pages = (("doc-b", 3), ("doc-a", 1))
    features = ({"doc_id": "doc-b", "page_index": 3}, {"doc_id": "doc-a", "page_index": 1})
    loaded = align_retrieved_page_context(
        pages,
        features,
        lambda doc_id, page_index: f"image:{doc_id}:{page_index}",
    )
    assert loaded == ("image:doc-b:3", "image:doc-a:1")
    with pytest.raises(ValueError, match="alignment"):
        align_retrieved_page_context(
            pages,
            (features[1], features[0]),
            lambda doc_id, page_index: f"image:{doc_id}:{page_index}",
        )


def test_gate_evidence_rejects_missing_or_fabricated_no_reencoding_evidence() -> None:
    semantic = _valid_semantic()
    with pytest.raises(ValueError, match="missing"):
        validate_gate_evidence_payload({"status": "passed"}, expected_qids=QIDS)
    semantic["colpali_counters"] = {
        "before_retrieval": {"processor_queries": 0, "processor_images": 20, "model_forwards": 20},
        "after_retrieval": {"processor_queries": 0, "processor_images": 20, "model_forwards": 20},
        "after_qa": {"processor_queries": 0, "processor_images": 20, "model_forwards": 20},
    }
    with pytest.raises(ValueError, match="query"):
        validate_gate_evidence_payload(semantic, expected_qids=QIDS)


def test_gate_evidence_rejects_one_of_five_retrieval_mismatches() -> None:
    semantic = _valid_semantic()
    semantic["retrieval_orders"]["q3"]["2"] = [{"doc_id": "wrong", "page_index": 0}]
    with pytest.raises(ValueError, match="retrieval"):
        validate_gate_evidence_payload(semantic, expected_qids=QIDS)


def test_gate_evidence_accepts_valid_sealed_fixture() -> None:
    assert validate_gate_evidence_payload(_valid_semantic(), expected_qids=QIDS) is None


def test_gate_evidence_rejects_missing_ctp_policy() -> None:
    semantic = _valid_semantic()
    semantic.pop("ctp_policy")
    with pytest.raises(ValueError, match="CTP policy"):
        validate_gate_evidence_payload(semantic, expected_qids=QIDS)


def test_gate_evidence_accepts_per_qid_trace_records() -> None:
    semantic = _valid_semantic()
    semantic["docprune_traces"] = {
        qid: {str(page): [100, 80, 60, 40] for page in (1, 2, 4)} for qid in QIDS
    }
    assert validate_gate_evidence_payload(semantic, expected_qids=QIDS) is None
