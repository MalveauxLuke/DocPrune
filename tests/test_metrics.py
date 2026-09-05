import json

import pytest

from docprune.m3docrag import SampleTiming
from docprune.metrics import (
    MEASUREMENT_DEFINITION,
    StageMetrics,
    append_result_jsonl,
    hardware_result_classification,
    measurement_identity,
    summarize_jsonl,
)
from docprune.qwen2vl.model import PruningTrace


def test_metrics_report_literal_drop_rates_and_throughput() -> None:
    metrics = StageMetrics()
    metrics.update(
        PruningTrace(100, 80, 50, 25, 7),
        SampleTiming(retrieval_seconds=1.0, qa_seconds=3.0),
    )
    metrics.update(
        PruningTrace(100, 60, 40, 20, 5),
        SampleTiming(retrieval_seconds=1.0, qa_seconds=5.0),
    )

    got = metrics.to_dict()

    assert got["samples"] == 2
    assert got["visual_tokens"] == {
        "original": 200,
        "post_btp": 140,
        "post_qtp": 90,
        "post_ctp": 45,
    }
    assert got["drop_rates"] == pytest.approx({"btp": 0.3, "qtp": 0.55, "ctp": 0.775})
    assert got["timing_seconds"] == {
        "retrieval": 2.0,
        "page_load": 0.0,
        "qa": 8.0,
        "encoder": 0.0,
        "decoder": 0.0,
        "total": 10.0,
    }
    assert got["original_visual_tokens_per_second"] == pytest.approx(20.0)


def test_sample_timing_omits_unmeasured_total_but_keeps_fixture_fallback() -> None:
    timing = SampleTiming(retrieval_seconds=1.0, qa_seconds=2.0)

    assert timing.total_seconds == pytest.approx(3.0)
    assert "total_sample_seconds" not in timing.to_dict()


def test_sample_timing_serializes_explicit_positive_total_exactly() -> None:
    timing = SampleTiming(retrieval_seconds=1.0, qa_seconds=2.0, total_sample_seconds=7.25)

    assert timing.to_dict()["total_sample_seconds"] == 7.25


def test_paper_drop_rates_are_arithmetic_mean_of_per_sample_ratios() -> None:
    metrics = StageMetrics()
    metrics.update(
        PruningTrace(100, 80, 50, 25, 7),
        SampleTiming(1.0, 3.0, encoder_seconds=0.5, decoder_seconds=1.0),
    )
    metrics.update(
        PruningTrace(200, 100, 100, 100, 5),
        SampleTiming(1.0, 3.0, encoder_seconds=0.5, decoder_seconds=1.0),
    )

    got = metrics.to_dict()

    assert got["drop_rates"] == pytest.approx({"btp": 0.35, "qtp": 0.5, "ctp": 0.625})
    assert got["token_weighted_drop_rates"] == pytest.approx(
        {"btp": 1 - 180 / 300, "qtp": 1 - 150 / 300, "ctp": 1 - 125 / 300}
    )
    assert got["timing_seconds"]["encoder"] == pytest.approx(1.0)
    assert got["timing_seconds"]["decoder"] == pytest.approx(2.0)


def test_measurement_identity_classifies_reconstruction_hardware_exactly() -> None:
    assert hardware_result_classification("NVIDIA A100-SXM4-80GB") == {
        "actual_hardware": "NVIDIA A100-SXM4-80GB",
        "paper_hardware": "NVIDIA RTX A6000",
        "classification": "reconstruction_measurement",
        "direct_hardware_parity": False,
    }
    assert hardware_result_classification("NVIDIA RTX A6000") == {
        "actual_hardware": "NVIDIA RTX A6000",
        "paper_hardware": "NVIDIA RTX A6000",
        "classification": "paper_hardware_parity",
        "direct_hardware_parity": True,
    }
    assert measurement_identity(sample_ids=("q1",))["result_classification"][
        "classification"
    ] == "reconstruction_measurement"


def test_jsonl_writer_refuses_overwrite_and_summary_is_reproducible(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    record = {
        "trace": {
            "original_visual_tokens": 10,
            "post_btp_visual_tokens": 8,
            "post_qtp_visual_tokens": 6,
            "post_ctp_visual_tokens": 4,
            "ctp_layer": 3,
        },
        "timing": {"retrieval_seconds": 0.25, "qa_seconds": 0.75},
    }
    append_result_jsonl(path, record)

    with pytest.raises(FileExistsError, match="resume"):
        append_result_jsonl(path, record)

    append_result_jsonl(path, record, resume=True)
    got = summarize_jsonl(path)

    assert got["samples"] == 2
    assert got["visual_tokens"]["post_ctp"] == 8
    assert len(path.read_text().splitlines()) == 2
    assert json.loads(path.read_text().splitlines()[0]) == record


def test_production_summary_rejects_missing_or_zero_stage_timings(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    record = {
        "trace": {
            "original_visual_tokens": 10,
            "post_btp_visual_tokens": 8,
            "post_qtp_visual_tokens": 6,
            "post_ctp_visual_tokens": 4,
            "ctp_layer": 3,
        },
        "timing": {
            "retrieval_seconds": 0.0,
            "qa_seconds": 0.0,
            "page_load_seconds": 0.0,
            "encoder_seconds": 0.0,
            "decoder_seconds": 0.0,
            "total_sample_seconds": 0.0,
        },
    }
    append_result_jsonl(path, record)

    with pytest.raises(ValueError, match="positive"):
        summarize_jsonl(path, require_positive=True)


def test_strict_summary_rejects_derived_total_and_missing_classification(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    record = {
        "trace": {
            "original_visual_tokens": 10,
            "post_btp_visual_tokens": 8,
            "post_qtp_visual_tokens": 6,
            "post_ctp_visual_tokens": 4,
            "ctp_layer": 3,
        },
        "timing": {
            "retrieval_seconds": 1.0,
            "qa_seconds": 1.0,
            "page_load_seconds": 1.0,
            "encoder_seconds": 1.0,
            "decoder_seconds": 1.0,
        },
    }
    append_result_jsonl(path, record)

    with pytest.raises(ValueError, match="total_sample_seconds"):
        summarize_jsonl(path, require_positive=True)

    record["timing"]["total_sample_seconds"] = 1.0
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="result classification"):
        summarize_jsonl(path, require_positive=True)


def test_strict_aggregate_carries_exact_result_classification() -> None:
    classification = hardware_result_classification("NVIDIA A100-SXM4-80GB")
    metrics = StageMetrics(result_classification=classification)
    metrics.update(
        PruningTrace(10, 8, 6, 4, None),
        SampleTiming(1.0, 1.0, page_load_seconds=1.0, encoder_seconds=1.0, decoder_seconds=1.0,
                     total_sample_seconds=1.0),
        require_positive=True,
    )

    assert metrics.to_dict(require_classification=True)["measurement"]["result_classification"] == classification


def test_measurement_aggregate_requires_warmup_flag_and_peak_gpu_bytes() -> None:
    metrics = StageMetrics()
    metrics.update(
        PruningTrace(10, 8, 6, 4, None),
        SampleTiming(
            1.0,
            2.0,
            peak_allocated_gpu_bytes=4096,
            warmup_excluded=True,
        ),
    )
    metrics.update(
        PruningTrace(10, 7, 5, 3, None),
        SampleTiming(
            1.0,
            2.0,
            peak_allocated_gpu_bytes=8192,
            warmup_excluded=True,
        ),
    )

    assert metrics.to_dict()["measurement"] == {
        "definition": MEASUREMENT_DEFINITION,
        "peak_allocated_gpu_bytes": 8192,
        "warmup_excluded": True,
        "profiler_enabled": False,
    }


def test_profiler_fields_are_serialized_only_when_enabled() -> None:
    metrics = StageMetrics()
    metrics.update(
        PruningTrace(10, 8, 6, 4, None),
        SampleTiming(
            1.0,
            2.0,
            peak_allocated_gpu_bytes=4096,
            warmup_excluded=True,
            profiler_enabled=True,
            profiler_definition="torch.profiler total FLOPs",
            flops=10.5,
        ),
    )

    assert metrics.to_dict()["measurement"] == {
        "definition": MEASUREMENT_DEFINITION,
        "peak_allocated_gpu_bytes": 4096,
        "warmup_excluded": True,
        "profiler_enabled": True,
        "profiler_definition": "torch.profiler total FLOPs",
        "flops": 10.5,
    }


@pytest.mark.parametrize("field", ["retrieval_seconds", "qa_seconds"])
def test_stage_metrics_rejects_nonfinite_timings(field: str) -> None:
    values = {"retrieval_seconds": 1.0, "qa_seconds": 2.0}
    values[field] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        StageMetrics().update(
            PruningTrace(10, 8, 6, 4, None),
            SampleTiming(**values),
        )
