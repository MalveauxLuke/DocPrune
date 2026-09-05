"""Immutable JSONL results and aggregate DocPrune efficiency metrics."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docprune.m3docrag import SampleTiming
from docprune.qwen2vl.model import PruningTrace

MEASUREMENT_DEFINITION = (
    "peak_allocated_gpu_bytes is torch.cuda.max_memory_allocated after a synchronized "
    "counter reset at the start of complete QA and synchronization after generation; "
    "encoder is synchronized Qwen vision execution and decoder is synchronized LM prefill "
    "plus greedy decode"
)


def hardware_result_classification(actual_hardware: str) -> dict[str, object]:
    """Classify measurements against the paper's RTX A6000 reference hardware."""

    if not isinstance(actual_hardware, str) or not actual_hardware.strip():
        raise ValueError("actual_hardware must be a non-empty string")
    direct_parity = "RTX A6000" in actual_hardware.upper()
    return {
        "actual_hardware": actual_hardware,
        "paper_hardware": "NVIDIA RTX A6000",
        "classification": "paper_hardware_parity"
        if direct_parity
        else "reconstruction_measurement",
        "direct_hardware_parity": direct_parity,
    }


def _validate_result_classification(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("result classification must be an object")
    actual = value.get("actual_hardware")
    if not isinstance(actual, str) or not actual.strip():
        raise ValueError("result classification hardware is invalid")
    expected = hardware_result_classification(actual)
    if value != expected:
        raise ValueError("result classification is not canonical")
    return expected


def measurement_identity(
    *, sample_ids: tuple[str, ...] = (), warmup_count: int = 1
) -> dict[str, object]:
    """Return the canonical identity used for paired efficiency measurements."""

    import torch

    cuda = bool(torch.cuda.is_available())
    if cuda:
        device = torch.cuda.current_device()
        gpu_model = torch.cuda.get_device_name(device)
        capability = list(torch.cuda.get_device_capability(device))
    else:
        gpu_model = "cpu"
        capability = None
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = "unavailable"
    identity_ids = tuple(str(value) for value in sample_ids)
    return {
        "definition": MEASUREMENT_DEFINITION,
        "warmup_required": True,
        "profiler_enabled": False,
        "hardware": {
            "gpu_model": gpu_model,
            "compute_capability": capability,
        },
        "result_classification": hardware_result_classification(gpu_model),
        "software": {
            "python": platform.python_version(),
            "cuda": torch.version.cuda or "unavailable",
            "pytorch": torch.__version__,
            "transformers": transformers_version,
        },
        "precision": {
            "weights": "bfloat16",
            "vision": "bfloat16",
            "attention_accumulation": "float32",
        },
        "attention_backend": "flash_attention_2",
        "allocator": {
            "peak_memory": "torch.cuda.max_memory_allocated",
            "reset": "torch.cuda.reset_peak_memory_stats",
        },
        "timer_boundaries": {
            "synchronization": "CUDA stage events resolve with one synchronization after generation; sparse stages synchronize at each boundary",
            "encoder": "Qwen visual encoder execution only",
            "decoder": "language-model prefill, first logits, and greedy decode",
            "qa": "complete page preparation, BTP/QTP, encoder, decoder, and answer decode",
            "sample": "retrieval, page loading, and complete QA wall time",
        },
        "warmup": {
            "count": warmup_count,
            "sample_id": identity_ids[0] if identity_ids else None,
            "sample_identity_sha256": hashlib.sha256(
                "\n".join(identity_ids).encode("utf-8")
            ).hexdigest(),
        },
    }


@dataclass
class StageMetrics:
    samples: int = 0
    original: int = 0
    post_btp: int = 0
    post_qtp: int = 0
    post_ctp: int = 0
    retrieval_seconds: float = 0.0
    page_load_seconds: float = 0.0
    qa_seconds: float = 0.0
    encoder_seconds: float = 0.0
    decoder_seconds: float = 0.0
    total_sample_seconds: float = 0.0
    peak_allocated_gpu_bytes: int = 0
    warmup_excluded: bool = False
    profiler_enabled: bool = False
    profiler_definition: str | None = None
    flops: float | None = None
    result_classification: dict[str, object] | None = None
    _drop_rates: dict[str, list[float]] = field(
        default_factory=lambda: {"btp": [], "qtp": [], "ctp": []}, repr=False
    )

    def update(
        self, trace: PruningTrace, timing: SampleTiming, *, require_positive: bool = False
    ) -> None:
        counts = (
            trace.original_visual_tokens,
            trace.post_btp_visual_tokens,
            trace.post_qtp_visual_tokens,
            trace.post_ctp_visual_tokens,
        )
        if any(value <= 0 for value in counts) or not all(
            left >= right for left, right in zip(counts, counts[1:])
        ):
            raise ValueError(
                "visual token counts must be nonnegative and monotonically nonincreasing"
            )
        stage_values = (
            timing.retrieval_seconds,
            timing.page_load_seconds,
            timing.qa_seconds,
            timing.encoder_seconds,
            timing.decoder_seconds,
            timing.total_seconds,
        )
        if any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or (float(value) <= 0 if require_positive else float(value) < 0)
            for value in stage_values
        ):
            raise ValueError(
                "timings must be finite and positive"
                if require_positive
                else "timings must be finite and nonnegative"
            )
        peak = getattr(timing, "peak_allocated_gpu_bytes", 0)
        warmup_excluded = getattr(timing, "warmup_excluded", False)
        profiler_enabled = getattr(timing, "profiler_enabled", False)
        profiler_definition = getattr(timing, "profiler_definition", None)
        flops = getattr(timing, "flops", None)
        if (
            not isinstance(peak, int)
            or isinstance(peak, bool)
            or peak < 0
            or not isinstance(warmup_excluded, bool)
            or not isinstance(profiler_enabled, bool)
        ):
            raise ValueError("measurement fields are invalid")
        if profiler_enabled:
            if not isinstance(profiler_definition, str) or not profiler_definition:
                raise ValueError("enabled profiler requires a definition")
            if (
                not isinstance(flops, int | float)
                or isinstance(flops, bool)
                or not math.isfinite(float(flops))
                or float(flops) < 0
            ):
                raise ValueError("enabled profiler requires finite nonnegative FLOPs")
            if (
                self.profiler_definition is not None
                and self.profiler_definition != profiler_definition
            ):
                raise ValueError("profiler definition changed across samples")
            self.profiler_definition = profiler_definition
            self.flops = (self.flops or 0.0) + float(flops)
            self.profiler_enabled = True
        elif profiler_definition is not None or flops is not None:
            raise ValueError("profiler definition and FLOPs require profiling")
        self.samples += 1
        self.original += counts[0]
        self.post_btp += counts[1]
        self.post_qtp += counts[2]
        self.post_ctp += counts[3]
        self.retrieval_seconds += timing.retrieval_seconds
        self.page_load_seconds += timing.page_load_seconds
        self.qa_seconds += timing.qa_seconds
        self.encoder_seconds += timing.encoder_seconds
        self.decoder_seconds += timing.decoder_seconds
        self.total_sample_seconds += timing.total_seconds
        self._drop_rates["btp"].append(1.0 - counts[1] / counts[0])
        self._drop_rates["qtp"].append(1.0 - counts[2] / counts[0])
        self._drop_rates["ctp"].append(1.0 - counts[3] / counts[0])
        self.peak_allocated_gpu_bytes = max(self.peak_allocated_gpu_bytes, peak)
        if self.samples == 1:
            self.warmup_excluded = warmup_excluded
        else:
            self.warmup_excluded = self.warmup_excluded and warmup_excluded

    def to_dict(self, *, require_classification: bool = False) -> dict[str, object]:
        if require_classification and self.result_classification is None:
            raise ValueError("strict aggregate requires a result classification")
        if self.result_classification is not None:
            self.result_classification = _validate_result_classification(
                self.result_classification
            )
        total_seconds = self.total_sample_seconds

        def drop(retained: int) -> float:
            return 0.0 if self.original == 0 else 1.0 - retained / self.original

        payload: dict[str, object] = {
            "samples": self.samples,
            "visual_tokens": {
                "original": self.original,
                "post_btp": self.post_btp,
                "post_qtp": self.post_qtp,
                "post_ctp": self.post_ctp,
            },
            "drop_rates": {
                name: (sum(values) / len(values) if values else 0.0)
                for name, values in self._drop_rates.items()
            },
            "token_weighted_drop_rates": {
                "btp": drop(self.post_btp),
                "qtp": drop(self.post_qtp),
                "ctp": drop(self.post_ctp),
            },
            "timing_seconds": {
                "retrieval": self.retrieval_seconds,
                "page_load": self.page_load_seconds,
                "qa": self.qa_seconds,
                "encoder": self.encoder_seconds,
                "decoder": self.decoder_seconds,
                "total": total_seconds,
            },
            "encoder_samples_per_second": (
                0.0 if self.encoder_seconds == 0 else self.samples / self.encoder_seconds
            ),
            "decoder_samples_per_second": (
                0.0 if self.decoder_seconds == 0 else self.samples / self.decoder_seconds
            ),
            "original_visual_tokens_per_second": (
                0.0 if total_seconds == 0 else self.original / total_seconds
            ),
            "measurement": {
                "definition": MEASUREMENT_DEFINITION,
                "peak_allocated_gpu_bytes": self.peak_allocated_gpu_bytes,
                "warmup_excluded": self.warmup_excluded,
                "profiler_enabled": self.profiler_enabled,
            },
        }
        if self.result_classification is not None:
            payload["measurement"]["result_classification"] = self.result_classification
        if self.profiler_enabled:
            measurement = payload["measurement"]
            if not isinstance(measurement, dict):
                raise AssertionError("measurement aggregate must be a dictionary")
            measurement["profiler_definition"] = self.profiler_definition
            measurement["flops"] = self.flops
        return payload


def append_result_jsonl(path: Path, record: dict[str, Any], *, resume: bool = False) -> None:
    path = Path(path)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"results JSONL must be a regular file: {path}")
    if path.exists() and not resume:
        raise FileExistsError(f"{path} exists; pass resume only after its manifest is verified")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def summarize_jsonl(
    path: Path,
    *,
    require_positive: bool = False,
    result_classification: dict[str, object] | None = None,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"results JSONL must be a regular file: {path}")
    metrics = StageMetrics()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            timing_mapping = record.get("timing") if isinstance(record, dict) else None
            if require_positive:
                required_timing = {
                    "retrieval_seconds",
                    "page_load_seconds",
                    "qa_seconds",
                    "total_sample_seconds",
                    "encoder_seconds",
                    "decoder_seconds",
                }
                if not isinstance(timing_mapping, dict) or not required_timing <= set(
                    timing_mapping
                ):
                    missing = sorted(
                        required_timing - set(timing_mapping)
                        if isinstance(timing_mapping, dict)
                        else required_timing
                    )
                    raise ValueError(f"production timing is missing fields: {missing!r}")
            try:
                trace = PruningTrace(**record["trace"])
                timing = SampleTiming(**record["timing"])
            except (KeyError, TypeError) as exc:
                raise ValueError(f"invalid result record on line {line_number}") from exc
            metrics.update(trace, timing, require_positive=require_positive)
    metrics.result_classification = result_classification
    return metrics.to_dict(require_classification=require_positive)
