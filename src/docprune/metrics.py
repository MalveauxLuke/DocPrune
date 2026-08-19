"""Immutable JSONL results and aggregate DocPrune efficiency metrics."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docprune.m3docrag import SampleTiming
from docprune.qwen2vl.model import PruningTrace

MEASUREMENT_DEFINITION = (
    "peak_allocated_gpu_bytes is torch.cuda.max_memory_allocated after a synchronized "
    "counter reset at the start of the complete QA path and synchronization after generation"
)


@dataclass
class StageMetrics:
    samples: int = 0
    original: int = 0
    post_btp: int = 0
    post_qtp: int = 0
    post_ctp: int = 0
    retrieval_seconds: float = 0.0
    qa_seconds: float = 0.0
    peak_allocated_gpu_bytes: int = 0
    warmup_excluded: bool = False
    profiler_enabled: bool = False
    profiler_definition: str | None = None
    flops: float | None = None

    def update(self, trace: PruningTrace, timing: SampleTiming) -> None:
        counts = (
            trace.original_visual_tokens,
            trace.post_btp_visual_tokens,
            trace.post_qtp_visual_tokens,
            trace.post_ctp_visual_tokens,
        )
        if any(value < 0 for value in counts) or not all(
            left >= right for left, right in zip(counts, counts[1:])
        ):
            raise ValueError(
                "visual token counts must be nonnegative and monotonically nonincreasing"
            )
        if any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in (timing.retrieval_seconds, timing.qa_seconds)
        ):
            raise ValueError("timings must be finite and nonnegative")
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
        self.qa_seconds += timing.qa_seconds
        self.peak_allocated_gpu_bytes = max(self.peak_allocated_gpu_bytes, peak)
        if self.samples == 1:
            self.warmup_excluded = warmup_excluded
        else:
            self.warmup_excluded = self.warmup_excluded and warmup_excluded

    def to_dict(self) -> dict[str, object]:
        total_seconds = self.retrieval_seconds + self.qa_seconds

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
                "btp": drop(self.post_btp),
                "qtp": drop(self.post_qtp),
                "ctp": drop(self.post_ctp),
            },
            "timing_seconds": {
                "retrieval": self.retrieval_seconds,
                "qa": self.qa_seconds,
                "total": total_seconds,
            },
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


def summarize_jsonl(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"results JSONL must be a regular file: {path}")
    metrics = StageMetrics()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                trace = PruningTrace(**record["trace"])
                timing = SampleTiming(**record["timing"])
            except (KeyError, TypeError) as exc:
                raise ValueError(f"invalid result record on line {line_number}") from exc
            metrics.update(trace, timing)
    return metrics.to_dict()
