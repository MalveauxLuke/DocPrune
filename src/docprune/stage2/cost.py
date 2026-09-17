"""Measured stage costs; synchronizes only a supplied CUDA tensor device."""

import math
import resource
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

import torch


@dataclass
class CostLedger:
    seconds: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)
    hardware: str = "unrecorded"
    precision: str = "unrecorded"
    cache_mode: str = "unrecorded"

    def add(self, stage, seconds, **counters):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid cost measurement")
        self.seconds[stage] = self.seconds.get(stage, 0.0) + seconds
        self.calls[stage] = self.calls.get(stage, 0) + 1
        for key, value in counters.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid measured counter")
            self.counters[key] = self.counters.get(key, 0.0) + value

    @contextmanager
    def measure(self, stage, *, synchronize=None):
        if synchronize is not None:
            synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if synchronize is not None:
                synchronize()
            self.add(stage, time.perf_counter() - start)

    def capture_process_memory(self, reference):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.counters["process_peak_rss_bytes"] = (
            rss if sys.platform == "darwin" else rss * 1024
        )
        if reference.is_cuda:
            self.counters["process_peak_cuda_allocated_bytes"] = (
                torch.cuda.max_memory_allocated(reference.device)
            )
            self.counters["process_peak_cuda_reserved_bytes"] = (
                torch.cuda.max_memory_reserved(reference.device)
            )

    def summary(self):
        required = (
            "retrieval",
            "preprocessing",
            "answerer_vision",
            "selector",
            "allocation",
            "answerer_decode",
        )
        missing = [
            s
            for s in required
            if s not in self.seconds
            and not (
                s == "selector" and any(k.startswith("selector_") for k in self.seconds)
            )
        ]
        return {
            "seconds": dict(self.seconds),
            "calls": dict(self.calls),
            "counters": dict(self.counters),
            "total_recorded_seconds": sum(self.seconds.values()),
            "missing_stages": missing,
            "whole_system_complete": not missing,
            "hardware": self.hardware,
            "precision": self.precision,
            "cache_mode": self.cache_mode,
        }


def measure(ledger, stage, reference):
    if ledger is None:
        return nullcontext()
    sync = (
        (lambda: torch.cuda.synchronize(reference.device))
        if reference.is_cuda
        else None
    )
    return ledger.measure(stage, synchronize=sync)


def execution_ledger(reference, *, cache_mode):
    """Record hardware and precision for an already allocated tensor, never initialize a GPU."""
    import platform
    hardware = (torch.cuda.get_device_name(reference.device) if reference.is_cuda
                else f"CPU {platform.machine()}")
    return CostLedger(hardware=hardware, precision=str(reference.dtype), cache_mode=cache_mode)
