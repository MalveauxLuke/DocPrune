"""Validated paper parameters and explicitly labeled reconstruction choices."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import tomllib

SUPPORTED_PAGE_COUNTS = (1, 2, 4)


@dataclass(frozen=True)
class PagePruningConfig:
    retrieval_background_threshold: float
    qa_background_threshold: float
    background_error_tolerance: float
    question_threshold: float
    comprehension_threshold: float
    attention_threshold: float

    def __post_init__(self) -> None:
        unit_interval_fields = (
            "retrieval_background_threshold",
            "qa_background_threshold",
            "question_threshold",
            "attention_threshold",
        )
        for name in unit_interval_fields:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        if self.background_error_tolerance < 0:
            raise ValueError("background_error_tolerance must be nonnegative")
        if self.comprehension_threshold < 0:
            raise ValueError("comprehension_threshold must be nonnegative")


@dataclass(frozen=True)
class ReconstructionDefaults:
    source: str = "reconstruction_default"
    grayscale: str = "bt601_uint8"
    gaussian_sigma: float = 1.0
    group_retention: str = "any"
    attention_head_aggregation: str = "mean"
    ctp_timing: str = "prefill_last_prompt_token"

    def __post_init__(self) -> None:
        if self.source != "reconstruction_default":
            raise ValueError("reconstruction choices must use source='reconstruction_default'")
        if self.gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be positive")
        if self.group_retention not in {"any", "all", "mean"}:
            raise ValueError("group_retention must be any, all, or mean")
        if self.attention_head_aggregation not in {"mean", "max"}:
            raise ValueError("attention_head_aggregation must be mean or max")


@dataclass(frozen=True)
class DocPruneConfig:
    page_settings: Mapping[int, PagePruningConfig]
    reconstruction_defaults: ReconstructionDefaults
    m3docrag_commit: str = "29e6ac2294d6b87075a1d45b8a8df175b214248a"

    def __post_init__(self) -> None:
        settings = dict(self.page_settings)
        if tuple(sorted(settings)) != SUPPORTED_PAGE_COUNTS:
            raise ValueError("page_settings must define exactly page counts 1, 2, and 4")
        object.__setattr__(self, "page_settings", MappingProxyType(settings))

    def for_pages(self, page_count: int) -> PagePruningConfig:
        try:
            return self.page_settings[page_count]
        except KeyError as exc:
            raise ValueError("page_count must be 1, 2, or 4") from exc

    @classmethod
    def paper_defaults(cls) -> DocPruneConfig:
        return cls(
            page_settings={
                1: PagePruningConfig(0.9, 0.9, 1.0, 0.3, 65.0, 0.5),
                2: PagePruningConfig(1.0, 1.0, 1.0, 0.3, 60.0, 0.25),
                4: PagePruningConfig(1.0, 0.8, 1.0, 0.4, 45.0, 0.075),
            },
            reconstruction_defaults=ReconstructionDefaults(),
        )


def _page_config(raw: Mapping[str, Any]) -> PagePruningConfig:
    return PagePruningConfig(
        retrieval_background_threshold=float(raw["retrieval_background_threshold"]),
        qa_background_threshold=float(raw["qa_background_threshold"]),
        background_error_tolerance=float(raw["background_error_tolerance"]),
        question_threshold=float(raw["question_threshold"]),
        comprehension_threshold=float(raw["comprehension_threshold"]),
        attention_threshold=float(raw["attention_threshold"]),
    )


def load_config(path: Path) -> DocPruneConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    paper = raw["paper"]
    reconstruction = ReconstructionDefaults(**raw["reconstruction_defaults"])
    return DocPruneConfig(
        page_settings={
            1: _page_config(paper["top1"]),
            2: _page_config(paper["top2"]),
            4: _page_config(paper["top4"]),
        },
        reconstruction_defaults=reconstruction,
        m3docrag_commit=str(raw["upstream"]["m3docrag_commit"]),
    )
