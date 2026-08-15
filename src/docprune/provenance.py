"""Run provenance that distinguishes paper facts from reconstruction choices."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from docprune.config import DocPruneConfig


@dataclass(frozen=True)
class RunProvenance:
    page_count: int
    source_commit: str
    model_revisions: Mapping[str, str]
    dataset_revisions: Mapping[str, str]
    dependency_versions: Mapping[str, str]
    hardware: Mapping[str, str]

    def __post_init__(self) -> None:
        DocPruneConfig.paper_defaults().for_pages(self.page_count)

    def to_dict(self) -> dict[str, object]:
        config = DocPruneConfig.paper_defaults()
        paper_values = asdict(config.for_pages(self.page_count))
        reconstruction_defaults = asdict(config.reconstruction_defaults)
        return {
            "page_count": self.page_count,
            "source_commit": self.source_commit,
            "upstream": {"m3docrag_commit": config.m3docrag_commit},
            "model_revisions": dict(sorted(self.model_revisions.items())),
            "dataset_revisions": dict(sorted(self.dataset_revisions.items())),
            "dependency_versions": dict(sorted(self.dependency_versions.items())),
            "hardware": dict(sorted(self.hardware.items())),
            "paper_values": paper_values,
            "reconstruction_defaults": reconstruction_defaults,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
