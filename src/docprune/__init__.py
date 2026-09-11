"""Training-free DocPrune reproduction components."""

from pathlib import Path as _Path

# Retained baseline modules live out of the active source list. Keep their public
# docprune.task*/segmentation/qwen2vl imports stable for callers and archived tools.
__path__.append(str(_Path(__file__).parent / "_legacy"))

from docprune.config import (
    DocPruneConfig,
    PagePruningConfig,
    ReconstructionDefaults,
    load_config,
)

__all__ = [
    "DocPruneConfig",
    "PagePruningConfig",
    "ReconstructionDefaults",
    "load_config",
]
