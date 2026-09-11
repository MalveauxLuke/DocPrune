"""Version-gated Qwen2-VL integration for DocPrune."""

from docprune.qwen2vl.compat import (
    QwenCompatibility,
    QwenCompatibilityError,
    assert_supported_qwen2vl,
)
from docprune.qwen2vl.model import (
    DocPruneQwen2VL,
    GenerationResult,
    PruningTrace,
    VisionPruningMasks,
)

__all__ = [
    "DocPruneQwen2VL",
    "GenerationResult",
    "PruningTrace",
    "QwenCompatibility",
    "QwenCompatibilityError",
    "VisionPruningMasks",
    "assert_supported_qwen2vl",
]
