"""Matched arm configuration and explicit Stage 1 admissions."""

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .contracts import fingerprint
from .qwen import ANSWERER_ID, SELECTOR_ID


@dataclass(frozen=True)
class Stage1Contract:
    answerer_revision: str | None = None
    selector_revision: str | None = None
    source_manifest: str | None = None
    document_split_manifest: str | None = None
    retrieval_and_pages: str | None = None
    rendering_and_partition: str | None = None
    teacher_bank_identity: str | None = None
    decoding_and_scoring: str | None = None
    interface_validation_receipt: str | None = None
    epsilon: float | None = None
    margin: float | None = None
    budget: int | None = None

    def pending(self):
        return [key for key, value in asdict(self).items() if value is None]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "rich-gold-aware-shared"
    architecture: str = "rich"
    supervision: str = "gold_aware"
    vision_mode: str = "shared_adapter"
    answerer: str = ANSWERER_ID
    selector: str = SELECTOR_ID
    head2: bool = False
    head2_weight: float = 1.0
    question_first: bool = True
    representation_layer: int = -1
    width: int = 128
    heads: int = 4
    slots: int = 4
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    compact_reread: bool = False
    retrieval_dim: int | None = None
    retrieval_schema: str | None = None
    metadata_dim: int = 8
    metadata_schema: str = "bbox_minmax_page_cost_v1"
    stream_map: tuple[int, ...] = (0, 1, 2)
    seed: int = 0
    weighting: str = "question"
    temperature: float = 1.0
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    stage1: Stage1Contract = field(default_factory=Stage1Contract)

    def validate(self, *, execution=False):
        if self.answerer != ANSWERER_ID or self.selector != SELECTOR_ID:
            raise ValueError("Owner-selected model identities must remain fixed")
        if self.architecture not in (
            "rich",
            "pooled",
            "compact",
        ) or self.supervision not in ("g_only", "gold_aware", "pure_contrast"):
            raise ValueError("Unknown architecture or supervision")
        if not isinstance(self.head2, bool):
            raise ValueError("Head 2 switch must be an explicit boolean")
        if self.head2_weight < 0 or not math.isfinite(self.head2_weight):
            raise ValueError("Head 2 weight must be finite and nonnegative")
        if not self.question_first:
            raise ValueError("Stage 2 uses question-first ordering")
        allowed = (
            ("shared_answerer",)
            if self.architecture == "compact"
            else ("native", "shared_adapter")
        )
        if self.vision_mode not in allowed:
            raise ValueError("Invalid vision path")
        if (self.retrieval_dim is None) != (self.retrieval_schema is None):
            raise ValueError("Retrieval width and schema must be specified together")
        if (
            self.width < 1
            or self.heads < 1
            or self.width % self.heads
            or self.slots < 1
        ):
            raise ValueError("Invalid readout dimensions")
        if (
            self.weighting not in ("question", "family")
            or self.temperature <= 0
            or not math.isfinite(self.temperature)
        ):
            raise ValueError("Invalid loss settings")
        if self.lora_rank < 1 or self.lora_alpha <= 0 or not 0 <= self.lora_dropout < 1:
            raise ValueError("Invalid LoRA configuration")
        if (
            self.learning_rate <= 0
            or self.weight_decay < 0
            or not math.isfinite(self.learning_rate + self.weight_decay)
        ):
            raise ValueError("Invalid optimization defaults")
        if (
            self.metadata_dim < 1
            or not self.metadata_schema
            or (self.retrieval_dim is not None and self.retrieval_dim < 1)
        ):
            raise ValueError("Invalid feature dimensions/schema")
        for value in (self.stage1.epsilon, self.stage1.margin):
            if value is not None and (value < 0 or not math.isfinite(value)):
                raise ValueError("Invalid Stage 1 preference threshold")
        if self.stage1.budget is not None and (
            not isinstance(self.stage1.budget, int) or self.stage1.budget <= 0
        ):
            raise ValueError("Invalid Stage 1 budget")
        if execution and self.stage1.pending():
            raise ValueError(
                "Stage 1 fields remain unresolved: " + ", ".join(self.stage1.pending())
            )
        return self

    @property
    def identity(self):
        return fingerprint(asdict(self))


def matched_configs(base=None):
    """Five central fits; native-vision diagnostics are explicitly separated."""
    base = replace(
        base or ExperimentConfig(),
        architecture="rich",
        vision_mode="shared_adapter",
        supervision="gold_aware",
        head2=False,
    )
    central = [
        replace(base, name=f"rich-{mode}-shared", supervision=mode)
        for mode in ("g_only", "gold_aware", "pure_contrast")
    ]
    central += [
        replace(base, name="pooled-gold-aware-shared", architecture="pooled"),
        replace(
            base,
            name="compact-gold-aware",
            architecture="compact",
            vision_mode="shared_answerer",
        ),
    ]
    diagnostic = [
        replace(c, name=c.name.replace("shared", "native"), vision_mode="native")
        for c in (central[1], central[3])
    ]
    return {c.name: c.validate() for c in central + diagnostic}


def head2_comparison_configs(base=None):
    """Opt-in Stage 3B comparison; never adds runs to the initial Stage 2 matrix."""
    base = base or ExperimentConfig()
    return {
        c.name: c.validate()
        for c in (
            replace(base, name=base.name + "-direct-only", head2=False),
            replace(base, name=base.name + "-linked-head2", head2=True),
        )
    }


def load_config(path):
    data = json.loads(Path(path).read_text())
    data["stage1"] = Stage1Contract(**data.get("stage1", {}))
    for key in ("lora_targets", "stream_map"):
        if key in data:
            data[key] = tuple(data[key])
    return ExperimentConfig(**data).validate()


def verify_split_integrity(train, validation, test, *, known_diagnostic_keys=()):
    """Guard parent-family and admitted-page exposure, including background distractors."""
    sets = [tuple(split) for split in (train, validation, test)]
    excluded = set(known_diagnostic_keys)
    for split in sets:
        keys = [x.key for x in split]
        if len(set(keys)) != len(keys) or excluded.intersection(keys):
            raise ValueError(
                "Duplicate identity or inspected diagnostic in learning split"
            )
    for i, first in enumerate(sets):
        for second in sets[i + 1 :]:
            families_a = {(x.source, x.document_family) for x in first}
            families_b = {(x.source, x.document_family) for x in second}
            pages_a = {p for x in first for p in x.ordered_pages}
            pages_b = {p for x in second for p in x.ordered_pages}
            if families_a & families_b or pages_a & pages_b:
                raise ValueError("Cross-split family or admitted-page exposure")
