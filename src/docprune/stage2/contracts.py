"""Deployment inputs and immutable measurement identities; answers live elsewhere."""

import hashlib
import json
from dataclasses import asdict, dataclass

import torch
from torch import Tensor


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class InstanceIdentity:
    source: str
    document_family: str
    question_id: str
    question_sha256: str
    ordered_pages: tuple[str, ...]
    rendering: str
    partition: str
    answerer_revision: str
    retrieval: str
    scoring: str
    intervention: str = "visual_decoder_input_deletion"

    @property
    def key(self):
        return fingerprint(asdict(self))

    def validate(self):
        if any(not v for v in asdict(self).values()):
            raise ValueError("Every teacher identity component must be frozen")
        if self.intervention != "visual_decoder_input_deletion":
            raise ValueError("Stage 2 requires decoder-input deletion")


@dataclass
class RegionLayout:
    """Canonical visual order; each token has one region and one page owner."""

    region_ids: tuple[str, ...]
    owner: Tensor  # [N], int64; region IDs are indices, not embeddings
    page: Tensor  # [N], int64
    coordinates: Tensor  # [N,3], spatial/temporal coordinates
    metadata: Tensor  # [R,D], caller-defined, schema pinned in config

    def validate(self):
        n, r = self.owner.numel(), len(self.region_ids)
        if n == 0 or r == 0 or len(set(self.region_ids)) != r:
            raise ValueError("Empty or duplicate region identity")
        if self.owner.shape != (n,) or self.owner.dtype != torch.long:
            raise ValueError("Region owners must be a 1D int64 tensor")
        if (
            self.page.shape != (n,)
            or self.page.dtype != torch.long
            or (self.page < 0).any()
        ):
            raise ValueError("Invalid page ownership")
        if (
            self.coordinates.shape != (n, 3)
            or self.metadata.ndim != 2
            or self.metadata.shape[0] != r
        ):
            raise ValueError("Invalid coordinates or region metadata")
        if (
            not torch.isfinite(self.coordinates).all()
            or not torch.isfinite(self.metadata).all()
        ):
            raise ValueError("Nonfinite geometry")
        if (self.owner < 0).any() or (self.owner >= r).any():
            raise ValueError("Uncovered or out-of-range visual token")
        if (torch.bincount(self.owner, minlength=r) == 0).any():
            raise ValueError("Every action must own at least one token")
        for i in range(r):
            if torch.unique(self.page[self.owner == i]).numel() != 1:
                raise ValueError("An atomic region cannot span pages")
        return self

    @property
    def costs(self):
        return torch.bincount(self.owner, minlength=len(self.region_ids))

    def retained_tokens(self, mask):
        if mask.dtype != torch.bool or mask.shape != (len(self.region_ids),):
            raise ValueError("Expected one boolean keep/drop decision per region")
        return torch.nonzero(mask.to(self.owner.device)[self.owner], as_tuple=True)[0]


@dataclass
class VisualMemory:
    merged: Tensor  # [N, decoder_width], before answerer decoding
    deepstack: tuple[Tensor, ...]  # each [N, decoder_width]
    grid_thw: Tensor  # [P,3], unmerged patch grid
    provenance: str

    def validate(self):
        if (
            self.merged.ndim != 2
            or self.grid_thw.ndim != 2
            or self.grid_thw.shape[1] != 3
        ):
            raise ValueError("Invalid visual-memory shape")
        if not self.provenance or not torch.isfinite(self.merged).all():
            raise ValueError("Missing provenance or nonfinite visual memory")
        for stream in self.deepstack:
            if stream.shape != self.merged.shape or not torch.isfinite(stream).all():
                raise ValueError(
                    "DeepStack streams must align with merged visual positions"
                )
        return self


@dataclass
class RetrievalFeatures:
    values: Tensor  # [R,F] or [R,T,F]; arbitrary configured feature schema
    schema: str
    valid: Tensor | None = None  # [R,T] for variable-length query profiles

    def validate(self, regions):
        if (
            not self.schema
            or self.values.ndim not in (2, 3)
            or self.values.shape[0] != regions
        ):
            raise ValueError("Invalid retrieval feature schema/shape")
        if not torch.isfinite(self.values).all():
            raise ValueError("Nonfinite retrieval features")
        if self.valid is not None:
            if (
                self.values.ndim != 3
                or self.valid.dtype != torch.bool
                or self.valid.shape != self.values.shape[:2]
            ):
                raise ValueError("Invalid retrieval profile mask")
            if not self.valid.any(dim=1).all():
                raise ValueError("A retrieval profile has no valid entries")


@dataclass
class SelectorInputs:
    identity: InstanceIdentity
    layout: RegionLayout
    vision: VisualMemory
    question_ids: Tensor  # token-level compact question input, [T]
    budget: int
    retrieval: RetrievalFeatures | None = None

    def validate(self):
        self.identity.validate()
        self.layout.validate()
        self.vision.validate()
        if self.vision.merged.shape[0] != self.layout.owner.numel():
            raise ValueError("Answerer memory and action partition differ")
        if (
            self.question_ids.dtype != torch.long
            or self.question_ids.ndim != 1
            or self.question_ids.numel() == 0
        ):
            raise ValueError("Question must contain int64 token IDs")
        if (
            not isinstance(self.budget, int)
            or not 0 < self.budget <= self.layout.owner.numel()
        ):
            raise ValueError("Invalid token budget")
        if self.retrieval is not None:
            self.retrieval.validate(len(self.layout.region_ids))
        return self


def collate_examples(examples):
    """Ragged microbatch: no accidental cross-document attention or padded-token cost."""
    return tuple(x.validate() for x in examples)
