"""Compact, pooled, and persistent-memory rich regional readers for Head 1."""

import torch
from torch import nn

from .contracts import RetrievalFeatures
from .cost import measure
from .policy import PolicyEncoding, PolicyHead, SetCorrection


class RetrievalFusion(nn.Module):
    """Optional extensible features; token-associated profiles use learned attention."""

    def __init__(self, feature_dim, width, heads):
        super().__init__()
        self.project = nn.Linear(feature_dim, width)
        self.attend = nn.MultiheadAttention(width, heads, batch_first=True)

    def forward(self, slots, features: RetrievalFeatures):
        features.validate(slots.shape[0])
        values = self.project(features.values.to(slots))
        if values.ndim == 2:
            return slots + values[:, None, :]
        padding = None if features.valid is None else ~features.valid.to(slots.device)
        return (
            slots
            + self.attend(
                slots, values, values, key_padding_mask=padding, need_weights=False
            )[0]
        )


class ReadBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.question = nn.MultiheadAttention(width, heads, batch_first=True)
        self.local = nn.MultiheadAttention(width, heads, batch_first=True)
        self.global_mix = nn.TransformerEncoderLayer(
            width, heads, width * 2, dropout=0.0, batch_first=True, norm_first=True
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, slots, question, memory, owner):
        r, s, d = slots.shape
        q = question.unsqueeze(0).expand(r, -1, -1)
        slots = slots + self.question(slots, q, q, need_weights=False)[0]
        # Ragged local reads avoid R x N padding and prevent crossing region boundaries.
        local = []
        for i in range(r):
            v = memory[owner == i].unsqueeze(0)
            local.append(
                slots[i : i + 1]
                + self.local(slots[i : i + 1], v, v, need_weights=False)[0]
            )
        slots = torch.cat(local, dim=0)
        return self.norm(self.global_mix(slots.reshape(1, r * s, d)).reshape(r, s, d))


class RegionReader(nn.Module):
    def __init__(
        self,
        memory_dim,
        question_dim,
        metadata_dim,
        *,
        width=128,
        heads=4,
        slots=4,
        mode="rich",
        compact_reread=False,
        retrieval_dim=None,
    ):
        super().__init__()
        if mode not in ("rich", "pooled", "compact") or width % heads or slots < 1:
            raise ValueError("Invalid reader configuration")
        self.mode = mode
        self.visual = nn.Linear(memory_dim, width)
        self.spatial = nn.Linear(3, width)
        self.question = nn.Linear(question_dim, width)
        self.metadata = nn.Linear(metadata_dim, width)
        self.budget = nn.Linear(1, width)
        self.slots = (
            nn.Parameter(torch.randn(slots, width) * 0.02) if mode != "pooled" else None
        )
        self.retrieval = (
            None
            if retrieval_dim is None
            else RetrievalFusion(retrieval_dim, width, heads)
        )
        self.question_attention = (
            nn.MultiheadAttention(width, heads, batch_first=True)
            if mode == "pooled"
            else None
        )
        self.pooled_mixer = (
            nn.TransformerEncoderLayer(
                width, heads, width * 2, dropout=0.0, batch_first=True, norm_first=True
            )
            if mode == "pooled"
            else None
        )
        rounds = 2 if mode == "rich" or (mode == "compact" and compact_reread) else 1
        self.reads = nn.ModuleList(
            [ReadBlock(width, heads) for _ in range(rounds if mode != "pooled" else 0)]
        )
        self.final = nn.LayerNorm(width)

    def forward(self, memory, question, layout, budget_fraction, retrieval=None):
        layout.validate()
        if memory.ndim != 2 or memory.shape[0] != layout.owner.numel():
            raise ValueError("Region readout/memory mismatch")
        v = self.visual(memory.to(self.visual.weight))
        q = self.question(question.to(self.question.weight))
        coords = layout.coordinates.to(v)
        # Coordinates are supplied in a frozen schema; rescale per axis without losing location.
        coords = coords / coords.abs().amax(dim=0).clamp_min(1)
        v = v + self.spatial(coords)
        meta = self.metadata(layout.metadata.to(v))
        budget = self.budget(v.new_tensor([[budget_fraction]]))
        if self.mode == "pooled":
            pooled = torch.stack(
                [v[layout.owner == i].mean(0) for i in range(len(layout.region_ids))]
            )
            state = (pooled + meta + budget)[:, None, :]
            query = q[None].expand(state.shape[0], -1, -1)
            state = (
                state
                + self.question_attention(state, query, query, need_weights=False)[0]
            )
            if retrieval is not None:
                if self.retrieval is None:
                    raise ValueError(
                        "Retrieval features supplied to a feature-disabled reader"
                    )
                state = self.retrieval(state, retrieval)
            # No fine memory is consumed beyond this pooling boundary.
            return self.final(self.pooled_mixer(state[:, 0][None])[0])
        state = self.slots[None] + meta[:, None] + budget
        if retrieval is not None:
            if self.retrieval is None:
                raise ValueError(
                    "Retrieval features supplied to a feature-disabled reader"
                )
            state = self.retrieval(state, retrieval)
        for block in self.reads:
            state = block(state, q, v, layout.owner)
        return self.final(state.mean(dim=1))


class CompactSelector(nn.Module):
    def __init__(
        self,
        vision_dim,
        vocab_size,
        metadata_dim,
        *,
        width=128,
        heads=4,
        slots=4,
        retrieval_dim=None,
        reread=False,
        max_question_tokens=4096,
        head2=False,
    ):
        super().__init__()
        self.word = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(max_question_tokens, width)
        self.question_encoder = nn.TransformerEncoderLayer(
            width, heads, width * 2, dropout=0.0, batch_first=True
        )
        self.reader = RegionReader(
            vision_dim,
            width,
            metadata_dim,
            width=width,
            heads=heads,
            slots=slots,
            mode="compact",
            compact_reread=reread,
            retrieval_dim=retrieval_dim,
        )
        self.head = PolicyHead(width)
        self.correction = SetCorrection(width) if head2 else None

    def forward(self, inputs, *, ledger=None, return_encoding=False):
        inputs.validate()
        ids = inputs.question_ids.to(self.word.weight.device)
        if ids.numel() > self.position.num_embeddings:
            raise ValueError(
                "Question exceeds configured length; truncation is forbidden"
            )
        with measure(ledger, "selector_question", self.word.weight):
            q = self.word(ids) + self.position(
                torch.arange(ids.numel(), device=ids.device)
            )
            q = self.question_encoder(q[None])[0]
        # The answerer visual stack is always frozen; train only selector components.
        with measure(ledger, "selector_readout", self.word.weight):
            e = self.reader(
                inputs.vision.merged.detach(),
                q,
                inputs.layout,
                inputs.budget / inputs.layout.owner.numel(),
                inputs.retrieval,
            )
        with measure(ledger, "selector_head", self.word.weight):
            scores = self.head(e)
        return PolicyEncoding(e, scores) if return_encoding else scores
