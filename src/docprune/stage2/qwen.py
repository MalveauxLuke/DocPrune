"""Explicit Qwen3-VL 4.57.3 bridge. Local checkpoints only; no implicit downloads."""

import re
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn

from .contracts import VisualMemory
from .cost import measure
from .policy import PolicyEncoding, PolicyHead, SetCorrection
from .readouts import RegionReader

ANSWERER_ID = "Qwen/Qwen3-VL-8B-Instruct"
SELECTOR_ID = "Qwen/Qwen3-VL-Reranker-2B"


def load_local_checkpoint(path, *, revision, role, dtype=torch.float32):
    """Explicit future entry point. Preparation/tests never call pretrained loading."""
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if transformers.__version__ != "4.57.3":
        raise RuntimeError("This bridge is verified against transformers 4.57.3 only")
    path = Path(path)
    if (
        not re.fullmatch(r"[0-9a-f]{40}", revision)
        or path.name != revision
        or not path.is_dir()
    ):
        raise ValueError(
            "Supply a local immutable HF snapshot directory and exact revision"
        )
    if role not in ("answerer", "selector"):
        raise ValueError("Unknown checkpoint role")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, local_files_only=True, torch_dtype=dtype, attn_implementation="sdpa"
    )
    expected = (
        (4096, 27, (8, 16, 24)) if role == "answerer" else (2048, 24, (5, 11, 17))
    )
    actual = (
        model.config.text_config.hidden_size,
        model.config.vision_config.depth,
        tuple(model.config.vision_config.deepstack_visual_indexes),
    )
    if actual != expected:
        raise ValueError(f"Wrong checkpoint architecture for {role}: {actual}")
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model.requires_grad_(False)
    model.eval()
    return model, processor


def apply_lora(
    model,
    *,
    rank=8,
    alpha=16,
    dropout=0.0,
    targets=("q_proj", "k_proj", "v_proj", "o_proj"),
):
    from peft import LoraConfig, get_peft_model

    if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1 or not targets:
        raise ValueError("Invalid LoRA settings")
    model.requires_grad_(False)
    model.model.language_model = get_peft_model(
        model.model.language_model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(targets),
            bias="none",
            task_type=None,
        ),
    )
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names or any("lora_" not in n for n in names):
        raise RuntimeError("Selector backbone must train only LoRA parameters")
    return model


@dataclass
class PackedPrompt:
    input_ids: torch.Tensor
    question_positions: torch.Tensor
    image_grid_thw: torch.Tensor
    pixel_values: torch.Tensor | None = None

    def validate(self, config, *, question_first=True):
        ids = self.input_ids
        if ids.dtype != torch.long or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError(
                "Use one unpadded, jointly encoded document instance per microbatch"
            )
        positions = self.question_positions
        if (
            positions.dtype != torch.long
            or positions.ndim != 1
            or not positions.numel()
        ):
            raise ValueError("Question positions are required")
        if (
            (positions < 0).any()
            or (positions >= ids.shape[1]).any()
            or not torch.equal(positions, positions.unique(sorted=True))
        ):
            raise ValueError("Invalid question positions")
        visual = ids[0] == config.image_token_id
        if (ids == config.video_token_id).any():
            raise ValueError(
                "Stage 2 document interface currently supports page images only"
            )
        if visual[positions].any() or not visual.any():
            raise ValueError("Question and visual positions must be separate")
        if question_first and int(positions.max()) >= int(torch.nonzero(visual)[0]):
            raise ValueError(
                "Primary selector comparisons require question-first ordering"
            )
        grid = self.image_grid_thw
        merge = config.vision_config.spatial_merge_size
        if (
            grid.dtype != torch.long
            or grid.ndim != 2
            or grid.shape[1] != 3
            or (grid <= 0).any()
        ):
            raise ValueError("Invalid page grid")
        if (grid[:, 0] != 1).any() or (grid[:, 1:] % merge).any():
            raise ValueError("Invalid document page/merge grid")
        if int((grid.prod(-1) // merge**2).sum()) != int(visual.sum()):
            raise ValueError("Image placeholders and grid token counts differ")
        return visual


def prepare_prompt(processor, question, images, *, instruction=None, system=None):
    """Question-before-document serialization; no teacher answers enter this path."""
    if not question or not images:
        raise ValueError("A question and admitted pages are required")
    if (instruction is None) != (system is None):
        raise ValueError("Instruction and system prompt must be supplied together")
    if instruction is None:
        message = [{
            "role": "user",
            "content": [{"type": "text", "text": question}]
            + [{"type": "image"} for _ in images],
        }]
    else:
        if not instruction.strip() or not system.strip():
            raise ValueError("Instruction and system prompt cannot be empty")
        message = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<Instruct>: " + instruction},
                    {"type": "text", "text": "<Query>:\n"},
                    {"type": "text", "text": question},
                    {"type": "text", "text": "\n<Document>:"},
                ] + [{"type": "image"} for _ in images],
            },
        ]
    text = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    encoded = processor(text=[text], images=images, return_tensors="pt", padding=False)
    ids = encoded["input_ids"][0].tolist()
    if instruction is None:
        query = processor.tokenizer.encode(question, add_special_tokens=False)
        starts = [
            i
            for i in range(len(ids) - len(query) + 1)
            if ids[i : i + len(query)] == query
        ]
        if len(starts) != 1:
            raise ValueError(
                "Cannot identify a unique exact question span; do not guess token positions"
            )
        positions = torch.arange(starts[0], starts[0] + len(query))
    else:
        query_marker = "<Query>:\n"
        document_marker = "\n<Document>:"
        if text.count(query_marker) != 1 or text.count(document_marker) != 1:
            raise ValueError("Structured prompt markers must each occur exactly once")
        question_start = text.index(query_marker) + len(query_marker)
        question_end = text.index(document_marker, question_start)
        if text[question_start:question_end] != question:
            raise ValueError("Structured prompt query field does not match the question")

        tokenized = processor.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        text_ids = tokenized["input_ids"]
        offsets = tokenized["offset_mapping"]
        if text_ids and isinstance(text_ids[0], list):
            text_ids, offsets = text_ids[0], offsets[0]
        query_indices = [
            i
            for i, (start, end) in enumerate(offsets)
            if end > question_start and start < question_end
        ]
        if not query_indices or query_indices != list(
            range(query_indices[0], query_indices[-1] + 1)
        ):
            raise ValueError("Cannot map the structured query to contiguous tokens")

        # Images occur after the query. Their placeholders may expand during
        # multimodal processing, so align only the rendered prefix through the
        # query instead of assuming identical full text/image tokenization.
        prefix = text_ids[: query_indices[-1] + 1]
        prefix_starts = [
            i
            for i in range(len(ids) - len(prefix) + 1)
            if ids[i : i + len(prefix)] == prefix
        ]
        if len(prefix_starts) != 1:
            raise ValueError(
                "Cannot align the structured prompt prefix with processed input tokens"
            )
        positions = torch.tensor(
            [prefix_starts[0] + i for i in query_indices], dtype=torch.long
        )
        extracted = processor.tokenizer.decode(
            [ids[i] for i in positions.tolist()],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if extracted.strip() != question.strip():
            raise ValueError("Structured query tokens do not decode to the question")
    return PackedPrompt(
        encoded["input_ids"],
        positions,
        encoded["image_grid_thw"],
        encoded["pixel_values"],
    )


def extract_vision(model, prompt, *, provenance):
    prompt.validate(model.config, question_first=False)
    if prompt.pixel_values is None:
        raise ValueError("Native vision requires preprocessed page pixels")
    model.model.visual.eval()
    with torch.no_grad():
        merged, deep = model.model.get_image_features(
            prompt.pixel_values, prompt.image_grid_thw
        )
    return VisualMemory(
        torch.cat(merged, dim=0).detach(),
        tuple(v.detach() for v in deep),
        prompt.image_grid_thw,
        provenance,
    ).validate()


class SharedVisionAdapter(nn.Module):
    """Separate learned projections for merged and each explicitly matched DeepStack stream."""

    def __init__(
        self, source_width, target_width, source_taps, target_taps, stream_map=None
    ):
        super().__init__()
        self.source_taps, self.target_taps = tuple(source_taps), tuple(target_taps)
        self.stream_map = (
            tuple(range(len(target_taps))) if stream_map is None else tuple(stream_map)
        )
        if len(self.stream_map) != len(target_taps) or any(
            i not in range(len(source_taps)) for i in self.stream_map
        ):
            raise ValueError("Every target DeepStack stream needs an explicit source")
        self.merged = nn.Linear(source_width, target_width)
        self.deep = nn.ModuleList(
            [nn.Linear(source_width, target_width) for _ in target_taps]
        )

    def forward(self, source):
        source.validate()
        if len(source.deepstack) != len(self.source_taps):
            raise ValueError("Source DeepStack contract mismatch")
        return VisualMemory(
            self.merged(source.merged.detach().to(self.merged.weight)),
            tuple(
                layer(source.deepstack[i].detach().to(layer.weight))
                for layer, i in zip(self.deep, self.stream_map)
            ),
            source.grid_thw,
            "adapted:" + source.provenance,
        ).validate()


def assemble_prefill(model, prompt, vision, retained=None, *, question_first=False):
    """Keep original multimodal positions; compact sequence/cache indices independently."""
    visual = prompt.validate(model.config, question_first=question_first)
    vision.validate()
    if not torch.equal(prompt.image_grid_thw.cpu(), vision.grid_thw.cpu()):
        raise ValueError(
            "Vision geometry differs; spatial remapping must be validated explicitly"
        )
    n = int(visual.sum())
    if vision.merged.shape != (n, model.config.text_config.hidden_size):
        raise ValueError("Vision width or token count differs from language interface")
    if len(vision.deepstack) != len(
        model.config.vision_config.deepstack_visual_indexes
    ):
        raise ValueError("Missing DeepStack streams")
    ids = prompt.input_ids
    embeddings = model.model.get_input_embeddings()(ids)
    embeddings = embeddings.clone()
    embeddings[0, visual] = vision.merged.to(embeddings)
    positions, _ = model.model.get_rope_index(
        ids, prompt.image_grid_thw, attention_mask=torch.ones_like(ids)
    )
    original_next_position = int(positions.max()) + 1
    if retained is None:
        retained = torch.arange(n, device=ids.device)
    if (
        retained.dtype != torch.long
        or retained.ndim != 1
        or (retained < 0).any()
        or (retained >= n).any()
        or not torch.equal(retained, retained.unique(sorted=True))
    ):
        raise ValueError("Retained original IDs must be sorted, unique and in range")
    keep = ~visual.clone()
    visual_indices = torch.nonzero(visual, as_tuple=True)[0]
    keep[visual_indices[retained.to(visual_indices.device)]] = True
    keep_indices = torch.nonzero(keep, as_tuple=True)[0]
    return (
        dict(
            inputs_embeds=embeddings[:, keep],
            position_ids=positions[:, :, keep],
            attention_mask=torch.ones(
                (1, int(keep.sum())), dtype=torch.long, device=ids.device
            ),
            visual_pos_masks=visual[keep][None],
            deepstack_visual_embeds=[
                v[retained.to(v.device)].to(embeddings) for v in vision.deepstack
            ],
            cache_position=torch.arange(int(keep.sum()), device=ids.device),
        ),
        keep_indices,
        original_next_position,
    )


class ProxySelector(nn.Module):
    def __init__(
        self,
        model,
        metadata_dim,
        *,
        mode,
        vision_mode,
        adapter=None,
        width=128,
        heads=4,
        slots=4,
        retrieval_dim=None,
        layer=-1,
        head2=False,
    ):
        super().__init__()
        if mode not in ("rich", "pooled") or vision_mode not in (
            "native",
            "shared_adapter",
        ):
            raise ValueError("Invalid proxy architecture or vision mode")
        if (vision_mode == "shared_adapter") != (adapter is not None):
            raise ValueError(
                "Shared vision requires a selector-side adapter only in that mode"
            )
        if not any(
            "lora_" in n and p.requires_grad for n, p in model.named_parameters()
        ):
            raise ValueError("Primary proxy must have trainable LoRA attached")
        self.backbone, self.adapter = model, adapter
        self.vision_mode, self.layer = vision_mode, layer
        d = model.config.text_config.hidden_size
        self.reader = RegionReader(
            d,
            d,
            metadata_dim,
            width=width,
            heads=heads,
            slots=slots,
            mode=mode,
            retrieval_dim=retrieval_dim,
        )
        self.head = PolicyHead(width)
        self.correction = SetCorrection(width) if head2 else None

    def train(self, mode=True):
        super().train(mode)
        self.backbone.model.visual.eval()
        return self

    def forward(
        self, inputs, prompt, *, native_layout=None, ledger=None, return_encoding=False
    ):
        inputs.validate()
        if self.vision_mode == "native":
            with measure(
                ledger,
                "selector_vision",
                self.backbone.model.get_input_embeddings().weight,
            ):
                vision = extract_vision(
                    self.backbone, prompt, provenance=inputs.identity.key + ":native"
                )
            layout = (
                inputs.layout
                if native_layout is None
                else replace(native_layout, metadata=inputs.layout.metadata)
            )
            layout.validate()
            if layout.region_ids != inputs.layout.region_ids:
                raise ValueError(
                    "Native mapping must describe exactly the original actions"
                )
        else:
            with measure(ledger, "selector_adapter", self.adapter.merged.weight):
                vision = self.adapter(inputs.vision)
            layout = inputs.layout
        with measure(ledger, "selector_input", vision.merged):
            packed, _, _ = assemble_prefill(
                self.backbone, prompt, vision, question_first=True
            )
        with measure(ledger, "selector_proxy", packed["inputs_embeds"]):
            output = self.backbone.model.language_model(
                **packed, use_cache=False, output_hidden_states=True, return_dict=True
            )
        states = output.hidden_states
        if states is None or not -len(states) <= self.layer < len(states):
            raise ValueError("Requested proxy representation layer is unavailable")
        hidden = states[self.layer][0]
        visual = prompt.input_ids[0] == self.backbone.config.image_token_id
        with measure(ledger, "selector_readout", hidden):
            e = self.reader(
                hidden[visual],
                hidden[prompt.question_positions],
                layout,
                inputs.budget / inputs.layout.owner.numel(),
                inputs.retrieval,
            )
        with measure(ledger, "selector_head", e):
            scores = self.head(e)
        return PolicyEncoding(e, scores) if return_encoding else scores
