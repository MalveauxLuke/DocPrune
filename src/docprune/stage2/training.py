"""Training primitives for the explicitly executed pipeline; no work occurs on import."""

import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .policy import score_candidate_masks
from .qwen import ProxySelector, SharedVisionAdapter, apply_lora
from .readouts import CompactSelector
from .supervision import grouped_loss, question_loss


def build_selector(config, *, answerer_config, selector_model=None):
    config.validate()
    torch.manual_seed(config.seed)
    common = dict(
        width=config.width,
        heads=config.heads,
        slots=config.slots,
        retrieval_dim=config.retrieval_dim,
        head2=config.head2,
    )
    if config.architecture == "compact":
        return CompactSelector(
            answerer_config.text_config.hidden_size,
            answerer_config.text_config.vocab_size,
            config.metadata_dim,
            reread=config.compact_reread,
            **common,
        )
    if selector_model is None:
        raise ValueError(
            "Supply an explicitly loaded local selector; no automatic model loading"
        )
    model = apply_lora(
        selector_model,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        targets=config.lora_targets,
    )
    adapter = None
    if config.vision_mode == "shared_adapter":
        adapter = SharedVisionAdapter(
            answerer_config.text_config.hidden_size,
            model.config.text_config.hidden_size,
            answerer_config.vision_config.deepstack_visual_indexes,
            model.config.vision_config.deepstack_visual_indexes,
            config.stream_map,
        )
    return ProxySelector(
        model,
        config.metadata_dim,
        mode=config.architecture,
        vision_mode=config.vision_mode,
        adapter=adapter,
        layer=config.representation_layer,
        **common,
    ).to(device=model.model.get_input_embeddings().weight.device)


def objective(
    model,
    examples,
    banks,
    config,
    *,
    proxy_prompts=None,
    native_layouts=None,
    ledger=None,
):
    """One encoder call/question, then all matched mask preferences; supports ragged batches."""
    config.validate()
    if (getattr(model, "correction", None) is not None) != config.head2:
        raise ValueError("Model/configuration Head 2 mismatch")
    if len(examples) != len(banks) or not examples:
        raise ValueError("Examples and teacher banks must align")
    if config.stage1.epsilon is None or config.stage1.margin is None:
        raise ValueError(
            "Choose audited epsilon and margin explicitly before computing the loss"
        )
    losses, families = [], []
    for i, (example, bank) in enumerate(zip(examples, banks)):
        example.validate()
        if (
            config.stage1.answerer_revision is not None
            and example.identity.answerer_revision != config.stage1.answerer_revision
        ):
            raise ValueError("Wrong teacher answerer revision")
        if config.stage1.budget is not None and example.budget != config.stage1.budget:
            raise ValueError("Wrong training budget")
        bank.validate(example)
        if (
            example.retrieval is not None
            and example.retrieval.schema != config.retrieval_schema
        ):
            raise ValueError("Retrieval feature schema mismatch")
        use_correction = config.head2 and config.head2_weight > 0
        if config.architecture == "compact":
            output = model(example, return_encoding=use_correction, ledger=ledger)
        else:
            if proxy_prompts is None:
                raise ValueError("Proxy prompts required")
            output = model(
                example,
                proxy_prompts[i],
                native_layout=None if native_layouts is None else native_layouts[i],
                return_encoding=use_correction,
                ledger=ledger,
            )
        correction_scores = None
        scores = output
        if use_correction:
            scores = output.scores
            correction_scores = score_candidate_masks(
                output, bank.masks, model.correction, ledger=ledger
            ).correction
        losses.append(
            question_loss(
                scores,
                bank,
                mode=config.supervision,
                epsilon=config.stage1.epsilon,
                margin=config.stage1.margin,
                temperature=config.temperature,
                correction_scores=correction_scores,
                auxiliary_weight=config.head2_weight,
            )
        )
        families.append((example.identity.source, bank.family))
    return (
        None
        if all(loss is None for loss in losses)
        else grouped_loss(losses, families, weighting=config.weighting)
    )


def make_optimizer(model, config):
    config.validate(execution=True)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable selector parameters")
    return torch.optim.AdamW(
        params, lr=config.learning_rate, weight_decay=config.weight_decay
    )


def train_epoch(model, batches, optimizer, config, *, execute_training=False):
    """Explicit epoch operation; CPU preparation checks never execute optimizer steps."""
    if not execute_training:
        raise PermissionError("Training requires explicit execution authorization")
    config.validate(execution=True)
    model.train()
    reference = next(model.parameters())
    if reference.is_cuda:
        torch.cuda.synchronize(reference.device)
    started = time.perf_counter()
    total, steps, questions, masks, unordered_batches = 0.0, 0, 0, 0, 0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, config=config, **batch)
        questions += len(batch["examples"])
        masks += sum(len(bank.outcomes) for bank in batch["banks"])
        if loss is None:
            unordered_batches += 1
            continue
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        total += float(loss.detach())
        steps += 1
    if reference.is_cuda:
        torch.cuda.synchronize(reference.device)
    return {
        "seconds": time.perf_counter() - started,
        "question_exposures": questions,
        "mask_exposures": masks,
        "steps": steps,
        "unordered_batches": unordered_batches,
        "mean_loss": total / steps if steps else None,
    }


def save_checkpoint(path, model, config, *, step, optimizer=None):
    """Save only trainable parameters: LoRA, adapter, reader and head; not frozen model weights."""
    path = Path(path)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}
    record = {
        "config": asdict(config),
        "config_identity": config.identity,
        "step": step,
        "trainable": state,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        "optimizer": None if optimizer is None else optimizer.state_dict(),
    }
    with path.open("xb") as stream:
        torch.save(record, stream)
        stream.flush()
        os.fsync(stream.fileno())


def restore_checkpoint(path, model, config, *, optimizer=None, restore_rng=True):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record["config_identity"] != config.identity:
        raise ValueError("Checkpoint/configuration identity mismatch")
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(record["trainable"]) != expected:
        raise ValueError("Trainable parameter identity mismatch")
    model.load_state_dict(record["trainable"], strict=False)
    if optimizer is not None:
        if record["optimizer"] is None:
            raise ValueError("No optimizer state in checkpoint")
        optimizer.load_state_dict(record["optimizer"])
    if restore_rng:
        torch.set_rng_state(record["torch_rng"])
        random.setstate(record["python_rng"])
        if record.get("cuda_rng") is not None:
            if not torch.cuda.is_initialized() or len(record["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("Resume needs the same initialized CUDA device topology")
            torch.cuda.set_rng_state_all(record["cuda_rng"])
    return record["step"]
