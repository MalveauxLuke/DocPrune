"""Sealed local examples and teacher banks; safetensors prevents pickle execution."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from safetensors.torch import load_file, save_file

from .contracts import (
    InstanceIdentity,
    RegionLayout,
    RetrievalFeatures,
    SelectorInputs,
    VisualMemory,
)
from .supervision import Outcome, TeacherBank


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_example(directory, inputs, bank=None):
    """Fresh directory only; completion manifest is published last."""
    inputs.validate()
    if bank is not None:
        bank.validate(inputs)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    tensors = dict(
        owner=inputs.layout.owner,
        page=inputs.layout.page,
        coordinates=inputs.layout.coordinates,
        metadata=inputs.layout.metadata,
        merged=inputs.vision.merged,
        grid_thw=inputs.vision.grid_thw,
        question_ids=inputs.question_ids,
    )
    for i, stream in enumerate(inputs.vision.deepstack):
        tensors[f"deepstack_{i}"] = stream
    record = {
        "schema": "docprune-stage2-example-v1",
        "identity": asdict(inputs.identity),
        "instance_key": inputs.identity.key,
        "region_ids": inputs.layout.region_ids,
        "vision_provenance": inputs.vision.provenance,
        "deepstack_count": len(inputs.vision.deepstack),
        "budget": inputs.budget,
        "retrieval_schema": None,
        "bank": None,
    }
    if inputs.retrieval is not None:
        record["retrieval_schema"] = inputs.retrieval.schema
        tensors["retrieval"] = inputs.retrieval.values
        if inputs.retrieval.valid is not None:
            tensors["retrieval_valid"] = inputs.retrieval.valid
    if bank is not None:
        tensors["masks"] = bank.masks
        record["bank"] = {k: v for k, v in asdict(bank).items() if k != "masks"}
    # Clone breaks tied storage, and no trainable graph is persisted.
    save_file(
        {k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()},
        root / "tensors.safetensors",
    )
    (root / "example.json").write_text(
        json.dumps(record, sort_keys=True, allow_nan=False)
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {n: file_hash(root / n) for n in ("example.json", "tensors.safetensors")}
        )
    )


def load_example(directory, *, device="cpu"):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if set(manifest) != {"example.json", "tensors.safetensors"}:
        raise ValueError("Unexpected manifest entries")
    for name, expected in manifest.items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Changed sealed input: {name}")
    record = json.loads((root / "example.json").read_text())
    if record["schema"] != "docprune-stage2-example-v1":
        raise ValueError("Unknown input schema")
    identity_fields = record["identity"]
    identity_fields["ordered_pages"] = tuple(identity_fields["ordered_pages"])
    identity = InstanceIdentity(**identity_fields)
    if identity.key != record["instance_key"]:
        raise ValueError("Instance identity mismatch")
    data = load_file(root / "tensors.safetensors", device=device)
    layout = RegionLayout(
        tuple(record["region_ids"]),
        *(data[k] for k in ("owner", "page", "coordinates", "metadata")),
    )
    vision = VisualMemory(
        data["merged"],
        tuple(data[f"deepstack_{i}"] for i in range(record["deepstack_count"])),
        data["grid_thw"],
        record["vision_provenance"],
    )
    retrieval = (
        None
        if record["retrieval_schema"] is None
        else RetrievalFeatures(
            data["retrieval"], record["retrieval_schema"], data.get("retrieval_valid")
        )
    )
    example = SelectorInputs(
        identity, layout, vision, data["question_ids"], record["budget"], retrieval
    ).validate()
    bank = None
    if record["bank"] is not None:
        b = record["bank"]
        bank = TeacherBank(
            b["instance_key"],
            data["masks"],
            tuple(Outcome(**o) for o in b["outcomes"]),
            Outcome(**b["reference"]),
            b["baseline_correct"],
            b["proposal_uses_s"],
            b["family"],
        )
        bank.validate(example)
    return example, bank
