"""Cached pages and layout boxes -> original Qwen3 actions and complete model inputs."""

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from .contracts import (
    InstanceIdentity,
    RegionLayout,
    RetrievalFeatures,
    SelectorInputs,
    fingerprint,
)
from .cost import execution_ledger, measure
from .data import file_hash, load_example, save_example
from .policy import achievable_budget
from .qwen import PackedPrompt, prepare_prompt
from .records import QuestionRecord
from .regions import partition_tokens
from .storage import publish, read_record, seal_files, verify_files


@dataclass(frozen=True)
class PageAsset:
    page_id: str
    image_path: str
    image_sha256: str
    layout_path: str
    layout_sha256: str
    rendering_identity: str

    def read(self):
        if not self.page_id or not self.rendering_identity:
            raise ValueError("Page identity and rendering contract are required")
        if (
            file_hash(self.image_path) != self.image_sha256
            or file_hash(self.layout_path) != self.layout_sha256
        ):
            raise ValueError("Changed page image or layout")
        layout = json.loads(Path(self.layout_path).read_text())
        if layout["schema"] != "docprune-layout-boxes-v1" or layout["page_id"] != self.page_id:
            raise ValueError("Wrong layout/page identity")
        if layout["image_sha256"] != self.image_sha256 or not layout.get("provenance"):
            raise ValueError("Layout must bind the rendered image and extractor provenance")
        regions = layout["regions"]
        if len({r["id"] for r in regions}) != len(regions):
            raise ValueError("Duplicate semantic region IDs")
        for r in regions:
            box = r["bbox"]
            if len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in box):
                raise ValueError("Layout boxes must use normalized page coordinates")
            if box[2] <= box[0] or box[3] <= box[1] or not r["id"]:
                raise ValueError("Invalid semantic box")
        with Image.open(self.image_path) as source:
            if source.getexif().get(274, 1) != 1:
                raise ValueError("Normalize image orientation before freezing layout coordinates")
            image = source.convert("RGB")
        return image, regions


@dataclass(frozen=True)
class PreparationConfig:
    answerer_revision: str
    selector_revision: str
    answerer_processor: str
    selector_processor: str
    retrieval_identity: str
    scoring_identity: str
    budget: int
    fallback_tile_size: int = 8
    max_fallback_tokens: int = 128
    partition_policy: str = "largest_positive_overlap_then_reading_order_v1"
    include_native: bool = True
    precision: str = "float32"

    def validate(self):
        if any(not v for k, v in asdict(self).items() if k != "include_native") or self.budget < 1:
            raise ValueError("Freeze all preparation identities and a positive budget")
        if self.fallback_tile_size < 1 or self.max_fallback_tokens < 1:
            raise ValueError("Invalid fallback limits")
        if self.partition_policy != "largest_positive_overlap_then_reading_order_v1":
            raise ValueError("Unsupported partition policy")
        if self.precision not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unknown frozen vision precision")
        return self


def grid_geometry(grid, merge):
    """Page-major merged-token order; temporal,row,column and normalized footprints."""
    grid = torch.as_tensor(grid, dtype=torch.long).cpu()
    if grid.ndim != 2 or grid.shape[1] != 3 or (grid <= 0).any() or merge < 1:
        raise ValueError("Invalid visual grid")
    if (grid[:, 0] != 1).any() or (grid[:, 1:] % merge).any():
        raise ValueError("Document images require one temporal slice and complete merge cells")
    pages, coordinates, boxes = [], [], []
    for page, (_, height, width) in enumerate(grid.tolist()):
        h, w = height // merge, width // merge
        for row in range(h):
            for col in range(w):
                pages.append(page)
                coordinates.append((0, row, col))
                boxes.append((col / w, row / h, (col + 1) / w, (row + 1) / h))
    return torch.tensor(pages), torch.tensor(coordinates).float(), torch.tensor(boxes).float()


def overlap(a, b):
    lower = torch.maximum(a[:, None, :2], b[None, :, :2])
    upper = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    return (upper - lower).clamp_min(0).prod(-1)


def region_layout(grid, merge, page_assets, regions, config):
    page, coordinates, footprints = grid_geometry(grid, merge)
    if len(page_assets) != len(grid) or len(regions) != len(grid):
        raise ValueError("Admitted pages, layouts and visual grids differ")
    memberships, covered = [], 0
    for i, (asset, boxes) in enumerate(zip(page_assets, regions)):
        positions = torch.where(page == i)[0]
        if not boxes:
            continue
        intersections = overlap(footprints[positions], torch.tensor([r["bbox"] for r in boxes]))
        area, assigned = intersections.max(dim=1)
        covered += int((area > 0).sum())
        for j, r in enumerate(boxes):
            memberships.append(
                (asset.page_id + ":" + r["id"], positions[(assigned == j) & (area > 0)])
            )
    layout = partition_tokens(
        page,
        coordinates,
        memberships,
        tile_size=config.fallback_tile_size,
        max_fallback=config.max_fallback_tokens,
    )
    return (
        layout,
        footprints,
        {
            "visual_tokens": page.numel(),
            "semantic_covered_tokens": covered,
            "fallback_tokens": page.numel() - covered,
            "fallback_fraction": (page.numel() - covered) / page.numel(),
            "regions": len(layout.region_ids),
            "largest_region_tokens": int(layout.costs.max()),
            "oversized_regions": int((layout.costs > config.budget).sum()),
            "achievable_budget": (
                achievable_budget(layout.costs, config.budget)
                if int(layout.costs.min()) <= config.budget
                else 0
            ),
        },
    )


def native_action_layout(original, original_boxes, native_grid, merge):
    """Map a different native grid onto the original owned action footprints.

    No action may silently disappear. A too-coarse native grid fails explicitly.
    """
    page, coords, boxes = grid_geometry(native_grid, merge)
    owner = torch.full((page.numel(),), -1, dtype=torch.long)
    for p in page.unique().tolist():
        target = torch.where(page == p)[0]
        source = torch.where(original.page.cpu() == p)[0]
        areas = overlap(boxes[target], original_boxes[source])
        by_region = torch.zeros(len(target), len(original.region_ids))
        by_region.scatter_add_(1, original.owner.cpu()[source][None].expand(len(target), -1), areas)
        positive, assigned = by_region.max(dim=1)
        if not (positive > 0).all():
            raise ValueError("Native grid has unaligned page content")
        owner[target] = assigned
    if (torch.bincount(owner, minlength=len(original.region_ids)) == 0).any():
        raise ValueError("Native grid cannot represent every original action")
    return RegionLayout(
        original.region_ids, owner, page, coords, original.metadata.cpu()
    ).validate()


def map_retrieval_profiles(
    layout, footprints, patch_pages, patch_boxes, query_patch_scores, query_vectors, *, schema
):
    """Optional cached query-associated MaxSim profile over actual owned footprints.

    No index access, fresh retrieval, area normalization, or hard region exclusion.
    Query vectors retain query-entry identity; the explicit schema describes them.
    """
    if not schema or query_patch_scores.ndim != 2 or query_vectors.ndim != 2:
        raise ValueError("Explicit query-associated feature schema required")
    if query_patch_scores.shape != (query_vectors.shape[0], len(patch_boxes)) or len(
        patch_pages
    ) != len(patch_boxes):
        raise ValueError("Patch/profile dimensions differ")
    if patch_boxes.ndim != 2 or patch_boxes.shape[1] != 4 or patch_pages.ndim != 1:
        raise ValueError("Invalid cached patch geometry")
    if (
        not torch.isfinite(patch_boxes).all()
        or (patch_boxes < 0).any()
        or (patch_boxes > 1).any()
        or (patch_boxes[:, 2:] <= patch_boxes[:, :2]).any()
    ):
        raise ValueError("Patch boxes must be positive normalized rectangles")
    if (
        not torch.isin(patch_pages.cpu(), layout.page.cpu().unique()).all()
        or not torch.isfinite(query_patch_scores).all()
        or not torch.isfinite(query_vectors).all()
    ):
        raise ValueError("Cached profile has invalid pages or nonfinite features")
    scores, vectors = query_patch_scores.cpu(), query_vectors.cpu()
    rows = []
    for i in range(len(layout.region_ids)):
        tokens = layout.owner.cpu() == i
        p = int(layout.page.cpu()[tokens][0])
        indices = torch.where(patch_pages.cpu() == p)[0]
        touched = overlap(footprints[tokens], patch_boxes.cpu()[indices]).sum(0) > 0
        if not touched.any():
            raise ValueError("No cached retrieval patch touches an original action")
        profile = scores[:, indices[touched]].max(-1).values
        rows.append(torch.cat((profile[:, None], vectors), -1))
    result = RetrievalFeatures(torch.stack(rows), schema)
    result.validate(len(rows))
    return result


@dataclass
class PreparedDocument:
    inputs: SelectorInputs
    answerer_prompt: PackedPrompt
    shared_prompt: PackedPrompt | None
    native_prompt: PackedPrompt | None
    native_layout: RegionLayout | None
    record: QuestionRecord
    preparation: dict
    audit: dict

    def validate(self):
        self.inputs.validate()
        self.record.validate()
        identity = self.inputs.identity
        if (
            identity.source,
            identity.document_family,
            identity.question_id,
            identity.question_sha256,
        ) != (
            self.record.source,
            self.record.document_family,
            self.record.question_id,
            fingerprint(self.record.question),
        ):
            raise ValueError("Prepared document and question identity differ")
        if identity.scoring != fingerprint(
            (
                self.preparation["scoring_identity"],
                self.record.annotation_identity,
                asdict(self.record.answer),
            )
        ):
            raise ValueError("Prepared answer/scoring identity differs")
        if identity.answerer_revision != self.preparation["answerer_revision"]:
            raise ValueError("Prepared answerer revision differs")
        if not torch.equal(
            self.inputs.question_ids.cpu(),
            self.answerer_prompt.input_ids[0, self.answerer_prompt.question_positions].cpu(),
        ):
            raise ValueError("Compact question tokens differ from the original answerer question")
        if tuple(p["page_id"] for p in self.preparation["page_assets"]) != identity.ordered_pages:
            raise ValueError("Prepared page order differs")
        if not torch.equal(
            self.inputs.vision.grid_thw.cpu(), self.answerer_prompt.image_grid_thw.cpu()
        ):
            raise ValueError("Answerer prompt and memory grids differ")
        if self.shared_prompt is not None and not torch.equal(
            self.shared_prompt.image_grid_thw.cpu(), self.inputs.vision.grid_thw.cpu()
        ):
            raise ValueError("Shared selector prompt and memory grids differ")
        return self

    def to(self, device):
        def tensor(t):
            return None if t is None else t.to(device)

        def prompt(p):
            return (
                None
                if p is None
                else PackedPrompt(
                    *(
                        tensor(getattr(p, k))
                        for k in (
                            "input_ids",
                            "question_positions",
                            "image_grid_thw",
                            "pixel_values",
                        )
                    )
                )
            )

        def layout(x):
            return (
                None
                if x is None
                else replace(
                    x,
                    owner=tensor(x.owner),
                    page=tensor(x.page),
                    coordinates=tensor(x.coordinates),
                    metadata=tensor(x.metadata),
                )
            )

        x = self.inputs
        v = replace(
            x.vision,
            merged=tensor(x.vision.merged),
            deepstack=tuple(tensor(s) for s in x.vision.deepstack),
            grid_thw=tensor(x.vision.grid_thw),
        )
        r = (
            None
            if x.retrieval is None
            else replace(
                x.retrieval, values=tensor(x.retrieval.values), valid=tensor(x.retrieval.valid)
            )
        )
        return replace(
            self,
            inputs=replace(
                x,
                layout=layout(x.layout),
                vision=v,
                retrieval=r,
                question_ids=tensor(x.question_ids),
            ),
            answerer_prompt=prompt(self.answerer_prompt),
            shared_prompt=prompt(self.shared_prompt),
            native_prompt=prompt(self.native_prompt),
            native_layout=layout(self.native_layout),
        )


def prepare_document(
    record,
    pages,
    config,
    *,
    answerer,
    answerer_processor,
    selector_processor=None,
    selector_config=None,
    retrieval=None,
):
    """Explicit future model operation; tests use tiny random CPU models/processors."""
    record.validate()
    config.validate()
    if len({p.page_id for p in pages}) != len(pages) or not set(p.page_id for p in pages).issubset(
        record.page_ids
    ):
        raise ValueError("Admitted pages must be unique members of the question scope")
    if str(next(answerer.parameters()).dtype) != "torch." + config.precision:
        raise ValueError("Answerer precision differs from preparation contract")
    ledger = execution_ledger(next(answerer.parameters()), cache_mode="preparing_frozen_vision")
    with ledger.measure("preprocessing"):
        images, regions = zip(*(p.read() for p in pages))
        ap = prepare_prompt(answerer_processor, record.question, images)
    device = next(answerer.parameters()).device
    ap = PackedPrompt(
        *(
            None if x is None else x.to(device)
            for x in (ap.input_ids, ap.question_positions, ap.image_grid_thw, ap.pixel_values)
        )
    )
    ap.validate(answerer.model.config)
    layout, footprints, audit = region_layout(
        ap.image_grid_thw.cpu(),
        answerer.model.config.vision_config.spatial_merge_size,
        pages,
        regions,
        config,
    )
    identity = InstanceIdentity(
        record.source,
        record.document_family,
        record.question_id,
        fingerprint(record.question),
        tuple(p.page_id for p in pages),
        fingerprint(
            [
                (p.image_sha256, p.rendering_identity, config.answerer_processor, config.precision)
                for p in pages
            ]
        ),
        fingerprint(
            (
                config.partition_policy,
                config.fallback_tile_size,
                config.max_fallback_tokens,
                [(p.page_id, p.layout_sha256) for p in pages],
            )
        ),
        config.answerer_revision,
        config.retrieval_identity,
        fingerprint((config.scoring_identity, record.annotation_identity, asdict(record.answer))),
    )
    with measure(ledger, "answerer_vision", next(answerer.parameters())):
        memory = answerer.vision(ap, identity.key)
    inputs = SelectorInputs(
        identity,
        layout,
        memory,
        torch.tensor(
            answerer_processor.tokenizer.encode(record.question, add_special_tokens=False)
        ),
        config.budget,
    )
    if retrieval is not None:
        inputs.retrieval = map_retrieval_profiles(layout, footprints, **retrieval)
    shared = native = native_layout = None
    if selector_processor is not None:
        if selector_config is None:
            raise ValueError("Selector geometry requires its checkpoint configuration")
        with ledger.measure("selector_preprocessing"):
            native = prepare_prompt(selector_processor, record.question, images)
        native.validate(selector_config)
        # Use selector text/control serialization while matching the shared source grid.
        if answerer.model.config.image_token_id != selector_config.image_token_id:
            raise ValueError("Shared prompt requires verified matching special token IDs")
        # Actual processor serialization is authoritative. If grids differ, explicitly
        # expand each native image run to the shared grid count without altering controls.
        ids, positions, cursor = [], [], 0
        old = native.input_ids[0].tolist()
        counts = (
            ap.image_grid_thw.cpu().prod(-1)
            // answerer.model.config.vision_config.spatial_merge_size**2
        ).tolist()
        image_id = selector_config.image_token_id
        i = 0
        while i < len(old):
            if old[i] == image_id:
                j = i
                while j < len(old) and old[j] == image_id:
                    j += 1
                ids.extend([image_id] * counts[cursor])
                cursor += 1
                i = j
            else:
                if i in native.question_positions.tolist():
                    positions.append(len(ids))
                ids.append(old[i])
                i += 1
        if cursor != len(pages):
            raise ValueError("Unexpected selector image serialization")
        shared = PackedPrompt(
            torch.tensor([ids]), torch.tensor(positions), ap.image_grid_thw.cpu(), None
        )
        shared.validate(selector_config)
        if config.include_native:
            native_layout = native_action_layout(
                layout,
                footprints,
                native.image_grid_thw,
                selector_config.vision_config.spatial_merge_size,
            )
        else:
            native = None
    ledger.capture_process_memory(memory.merged)
    audit["cost"] = ledger.summary()
    return (
        PreparedDocument(
            inputs.validate(),
            ap,
            shared,
            native,
            native_layout,
            record,
            {**asdict(config), "page_assets": [asdict(p) for p in pages]},
            audit,
        )
        .to("cpu")
        .validate()
    )


def _write_document(root, document):
    document.validate()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    save_example(root / "input", document.inputs)
    tensors, prompts = {}, []
    for name in ("answerer_prompt", "shared_prompt", "native_prompt"):
        p = getattr(document, name)
        if p is None:
            continue
        prompts.append(name)
        for key in ("input_ids", "question_positions", "image_grid_thw", "pixel_values"):
            v = getattr(p, key)
            if v is not None:
                tensors[name + "." + key] = v
    if document.native_layout is not None:
        for key in ("owner", "page", "coordinates", "metadata"):
            tensors["native_layout." + key] = getattr(document.native_layout, key)
    save_file(
        {k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()},
        root / "prompts.safetensors",
    )
    publish(
        root / "document.json",
        {
            "schema": "docprune-prepared-document-v1",
            "prompts": prompts,
            "native_layout": document.native_layout is not None,
            "question": document.record.to_dict(),
            "preparation": document.preparation,
            "audit": document.audit,
        },
    )
    seal_files(
        root,
        [
            "input/example.json",
            "input/tensors.safetensors",
            "input/manifest.json",
            "prompts.safetensors",
            "document.json",
        ],
    )


def save_document(root, document):
    """Publish only a fully sealed directory; incomplete writes never look resumable."""
    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        raise FileExistsError(root)
    with tempfile.TemporaryDirectory(prefix=".pending-bundle-", dir=root.parent) as pending:
        directory = Path(pending) / "bundle"
        _write_document(directory, document)
        if root.exists():
            raise FileExistsError(root)
        os.rename(directory, root)


def load_document(root, *, device="cpu"):
    root = Path(root)
    verify_files(root)
    record = read_record(root / "document.json")
    if record["schema"] != "docprune-prepared-document-v1":
        raise ValueError("Unknown prepared document schema")
    inputs, _ = load_example(root / "input")
    data = load_file(root / "prompts.safetensors")
    prompts = {
        name: PackedPrompt(
            *(
                data.get(name + "." + k)
                for k in ("input_ids", "question_positions", "image_grid_thw", "pixel_values")
            )
        )
        if name in record["prompts"]
        else None
        for name in ("answerer_prompt", "shared_prompt", "native_prompt")
    }
    native = (
        RegionLayout(
            inputs.layout.region_ids,
            *(data["native_layout." + k] for k in ("owner", "page", "coordinates", "metadata")),
        )
        if record["native_layout"]
        else None
    )
    return (
        PreparedDocument(
            inputs,
            **prompts,
            native_layout=native,
            record=QuestionRecord.from_dict(record["question"]),
            preparation=record["preparation"],
            audit=record["audit"],
        )
        .to(device)
        .validate()
    )


def import_mineru_layout(raw_path, *, local_page_index, page_asset, output):
    """Convert existing MinerU middle.json; does not run or download a segmenter."""
    raw = json.loads(Path(raw_path).read_text())
    if not raw.get("_backend") or not raw.get("_version_name"):
        raise ValueError("MinerU output must declare backend/version")
    pages = [p for p in raw["pdf_info"] if p["page_idx"] == local_page_index]
    if len(pages) != 1:
        raise ValueError("Missing or duplicate MinerU page")
    page = pages[0]
    width, height = page["page_size"]
    if width <= 0 or height <= 0:
        raise ValueError("Invalid MinerU page dimensions")
    value = {
        "schema": "docprune-layout-boxes-v1",
        "page_id": page_asset["page_id"],
        "image_sha256": page_asset["image_sha256"],
        "provenance": {
            "raw_sha256": file_hash(raw_path),
            "backend": raw["_backend"],
            "version": raw["_version_name"],
            "local_page_index": local_page_index,
        },
        "regions": [
            {
                "id": f"region-{i}",
                "type": r["type"],
                "bbox": [
                    r["bbox"][0] / width,
                    r["bbox"][1] / height,
                    r["bbox"][2] / width,
                    r["bbox"][3] / height,
                ],
            }
            for i, r in enumerate(page["para_blocks"])
        ],
    }
    with Path(output).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    return value


def render_pdf_page(pdf, page_index, output_directory, *, dpi):
    """Explicit CPU rasterization of a local PDF page; source and output never overwritten."""
    binary = shutil.which("pdftoppm")
    if binary is None:
        raise RuntimeError("Install Poppler's pdftoppm or supply cached rendered page images")
    if page_index < 0 or dpi <= 0:
        raise ValueError("Invalid zero-based page index or DPI")
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=False)
    version = subprocess.run(
        [binary, "-v"], capture_output=True, text=True, check=True
    ).stderr.strip()
    subprocess.run(
        [
            binary,
            "-f",
            str(page_index + 1),
            "-l",
            str(page_index + 1),
            "-r",
            str(dpi),
            "-singlefile",
            "-png",
            str(Path(pdf).resolve()),
            str(root / "page"),
        ],
        check=True,
        capture_output=True,
    )
    receipt = {
        "pdf_sha256": file_hash(pdf),
        "source_page_index": page_index,
        "dpi": dpi,
        "renderer": version,
        "image_sha256": file_hash(root / "page.png"),
    }
    publish(root / "rendering.json", receipt)
    return receipt
