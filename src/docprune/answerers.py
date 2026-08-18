"""Stock and DocPrune Qwen2-VL answerers for the M3DocRAG boundary."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from docprune.benchmark_config import MAX_NEW_TOKENS, SHORT_ANSWER_TEMPLATE
from docprune.btp import background_scores, threshold_keep_mask
from docprune.config import DocPruneConfig, PagePruningConfig, ReconstructionDefaults
from docprune.indexing import colpali_uint8_raster
from docprune.m3docrag import AnswerOutput
from docprune.pipeline import prepare_qa_pruning_masks
from docprune.processor_probe import resolve_colpali_visual_mapping
from docprune.qwen2vl.model import DocPruneQwen2VL, PruningTrace, VisionPruningMasks
from docprune.qwen2vl.preprocessing import (
    PreparedQwenPage,
    prepare_qwen_page,
    prepared_raster_image,
)


def _value(container: object, name: str) -> object:
    if isinstance(container, Mapping):
        if name not in container:
            raise ValueError(f"processor output is missing {name}")
        return container[name]
    if not hasattr(container, name):
        raise ValueError(f"processor output is missing {name}")
    return getattr(container, name)


def _model_device(model: object) -> torch.device | None:
    parameters = getattr(model, "parameters", None)
    if parameters is None:
        return None
    try:
        return next(parameters()).device
    except StopIteration:
        return None


def _move_batch(batch: object, device: torch.device | None) -> dict[str, object]:
    if not isinstance(batch, Mapping):
        raise ValueError("Qwen processor must return a mapping")
    return {
        name: value.to(device) if device is not None and isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def _move_colpali_batch(batch: Mapping[str, object], model: object) -> dict[str, object]:
    """Move ColPali tensors while matching only vision pixels to model dtype."""

    device = _model_device(model)
    vision = getattr(getattr(model, "model", model), "vision_tower", None)
    dtype = None
    parameters = getattr(vision, "parameters", None)
    if parameters is not None:
        try:
            dtype = next(parameters()).dtype
        except StopIteration:
            pass
    moved: dict[str, object] = {}
    for name, value in batch.items():
        if not isinstance(value, torch.Tensor):
            moved[name] = value
        elif device is None:
            moved[name] = value
        elif name == "pixel_values" and dtype is not None:
            moved[name] = value.to(device=device, dtype=dtype)
        else:
            moved[name] = value.to(device=device)
    return moved


def _prompt(processor: object, page_count: int, question: str) -> str:
    content = [{"type": "image", "image": "dummy_content"} for _ in range(page_count)]
    content.append({"type": "text", "text": SHORT_ANSWER_TEMPLATE.replace("$question", question)})
    messages = [{"role": "user", "content": content}]
    apply = getattr(processor, "apply_chat_template", None)
    if not callable(apply):
        raise ValueError("Qwen processor must expose apply_chat_template")
    return str(apply(messages, tokenize=False, add_generation_prompt=True))


def _prepare_batch(processor: object, images: Sequence[object], question: str) -> dict[str, object]:
    if not images:
        raise ValueError("at least one page image is required")
    prompt = _prompt(processor, len(images), question)
    call = getattr(processor, "__call__", None)
    if not callable(call):
        raise ValueError("Qwen processor is not callable")
    return dict(call(text=[prompt], images=list(images), padding=True, return_tensors="pt"))


def _grid(batch: Mapping[str, object]) -> torch.Tensor:
    grid = torch.as_tensor(_value(batch, "image_grid_thw"), dtype=torch.long)
    if grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError("Qwen image_grid_thw must have shape [pages, 3]")
    return grid


def _merged_count(grid: torch.Tensor, merge_size: int = 2) -> int:
    if torch.any(grid[:, 0] != 1) or torch.any(grid[:, 1:] % merge_size != 0):
        raise ValueError("Qwen page grids must be temporal-one and merge divisible")
    return int((grid.prod(dim=1) // (merge_size**2)).sum().item())


def _image_token_id(model: object, processor: object) -> int:
    for owner in (getattr(model, "config", None), processor, getattr(processor, "tokenizer", None)):
        for name in ("image_token_id", "image_token_index"):
            value = getattr(owner, name, None) if owner is not None else None
            if value is not None:
                return int(value)
    raise ValueError("Qwen image token ID could not be resolved")


def _decode_new_tokens(processor: object, generated: torch.Tensor, prompt_length: int) -> str:
    if generated.ndim != 2 or generated.shape[0] != 1:
        raise ValueError("Qwen generation must return shape [1, sequence]")
    suffix = generated[:, prompt_length:]
    decode = getattr(processor, "batch_decode", None)
    if not callable(decode):
        raise ValueError("Qwen processor must expose batch_decode")
    decoded = decode(
        suffix,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not decoded:
        raise ValueError("Qwen processor returned no decoded answer")
    return str(decoded[0]).strip()


def _all_kept_trace(grid: torch.Tensor) -> PruningTrace:
    count = _merged_count(grid)
    return PruningTrace(count, count, count, count, None)


def _validate_placeholder_count(model: object, processor: object, input_ids: torch.Tensor, grid: torch.Tensor) -> None:
    expected = _merged_count(grid)
    observed = int((input_ids == _image_token_id(model, processor)).sum().item())
    if observed != expected:
        raise ValueError(f"Qwen image placeholders ({observed}) do not match merge groups ({expected})")


@dataclass
class AllKeptQwenAnswerer:
    """Stock Qwen2-VL generation with the M3DocRAG short-answer prompt."""

    model: object
    processor: object
    max_new_tokens: int = MAX_NEW_TOKENS

    def answer(self, images: Sequence[object], question: str) -> AnswerOutput:
        started = time.perf_counter()
        batch = _prepare_batch(self.processor, images, question)
        input_ids = torch.as_tensor(_value(batch, "input_ids"), dtype=torch.long)
        grid = _grid(batch)
        _validate_placeholder_count(self.model, self.processor, input_ids, grid)
        batch = _move_batch(batch, _model_device(self.model))
        with torch.no_grad():
            generated = self.model.generate(
                **batch,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        answer = _decode_new_tokens(self.processor, torch.as_tensor(generated), input_ids.shape[1])
        return AnswerOutput(answer, _all_kept_trace(grid), time.perf_counter() - started)


class DocPruneQwenAnswerer(AllKeptQwenAnswerer):
    """Qwen generation through the sparse DocPrune vision/decoder adapter."""

    def __init__(
        self,
        model: object,
        processor: object,
        *,
        colpali_model: object | None = None,
        colpali_processor: object | None = None,
        page_config: PagePruningConfig | None = None,
        reconstruction: ReconstructionDefaults | None = None,
        comprehension_threshold: float | None = None,
        attention_threshold: float | None = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:
        super().__init__(model=model, processor=processor, max_new_tokens=max_new_tokens)
        self.colpali_model = colpali_model
        self.colpali_processor = colpali_processor
        self.page_config = page_config
        self.reconstruction = reconstruction or ReconstructionDefaults()
        self.comprehension_threshold = comprehension_threshold
        self.attention_threshold = attention_threshold

    def _effective_page_config(self, page_count: int) -> PagePruningConfig | None:
        if self.page_config is not None:
            return self.page_config
        if self.colpali_model is not None and self.colpali_processor is not None:
            return DocPruneConfig.paper_defaults().for_pages(page_count)
        return None

    def _colpali_page_embeddings(
        self,
        images: Sequence[object],
        question: str,
        page_config: PagePruningConfig | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor] | None:
        if self.colpali_model is None or self.colpali_processor is None:
            return None
        process_images = getattr(self.colpali_processor, "process_images", None)
        process_queries = getattr(self.colpali_processor, "process_queries", None)
        if not callable(process_images) or not callable(process_queries):
            raise ValueError("pinned ColPali processor must expose process_images and process_queries")
        query_batch = process_queries([question])
        if not isinstance(query_batch, Mapping):
            raise ValueError("ColPali processor outputs must be mappings")
        with torch.no_grad():
            query = self.colpali_model(**_move_colpali_batch(query_batch, self.colpali_model))
        query = torch.as_tensor(query if isinstance(query, torch.Tensor) else query[0])
        query_attention = torch.as_tensor(_value(query_batch, "attention_mask"), device=query.device)
        if query.ndim != 3 or query.shape[0] != 1:
            raise ValueError("ColPali query embeddings have unsupported shape")
        image_token_id = getattr(self.colpali_processor, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(getattr(self.colpali_processor, "tokenizer", None), "image_token_id", None)
        tokenizer = getattr(self.colpali_processor, "tokenizer", None)
        if image_token_id is None and callable(getattr(tokenizer, "convert_tokens_to_ids", None)):
            candidate = tokenizer.convert_tokens_to_ids("<image>")
            unknown = getattr(tokenizer, "unk_token_id", None)
            if candidate is not None and candidate != unknown:
                image_token_id = int(candidate)
        if image_token_id is None:
            raise ValueError("ColPali image token ID could not be resolved")
        docs: list[torch.Tensor] = []
        rasters: list[torch.Tensor] = []
        for image in images:
            image_batch = self.colpali_processor.process_images([image])
            if not isinstance(image_batch, Mapping):
                raise ValueError("ColPali image processor output must be a mapping")
            input_ids = torch.as_tensor(_value(image_batch, "input_ids"), dtype=torch.long)
            attention = torch.as_tensor(_value(image_batch, "attention_mask"))
            mapping = resolve_colpali_visual_mapping(
                input_ids=input_ids,
                attention_mask=attention,
                image_token_id=int(image_token_id),
                image_seq_length=int(getattr(self.colpali_processor, "image_seq_length", 1024)),
            )
            if page_config is None:
                retrieval_keep = torch.ones(len(mapping.raster_indices), dtype=torch.bool)
            else:
                raster = colpali_uint8_raster(self.colpali_processor, image)
                scores = background_scores(
                    raster,
                    patch_size=14,
                    error_tolerance=page_config.background_error_tolerance,
                )
                retrieval_keep = threshold_keep_mask(
                    scores,
                    page_config.retrieval_background_threshold,
                )[0]
                if not bool(retrieval_keep.any()):
                    raise ValueError("retrieval BTP rejected every page patch")
            from docprune.colpali.embedding import encode_colpali_page

            encoded = encode_colpali_page(
                self.colpali_model,
                _move_colpali_batch(image_batch, self.colpali_model),
                mapping,
                retrieval_keep,
            )
            docs.append(encoded.visual_embeddings[0])
            rasters.append(encoded.raster_indices)
        query_mask = query_attention[0].bool()
        return docs, rasters, query[0, query_mask]

    def _masks(
        self,
        images: Sequence[object],
        prepared: Sequence[PreparedQwenPage],
        batch: Mapping[str, object],
        question: str,
    ) -> VisionPruningMasks:
        grid = _grid(batch)
        page_config = self._effective_page_config(len(images))
        if page_config is None:
            groups = _merged_count(grid)
            return VisionPruningMasks(torch.ones(groups, dtype=torch.bool), torch.ones(groups, dtype=torch.bool))
        embeddings = self._colpali_page_embeddings(images, question, page_config)
        if embeddings is None:
            raise ValueError("DocPrune answerer requires a ColPali model and processor")
        documents, rasters, query = embeddings
        masks = prepare_qa_pruning_masks(
            resized_images=[page.raster for page in prepared],
            image_grid_thw=grid,
            document_tokens=documents,
            document_source_hw=[(32, 32)] * len(images),
            document_raster_indices=rasters,
            question_tokens=query,
            patch_size=prepared[0].patch_size,
            page_config=page_config,
            reconstruction=self.reconstruction,
        )
        return masks

    def answer(self, images: Sequence[object], question: str) -> AnswerOutput:
        started = time.perf_counter()
        prepared = [prepare_qwen_page(self.processor, image) for image in images]
        qwen_images = [prepared_raster_image(page) for page in prepared]
        batch = _prepare_batch(self.processor, qwen_images, question)
        grid = _grid(batch)
        model_device = _model_device(self.model)
        moved = _move_batch(batch, model_device)
        input_ids = torch.as_tensor(_value(moved, "input_ids"), dtype=torch.long)
        _validate_placeholder_count(self.model, self.processor, input_ids, grid)
        attention_mask = torch.as_tensor(_value(moved, "attention_mask"), dtype=torch.long)
        pixel_values = torch.as_tensor(_value(moved, "pixel_values"))
        masks = self._masks(images, prepared, moved, question)
        page_config = self._effective_page_config(len(images))
        adapter = (
            self.model
            if callable(getattr(self.model, "generate_with_trace", None))
            else DocPruneQwen2VL(self.model)
        )
        eos = tuple(
            value
            for value in (
                getattr(getattr(self.model, "config", None), "eos_token_id", None),
            )
            if isinstance(value, int)
        )
        with torch.no_grad():
            result = adapter.generate_with_trace(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=grid,
                pruning_masks=masks,
                comprehension_threshold=(
                    self.comprehension_threshold
                    if self.comprehension_threshold is not None
                    else (page_config.comprehension_threshold if page_config else 1e9)
                ),
                attention_threshold=(
                    self.attention_threshold
                    if self.attention_threshold is not None
                    else (page_config.attention_threshold if page_config else 0.0)
                ),
                max_new_tokens=self.max_new_tokens,
                eos_token_ids=eos,
            )
        # The adapter intentionally returns only generated IDs; stock Qwen
        # returns the prompt plus its generated suffix.
        answer = _decode_new_tokens(self.processor, result.generated_ids, 0)
        return AnswerOutput(answer, result.trace, time.perf_counter() - started)
