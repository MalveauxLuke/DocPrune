"""Stock and DocPrune Qwen2-VL answerers for the M3DocRAG boundary."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from docprune.benchmark_config import MAX_NEW_TOKENS, SHORT_ANSWER_TEMPLATE
from docprune.config import DocPruneConfig, PagePruningConfig, ReconstructionDefaults
from docprune.ctp_controls import VisualTokenGeometry
from docprune.ctp_policy import CTPPolicy, PolicySelectionContext, btp_qtp_no_ctp_policy
from docprune.m3docrag import AnswerOutput, RetrievalOutput
from docprune.pipeline import prepare_qa_pruning_masks
from docprune.qwen2vl.decoder import ForcedVisualIntervention
from docprune.qwen2vl.model import (
    DocPruneQwen2VL,
    PruningTrace,
    SharedBoundaryLikelihoodResult,
    VisionPruningMasks,
    _begin_synchronized_timer,
    _end_synchronized_timer,
    _module_timer_hooks,
    _remove_module_timer_hooks,
    _resolve_module_timer_groups,
)
from docprune.qwen2vl.preprocessing import (
    PreparedQwenPage,
    prepare_qwen_page,
    prepared_raster_image,
)
from docprune.task6_runtime import derive_post_qtp_geometry, frozen_geometry_keep_mask


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


def _begin_gpu_measurement(model: object) -> torch.device | None:
    """Reset the CUDA peak counter immediately before the measured model call."""

    device = _model_device(model)
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return device


def _end_gpu_measurement(device: torch.device | None) -> int:
    if device is None:
        return 0
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def _encoder_timer_hooks(model: object) -> tuple[list[float], list[object]]:
    """Time only calls to Qwen's vision module during stock generation."""

    visual = getattr(model, "visual", None)
    return _module_timer_hooks(model, [visual] if visual is not None else [])


def _decoder_timer_hooks(model: object) -> tuple[list[float], list[object]]:
    """Time only calls to Qwen's language-model module during stock generation."""

    decoder = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    return _module_timer_hooks(
        model,
        [module for module in (decoder, lm_head) if module is not None],
    )


def _remove_hooks(handles: Sequence[object]) -> None:
    for handle in handles:
        remove = getattr(handle, "remove", None)
        if callable(remove):
            remove()


def _move_batch(batch: object, device: torch.device | None) -> dict[str, object]:
    if not isinstance(batch, Mapping):
        raise ValueError("Qwen processor must return a mapping")
    return {
        name: value.to(device) if device is not None and isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def _prompt(processor: object, page_count: int, question: str) -> str:
    content = [{"type": "image", "image": "dummy_content"} for _ in range(page_count)]
    content.append({"type": "text", "text": SHORT_ANSWER_TEMPLATE.replace("$question", question)})
    messages = [{"role": "user", "content": content}]
    apply = getattr(processor, "apply_chat_template", None)
    if not callable(apply):
        raise ValueError("Qwen processor must expose apply_chat_template")
    return str(apply(messages, tokenize=False, add_generation_prompt=True))


def prepare_task7_likelihood_target_for_question(
    processor: object,
    *,
    page_count: int,
    question: str,
    accepted_references: Sequence[str],
) -> dict[str, object]:
    """Derive a Task 7 target from the production prompt constructor only."""

    if type(page_count) is not int or page_count <= 0:
        raise ValueError("Task 7 likelihood target requires a positive page count")
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Task 7 likelihood target requires the production tokenizer")
    from docprune.task7_runtime import prepare_task7_likelihood_target

    return prepare_task7_likelihood_target(
        tokenizer,
        accepted_references,
        assistant_prompt=_prompt(processor, page_count, question),
    )


def _prepare_batch(processor: object, images: Sequence[object], question: str) -> dict[str, object]:
    return _prepare_batch_with_prompt(processor, images, question)[1]


def _prepare_batch_with_prompt(
    processor: object, images: Sequence[object], question: str
) -> tuple[str, dict[str, object]]:
    if not images:
        raise ValueError("at least one page image is required")
    prompt = _prompt(processor, len(images), question)
    call = getattr(processor, "__call__", None)
    if not callable(call):
        raise ValueError("Qwen processor is not callable")
    return prompt, dict(call(text=[prompt], images=list(images), padding=True, return_tensors="pt"))


def _input_ids_identity(input_ids: torch.Tensor) -> tuple[tuple[int, int], str]:
    canonical = input_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if canonical.ndim != 2 or canonical.shape[0] != 1 or not canonical.shape[1]:
        raise ValueError("Qwen input_ids must have nonempty batch-one shape")
    return (int(canonical.shape[0]), int(canonical.shape[1])), hashlib.sha256(
        canonical.numpy().tobytes()
    ).hexdigest()


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


def _response_ids_and_terminal(
    generated: torch.Tensor, prompt_length: int, eos_token_ids: Sequence[int]
) -> tuple[tuple[int, ...], int | None]:
    values = tuple(int(value) for value in generated[0, prompt_length:].tolist())
    terminal = values[-1] if values and values[-1] in set(eos_token_ids) else None
    return (values[:-1] if terminal is not None else values), terminal


def _resolved_eos_token_ids(model: object) -> tuple[int, ...]:
    """Resolve the EOS set used by stock ``generate`` for the pinned model."""

    generation_config = getattr(model, "generation_config", None)
    value = getattr(generation_config, "eos_token_id", None)
    if value is None:
        value = getattr(getattr(model, "config", None), "eos_token_id", None)
    values = (value,) if isinstance(value, int) and not isinstance(value, bool) else value
    if values is None:
        return ()
    if (
        not isinstance(values, Sequence)
        or isinstance(values, str | bytes)
        or any(not isinstance(item, int) or isinstance(item, bool) for item in values)
    ):
        raise ValueError("Qwen EOS token IDs must be an integer or a sequence of integers")
    return tuple(dict.fromkeys(int(item) for item in values))


def _all_kept_trace(grid: torch.Tensor) -> PruningTrace:
    count = _merged_count(grid)
    return PruningTrace(count, count, count, count, None)


def _mask_sha256(value: object) -> str:
    mask = torch.as_tensor(value, dtype=torch.bool)
    if mask.ndim != 1:
        raise ValueError("vision pruning masks must be one-dimensional")
    return hashlib.sha256(bytes(int(item) for item in mask.tolist())).hexdigest()


@dataclass(frozen=True, slots=True)
class BTPQTPGeometryCapture:
    """Exact post-BTP+QTP geometry derived without loading Qwen model weights."""

    geometry: tuple[VisualTokenGeometry, ...]
    geometry_count: int
    geometry_sha256: str
    image_grid_thw: tuple[tuple[int, int, int], ...]
    original_visual_tokens: int
    post_btp_visual_tokens: int
    post_qtp_visual_tokens: int
    background_keep_sha256: str
    question_keep_sha256: str
    combined_keep_sha256: str


@dataclass(frozen=True, slots=True)
class RegionalLikelihoodOutput:
    """Shared-boundary likelihoods bound to the exact production prompt batch."""

    result: SharedBoundaryLikelihoodResult
    assistant_prompt_sha256: str
    prefill_input_ids_shape: tuple[int, int]
    prefill_input_ids_sha256: str
    peak_allocated_gpu_bytes: int
    unpruned_generated_response_token_ids: tuple[int, ...] | None = None
    unpruned_terminal_eos_token_id: int | None = None
    unpruned_generation_trace: PruningTrace | None = None
    unpruned_generation_encoder_seconds: float | None = None
    unpruned_generation_decoder_seconds: float | None = None


def _prepare_pruning_masks_from_context(
    images: Sequence[object],
    prepared: Sequence[PreparedQwenPage],
    batch: Mapping[str, object],
    retrieval_output: RetrievalOutput,
    *,
    page_config: PagePruningConfig,
    reconstruction: ReconstructionDefaults,
) -> VisionPruningMasks:
    grid = _grid(batch)
    if len(retrieval_output.pages) != len(images):
        raise ValueError("retrieval_output page count must align with images")
    if len(retrieval_output.page_features) != len(images):
        raise ValueError("retrieval_output page features must align with images")
    documents = []
    rasters = []
    source_hws = []
    for page, feature in zip(retrieval_output.pages, retrieval_output.page_features, strict=True):
        if (page.doc_id, page.page_index) != (feature.doc_id, feature.page_index):
            raise ValueError("retrieval_output page features do not align with pages")
        if tuple(feature.source_hw) != (32, 32):
            raise ValueError("retrieval_output source grid must be (32, 32)")
        document = torch.as_tensor(feature.visual_embeddings)
        raster = torch.as_tensor(feature.raster_indices, dtype=torch.long)
        if document.ndim != 2 or document.shape[1] != 128:
            raise ValueError("retrieval_output visual embeddings must have shape [tokens, 128]")
        if raster.ndim != 1 or raster.numel() != document.shape[0] or not raster.numel():
            raise ValueError("retrieval_output raster rows must align with visual embeddings")
        if bool((raster < 0).any()) or bool((raster >= 1024).any()):
            raise ValueError("retrieval_output raster indices are out of range")
        if raster.numel() > 1 and not bool((raster[1:] > raster[:-1]).all()):
            raise ValueError("retrieval_output raster indices must be increasing")
        documents.append(document)
        rasters.append(raster)
        source_hws.append(tuple(feature.source_hw))
    query = torch.as_tensor(retrieval_output.query_embeddings)
    if query.ndim != 2 or query.shape[1] != 128:
        raise ValueError("retrieval_output query embeddings must have shape [tokens, 128]")
    return prepare_qa_pruning_masks(
        resized_images=[page.raster for page in prepared],
        image_grid_thw=grid,
        document_tokens=documents,
        document_source_hw=source_hws,
        document_raster_indices=rasters,
        question_tokens=query,
        patch_size=prepared[0].patch_size,
        page_config=page_config,
        reconstruction=reconstruction,
    )


def derive_btp_qtp_geometry_without_qwen_model(
    processor: object,
    images: Sequence[object],
    question: str,
    retrieval_output: RetrievalOutput,
    *,
    page_config: PagePruningConfig | None = None,
    reconstruction: ReconstructionDefaults | None = None,
) -> BTPQTPGeometryCapture:
    """Run the answerer's exact page/BTP/QTP path without Qwen generation weights."""

    prepared = tuple(prepare_qwen_page(processor, image) for image in images)
    qwen_images = tuple(prepared_raster_image(page) for page in prepared)
    batch = _prepare_batch(processor, qwen_images, question)
    _validate_prepared_batch(prepared, batch, None, processor)
    grid = _grid(batch)
    selected_config = page_config or DocPruneConfig.paper_defaults().for_pages(len(images))
    masks = _prepare_pruning_masks_from_context(
        images,
        prepared,
        batch,
        retrieval_output,
        page_config=selected_config,
        reconstruction=reconstruction or ReconstructionDefaults(),
    )
    combined = masks.combined()
    identity = derive_post_qtp_geometry(
        grid,
        combined,
        merge_size=prepared[0].merge_size,
    )
    return BTPQTPGeometryCapture(
        identity.geometry,
        identity.count,
        identity.sha256,
        tuple(tuple(int(item) for item in row) for row in grid.tolist()),
        _merged_count(grid, merge_size=prepared[0].merge_size),
        int(torch.as_tensor(masks.background_keep, dtype=torch.bool).sum().item()),
        int(torch.as_tensor(combined, dtype=torch.bool).sum().item()),
        _mask_sha256(masks.background_keep),
        _mask_sha256(masks.question_keep),
        _mask_sha256(combined),
    )


def derive_full_context_geometry_without_qwen_model(
    processor: object,
    images: Sequence[object],
    question: str,
) -> BTPQTPGeometryCapture:
    """Derive the complete merged Qwen visual grid without BTP or QTP."""

    prepared = tuple(prepare_qwen_page(processor, image) for image in images)
    qwen_images = tuple(prepared_raster_image(page) for page in prepared)
    batch = _prepare_batch(processor, qwen_images, question)
    _validate_prepared_batch(prepared, batch, None, processor)
    grid = _grid(batch)
    count = _merged_count(grid, merge_size=prepared[0].merge_size)
    keep = torch.ones(count, dtype=torch.bool)
    identity = derive_post_qtp_geometry(
        grid,
        keep,
        merge_size=prepared[0].merge_size,
    )
    mask_sha256 = _mask_sha256(keep)
    return BTPQTPGeometryCapture(
        identity.geometry,
        identity.count,
        identity.sha256,
        tuple(tuple(int(item) for item in row) for row in grid.tolist()),
        count,
        count,
        count,
        mask_sha256,
        mask_sha256,
        mask_sha256,
    )


def _validate_placeholder_count(
    model: object, processor: object, input_ids: torch.Tensor, grid: torch.Tensor
) -> None:
    expected = _merged_count(grid)
    observed = int((input_ids == _image_token_id(model, processor)).sum().item())
    if observed != expected:
        raise ValueError(
            f"Qwen image placeholders ({observed}) do not match merge groups ({expected})"
        )


def _validate_prepared_batch(
    prepared: Sequence[PreparedQwenPage],
    batch: Mapping[str, object],
    model: object,
    processor: object,
) -> None:
    """Ensure final chat batching did not alter page order or vision values."""

    if not prepared:
        raise ValueError("at least one prepared Qwen page is required")
    final_grid = _grid(batch)
    expected_grid = torch.cat([page.image_grid_thw for page in prepared], dim=0)
    if not torch.equal(final_grid.cpu(), expected_grid.cpu()):
        raise ValueError("batched Qwen image_grid_thw does not match page order")
    final_pixels = torch.as_tensor(_value(batch, "pixel_values"))
    expected_pixels = torch.cat([page.pixel_values for page in prepared], dim=0)
    if final_pixels.shape != expected_pixels.shape or not torch.equal(
        final_pixels.cpu(), expected_pixels.cpu()
    ):
        raise ValueError("batched Qwen pixel_values do not match page order or values")
    if (
        len({(page.patch_size, page.temporal_patch_size, page.merge_size) for page in prepared})
        != 1
    ):
        raise ValueError("Qwen pages use inconsistent patch or merge geometry")
    image_ids = torch.as_tensor(_value(batch, "input_ids"), dtype=torch.long)
    if image_ids.ndim != 2 or image_ids.shape[0] != 1:
        raise ValueError("Qwen input_ids must have shape [1, sequence]")
    image_positions = (
        (image_ids[0] == _image_token_id(model, processor)).nonzero(as_tuple=False).flatten()
    )
    expected_counts = [page.placeholder_count for page in prepared]
    if image_positions.numel() != sum(expected_counts):
        raise ValueError("batched Qwen image placeholder count does not match page groups")
    runs: list[torch.Tensor] = []
    if image_positions.numel():
        breaks = torch.where(image_positions[1:] != image_positions[:-1] + 1)[0] + 1
        boundaries = torch.cat(
            [
                image_positions.new_tensor([0]),
                breaks,
                image_positions.new_tensor([image_positions.numel()]),
            ]
        )
        runs = [
            image_positions[int(start) : int(end)] for start, end in zip(boundaries, boundaries[1:])
        ]
    if len(runs) != len(expected_counts) or any(
        run.numel() != count for run, count in zip(runs, expected_counts)
    ):
        raise ValueError("batched Qwen page placeholders do not match page order or counts")


@dataclass
class AllKeptQwenAnswerer:
    """Stock Qwen2-VL generation with the M3DocRAG short-answer prompt."""

    model: object
    processor: object
    max_new_tokens: int = MAX_NEW_TOKENS

    def answer(
        self,
        images: Sequence[object],
        question: str,
        *,
        retrieval_output: RetrievalOutput | None = None,
    ) -> AnswerOutput:
        del retrieval_output
        started = time.perf_counter()
        measurement_device = _begin_gpu_measurement(self.model)
        prompt, batch = _prepare_batch_with_prompt(self.processor, images, question)
        input_ids = torch.as_tensor(_value(batch, "input_ids"), dtype=torch.long)
        input_ids_shape, input_ids_sha256 = _input_ids_identity(input_ids)
        grid = _grid(batch)
        _validate_placeholder_count(self.model, self.processor, input_ids, grid)
        batch = _move_batch(batch, _model_device(self.model))
        generation_started, generation_device = _begin_synchronized_timer(self.model)
        encoder_times, encoder_hooks = _encoder_timer_hooks(self.model)
        decoder_times, decoder_hooks = _decoder_timer_hooks(self.model)
        eos = _resolved_eos_token_ids(self.model)
        with torch.no_grad():
            try:
                generation_kwargs = {
                    **batch,
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": False,
                    "num_beams": 1,
                }
                if eos:
                    generation_kwargs["eos_token_id"] = list(eos)
                generated = self.model.generate(
                    **generation_kwargs,
                )
            finally:
                _remove_module_timer_hooks(encoder_hooks)
                _remove_module_timer_hooks(decoder_hooks)
        generation_elapsed = _end_synchronized_timer(generation_started, generation_device)
        encoder_seconds, decoder_seconds = _resolve_module_timer_groups(
            self.model, (encoder_times, decoder_times)
        )
        if not encoder_seconds or not decoder_seconds:
            # CPU fakes do not expose Qwen's visual module.  Keep their
            # stage fields deterministic while production CUDA models use
            # the forward-hook measurement above.
            if generation_device is not None and generation_device.type == "cuda":
                raise RuntimeError("Qwen stage timing hooks did not observe model execution")
            if not encoder_seconds:
                encoder_seconds = max(generation_elapsed / 2.0, 1e-12)
            if not decoder_seconds:
                decoder_seconds = max(generation_elapsed - encoder_seconds, 1e-12)
        peak_allocated_gpu_bytes = _end_gpu_measurement(measurement_device)
        generated_tensor = torch.as_tensor(generated)
        answer = _decode_new_tokens(self.processor, generated_tensor, input_ids.shape[1])
        response_ids, terminal_eos_token_id = _response_ids_and_terminal(
            generated_tensor, input_ids.shape[1], eos
        )
        qa_elapsed = max(time.perf_counter() - started, 1e-12)
        return AnswerOutput(
            answer,
            _all_kept_trace(grid),
            qa_elapsed,
            peak_allocated_gpu_bytes,
            False,
            False,
            None,
            None,
            encoder_seconds,
            decoder_seconds,
            assistant_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            prefill_input_ids_shape=input_ids_shape,
            prefill_input_ids_sha256=input_ids_sha256,
            generated_response_token_ids=response_ids,
            terminal_eos_token_id=terminal_eos_token_id,
        )


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
        qa_stage: str = "full",
        max_new_tokens: int = MAX_NEW_TOKENS,
        forced_intervention: ForcedVisualIntervention | None = None,
        ctp_policy: CTPPolicy | None = None,
        selection_context: PolicySelectionContext | None = None,
        policy_experiment_version: object | None = None,
        policy_repetition: object | None = None,
        teacher_forced_target_token_ids: tuple[tuple[int, ...], ...] | None = None,
        frozen_post_qtp_geometry: tuple[VisualTokenGeometry, ...] | None = None,
        full_context: bool = False,
    ) -> None:
        del colpali_model, colpali_processor
        super().__init__(model=model, processor=processor, max_new_tokens=max_new_tokens)
        self.page_config = page_config
        self.reconstruction = reconstruction or ReconstructionDefaults()
        self.comprehension_threshold = comprehension_threshold
        self.attention_threshold = attention_threshold
        if qa_stage not in {"full", "btp-only", "btp-qtp"}:
            raise ValueError("qa_stage must be full, btp-only, or btp-qtp")
        self.qa_stage = qa_stage
        self.forced_intervention = forced_intervention
        self.ctp_policy = ctp_policy
        self.selection_context = selection_context
        self._policy_experiment_version = policy_experiment_version
        self._policy_repetition = policy_repetition
        self.teacher_forced_target_token_ids = teacher_forced_target_token_ids
        self.frozen_post_qtp_geometry = frozen_post_qtp_geometry
        self.full_context = bool(full_context)
        if frozen_post_qtp_geometry is not None and qa_stage != "full":
            raise ValueError("frozen post-QTP geometry requires the full QA stage")
        if self.forced_intervention is not None and self.ctp_policy is not None:
            raise ValueError("forced intervention and corrected CTP policy cannot be combined")
        if self.full_context and self.ctp_policy is not None:
            raise ValueError("full-context intervention mode cannot combine a CTP policy")
        if (
            self.ctp_policy is not None
            and self.ctp_policy.family in {"random-top-m", "coverage-top-m"}
            and (policy_experiment_version is None or policy_repetition is None)
        ):
            raise ValueError("random CTP policy requires experiment version and repetition context")

    def set_policy_question_id(self, qid: str) -> None:
        """Receive the source QID from the runner without inspecting question or answer text."""

        if self.ctp_policy is None or self.ctp_policy.family not in {
            "random-top-m",
            "coverage-top-m",
        }:
            return
        if not isinstance(qid, str) or not qid:
            raise ValueError("random CTP policy requires a nonempty source QID")
        geometry = None if self.selection_context is None else self.selection_context.geometry
        geometry_count = (
            None if self.selection_context is None else self.selection_context.geometry_count
        )
        geometry_sha256 = (
            None if self.selection_context is None else self.selection_context.geometry_sha256
        )
        self.selection_context = PolicySelectionContext(
            self._policy_experiment_version,
            qid,
            None,
            self._policy_repetition,
            geometry,
            geometry_count,
            geometry_sha256,
        )

    def _effective_page_config(self, page_count: int) -> PagePruningConfig:
        if self.page_config is not None:
            return self.page_config
        return DocPruneConfig.paper_defaults().for_pages(page_count)

    def _masks(
        self,
        images: Sequence[object],
        prepared: Sequence[PreparedQwenPage],
        batch: Mapping[str, object],
        question: str,
        retrieval_output: RetrievalOutput,
    ) -> VisionPruningMasks:
        del question
        page_config = self._effective_page_config(len(images))
        return _prepare_pruning_masks_from_context(
            images,
            prepared,
            batch,
            retrieval_output,
            page_config=page_config,
            reconstruction=self.reconstruction,
        )

    def _frozen_masks(
        self,
        masks: VisionPruningMasks,
        grid: torch.Tensor,
        *,
        merge_size: int,
    ) -> VisionPruningMasks:
        if self.frozen_post_qtp_geometry is None:
            return masks
        frozen = frozen_geometry_keep_mask(
            grid,
            self.frozen_post_qtp_geometry,
            merge_size=merge_size,
        ).to(device=torch.as_tensor(masks.background_keep).device)
        background = torch.as_tensor(masks.background_keep, dtype=torch.bool)
        if background.shape != frozen.shape or bool((frozen & ~background).any()):
            raise ValueError("frozen post-QTP geometry is outside the live BTP population")
        return VisionPruningMasks(background_keep=background, question_keep=frozen)

    def _base_masks(
        self,
        images: Sequence[object],
        prepared: Sequence[PreparedQwenPage],
        batch: Mapping[str, object],
        question: str,
        retrieval_output: RetrievalOutput,
        grid: torch.Tensor,
    ) -> VisionPruningMasks:
        if not self.full_context:
            return self._masks(images, prepared, batch, question, retrieval_output)
        count = _merged_count(grid, merge_size=prepared[0].merge_size)
        keep = torch.ones(count, dtype=torch.bool, device=grid.device)
        return VisionPruningMasks(background_keep=keep, question_keep=keep)

    def score_forced_intervention_likelihoods(
        self,
        images: Sequence[object],
        question: str,
        *,
        retrieval_output: RetrievalOutput | None,
        forced_interventions: tuple[ForcedVisualIntervention, ...],
        teacher_forced_target_token_ids: tuple[tuple[int, ...], ...],
        include_unpruned_generated_response: bool = False,
    ) -> RegionalLikelihoodOutput:
        """Score Task 9 masks through the selected frozen visual-context path."""

        if retrieval_output is None:
            raise ValueError("Task 9 regional scoring requires retrieval_output")
        if self.qa_stage != "full":
            raise ValueError("Task 9 regional scoring requires the full BTP+QTP stage")
        if self.forced_intervention is not None or (
            self.ctp_policy is not None and self.ctp_policy.family != "no-ctp"
        ):
            raise ValueError("Task 9 regional scoring cannot combine another CTP intervention")
        prepared = [prepare_qwen_page(self.processor, image) for image in images]
        qwen_images = [prepared_raster_image(page) for page in prepared]
        prompt, batch = _prepare_batch_with_prompt(self.processor, qwen_images, question)
        grid = _grid(batch)
        _validate_prepared_batch(prepared, batch, self.model, self.processor)
        moved = _move_batch(batch, _model_device(self.model))
        input_ids = torch.as_tensor(_value(moved, "input_ids"), dtype=torch.long)
        input_shape, input_sha256 = _input_ids_identity(input_ids)
        _validate_placeholder_count(self.model, self.processor, input_ids, grid)
        masks = self._base_masks(
            images, prepared, moved, question, retrieval_output, grid
        )
        masks = self._frozen_masks(masks, grid, merge_size=prepared[0].merge_size)
        adapter = (
            self.model
            if callable(getattr(self.model, "score_forced_intervention_likelihoods", None))
            else DocPruneQwen2VL(self.model)
        )
        measurement_device = _begin_gpu_measurement(self.model)
        generated_response_token_ids: tuple[int, ...] | None = None
        terminal_eos_token_id: int | None = None
        unpruned_generation = None
        with torch.no_grad():
            if include_unpruned_generated_response:
                eos = _resolved_eos_token_ids(self.model)
                unpruned_generation = adapter.generate_with_trace(
                    input_ids=input_ids,
                    attention_mask=torch.as_tensor(
                        _value(moved, "attention_mask"), dtype=torch.long
                    ),
                    pixel_values=torch.as_tensor(_value(moved, "pixel_values")),
                    image_grid_thw=grid,
                    pruning_masks=masks,
                    comprehension_threshold=1e9,
                    attention_threshold=0.0,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_ids=eos,
                    forced_intervention=None,
                    ctp_policy=btp_qtp_no_ctp_policy(),
                )
                generated = torch.as_tensor(unpruned_generation.generated_ids).detach().cpu()
                if generated.ndim != 2 or generated.shape[0] != 1 or generated.shape[1] == 0:
                    raise ValueError("Task 9 unpruned generated response IDs are invalid")
                generated_ids = tuple(int(token) for token in generated[0].tolist())
                if generated_ids[-1] in set(eos):
                    terminal_eos_token_id = generated_ids[-1]
                    generated_ids = generated_ids[:-1]
                if (
                    not generated_ids
                    or unpruned_generation.trace.ctp_layer is not None
                    or unpruned_generation.trace.post_ctp_visual_tokens
                    != unpruned_generation.trace.post_qtp_visual_tokens
                    or unpruned_generation.forced_intervention is not None
                ):
                    raise ValueError("Task 9 unpruned generated response is invalid")
                generated_response_token_ids = generated_ids
                teacher_forced_target_token_ids = (
                    *teacher_forced_target_token_ids,
                    generated_response_token_ids,
                )
            result = adapter.score_forced_intervention_likelihoods(
                input_ids=input_ids,
                attention_mask=torch.as_tensor(_value(moved, "attention_mask"), dtype=torch.long),
                pixel_values=torch.as_tensor(_value(moved, "pixel_values")),
                image_grid_thw=grid,
                pruning_masks=masks,
                forced_interventions=forced_interventions,
                teacher_forced_target_token_ids=teacher_forced_target_token_ids,
            )
        return RegionalLikelihoodOutput(
            result=result,
            assistant_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            prefill_input_ids_shape=input_shape,
            prefill_input_ids_sha256=input_sha256,
            peak_allocated_gpu_bytes=_end_gpu_measurement(measurement_device),
            unpruned_generated_response_token_ids=generated_response_token_ids,
            unpruned_terminal_eos_token_id=terminal_eos_token_id,
            unpruned_generation_trace=(
                None if unpruned_generation is None else unpruned_generation.trace
            ),
            unpruned_generation_encoder_seconds=(
                None if unpruned_generation is None else unpruned_generation.encoder_seconds
            ),
            unpruned_generation_decoder_seconds=(
                None if unpruned_generation is None else unpruned_generation.decoder_seconds
            ),
        )

    def answer(
        self,
        images: Sequence[object],
        question: str,
        *,
        retrieval_output: RetrievalOutput | None = None,
    ) -> AnswerOutput:
        if retrieval_output is None:
            raise ValueError("DocPrune answerer requires retrieval_output")
        started = time.perf_counter()
        measurement_device = _begin_gpu_measurement(self.model)
        prepared = [prepare_qwen_page(self.processor, image) for image in images]
        qwen_images = [prepared_raster_image(page) for page in prepared]
        prompt, batch = _prepare_batch_with_prompt(self.processor, qwen_images, question)
        grid = _grid(batch)
        _validate_prepared_batch(prepared, batch, self.model, self.processor)
        model_device = _model_device(self.model)
        moved = _move_batch(batch, model_device)
        input_ids = torch.as_tensor(_value(moved, "input_ids"), dtype=torch.long)
        capture_likelihood = self.teacher_forced_target_token_ids is not None
        input_ids_shape, input_ids_sha256 = (
            _input_ids_identity(input_ids) if capture_likelihood else (None, None)
        )
        _validate_placeholder_count(self.model, self.processor, input_ids, grid)
        attention_mask = torch.as_tensor(_value(moved, "attention_mask"), dtype=torch.long)
        pixel_values = torch.as_tensor(_value(moved, "pixel_values"))
        masks = self._base_masks(
            images, prepared, moved, question, retrieval_output, grid
        )
        if self.qa_stage == "btp-only":
            masks = VisionPruningMasks(
                background_keep=masks.background_keep,
                question_keep=torch.ones_like(masks.question_keep, dtype=torch.bool),
            )
        masks = self._frozen_masks(masks, grid, merge_size=prepared[0].merge_size)
        if self.ctp_policy is not None:
            if self.ctp_policy.family in {"random-top-m", "coverage-top-m"} and (
                self.selection_context is None or not self.selection_context.qid
            ):
                raise ValueError("random CTP policy requires source QID context before generation")
            geometry = derive_post_qtp_geometry(
                grid,
                masks.combined(),
                merge_size=prepared[0].merge_size,
            )
            prior = self.selection_context
            self.selection_context = PolicySelectionContext(
                None if prior is None else prior.experiment_version,
                None if prior is None else prior.qid,
                None if prior is None else prior.boundary,
                None if prior is None else prior.repetition,
                geometry.geometry,
                geometry.count,
                geometry.sha256,
            )
        page_config = self._effective_page_config(len(images))
        adapter = (
            self.model
            if callable(getattr(self.model, "generate_with_trace", None))
            else DocPruneQwen2VL(self.model)
        )
        eos = _resolved_eos_token_ids(self.model)
        with torch.no_grad():
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": grid,
                "pruning_masks": masks,
                "comprehension_threshold": (
                    1e9
                    if self.qa_stage != "full"
                    else (
                        self.comprehension_threshold
                        if self.comprehension_threshold is not None
                        else (page_config.comprehension_threshold if page_config else 1e9)
                    )
                ),
                "attention_threshold": (
                    self.attention_threshold
                    if self.attention_threshold is not None
                    else (page_config.attention_threshold if page_config else 0.0)
                ),
                "max_new_tokens": self.max_new_tokens,
                "eos_token_ids": eos,
            }
            if self.forced_intervention is not None:
                generation_kwargs["forced_intervention"] = self.forced_intervention
            if self.ctp_policy is not None:
                generation_kwargs["ctp_policy"] = self.ctp_policy
            if self.selection_context is not None:
                generation_kwargs["selection_context"] = self.selection_context
            if self.teacher_forced_target_token_ids is not None:
                generation_kwargs["teacher_forced_target_token_ids"] = (
                    self.teacher_forced_target_token_ids
                )
            result = adapter.generate_with_trace(
                **generation_kwargs,
            )
        peak_allocated_gpu_bytes = _end_gpu_measurement(measurement_device)
        # The adapter intentionally returns only generated IDs; stock Qwen
        # returns the prompt plus its generated suffix.
        answer = _decode_new_tokens(self.processor, result.generated_ids, 0)
        return AnswerOutput(
            answer,
            result.trace,
            max(time.perf_counter() - started, 1e-12),
            peak_allocated_gpu_bytes,
            False,
            False,
            None,
            None,
            max(float(result.encoder_seconds), 1e-12),
            max(float(result.decoder_seconds), 1e-12),
            result.forced_intervention,
            result.policy_selection,
            result.teacher_forced_loglikelihoods,
            hashlib.sha256(prompt.encode("utf-8")).hexdigest() if capture_likelihood else None,
            input_ids_shape,
            input_ids_sha256,
            generated_response_token_ids=_response_ids_and_terminal(
                torch.as_tensor(result.generated_ids), 0, eos
            )[0],
            terminal_eos_token_id=_response_ids_and_terminal(
                torch.as_tensor(result.generated_ids), 0, eos
            )[1],
        )
