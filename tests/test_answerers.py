from __future__ import annotations

import hashlib

import pytest
import torch
from PIL import Image

from docprune import answerers
from docprune.answerers import AllKeptQwenAnswerer, DocPruneQwenAnswerer
from docprune.config import PagePruningConfig
from docprune.ctp import ComprehensionController
from docprune.m3docrag import RetrievalOutput, RetrievedPage, RetrievedPageFeatures
from docprune.qwen2vl.decoder import ForcedInterventionRecord, ForcedVisualIntervention
from docprune.qwen2vl.model import GenerationResult, PruningTrace, VisionPruningMasks
from docprune.qwen2vl.preprocessing import PreparedQwenPage


class RecordingProcessor:
    image_token = 100

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages[0]["content"][0] == {"type": "image", "image": "dummy_content"}
        return "<|im_start|>user\nquestion: what?\noutput only answer.<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, *, text, images, padding, return_tensors):
        self.calls.append(
            {"text": text, "images": images, "padding": padding, "return_tensors": return_tensors}
        )
        return {
            "input_ids": torch.tensor([[10, 100, 100, 100, 100, 11]]),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "pixel_values": torch.zeros((16, 24)),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }

    def batch_decode(
        self, values, *, skip_special_tokens: bool, clean_up_tokenization_spaces: bool
    ):
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return ["answer"]


class RecordingModel:
    config = type("Config", (), {"image_token_id": 100, "eos_token_id": 151645})()
    generation_config = type("GenerationConfig", (), {"eos_token_id": [151645, 151643]})()

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[10, 100, 100, 11, 55]])


def retrieval_context(page_count: int = 1) -> RetrievalOutput:
    pages = tuple(RetrievedPage(f"doc-{index}", index, 1.0) for index in range(page_count))
    return RetrievalOutput(
        pages=pages,
        query_embeddings=torch.ones((2, 128)),
        page_features=tuple(
            RetrievedPageFeatures(
                page.doc_id,
                page.page_index,
                torch.ones((4, 128)),
                torch.arange(4),
                (32, 32),
            )
            for page in pages
        ),
    )


def test_all_kept_answerer_uses_exact_prompt_greedy_settings_and_new_tokens() -> None:
    processor = RecordingProcessor()
    model = RecordingModel()
    answerer = AllKeptQwenAnswerer(model=model, processor=processor)

    output = answerer.answer(["page"], "what?")

    assert output.answer == "answer"
    assert output.trace.original_visual_tokens == 4
    assert output.trace.post_btp_visual_tokens == 4
    assert output.trace.post_qtp_visual_tokens == 4
    assert output.trace.post_ctp_visual_tokens == 4
    assert output.warmup_excluded is False
    assert output.peak_allocated_gpu_bytes >= 0
    assert output.encoder_seconds > 0
    assert output.decoder_seconds > 0
    call = model.calls[0]
    assert call["max_new_tokens"] == 128
    assert call["do_sample"] is False
    assert call["num_beams"] == 1
    assert call["eos_token_id"] == [151645, 151643]
    assert call["input_ids"].tolist() == [[10, 100, 100, 100, 100, 11]]


def test_stock_stage_timers_observe_only_visual_and_language_modules() -> None:
    class HookedModel(RecordingModel):
        def __init__(self) -> None:
            super().__init__()
            self.visual = torch.nn.Identity()
            self.model = torch.nn.Identity()

        def generate(self, **kwargs):
            self.visual(torch.ones(1))
            self.model(torch.ones(1))
            self.model(torch.ones(1))
            return super().generate(**kwargs)

    output = AllKeptQwenAnswerer(model=HookedModel(), processor=RecordingProcessor()).answer(
        ["page"], "what?"
    )

    assert output.encoder_seconds > 0
    assert output.decoder_seconds > 0


def test_docprune_answerer_decodes_adapter_suffix_without_prompt(monkeypatch) -> None:
    from transformers import Qwen2VLImageProcessor

    class FakeAdapter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_with_trace(self, **kwargs):
            self.calls.append(kwargs)
            return GenerationResult(
                generated_ids=torch.tensor([[55]]),
                trace=PruningTrace(2, 2, 2, 1, None),
            )

    processor = RecordingProcessor()
    processor.image_processor = Qwen2VLImageProcessor()

    monkeypatch.setattr(answerers, "_validate_prepared_batch", lambda *args: None)
    monkeypatch.setattr(
        DocPruneQwenAnswerer,
        "_masks",
        lambda self, images, prepared, batch, question, retrieval_output: VisionPruningMasks(
            torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool)
        ),
    )
    adapter = FakeAdapter()
    monkeypatch.setattr(answerers, "DocPruneQwen2VL", lambda _: adapter)
    answerer = DocPruneQwenAnswerer(
        model=RecordingModel(),
        processor=processor,
        page_config=PagePruningConfig(1.0, 1.0, 1.0, 0.3, 45.0, 0.075),
    )

    output = answerer.answer(
        [Image.new("RGB", (112, 56))], "what?", retrieval_output=retrieval_context()
    )

    assert output.answer == "answer"
    assert output.trace.post_ctp_visual_tokens == 1
    assert output.encoder_seconds > 0
    assert output.decoder_seconds > 0
    assert adapter.calls[0]["eos_token_ids"] == (151645, 151643)
    assert output.assistant_prompt_sha256 is None
    assert output.prefill_input_ids_shape is None
    assert output.prefill_input_ids_sha256 is None


def test_task7_likelihood_capture_derives_exact_prompt_and_processor_input_internally(
    monkeypatch,
) -> None:
    """Catch accepting a caller-supplied prompt/hash or scoring a different batch."""

    from transformers import Qwen2VLImageProcessor

    class FakeAdapter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_with_trace(self, **kwargs):
            self.calls.append(kwargs)
            return GenerationResult(
                generated_ids=torch.tensor([[55]]),
                trace=PruningTrace(2, 2, 2, 2, None),
                teacher_forced_loglikelihoods=(-0.25, -0.75),
            )

    processor = RecordingProcessor()
    processor.image_processor = Qwen2VLImageProcessor()
    adapter = FakeAdapter()
    monkeypatch.setattr(answerers, "_validate_prepared_batch", lambda *args: None)
    monkeypatch.setattr(answerers, "DocPruneQwen2VL", lambda _: adapter)
    monkeypatch.setattr(
        DocPruneQwenAnswerer,
        "_masks",
        lambda self, images, prepared, batch, question, retrieval_output: VisionPruningMasks(
            torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool)
        ),
    )
    answerer = DocPruneQwenAnswerer(
        model=RecordingModel(),
        processor=processor,
        page_config=PagePruningConfig(1.0, 1.0, 1.0, 0.3, 45.0, 0.075),
        teacher_forced_target_token_ids=((12, 13), (14,)),
    )

    output = answerer.answer(
        [Image.new("RGB", (112, 56))], "what?", retrieval_output=retrieval_context()
    )

    exact_prompt = (
        "<|im_start|>user\nquestion: what?\noutput only answer.<|im_end|>\n<|im_start|>assistant\n"
    )
    expected_ids = torch.tensor([[10, 100, 100, 100, 100, 11]], dtype=torch.int64)
    assert adapter.calls[0]["teacher_forced_target_token_ids"] == ((12, 13), (14,))
    assert output.teacher_forced_loglikelihoods == (-0.25, -0.75)
    assert output.assistant_prompt_sha256 == hashlib.sha256(exact_prompt.encode()).hexdigest()
    assert output.prefill_input_ids_shape == (1, 6)
    assert (
        output.prefill_input_ids_sha256
        == hashlib.sha256(expected_ids.numpy().tobytes()).hexdigest()
    )


def test_task7_target_preparation_has_no_free_form_prompt_boundary() -> None:
    """Catch a producer accepting a caller-selected prompt or standalone answer tokens."""

    processor = RecordingProcessor()
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": "dummy_content"}]}],
        tokenize=False,
        add_generation_prompt=True,
    )

    class Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            if text == prompt:
                return [1, 2, 3]
            if text == prompt + "42":
                return [1, 2, 3, 19]
            raise AssertionError(f"unexpected tokenization input: {text!r}")

    processor.tokenizer = Tokenizer()

    target = answerers.prepare_task7_likelihood_target_for_question(
        processor, page_count=1, question="what?", accepted_references=("42",)
    )

    assert target["assistant_prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert target["target_token_ids"] == [[19]]
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        answerers.prepare_task7_likelihood_target_for_question(
            processor,
            page_count=1,
            question="what?",
            accepted_references=("42",),
            assistant_prompt="caller-selected prompt",
            assistant_prompt_sha256="0" * 64,
        )


def test_docprune_answerer_passes_forced_intervention_only_when_requested(monkeypatch) -> None:
    """Catch the public answerer dropping a selected fixed-boundary intervention."""

    from transformers import Qwen2VLImageProcessor

    class FakeAdapter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_with_trace(self, **kwargs):
            self.calls.append(kwargs)
            return GenerationResult(
                torch.tensor([[55]]),
                PruningTrace(4, 4, 4, 2, None),
                forced_intervention=ForcedInterventionRecord(
                    boundary="B_input",
                    mode="physical_delete",
                    selection_kind="forced",
                    visual_population=4,
                    requested_budget=2,
                    achieved_budget=2,
                    retained_visual_ids=(0, 2),
                    logical_retained_sequence_ids=(0, 1, 2, 4, 5),
                    prefill_cache_lengths=(5, 5, 5, 5),
                    retained_mrope_position_shape=(3, 1, 5),
                    retained_mrope_position_sha256="a" * 64,
                ),
            )

    processor = RecordingProcessor()
    processor.image_processor = Qwen2VLImageProcessor()
    adapter = FakeAdapter()
    monkeypatch.setattr(answerers, "_validate_prepared_batch", lambda *args: None)
    monkeypatch.setattr(answerers, "DocPruneQwen2VL", lambda _: adapter)
    monkeypatch.setattr(
        DocPruneQwenAnswerer,
        "_masks",
        lambda self, images, prepared, batch, question, retrieval_output: VisionPruningMasks(
            torch.ones(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool)
        ),
    )
    forced = ForcedVisualIntervention("input", "physical_delete", (0, 2))
    answerer = DocPruneQwenAnswerer(
        model=RecordingModel(),
        processor=processor,
        page_config=PagePruningConfig(1.0, 1.0, 1.0, 0.3, 45.0, 0.075),
        forced_intervention=forced,
    )

    output = answerer.answer(
        [Image.new("RGB", (112, 56))], "what?", retrieval_output=retrieval_context()
    )

    assert adapter.calls[0]["forced_intervention"] is forced
    assert output.forced_intervention is not None
    assert output.forced_intervention.selection_kind == "forced"
    assert output.trace.ctp_layer is None


def test_docprune_answerer_passes_corrected_policy_and_keeps_selection_separate(
    monkeypatch,
) -> None:
    """A controlled policy must reach the adapter without becoming a Task 3 forced request."""

    from transformers import Qwen2VLImageProcessor

    from docprune.ctp_policy import aggregate_native_threshold_policy, select_boundary_policy

    selection = select_boundary_policy(
        aggregate_native_threshold_policy(),
        literal_scores=(0.1, 0.9),
        aggregate_scores=(0.2, 0.8),
        attention_threshold=0.5,
        boundary="B_0",
        native_layer=0,
    )

    class FakeAdapter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_with_trace(self, **kwargs):
            self.calls.append(kwargs)
            return GenerationResult(
                torch.tensor([[55]]),
                PruningTrace(4, 4, 4, 1, 0),
                policy_selection=selection,
            )

    processor = RecordingProcessor()
    processor.image_processor = Qwen2VLImageProcessor()
    adapter = FakeAdapter()
    monkeypatch.setattr(answerers, "_validate_prepared_batch", lambda *args: None)
    monkeypatch.setattr(answerers, "DocPruneQwen2VL", lambda _: adapter)
    monkeypatch.setattr(
        DocPruneQwenAnswerer,
        "_masks",
        lambda self, images, prepared, batch, question, retrieval_output: VisionPruningMasks(
            torch.ones(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool)
        ),
    )
    answerer = DocPruneQwenAnswerer(
        model=RecordingModel(),
        processor=processor,
        page_config=PagePruningConfig(1.0, 1.0, 1.0, 0.3, 45.0, 0.075),
        ctp_policy=aggregate_native_threshold_policy(),
    )

    output = answerer.answer(
        [Image.new("RGB", (112, 56))], "what?", retrieval_output=retrieval_context()
    )

    assert adapter.calls[0]["ctp_policy"] == aggregate_native_threshold_policy()
    assert "forced_intervention" not in adapter.calls[0]
    assert output.forced_intervention is None
    assert output.policy_selection is selection


def test_docprune_answerer_uses_runner_qid_for_random_policy_seed_context() -> None:
    """Random policy identity must use the source QID, never question or answer text."""

    from docprune.ctp_policy import fixed_retention_random_policy

    answerer = DocPruneQwenAnswerer(
        model=RecordingModel(),
        processor=RecordingProcessor(),
        ctp_policy=fixed_retention_random_policy("global-uniform-random", "11/20"),
        policy_experiment_version="dev-v1",
        policy_repetition=3,
    )

    answerer.set_policy_question_id("source-qid-17")

    assert answerer.selection_context is not None
    assert answerer.selection_context.experiment_version == "dev-v1"
    assert answerer.selection_context.qid == "source-qid-17"
    assert answerer.selection_context.repetition == 3
    assert answerer.selection_context.geometry is None
    with pytest.raises(ValueError, match="source QID"):
        answerer.set_policy_question_id("")


def test_docprune_answerer_derives_random_policy_geometry_after_exact_qtp_mask(
    monkeypatch,
) -> None:
    """Geometry must be per-question, compact, and derived after BTP and QTP."""

    from transformers import Qwen2VLImageProcessor

    from docprune.ctp_policy import random_top_m_policy

    class GeometryAdapter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_with_trace(self, **kwargs):
            self.calls.append(kwargs)
            return GenerationResult(torch.tensor([[55]]), PruningTrace(4, 3, 2, 2, None))

    processor = RecordingProcessor()
    processor.image_processor = Qwen2VLImageProcessor()
    adapter = GeometryAdapter()
    monkeypatch.setattr(answerers, "_validate_prepared_batch", lambda *args: None)
    monkeypatch.setattr(answerers, "DocPruneQwen2VL", lambda _: adapter)
    monkeypatch.setattr(
        DocPruneQwenAnswerer,
        "_masks",
        lambda self, images, prepared, batch, question, retrieval_output: VisionPruningMasks(
            torch.tensor([True, True, False, True]),
            torch.tensor([True, False, True, True]),
        ),
    )
    answerer = DocPruneQwenAnswerer(
        model=RecordingModel(),
        processor=processor,
        ctp_policy=random_top_m_policy("page-stratified-random"),
        policy_experiment_version="task6-v1",
        policy_repetition=0,
    )
    answerer.set_policy_question_id("q-geometry")

    answerer.answer([Image.new("RGB", (112, 56))], "what?", retrieval_output=retrieval_context())

    context = adapter.calls[0]["selection_context"]
    assert context.qid == "q-geometry"
    assert [
        (token.page_index, token.row, token.column, token.height, token.width)
        for token in context.geometry
    ] == [(0, 0, 0, 2, 2), (0, 1, 1, 2, 2)]
    assert context.geometry_count == 2
    assert context.geometry_sha256 == hashlib.sha256(b"[[0,0,0,2,2],[0,1,1,2,2]]").hexdigest()


def test_docprune_answerer_requires_retrieval_context() -> None:
    answerer = DocPruneQwenAnswerer(model=RecordingModel(), processor=RecordingProcessor())
    with pytest.raises(ValueError, match="retrieval_output"):
        answerer.answer([Image.new("RGB", (112, 56))], "what?")


def test_docprune_answerer_reuses_retrieval_features_without_colpali_forward(monkeypatch) -> None:
    from transformers import Qwen2VLImageProcessor

    class FakeAdapter:
        def generate_with_trace(self, **kwargs):
            return GenerationResult(
                generated_ids=torch.tensor([[55]]),
                trace=PruningTrace(4, 4, 4, 4, None),
            )

    processor = RecordingProcessor()
    processor.image_processor = Qwen2VLImageProcessor()
    monkeypatch.setattr(answerers, "DocPruneQwen2VL", lambda _: FakeAdapter())
    monkeypatch.setattr(answerers, "_validate_prepared_batch", lambda *args: None)
    monkeypatch.setattr(
        DocPruneQwenAnswerer,
        "_masks",
        lambda self, images, prepared, batch, question, retrieval_output: VisionPruningMasks(
            torch.ones(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool)
        ),
    )

    answerer = DocPruneQwenAnswerer(
        model=RecordingModel(),
        processor=processor,
        page_config=PagePruningConfig(1.0, 1.0, 1.0, 0.3, 45.0, 0.075),
    )

    output = answerer.answer(
        [Image.new("RGB", (112, 56))], "what?", retrieval_output=retrieval_context()
    )

    assert output.answer == "answer"


def test_docprune_constructor_does_not_load_colpali() -> None:
    DocPruneQwenAnswerer(model=RecordingModel(), processor=RecordingProcessor())


def test_docprune_answerer_rejects_unknown_qa_stage() -> None:
    with pytest.raises(ValueError, match="qa_stage"):
        DocPruneQwenAnswerer(
            model=RecordingModel(),
            processor=RecordingProcessor(),
            qa_stage="unknown",
        )


@pytest.mark.parametrize(
    ("qa_stage", "expected_question_keep"),
    (
        ("btp-only", [True, True, True, True]),
        ("btp-qtp", [True, True, False, False]),
    ),
)
def test_docprune_diagnostic_stage_disables_later_pruning(
    monkeypatch, qa_stage: str, expected_question_keep: list[bool]
) -> None:
    from transformers import Qwen2VLImageProcessor

    class RecordingAdapter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_with_trace(self, **kwargs):
            ComprehensionController(float(kwargs["comprehension_threshold"]))
            self.calls.append(kwargs)
            return GenerationResult(
                generated_ids=torch.tensor([[55]]),
                trace=PruningTrace(
                    4, 2, sum(expected_question_keep), sum(expected_question_keep), None
                ),
            )

    processor = RecordingProcessor()
    processor.image_processor = Qwen2VLImageProcessor()
    adapter = RecordingAdapter()
    monkeypatch.setattr(answerers, "_validate_prepared_batch", lambda *args: None)
    monkeypatch.setattr(answerers, "DocPruneQwen2VL", lambda _: adapter)
    monkeypatch.setattr(
        DocPruneQwenAnswerer,
        "_masks",
        lambda self, images, prepared, batch, question, retrieval_output: VisionPruningMasks(
            torch.tensor([True, False, True, False]),
            torch.tensor([True, True, False, False]),
        ),
    )
    answerer = DocPruneQwenAnswerer(
        model=RecordingModel(),
        processor=processor,
        page_config=PagePruningConfig(1.0, 1.0, 1.0, 0.3, 45.0, 0.075),
        qa_stage=qa_stage,
    )

    answerer.answer([Image.new("RGB", (112, 56))], "what?", retrieval_output=retrieval_context())

    call = adapter.calls[0]
    assert call["comprehension_threshold"] == 1e9
    assert call["pruning_masks"].question_keep.tolist() == expected_question_keep


def test_docprune_rejects_batched_qwen_page_order_drift(monkeypatch) -> None:
    from transformers import Qwen2VLImageProcessor

    class DriftProcessor(RecordingProcessor):
        def __init__(self) -> None:
            super().__init__()
            self.image_processor = Qwen2VLImageProcessor()

        def __call__(self, *, text, images, padding, return_tensors):
            del text, padding, return_tensors
            pages = [answerers.prepare_qwen_page(self, image) for image in images]
            swapped = list(reversed(pages))
            input_ids = torch.tensor([[10, *([100] * 8), 11, *([100] * 8), 12]])
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "pixel_values": torch.cat([page.pixel_values for page in swapped]),
                "image_grid_thw": torch.cat([page.image_grid_thw for page in swapped]),
            }

    monkeypatch.setattr(
        answerers.DocPruneQwenAnswerer,
        "_masks",
        lambda self, images, prepared, batch, question, retrieval_output: VisionPruningMasks(
            torch.ones(16, dtype=torch.bool), torch.ones(16, dtype=torch.bool)
        ),
    )
    monkeypatch.setattr(answerers, "DocPruneQwen2VL", lambda _: object(), raising=False)

    answerer = DocPruneQwenAnswerer(
        model=RecordingModel(),
        processor=DriftProcessor(),
        page_config=PagePruningConfig(1.0, 1.0, 1.0, 0.3, 45.0, 0.075),
    )
    with pytest.raises(ValueError, match="batched Qwen"):
        answerer.answer(
            [
                Image.new("RGB", (112, 56), color=(1, 2, 3)),
                Image.new("RGB", (56, 112), color=(4, 5, 6)),
            ],
            "what?",
            retrieval_output=retrieval_context(2),
        )


def test_valid_two_page_qwen_placeholders_allow_separator_tokens() -> None:
    pages = [
        PreparedQwenPage(
            raster=torch.zeros((1, 3, 56, 56), dtype=torch.uint8),
            pixel_values=torch.zeros((16, 392)),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
            placeholder_count=4,
            placeholder_raster_indices=torch.arange(4),
        ),
        PreparedQwenPage(
            raster=torch.ones((1, 3, 56, 56), dtype=torch.uint8),
            pixel_values=torch.ones((16, 392)),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
            placeholder_count=4,
            placeholder_raster_indices=torch.arange(4),
        ),
    ]
    batch = {
        "input_ids": torch.tensor([[10, 100, 100, 100, 100, 11, 100, 100, 100, 100, 12]]),
        "pixel_values": torch.cat([page.pixel_values for page in pages]),
        "image_grid_thw": torch.cat([page.image_grid_thw for page in pages]),
    }
    model = type("Model", (), {"config": type("Config", (), {"image_token_id": 100})()})()

    answerers._validate_prepared_batch(pages, batch, model, object())
