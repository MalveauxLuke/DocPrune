from __future__ import annotations

import pytest
import torch
from PIL import Image

from docprune import answerers
from docprune.answerers import AllKeptQwenAnswerer, DocPruneQwenAnswerer
from docprune.config import PagePruningConfig
from docprune.ctp import ComprehensionController
from docprune.m3docrag import RetrievalOutput, RetrievedPage, RetrievedPageFeatures
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
    config = type("Config", (), {"image_token_id": 100})()

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

    output = AllKeptQwenAnswerer(
        model=HookedModel(), processor=RecordingProcessor()
    ).answer(["page"], "what?")

    assert output.encoder_seconds > 0
    assert output.decoder_seconds > 0


def test_docprune_answerer_decodes_adapter_suffix_without_prompt(monkeypatch) -> None:
    from transformers import Qwen2VLImageProcessor

    class FakeAdapter:
        def generate_with_trace(self, **kwargs):
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
    monkeypatch.setattr(answerers, "DocPruneQwen2VL", lambda _: FakeAdapter())
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
                trace=PruningTrace(4, 2, sum(expected_question_keep), sum(expected_question_keep), None),
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

    answerer.answer(
        [Image.new("RGB", (112, 56))], "what?", retrieval_output=retrieval_context()
    )

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
