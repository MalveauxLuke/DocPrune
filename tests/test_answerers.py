from __future__ import annotations

import torch
from PIL import Image

from docprune import answerers
from docprune.answerers import AllKeptQwenAnswerer, DocPruneQwenAnswerer
from docprune.qwen2vl.model import GenerationResult, PruningTrace


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
        self.calls.append({"text": text, "images": images, "padding": padding, "return_tensors": return_tensors})
        return {
            "input_ids": torch.tensor([[10, 100, 100, 100, 100, 11]]),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "pixel_values": torch.zeros((16, 24)),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }

    def batch_decode(self, values, *, skip_special_tokens: bool, clean_up_tokenization_spaces: bool):
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
    call = model.calls[0]
    assert call["max_new_tokens"] == 128
    assert call["do_sample"] is False
    assert call["num_beams"] == 1
    assert call["input_ids"].tolist() == [[10, 100, 100, 100, 100, 11]]


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
    monkeypatch.setattr(answerers, "DocPruneQwen2VL", lambda _: FakeAdapter())
    answerer = DocPruneQwenAnswerer(model=RecordingModel(), processor=processor)

    output = answerer.answer([Image.new("RGB", (112, 56))], "what?")

    assert output.answer == "answer"
    assert output.trace.post_ctp_visual_tokens == 1
