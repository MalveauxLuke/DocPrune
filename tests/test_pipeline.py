import torch

from docprune.config import PagePruningConfig, ReconstructionDefaults
from docprune.pipeline import prepare_qa_pruning_masks


def test_pipeline_supports_page_specific_qwen_grids() -> None:
    images = [
        torch.full((1, 1, 4, 4), 255, dtype=torch.uint8),
        torch.tensor([[[[0, 0, 255, 255], [0, 0, 255, 255]]]], dtype=torch.uint8),
    ]
    document_tokens = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    ]
    question_tokens = torch.tensor([[1.0, 0.0]])
    config = PagePruningConfig(1.0, 0.5, 1.0, 0.5, 45.0, 0.075)

    got = prepare_qa_pruning_masks(
        resized_images=images,
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 2, 4]]),
        document_tokens=document_tokens,
        document_source_hw=((2, 2), (1, 2)),
        question_tokens=question_tokens,
        patch_size=1,
        page_config=config,
        reconstruction=ReconstructionDefaults(gaussian_sigma=0.01),
    )

    assert got.background_keep.shape == (6,)
    assert got.question_keep.shape == (6,)
    assert got.combined().shape == (6,)


def test_pipeline_rejects_nonvisual_retrieval_tokens_without_explicit_slicing() -> None:
    try:
        prepare_qa_pruning_masks(
            resized_images=[torch.zeros((1, 1, 2, 2), dtype=torch.uint8)],
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            document_tokens=[torch.zeros((5, 2))],
            document_source_hw=((2, 2),),
            question_tokens=torch.zeros((1, 2)),
            patch_size=1,
            page_config=PagePruningConfig(1.0, 1.0, 1.0, 0.0, 45.0, 0.075),
            reconstruction=ReconstructionDefaults(),
        )
    except ValueError as error:
        assert "visual-only" in str(error)
    else:
        raise AssertionError("ambiguous ColPali special tokens should be rejected")
