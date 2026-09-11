from dataclasses import asdict
from pathlib import Path

import pytest

from docprune.config import DocPruneConfig, load_config


def test_paper_page_settings_are_exact() -> None:
    config = DocPruneConfig.paper_defaults()

    assert asdict(config.for_pages(4)) == {
        "retrieval_background_threshold": 1.0,
        "qa_background_threshold": 0.8,
        "background_error_tolerance": 1.0,
        "question_threshold": 0.4,
        "comprehension_threshold": 45.0,
        "attention_threshold": 0.075,
    }


@pytest.mark.parametrize(
    ("page_count", "expected"),
    [
        (1, (0.9, 0.9, 1.0, 0.3, 65.0, 0.5)),
        (2, (1.0, 1.0, 1.0, 0.3, 60.0, 0.25)),
        (4, (1.0, 0.8, 1.0, 0.4, 45.0, 0.075)),
    ],
)
def test_all_paper_page_settings(page_count: int, expected: tuple[float, ...]) -> None:
    setting = DocPruneConfig.paper_defaults().for_pages(page_count)

    assert tuple(asdict(setting).values()) == expected


def test_unsupported_page_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="1, 2, or 4"):
        DocPruneConfig.paper_defaults().for_pages(3)


def test_reconstruction_defaults_are_labeled() -> None:
    defaults = DocPruneConfig.paper_defaults().reconstruction_defaults

    assert defaults.source == "reconstruction_default"
    assert defaults.grayscale == "bt601_uint8"
    assert defaults.gaussian_sigma == 1.0
    assert defaults.group_retention == "any"
    assert defaults.attention_head_aggregation == "mean"
    assert defaults.ctp_timing == "prefill_last_prompt_token"


def test_repository_toml_loads_the_same_paper_values() -> None:
    config = load_config(Path("legacy/configs/docprune-m3docvqa.toml"))

    assert config.for_pages(1) == DocPruneConfig.paper_defaults().for_pages(1)
    assert config.for_pages(2) == DocPruneConfig.paper_defaults().for_pages(2)
    assert config.for_pages(4) == DocPruneConfig.paper_defaults().for_pages(4)
