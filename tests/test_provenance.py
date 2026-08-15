import json

import pytest

from docprune.provenance import RunProvenance


def test_provenance_separates_paper_values_from_reconstruction_defaults() -> None:
    provenance = RunProvenance(
        page_count=4,
        source_commit="abc123",
        model_revisions={"qa": "qwen-revision"},
        dataset_revisions={"m3docvqa": "dataset-revision"},
        dependency_versions={"torch": "2.6.0"},
        hardware={"device": "cpu"},
    )

    payload = provenance.to_dict()

    assert payload["paper_values"]["comprehension_threshold"] == 45.0
    assert payload["reconstruction_defaults"]["source"] == "reconstruction_default"
    assert "gaussian_sigma" not in payload["paper_values"]
    assert "attention_threshold" not in payload["reconstruction_defaults"]


def test_provenance_serialization_is_deterministic_json() -> None:
    provenance = RunProvenance(
        page_count=1,
        source_commit="abc123",
        model_revisions={"retriever": "colpali-revision", "qa": "qwen-revision"},
        dataset_revisions={"m3docvqa": "dataset-revision"},
        dependency_versions={"torch": "2.6.0"},
        hardware={"device": "cpu"},
    )

    first = provenance.to_json()
    second = provenance.to_json()

    assert first == second
    assert json.loads(first)["source_commit"] == "abc123"


def test_provenance_rejects_an_unsupported_page_count() -> None:
    with pytest.raises(ValueError, match="1, 2, or 4"):
        RunProvenance(
            page_count=3,
            source_commit="abc123",
            model_revisions={},
            dataset_revisions={},
            dependency_versions={},
            hardware={},
        )
