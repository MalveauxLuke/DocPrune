from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import docprune.task8_mapping as mapping_module
from docprune.segmentation import InputPageIdentity, MinerUArtifactIdentity
from docprune.task6_runtime import TASK6_RENDERER_CONTRACT
from docprune.task8_mapping import publish_task8_region_mapping


def _artifact(tmp_path: Path, page_index: int) -> tuple[MinerUArtifactIdentity, bytes]:
    raw = json.dumps(
        {
            "_backend": "vlm",
            "_version_name": "3.0.9",
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [100, 100],
                    "para_blocks": [
                        {"type": "text", "bbox": [0, 0, 50, 100]},
                    ],
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    raw_path = tmp_path / f"page-{page_index:02d}_middle.json"
    raw_path.write_bytes(raw)
    page = InputPageIdentity(
        page_index,
        0,
        f"doc-{page_index}",
        page_index,
        tmp_path / f"doc-{page_index}.pdf",
        "1" * 64,
        "q-1",
        page_index,
        tmp_path / "fixture.json",
        "2" * 64,
        str(page_index) * 64,
        100,
        100,
        "3" * 64,
        TASK6_RENDERER_CONTRACT,
    )
    artifact = MinerUArtifactIdentity(
        raw_path,
        hashlib.sha256(raw).hexdigest(),
        tmp_path / "smoke.json",
        "4" * 64,
        tmp_path / f"page-{page_index:02d}.png",
        "5" * 64,
        "vlm",
        "3.0.9",
        "https://github.com/opendatalab/MinerU",
        "d9cd58add047c2364c1198eefcb1ee9cd63a971a",
        "opendatalab/MinerU2.5-Pro-2604-1.2B",
        "d3f5e08d073c21466bbabe21c71bb1e9c2e595da",
        tmp_path / "config.json",
        "6" * 64,
        tmp_path / "tool.json",
        "7" * 64,
        (page,),
    )
    return artifact, raw


def _manifests(tmp_path: Path):
    pairs = tuple(_artifact(tmp_path, page) for page in range(2))
    completion = {
        "global_index_loaded": False,
        "retrieval_run": False,
        "artifacts": [
            {"identity": artifact.to_dict(), "region_count": 1, "region_types": ["text"]}
            for artifact, _raw in pairs
        ],
    }
    rows = [[0, 0, 0, 1, 1], [1, 0, 0, 1, 1]]
    geometry_sha = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    geometry = {
        "qid": "q-1",
        "global_index_loaded": False,
        "retrieval_search_run": False,
        "qwen_generation_model_loaded": False,
        "fixed_page_fixture_path": str(tmp_path / "fixture.json"),
        "fixed_page_fixture_sha256": "2" * 64,
        "smoke_input_manifest_path": str(tmp_path / "smoke.json"),
        "smoke_input_manifest_sha256": "4" * 64,
        "gpu": {"device_index": 0, "device_name": "NVIDIA L40S"},
        "geometry_count": 2,
        "geometry_sha256": geometry_sha,
        "geometry": rows,
    }
    return completion, geometry, pairs


def test_publish_task8_mapping_composes_only_cross_authenticated_live_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    completion, geometry, pairs = _manifests(tmp_path)
    monkeypatch.setattr(
        mapping_module,
        "load_task8_mineru_completion",
        lambda path, expected_sha256: completion,
    )
    monkeypatch.setattr(
        mapping_module,
        "load_task8_geometry_capture",
        lambda path, expected_sha256: geometry,
    )
    published = []
    monkeypatch.setattr(
        mapping_module,
        "publish_region_mapping",
        lambda path, mapping: published.append((path, mapping)),
    )
    output = tmp_path / "mapping.json"

    mapping = publish_task8_region_mapping(
        mineru_completion_path=tmp_path / "completion.json",
        mineru_completion_sha256="8" * 64,
        geometry_capture_path=tmp_path / "geometry.json",
        geometry_capture_sha256="9" * 64,
        output_path=output,
        expected_qid="q-1",
        expected_geometry_count=2,
        expected_geometry_sha256=geometry["geometry_sha256"],
    )

    assert published == [(output, mapping)]
    assert mapping.geometry_count == 2
    assert len(mapping.artifacts) == 2
    assert mapping.token_to_source == (
        "mineru:q0:doc-0:0:m0-r0",
        "mineru:q1:doc-1:1:m0-r0",
    )
    assert [artifact.raw_middle_json_path for artifact in mapping.artifacts] == [
        pair[0].raw_middle_json_path for pair in pairs
    ]


def test_publish_task8_mapping_rejects_cross_manifest_page_or_gpu_drift(
    monkeypatch, tmp_path: Path
) -> None:
    completion, geometry, _pairs = _manifests(tmp_path)
    geometry["gpu"]["device_name"] = "NVIDIA H100"
    monkeypatch.setattr(
        mapping_module,
        "load_task8_mineru_completion",
        lambda path, expected_sha256: completion,
    )
    monkeypatch.setattr(
        mapping_module,
        "load_task8_geometry_capture",
        lambda path, expected_sha256: geometry,
    )

    with pytest.raises(ValueError, match="L40S"):
        publish_task8_region_mapping(
            mineru_completion_path=tmp_path / "completion.json",
            mineru_completion_sha256="8" * 64,
            geometry_capture_path=tmp_path / "geometry.json",
            geometry_capture_sha256="9" * 64,
            output_path=tmp_path / "mapping.json",
            expected_qid="q-1",
            expected_geometry_count=2,
            expected_geometry_sha256=geometry["geometry_sha256"],
        )


def test_task9_mapping_can_accept_any_authenticated_cuda_gpu(
    monkeypatch, tmp_path: Path
) -> None:
    completion, geometry, _pairs = _manifests(tmp_path)
    geometry["gpu"]["device_name"] = "NVIDIA A100-SXM4-40GB"
    monkeypatch.setattr(
        mapping_module,
        "load_task8_mineru_completion",
        lambda path, expected_sha256: completion,
    )
    monkeypatch.setattr(
        mapping_module,
        "load_task8_geometry_capture",
        lambda path, expected_sha256: geometry,
    )
    published = []
    monkeypatch.setattr(
        mapping_module,
        "publish_region_mapping",
        lambda path, mapping: published.append((path, mapping)),
    )

    mapping = publish_task8_region_mapping(
        mineru_completion_path=tmp_path / "completion.json",
        mineru_completion_sha256="8" * 64,
        geometry_capture_path=tmp_path / "geometry.json",
        geometry_capture_sha256="9" * 64,
        output_path=tmp_path / "mapping.json",
        expected_qid="q-1",
        expected_geometry_count=2,
        expected_geometry_sha256=geometry["geometry_sha256"],
        required_geometry_gpu_substring=None,
    )

    assert published == [(tmp_path / "mapping.json", mapping)]


def test_mapping_cli_is_artifact_only_and_pins_canonical_geometry() -> None:
    script = Path("archive/experiments/task6_9_2026_09_10/examples/build_task8_region_mapping.py").read_text(encoding="utf-8")

    assert "publish_task8_region_mapping" in script
    assert "e1e6ed53f9ad11813845088f4cf2f6b1" in script
    assert "3586" in script
    assert "47b32cf6156dcee41da7eab1686219c760dd73803a135534546c1a50519086e8" in script
    assert "load_pinned_colpali" not in script
    assert "load_pinned_qwen" not in script


def test_task9_mapping_cli_explicitly_removes_only_the_gpu_model_name_gate() -> None:
    script = Path("archive/experiments/task6_9_2026_09_10/examples/build_task9_preliminary_region_mapping.py").read_text(
        encoding="utf-8"
    )

    assert "publish_task8_region_mapping" in script
    assert "required_geometry_gpu_substring=None" in script
    assert "expected_geometry_count" in script
    assert "expected_geometry_sha256" in script
    assert "load_pinned_colpali" not in script
    assert "load_pinned_qwen" not in script
