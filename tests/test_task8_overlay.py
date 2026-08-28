from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

import docprune.task8_overlay as overlay_module
from docprune.ctp_controls import VisualTokenGeometry
from docprune.segmentation import (
    InputPageIdentity,
    MinerUArtifactIdentity,
    build_region_mapping_from_artifacts,
    region_mapping_json_bytes,
)
from docprune.task6_runtime import TASK6_RENDERER_CONTRACT
from docprune.task8_overlay import publish_task8_region_overlays


def _mapping(tmp_path: Path):
    image = Image.new("RGB", (20, 10), color=(240, 240, 240))
    input_path = tmp_path / "page-00.png"
    image.save(input_path)
    raw = json.dumps(
        {
            "_backend": "vlm",
            "_version_name": "3.0.9",
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [200, 100],
                    "para_blocks": [
                        {"type": "text", "bbox": [0, 0, 100, 100]},
                        {"type": "footnote", "bbox": [20, 20, 40, 40]},
                    ],
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    raw_path = tmp_path / "page-00_middle.json"
    raw_path.write_bytes(raw)
    page = InputPageIdentity(
        0,
        0,
        "doc-a",
        7,
        tmp_path / "doc-a.pdf",
        "1" * 64,
        "q-1",
        0,
        tmp_path / "fixture.json",
        "2" * 64,
        "3" * 64,
        image.width,
        image.height,
        hashlib.sha256(image.tobytes()).hexdigest(),
        TASK6_RENDERER_CONTRACT,
    )
    artifact = MinerUArtifactIdentity(
        raw_path,
        hashlib.sha256(raw).hexdigest(),
        tmp_path / "smoke.json",
        "4" * 64,
        input_path,
        hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "vlm",
        "3.0.9",
        "https://github.com/opendatalab/MinerU",
        "d9cd58add047c2364c1198eefcb1ee9cd63a971a",
        "opendatalab/MinerU2.5-Pro-2604-1.2B",
        "d3f5e08d073c21466bbabe21c71bb1e9c2e595da",
        tmp_path / "config.json",
        "5" * 64,
        tmp_path / "tool.json",
        "6" * 64,
        (page,),
    )
    geometry = tuple(
        VisualTokenGeometry(0, row, column, 2, 2) for row in range(2) for column in range(2)
    )
    return build_region_mapping_from_artifacts(geometry, ((artifact, raw),))


def test_overlay_publishes_deterministic_region_and_token_panels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mapping = _mapping(tmp_path)
    monkeypatch.setattr(overlay_module, "load_region_mapping", lambda path: mapping)
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_bytes(region_mapping_json_bytes(mapping))
    output = tmp_path / "overlays"

    manifest = publish_task8_region_overlays(mapping_path, output)

    assert manifest["mapping_sha256"] == mapping.sha256
    assert manifest["mapping_file_sha256"] == hashlib.sha256(mapping_path.read_bytes()).hexdigest()
    assert manifest["page_count"] == 1
    assert manifest["overlay_contract"].startswith("left=source-boxes;right=post-qtp-token-masks")
    page = manifest["pages"][0]
    assert page["input_page_index"] == 0
    assert page["token_count"] == 4
    assert sum(source["token_cost"] for source in page["sources"]) == 4
    assert page["empty_regions"] == [
        {
            "source_id": "mineru:q0:doc-a:7:m0-r1",
            "region_type": "footnote",
            "color_rgb": list(overlay_module._source_color("mineru:q0:doc-a:7:m0-r1")),
            "token_cost": 0,
        }
    ]
    overlay_path = output / page["output_filename"]
    assert hashlib.sha256(overlay_path.read_bytes()).hexdigest() == page["output_sha256"]
    with Image.open(overlay_path) as rendered:
        assert rendered.size == (48, 10)
        assert rendered.convert("RGB").getpixel((5, 5)) != (240, 240, 240)
        assert rendered.convert("RGB").getpixel((33, 2)) != (240, 240, 240)
    manifest_bytes = (output / "manifest.json").read_bytes()
    assert manifest_bytes.endswith(b"\n")
    assert json.loads(manifest_bytes) == manifest

    replay_output = tmp_path / "overlays-replay"
    replay = publish_task8_region_overlays(mapping_path, replay_output)
    assert replay == manifest
    assert (replay_output / page["output_filename"]).read_bytes() == overlay_path.read_bytes()

    with pytest.raises(FileExistsError):
        publish_task8_region_overlays(mapping_path, output)


def test_overlay_rejects_input_pixel_or_file_identity_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mapping = _mapping(tmp_path)
    monkeypatch.setattr(overlay_module, "load_region_mapping", lambda path: mapping)
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_bytes(region_mapping_json_bytes(mapping))
    Image.new("RGB", (20, 10), color=(1, 2, 3)).save(mapping.artifacts[0].mineru_input_path)

    with pytest.raises(ValueError, match="input image SHA-256"):
        publish_task8_region_overlays(mapping_path, tmp_path / "overlays")
    assert not (tmp_path / "overlays" / "manifest.json").exists()
    assert not (tmp_path / "overlays").exists()
    assert not list(tmp_path.glob(".overlays.stage-*"))


def test_overlay_rejects_mapping_two_open_drift_before_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mapping = _mapping(tmp_path)
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_bytes(region_mapping_json_bytes(mapping))

    def drift_during_load(path: Path):
        path.write_bytes(b"replaced between reads\n")
        return mapping

    monkeypatch.setattr(overlay_module, "load_region_mapping", drift_during_load)

    with pytest.raises(ValueError, match="changed during authentication"):
        publish_task8_region_overlays(mapping_path, tmp_path / "overlays")
    assert not (tmp_path / "overlays").exists()


def test_overlay_stages_outputs_and_cleans_failed_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mapping = _mapping(tmp_path)
    monkeypatch.setattr(overlay_module, "load_region_mapping", lambda path: mapping)
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_bytes(region_mapping_json_bytes(mapping))
    original_write = overlay_module._write_new

    def fail_manifest(directory_descriptor: int, filename: str, content: bytes) -> None:
        if filename == "manifest.json":
            raise OSError("injected publication failure")
        original_write(directory_descriptor, filename, content)

    monkeypatch.setattr(overlay_module, "_write_new", fail_manifest)

    with pytest.raises(OSError, match="injected publication failure"):
        publish_task8_region_overlays(mapping_path, tmp_path / "overlays")
    assert not (tmp_path / "overlays").exists()
    assert not list(tmp_path.glob(".overlays.stage-*"))


def test_overlay_cli_is_artifact_only() -> None:
    script = Path("examples/render_task8_region_overlays.py").read_text(encoding="utf-8")

    assert "publish_task8_region_overlays" in script
    assert "retriev" not in script.lower()
    assert "load_pinned_colpali" not in script
    assert "load_pinned_qwen" not in script
