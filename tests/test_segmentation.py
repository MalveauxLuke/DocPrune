"""Tests for deterministic MinerU-region to post-QTP token masks."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from docprune.ctp_controls import VisualTokenGeometry
from docprune.segmentation import (
    InputPageIdentity,
    MinerUArtifactIdentity,
    RegionTokenMapping,
    build_region_mapping_from_artifacts,
    load_region_mapping,
    map_post_qtp_tokens_to_regions,
    mineru_regions_from_middle_json,
    publish_region_mapping,
    region_mapping_json_bytes,
)
from docprune.task6_runtime import (
    TASK6_RENDERER_CONTRACT,
    FixedPageFixture,
    FixedPageQuestion,
    FixedPageRecord,
)
from docprune.task8_runtime import seal_task8_smoke_inputs


def _middle_bytes() -> bytes:
    return json.dumps(
        {
            "_backend": "vlm",
            "_version_name": "3.0.9",
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [200, 100],
                    "para_blocks": [
                        {"type": "title", "bbox": [10, 5, 190, 25]},
                        {"type": "text", "bbox": [10, 30, 90, 90]},
                    ],
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _page(
    input_page_index: int = 0,
    mineru_page_index: int = 0,
    document_id: str = "doc-a",
    source_page_index: int = 7,
    *,
    source_pdf_path: Path = Path("/scratch/task8/doc-a.pdf"),
    source_pdf_sha256: str = "a" * 64,
    fixture_path: Path = Path("/scratch/task8/fixture.json"),
    fixture_sha256: str = "b" * 64,
    page_record_sha256: str = "c" * 64,
    rendered_rgb_sha256: str = "d" * 64,
) -> InputPageIdentity:
    return InputPageIdentity(
        input_page_index,
        mineru_page_index,
        document_id,
        source_page_index,
        source_pdf_path,
        source_pdf_sha256,
        "q-1",
        input_page_index,
        fixture_path,
        fixture_sha256,
        page_record_sha256,
        100,
        200,
        rendered_rgb_sha256,
        TASK6_RENDERER_CONTRACT,
    )


def _artifact(
    raw: bytes,
    pages: tuple[InputPageIdentity, ...] | None = None,
    path: Path = Path("/scratch/task8/doc-a_middle.json"),
    smoke_manifest_path: Path = Path("/scratch/task8/smoke-input-manifest.json"),
    smoke_manifest_sha256: str = "2" * 64,
    input_path: Path = Path("/scratch/task8/doc-a.png"),
    input_sha256: str = "1" * 64,
    configuration_path: Path = Path("/scratch/task8/mineru-config.json"),
    tool_manifest_path: Path = Path("/scratch/task8/mineru-tool-manifest.json"),
) -> MinerUArtifactIdentity:
    if pages is None:
        pages = (_page(),)
    return MinerUArtifactIdentity(
        raw_middle_json_path=path,
        raw_middle_json_sha256=hashlib.sha256(raw).hexdigest(),
        smoke_input_manifest_path=smoke_manifest_path,
        smoke_input_manifest_sha256=smoke_manifest_sha256,
        mineru_input_path=input_path,
        mineru_input_sha256=input_sha256,
        backend="vlm",
        version="3.0.9",
        repository_id="https://github.com/opendatalab/MinerU",
        repository_revision="d9cd58add047c2364c1198eefcb1ee9cd63a971a",
        model_repository_id="opendatalab/MinerU2.5-Pro-2604-1.2B",
        model_revision="d3f5e08d073c21466bbabe21c71bb1e9c2e595da",
        configuration_path=configuration_path,
        configuration_sha256="f" * 64,
        tool_manifest_path=tool_manifest_path,
        tool_manifest_sha256="e" * 64,
        pages=pages,
    )


def test_mineru_parser_separates_qwen_local_and_source_page_identities() -> None:
    raw = _middle_bytes()
    artifact = _artifact(raw)

    regions = mineru_regions_from_middle_json(raw, artifact)

    assert [region.to_dict() for region in regions] == [
        {
            "input_page_index": 0,
            "mineru_page_index": 0,
            "document_id": "doc-a",
            "source_page_index": 7,
            "region_id": "m0-r0",
            "region_type": "title",
            "bbox": [10.0, 5.0, 190.0, 25.0],
            "page_size": [200.0, 100.0],
            "reading_order": 0,
        },
        {
            "input_page_index": 0,
            "mineru_page_index": 0,
            "document_id": "doc-a",
            "source_page_index": 7,
            "region_id": "m0-r1",
            "region_type": "text",
            "bbox": [10.0, 30.0, 90.0, 90.0],
            "page_size": [200.0, 100.0],
            "reading_order": 1,
        },
    ]

    rebound = mineru_regions_from_middle_json(
        raw,
        _artifact(raw, pages=(_page(source_page_index=0),)),
    )
    assert rebound[0].mineru_page_index == 0
    assert rebound[0].source_page_index == 0


def test_mineru_parser_rejects_raw_hash_page_coverage_or_bbox_drift() -> None:
    raw = _middle_bytes()
    with pytest.raises(ValueError, match="SHA-256"):
        mineru_regions_from_middle_json(raw + b" ", _artifact(raw))

    missing_page = _artifact(
        raw,
        pages=(_page(mineru_page_index=1),),
    )
    with pytest.raises(ValueError, match="page coverage"):
        mineru_regions_from_middle_json(raw, missing_page)

    with pytest.raises(ValueError, match="exactly one"):
        _artifact(raw, pages=(_page(), _page(1, 1, "doc-a", 8)))

    payload = json.loads(raw)
    payload["pdf_info"][0]["para_blocks"][0]["bbox"] = [10, 5, 205, 25]
    malformed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="bbox"):
        mineru_regions_from_middle_json(malformed, _artifact(malformed))


def test_region_mapping_assigns_each_token_once_with_deterministic_overlap_ties() -> None:
    raw = _middle_bytes()
    artifact = _artifact(raw)
    earlier, later = mineru_regions_from_middle_json(raw, artifact)
    tied = (
        later,
        type(earlier)(
            earlier.input_page_index,
            earlier.mineru_page_index,
            earlier.document_id,
            earlier.source_page_index,
            earlier.region_id,
            earlier.region_type,
            later.bbox,
            earlier.page_size,
            earlier.reading_order,
        ),
    )
    geometry = tuple(
        VisualTokenGeometry(0, row, column, 2, 2) for row in range(2) for column in range(2)
    )

    mapping = map_post_qtp_tokens_to_regions(geometry, (artifact,), tied)

    assert mapping.token_to_source == (
        "mineru:q0:doc-a:7:m0-r0",
        "residual:q0:doc-a:7:r1:c3",
        "mineru:q0:doc-a:7:m0-r0",
        "residual:q0:doc-a:7:r3:c3",
    )
    assigned = [token_id for source in mapping.sources for token_id in source.token_ids]
    assert sorted(assigned) == list(range(len(geometry)))
    assert len(assigned) == len(set(assigned))
    assert sum(source.token_cost for source in mapping.sources) == len(geometry)


def test_mapping_preserves_empty_regions_only_as_audit_evidence() -> None:
    payload = json.loads(_middle_bytes())
    payload["pdf_info"][0]["para_blocks"] = [{"type": "text", "bbox": [80, 40, 120, 60]}]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    artifact = _artifact(raw)
    regions = mineru_regions_from_middle_json(raw, artifact)
    geometry = (
        VisualTokenGeometry(0, 0, 0, 4, 4),
        VisualTokenGeometry(0, 3, 3, 4, 4),
    )

    mapping = map_post_qtp_tokens_to_regions(geometry, (artifact,), regions)

    assert mapping.empty_region_source_ids == ("mineru:q0:doc-a:7:m0-r0",)
    assert all(source.source_kind == "residual-grid" for source in mapping.sources)
    assert all(source.token_cost > 0 for source in mapping.sources)
    assert mapping.token_to_source == (
        "residual:q0:doc-a:7:r0:c0",
        "residual:q0:doc-a:7:r3:c3",
    )


def test_mapping_digest_binds_geometry_artifacts_grid_and_assignment_contract() -> None:
    raw = _middle_bytes()
    artifact = _artifact(raw)
    regions = mineru_regions_from_middle_json(raw, artifact)
    geometry = (VisualTokenGeometry(0, 0, 0, 1, 1),)
    baseline = map_post_qtp_tokens_to_regions(geometry, (artifact,), regions)

    assert baseline.geometry_count == 1
    assert len(baseline.geometry_sha256) == 64
    assert baseline.residual_grid_size == 4
    assert "maximum-positive-normalized-intersection" in baseline.assignment_contract
    assert baseline.artifacts == (artifact,)
    assert baseline == map_post_qtp_tokens_to_regions(geometry, (artifact,), regions)
    assert (
        baseline.sha256
        != map_post_qtp_tokens_to_regions(
            geometry, (artifact,), regions, residual_grid_size=5
        ).sha256
    )

    shifted = (VisualTokenGeometry(0, 0, 0, 1, 2),)
    assert baseline.sha256 != map_post_qtp_tokens_to_regions(shifted, (artifact,), regions).sha256


def test_mapping_replay_is_strict_and_rejects_corruption() -> None:
    raw = _middle_bytes()
    artifact = _artifact(raw)
    regions = mineru_regions_from_middle_json(raw, artifact)
    geometry = tuple(VisualTokenGeometry(0, 0, column, 1, 2) for column in range(2))
    mapping = map_post_qtp_tokens_to_regions(geometry, (artifact,), regions)
    serialized = mapping.to_dict()

    assert RegionTokenMapping.from_dict(serialized) == mapping
    assert json.dumps(serialized, sort_keys=True, separators=(",", ":")) == json.dumps(
        mapping.to_dict(), sort_keys=True, separators=(",", ":")
    )

    for mutation in ("digest", "duplicate", "token-map"):
        corrupted = deepcopy(serialized)
        if mutation == "digest":
            corrupted["sha256"] = "0" * 64
        elif mutation == "duplicate":
            corrupted["sources"][0]["token_ids"].append(corrupted["sources"][0]["token_ids"][0])
            corrupted["sources"][0]["token_cost"] += 1
        else:
            corrupted["token_to_source"][0] = "fabricated"
        with pytest.raises(ValueError, match="mapping"):
            RegionTokenMapping.from_dict(corrupted)


def test_mapping_publication_and_load_reauthenticate_raw_mineru_bytes(tmp_path: Path) -> None:
    raw = _middle_bytes()
    raw_path = tmp_path / "doc-a_middle.json"
    raw_path.write_bytes(raw)
    source_path = tmp_path / "doc-a.pdf"
    source_path.write_bytes(b"source-pdf")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    feature_path = tmp_path / "doc-a.safetensors"
    feature_path.write_bytes(b"fixed-feature-shard")
    feature_sha = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    rendered_image = Image.new("RGB", (100, 200), color=(7, 8, 9))
    rendered_rgb_sha = hashlib.sha256(rendered_image.tobytes()).hexdigest()
    page_record = FixedPageRecord(
        rank=0,
        doc_id="doc-a",
        page_index=7,
        score=1.0,
        source_pdf_path=source_path,
        source_pdf_sha256=source_sha,
        rendered_rgb_width=100,
        rendered_rgb_height=200,
        rendered_rgb_sha256=rendered_rgb_sha,
        renderer_contract=TASK6_RENDERER_CONTRACT,
        feature_shard_path=feature_path,
        feature_shard_sha256=feature_sha,
        feature_page_index=7,
    )
    page_record_sha = hashlib.sha256(
        json.dumps(page_record.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    reference_path = tmp_path / "reference.json"
    eligible_path = tmp_path / "eligible.jsonl"
    feature_manifest_path = tmp_path / "feature-manifest.json"
    completion_ledger_path = tmp_path / "completion-ledger.jsonl"
    for path, content in (
        (reference_path, b"{}\n"),
        (eligible_path, b'{"qid":"q-1"}\n'),
        (feature_manifest_path, b'{"fixed":true}\n'),
        (completion_ledger_path, b'{"complete":true}\n'),
    ):
        path.write_bytes(content)
    fixture = FixedPageFixture(
        fixture_version="task8-test-v1",
        reference_path=reference_path,
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        eligible_questions_path=eligible_path,
        eligible_questions_sha256=hashlib.sha256(eligible_path.read_bytes()).hexdigest(),
        feature_manifest_path=feature_manifest_path,
        feature_manifest_sha256=hashlib.sha256(feature_manifest_path.read_bytes()).hexdigest(),
        completion_ledger_path=completion_ledger_path,
        completion_ledger_sha256=hashlib.sha256(completion_ledger_path.read_bytes()).hexdigest(),
        questions=(FixedPageQuestion("q-1", "9" * 64, (page_record,)),),
    )
    fixture_path = tmp_path / "fixture.json"
    fixture_bytes = json.dumps(fixture.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    fixture_path.write_bytes(fixture_bytes)
    fixture_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    smoke_root = tmp_path / "smoke-inputs"
    smoke_manifest = seal_task8_smoke_inputs(
        fixture_path=fixture_path,
        fixture_sha256=fixture_sha,
        qid="q-1",
        output_root=smoke_root,
        runtime_commit="a" * 40,
        render_page=lambda _path, _page: rendered_image.copy(),
    )
    smoke_manifest_path = smoke_root / "smoke-input-manifest.json"
    smoke_manifest_sha = hashlib.sha256(smoke_manifest_path.read_bytes()).hexdigest()
    mineru_input_path = Path(smoke_manifest["pages"][0]["mineru_input_path"])
    mineru_input_sha = smoke_manifest["pages"][0]["mineru_input_sha256"]
    config_path = tmp_path / "mineru-config.json"
    config_path.write_bytes(b'{"device":"cpu"}\n')
    manifest_path = tmp_path / "mineru-tool-manifest.json"
    configuration_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest_path.write_bytes(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pinned-mineru-tool",
                "backend": "vlm",
                "version": "3.0.9",
                "repository_id": "https://github.com/opendatalab/MinerU",
                "repository_revision": "d9cd58add047c2364c1198eefcb1ee9cd63a971a",
                "model_repository_id": "opendatalab/MinerU2.5-Pro-2604-1.2B",
                "model_revision": "d3f5e08d073c21466bbabe21c71bb1e9c2e595da",
                "configuration_path": str(config_path),
                "configuration_sha256": configuration_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    canonical_tool_manifest_bytes = manifest_path.read_bytes()
    page = _page(
        source_pdf_path=source_path,
        source_pdf_sha256=source_sha,
        fixture_path=fixture_path,
        fixture_sha256=fixture_sha,
        page_record_sha256=page_record_sha,
        rendered_rgb_sha256=rendered_rgb_sha,
    )
    artifact = replace(
        _artifact(
            raw,
            pages=(page,),
            path=raw_path,
            smoke_manifest_path=smoke_manifest_path,
            smoke_manifest_sha256=smoke_manifest_sha,
            input_path=mineru_input_path,
            input_sha256=mineru_input_sha,
            configuration_path=config_path,
            tool_manifest_path=manifest_path,
        ),
        configuration_sha256=configuration_sha,
        tool_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    geometry = tuple(VisualTokenGeometry(0, 0, column, 1, 2) for column in range(2))
    mapping = build_region_mapping_from_artifacts(
        geometry,
        ((artifact, raw),),
    )
    output = tmp_path / "mapping.json"

    publish_region_mapping(output, mapping)

    assert output.read_bytes() == region_mapping_json_bytes(mapping)
    assert load_region_mapping(output) == mapping
    with pytest.raises(FileExistsError):
        publish_region_mapping(output, mapping)

    invalid_output = tmp_path / "invalid-mapping.json"
    with pytest.raises(ValueError, match="mapping"):
        publish_region_mapping(invalid_output, replace(mapping, sha256="0" * 64))
    assert not invalid_output.exists()

    raw_path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="raw MinerU artifact"):
        load_region_mapping(output)
    unauthenticated_output = tmp_path / "unauthenticated-mapping.json"
    with pytest.raises(ValueError, match="raw MinerU artifact"):
        publish_region_mapping(unauthenticated_output, mapping)
    assert not unauthenticated_output.exists()

    raw_path.write_bytes(raw)
    canonical_mineru_input = mineru_input_path.read_bytes()
    mineru_input_path.write_bytes(b"drifted-mineru-input")
    with pytest.raises(ValueError, match="authentication"):
        load_region_mapping(output)
    mineru_input_path.write_bytes(canonical_mineru_input)
    source_path.write_bytes(b"different-source-pdf")
    with pytest.raises(ValueError, match="authentication"):
        load_region_mapping(output)
    source_path.write_bytes(b"source-pdf")
    config_path.write_bytes(b'{"device":"not-the-pinned-config"}\n')
    with pytest.raises(ValueError, match="authentication"):
        load_region_mapping(output)
    config_path.write_bytes(b'{"device":"cpu"}\n')
    fixture_path.write_bytes(fixture_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="authentication"):
        load_region_mapping(output)
    fixture_path.write_bytes(fixture_bytes)
    feature_path.write_bytes(b"drifted-feature-shard")
    with pytest.raises(ValueError, match="authentication"):
        load_region_mapping(output)

    feature_path.write_bytes(b"fixed-feature-shard")
    for name, mutate in (
        (
            "global-index",
            lambda payload: payload.__setitem__("global_index_loaded", True),
        ),
        (
            "missing-feature",
            lambda payload: payload["questions"][0]["pages"][0].pop("feature_shard_sha256"),
        ),
    ):
        malformed = fixture.to_dict()
        mutate(malformed)
        malformed_path = tmp_path / f"fixture-{name}.json"
        malformed_bytes = json.dumps(malformed, sort_keys=True, separators=(",", ":")).encode()
        malformed_path.write_bytes(malformed_bytes)
        malformed_page = replace(
            page,
            fixed_page_fixture_path=malformed_path,
            fixed_page_fixture_sha256=hashlib.sha256(malformed_bytes).hexdigest(),
        )
        malformed_artifact = replace(artifact, pages=(malformed_page,))
        malformed_mapping = build_region_mapping_from_artifacts(
            geometry,
            ((malformed_artifact, raw),),
        )
        malformed_output = tmp_path / f"mapping-{name}.json"
        with pytest.raises(ValueError, match="authentication"):
            publish_region_mapping(malformed_output, malformed_mapping)
        assert not malformed_output.exists()

    colliding_output = tmp_path / "mapping-colliding-tool-roles.json"
    with pytest.raises(ValueError, match="configuration and tool manifest paths"):
        replace(
            artifact,
            tool_manifest_path=config_path,
            tool_manifest_sha256=configuration_sha,
        )
    assert not colliding_output.exists()

    contradictory_manifest = json.loads(manifest_path.read_bytes())
    contradictory_manifest["model_revision"] = "0" * 40
    contradictory_bytes = json.dumps(
        contradictory_manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest_path.write_bytes(contradictory_bytes)
    contradictory_artifact = replace(
        artifact,
        tool_manifest_sha256=hashlib.sha256(contradictory_bytes).hexdigest(),
    )
    contradictory_mapping = build_region_mapping_from_artifacts(
        geometry,
        ((contradictory_artifact, raw),),
    )
    contradictory_output = tmp_path / "mapping-contradictory-tool.json"
    with pytest.raises(ValueError, match="authentication") as error:
        publish_region_mapping(contradictory_output, contradictory_mapping)
    assert "contradicts" in str(error.value.__cause__)
    assert not contradictory_output.exists()

    manifest_path.write_bytes(canonical_tool_manifest_bytes)
    wrong_input = tmp_path / "wrong-but-valid.png"
    Image.new("RGB", (100, 200), color=(10, 11, 12)).save(wrong_input, format="PNG")
    wrong_artifact = replace(
        artifact,
        mineru_input_path=wrong_input,
        mineru_input_sha256=hashlib.sha256(wrong_input.read_bytes()).hexdigest(),
    )
    wrong_mapping = build_region_mapping_from_artifacts(
        geometry,
        ((wrong_artifact, raw),),
    )
    wrong_output = tmp_path / "mapping-wrong-valid-input.json"
    with pytest.raises(ValueError, match="authentication"):
        publish_region_mapping(wrong_output, wrong_mapping)
    assert not wrong_output.exists()


def test_mapping_supports_two_single_page_mineru_outputs_without_page_conflation() -> None:
    raw = _middle_bytes()
    first = _artifact(raw, pages=(_page(),), path=Path("/scratch/task8/a.json"))
    second = _artifact(
        raw,
        pages=(_page(1, 0, "doc-b", 9),),
        path=Path("/scratch/task8/b.json"),
    )
    regions = (
        *mineru_regions_from_middle_json(raw, first),
        *mineru_regions_from_middle_json(raw, second),
    )
    geometry = (
        VisualTokenGeometry(0, 0, 0, 1, 1),
        VisualTokenGeometry(1, 0, 0, 1, 1),
    )

    mapping = map_post_qtp_tokens_to_regions(geometry, (first, second), regions)

    assert mapping.artifacts[0].pages[0].mineru_page_index == 0
    assert mapping.artifacts[1].pages[0].mineru_page_index == 0
    assert mapping.artifacts[0].pages[0].source_page_index == 7
    assert mapping.artifacts[1].pages[0].source_page_index == 9
    with pytest.raises(ValueError, match="homogeneous"):
        map_post_qtp_tokens_to_regions(
            geometry,
            (first, replace(second, model_revision="0" * 40)),
            regions,
        )


def test_mapping_rejects_duplicate_or_non_page_major_geometry() -> None:
    raw = _middle_bytes()
    artifact = _artifact(raw)
    regions = mineru_regions_from_middle_json(raw, artifact)

    for geometry in (
        (
            VisualTokenGeometry(0, 0, 0, 1, 1),
            VisualTokenGeometry(0, 0, 0, 1, 1),
        ),
        (
            VisualTokenGeometry(0, 0, 1, 1, 2),
            VisualTokenGeometry(0, 0, 0, 1, 2),
        ),
        (
            VisualTokenGeometry(0, 0, 0, 1, 1),
            VisualTokenGeometry(0, 0, 1, 1, 2),
        ),
    ):
        with pytest.raises(ValueError, match="geometry"):
            map_post_qtp_tokens_to_regions(geometry, (artifact,), regions)
