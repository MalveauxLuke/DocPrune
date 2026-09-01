from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

import docprune.task8_geometry as geometry_module
from docprune.answerers import BTPQTPGeometryCapture
from docprune.ctp_controls import VisualTokenGeometry
from docprune.task6_runtime import (
    TASK6_RENDERER_CONTRACT,
    FixedPageFixture,
    FixedPageQuestion,
    FixedPageRecord,
)
from docprune.task8_geometry import (
    capture_task8_btp_qtp_geometry,
    load_task8_geometry_capture,
)
from docprune.task8_runtime import seal_task8_smoke_inputs


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> dict[str, object]:
    qid = "q-geometry"
    question = "Which club has a flower?"
    source = tmp_path / "source.pdf"
    source.write_bytes(b"sealed-pdf")
    feature = tmp_path / "features.safetensors"
    save_file(
        {
            "embeddings": torch.ones((2, 128), dtype=torch.float32),
            "page_offsets": torch.tensor([0, 1, 2], dtype=torch.int64),
            "raster_indices": torch.tensor([0, 0], dtype=torch.int64),
        },
        feature,
    )
    eligible = tmp_path / "eligible.jsonl"
    eligible.write_text(
        json.dumps({"qid": qid, "question": question, "answers": [{"answer": "A"}]}) + "\n"
    )
    dependencies = []
    for name in ("reference.json", "feature-manifest.json", "ledger.jsonl"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        dependencies.append(path)
    images = (
        Image.new("RGB", (8, 6), color=(1, 2, 3)),
        Image.new("RGB", (8, 6), color=(4, 5, 6)),
    )
    pages = tuple(
        FixedPageRecord(
            rank=rank,
            doc_id=f"doc-{rank}",
            page_index=rank,
            score=float(2 - rank),
            source_pdf_path=source,
            source_pdf_sha256=_sha(source),
            rendered_rgb_width=image.width,
            rendered_rgb_height=image.height,
            rendered_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            renderer_contract=TASK6_RENDERER_CONTRACT,
            feature_shard_path=feature,
            feature_shard_sha256=_sha(feature),
            feature_page_index=rank,
        )
        for rank, image in enumerate(images)
    )
    fixture = FixedPageFixture(
        fixture_version="task8-geometry-test-v1",
        reference_path=dependencies[0],
        reference_sha256=_sha(dependencies[0]),
        eligible_questions_path=eligible,
        eligible_questions_sha256=_sha(eligible),
        feature_manifest_path=dependencies[1],
        feature_manifest_sha256=_sha(dependencies[1]),
        completion_ledger_path=dependencies[2],
        completion_ledger_sha256=_sha(dependencies[2]),
        questions=(
            FixedPageQuestion(
                qid,
                hashlib.sha256(question.encode()).hexdigest(),
                pages,
            ),
        ),
    )
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture.to_dict(), sort_keys=True, separators=(",", ":")))
    smoke_root = tmp_path / "smoke"
    seal_task8_smoke_inputs(
        fixture_path=fixture_path,
        fixture_sha256=_sha(fixture_path),
        qid=qid,
        output_root=smoke_root,
        runtime_commit="a" * 40,
        render_page=lambda _path, page: images[page].copy(),
    )
    geometry = (
        VisualTokenGeometry(0, 0, 0, 1, 1),
        VisualTokenGeometry(1, 0, 0, 1, 1),
    )
    geometry_sha = hashlib.sha256(b"[[0,0,0,1,1],[1,0,0,1,1]]").hexdigest()
    reference_path = tmp_path / "results.jsonl"
    reference_path.write_text(
        json.dumps(
            {
                "question_id": qid,
                "question": question,
                "fixed_page_fixture_sha256": _sha(fixture_path),
                "fixed_page_provenance": True,
                "global_index_loaded": False,
                "retrieved_pages": [
                    {"doc_id": page.doc_id, "page_index": page.page_index, "score": page.score}
                    for page in pages
                ],
                "trace": {
                    "original_visual_tokens": 4,
                    "post_btp_visual_tokens": 3,
                    "post_qtp_visual_tokens": 2,
                    "post_ctp_visual_tokens": 2,
                    "ctp_layer": None,
                },
                "policy_selection": {
                    "geometry_count": 2,
                    "geometry_sha256": geometry_sha,
                    "policy": {"name": "btp-qtp-no-ctp"},
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "qid": qid,
        "fixture_path": fixture_path,
        "fixture_sha256": _sha(fixture_path),
        "smoke_path": smoke_root / "smoke-input-manifest.json",
        "smoke_sha256": _sha(smoke_root / "smoke-input-manifest.json"),
        "reference_path": reference_path,
        "reference_sha256": _sha(reference_path),
        "geometry": geometry,
        "geometry_sha256": geometry_sha,
    }


class _QueryEncoder:
    def encode_queries(self, queries):
        assert queries == ["Which club has a flower?"]
        return [torch.ones((2, 128), dtype=torch.float32)]


def test_geometry_capture_is_fixed_page_exact_no_replace_and_no_qwen_model(
    monkeypatch, tmp_path: Path
) -> None:
    inputs = _inputs(tmp_path)
    capture = BTPQTPGeometryCapture(
        geometry=inputs["geometry"],
        geometry_count=2,
        geometry_sha256=inputs["geometry_sha256"],
        image_grid_thw=((1, 2, 2), (1, 2, 2)),
        original_visual_tokens=4,
        post_btp_visual_tokens=3,
        post_qtp_visual_tokens=2,
        background_keep_sha256="1" * 64,
        question_keep_sha256="2" * 64,
        combined_keep_sha256="3" * 64,
    )
    observed = {}

    def derive(processor, images, question, retrieval_output):
        observed.update(
            processor=processor,
            image_count=len(images),
            question=question,
            pages=retrieval_output.pages,
        )
        return capture

    monkeypatch.setattr(geometry_module, "derive_btp_qtp_geometry_without_qwen_model", derive)
    output = tmp_path / "geometry.json"
    payload = capture_task8_btp_qtp_geometry(
        fixture_path=inputs["fixture_path"],
        fixture_sha256=inputs["fixture_sha256"],
        smoke_input_manifest_path=inputs["smoke_path"],
        smoke_input_manifest_sha256=inputs["smoke_sha256"],
        reference_results_path=inputs["reference_path"],
        reference_results_sha256=inputs["reference_sha256"],
        qid=inputs["qid"],
        expected_geometry_count=2,
        expected_geometry_sha256=inputs["geometry_sha256"],
        output_path=output,
        runtime_commit="b" * 40,
        query_encoder=_QueryEncoder(),
        qwen_processor=object(),
        require_cuda=False,
    )

    assert payload == json.loads(output.read_text())
    assert load_task8_geometry_capture(output, expected_sha256=_sha(output)) == payload
    assert payload["global_index_loaded"] is False
    assert payload["retrieval_search_run"] is False
    assert payload["qwen_generation_model_loaded"] is False
    assert payload["geometry_count"] == 2
    assert payload["geometry"] == [[0, 0, 0, 1, 1], [1, 0, 0, 1, 1]]
    assert observed["image_count"] == 2
    assert observed["question"] == "Which club has a flower?"
    with pytest.raises(FileExistsError):
        capture_task8_btp_qtp_geometry(
            fixture_path=inputs["fixture_path"],
            fixture_sha256=inputs["fixture_sha256"],
            smoke_input_manifest_path=inputs["smoke_path"],
            smoke_input_manifest_sha256=inputs["smoke_sha256"],
            reference_results_path=inputs["reference_path"],
            reference_results_sha256=inputs["reference_sha256"],
            qid=inputs["qid"],
            expected_geometry_count=2,
            expected_geometry_sha256=inputs["geometry_sha256"],
            output_path=output,
            runtime_commit="b" * 40,
            query_encoder=_QueryEncoder(),
            qwen_processor=object(),
            require_cuda=False,
        )


def test_preliminary_geometry_capture_accepts_authenticated_stage_reference_without_hash(
    monkeypatch, tmp_path: Path
) -> None:
    """Catch requiring a circular predeclared geometry hash for the new random-48 cohort."""

    inputs = _inputs(tmp_path)
    reference = json.loads(Path(inputs["reference_path"]).read_text().strip())
    for key in (
        "fixed_page_fixture_sha256",
        "fixed_page_provenance",
        "global_index_loaded",
        "policy_selection",
    ):
        reference.pop(key)
    Path(inputs["reference_path"]).write_text(json.dumps(reference) + "\n")
    capture = BTPQTPGeometryCapture(
        geometry=inputs["geometry"],
        geometry_count=2,
        geometry_sha256=inputs["geometry_sha256"],
        image_grid_thw=((1, 2, 2), (1, 2, 2)),
        original_visual_tokens=4,
        post_btp_visual_tokens=3,
        post_qtp_visual_tokens=2,
        background_keep_sha256="1" * 64,
        question_keep_sha256="2" * 64,
        combined_keep_sha256="3" * 64,
    )
    monkeypatch.setattr(
        geometry_module,
        "derive_btp_qtp_geometry_without_qwen_model",
        lambda *_args: capture,
    )
    output = tmp_path / "preliminary-geometry.json"

    payload = capture_task8_btp_qtp_geometry(
        fixture_path=inputs["fixture_path"],
        fixture_sha256=inputs["fixture_sha256"],
        smoke_input_manifest_path=inputs["smoke_path"],
        smoke_input_manifest_sha256=inputs["smoke_sha256"],
        reference_results_path=inputs["reference_path"],
        reference_results_sha256=_sha(inputs["reference_path"]),
        qid=inputs["qid"],
        expected_geometry_count=2,
        expected_geometry_sha256=None,
        output_path=output,
        runtime_commit="c" * 40,
        query_encoder=_QueryEncoder(),
        qwen_processor=object(),
        require_cuda=False,
    )

    assert payload["schema_version"] == "docprune-task9-preliminary-btp-qtp-geometry-v1"
    assert payload["geometry_sha256"] == inputs["geometry_sha256"]
    assert load_task8_geometry_capture(output, expected_sha256=_sha(output)) == payload
    canonical = output.read_bytes()
    contradicted = json.loads(canonical)
    contradicted["retrieved_pages"][0]["score"] += 1.0
    unsigned = {key: value for key, value in contradicted.items() if key != "manifest_sha256"}
    contradicted["manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(contradicted, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError, match="reference"):
        load_task8_geometry_capture(output, expected_sha256=_sha(output))
    output.write_bytes(canonical)
    contradicted = json.loads(canonical)
    contradicted["resources"]["qwen_processor_revision"] = "0" * 40
    unsigned = {key: value for key, value in contradicted.items() if key != "manifest_sha256"}
    contradicted["manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(contradicted, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError, match="resource"):
        load_task8_geometry_capture(output, expected_sha256=_sha(output))
    output.write_bytes(canonical)
    output.write_bytes(canonical + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        load_task8_geometry_capture(output, expected_sha256=_sha(output))


def test_geometry_capture_rejects_reference_or_live_geometry_drift_without_output(
    monkeypatch, tmp_path: Path
) -> None:
    inputs = _inputs(tmp_path)
    drifted = BTPQTPGeometryCapture(
        geometry=inputs["geometry"][:1],
        geometry_count=2,
        geometry_sha256=inputs["geometry_sha256"],
        image_grid_thw=((1, 2, 2), (1, 2, 2)),
        original_visual_tokens=4,
        post_btp_visual_tokens=3,
        post_qtp_visual_tokens=2,
        background_keep_sha256="1" * 64,
        question_keep_sha256="2" * 64,
        combined_keep_sha256="3" * 64,
    )
    monkeypatch.setattr(
        geometry_module,
        "derive_btp_qtp_geometry_without_qwen_model",
        lambda *_args, **_kwargs: drifted,
    )
    output = tmp_path / "drifted.json"
    with pytest.raises(ValueError, match="geometry rows"):
        capture_task8_btp_qtp_geometry(
            fixture_path=inputs["fixture_path"],
            fixture_sha256=inputs["fixture_sha256"],
            smoke_input_manifest_path=inputs["smoke_path"],
            smoke_input_manifest_sha256=inputs["smoke_sha256"],
            reference_results_path=inputs["reference_path"],
            reference_results_sha256=inputs["reference_sha256"],
            qid=inputs["qid"],
            expected_geometry_count=2,
            expected_geometry_sha256=inputs["geometry_sha256"],
            output_path=output,
            runtime_commit="b" * 40,
            query_encoder=_QueryEncoder(),
            qwen_processor=object(),
            require_cuda=False,
        )
    assert not output.exists()


def test_geometry_launcher_is_single_short_l40s_fixed_page_capture() -> None:
    launcher = Path("examples/sbatch/37_docprune_task8_geometry_capture.sbatch").read_text(
        encoding="utf-8"
    )
    wrapper = Path("examples/run_task8_geometry_capture.py").read_text(encoding="utf-8")

    assert "#SBATCH --constraint=l40s" in launcher
    assert "#SBATCH --time=00:15:00" in launcher
    assert "#SBATCH --no-requeue" in launcher
    assert "#SBATCH --array" not in launcher
    assert "fixture-stage64-top4-v2.json" in launcher
    assert "task8-mineru-smoke-inputs-d2429ba-v1" in launcher
    assert "task6-native-l40s-2d2bea2-v1/shard-0000/results.jsonl" in launcher
    assert "e1e6ed53f9ad11813845088f4cf2f6b1" in launcher
    assert "3586" in launcher
    assert "47b32cf6156dcee41da7eab1686219c760dd73803a135534546c1a50519086e8" in launcher
    assert "HF_HUB_OFFLINE=1" in launcher
    assert "TRANSFORMERS_OFFLINE=1" in launcher
    assert "run_task8_geometry_capture.py" in launcher
    assert "load_pinned_colpali_query_encoder" in wrapper
    assert "load_pinned_qwen_processor" in wrapper
    assert "_load_qwen(" not in wrapper


def test_task9_preliminary_geometry_launcher_is_question_sharded_and_hash_discovering() -> None:
    launcher = Path(
        "examples/sbatch/43_docprune_task9_preliminary_geometry.sbatch"
    ).read_text(encoding="utf-8")
    wrapper = Path("examples/run_task9_preliminary_geometry_capture.py").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --array=0-47%8" in launcher
    assert "#SBATCH --constraint=" not in launcher
    assert "#SBATCH --time=00:15:00" in launcher
    assert "expected_post_qtp_visual_tokens" in launcher
    assert "selected-source-results.jsonl" in launcher
    assert "probe_task8_gpu.sh" in launcher
    assert "probe_task7_l40s_gpu.sh" not in launcher
    assert "HF_HUB_OFFLINE=1" in launcher
    assert "TRANSFORMERS_OFFLINE=1" in launcher
    assert "expected_geometry_sha256=None" in wrapper
    assert "load_pinned_colpali_query_encoder" in wrapper
    assert "load_pinned_qwen_processor" in wrapper
