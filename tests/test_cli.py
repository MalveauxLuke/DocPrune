import json

from docprune.cli import main


def test_inspect_validates_config_without_loading_models(capsys) -> None:
    exit_code = main(["inspect", "--config", "configs/docprune-m3docvqa.toml", "--pages", "4"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["page_count"] == 4
    assert payload["paper_values"]["question_threshold"] == 0.4
    assert payload["upstream"]["m3docrag_commit"].startswith("29e6ac")


def test_summarize_prints_aggregate_json(tmp_path, capsys) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(
            {
                "trace": {
                    "original_visual_tokens": 10,
                    "post_btp_visual_tokens": 8,
                    "post_qtp_visual_tokens": 6,
                    "post_ctp_visual_tokens": 4,
                    "ctp_layer": 3,
                },
                "timing": {"retrieval_seconds": 1.0, "qa_seconds": 1.0},
            }
        )
        + "\n"
    )

    exit_code = main(["summarize", "--results", str(results)])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["samples"] == 1


def test_evaluate_dry_run_refuses_existing_output(tmp_path, capsys) -> None:
    output = tmp_path / "run"
    output.mkdir()

    exit_code = main(
        [
            "evaluate",
            "--config",
            "configs/docprune-m3docvqa.toml",
            "--pages",
            "1",
            "--output",
            str(output),
            "--factory",
            "fake.module:build",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "already exists" in capsys.readouterr().err


def test_evaluate_dry_run_emits_manifest_for_new_output(tmp_path, capsys) -> None:
    output = tmp_path / "run"

    exit_code = main(
        [
            "evaluate",
            "--config",
            "configs/docprune-m3docvqa.toml",
            "--pages",
            "2",
            "--output",
            str(output),
            "--factory",
            "bridge:build",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "dry_run"
    assert payload["factory"] == "bridge:build"
    assert payload["page_count"] == 2
    assert not output.exists()
