#!/usr/bin/env python3
"""Create one authenticated B13-versus-B_input Task 9 diagnostic analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import warnings
from pathlib import Path

from docprune.segmentation import load_region_mapping
from docprune.task9_attribution import (
    _canonical_sha256,
    _root_mean_square,
    _spearman_rank_correlation,
    analyze_contextcite_development_question,
    contextcite_logit_per_token_from_mean_loglikelihood,
    whole_region_knapsack,
)
from docprune.task9_live import admit_task9_regional_development

_TARGET_SCALE = "contextcite-sequence-logit-per-generated-token"
_SECONDARY_KIND = "unpruned-generated-response-mean-loglikelihood"


def _publish(path: Path, payload: dict[str, object]) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        os.write(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _exact_generated_outcomes(
    target: dict[str, object], generated_token_count: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if target.get("target_kind") != _SECONDARY_KIND:
        raise ValueError("paired diagnostic requires the generated-response target")
    outcomes = target.get("outcomes")
    rows = target.get("per_sequence_mean_loglikelihoods")
    if not isinstance(outcomes, list) or not isinstance(rows, list) or len(outcomes) != len(rows):
        raise ValueError("generated-response target rows are invalid")
    transformed: list[dict[str, object]] = []
    corrections: list[float] = []
    for outcome, row in zip(outcomes, rows, strict=True):
        if not isinstance(outcome, dict) or not isinstance(row, list) or len(row) != 1:
            raise ValueError("generated-response target row is not a singleton sequence")
        mean_loglikelihood = float(row[0])
        if float(outcome.get("normalized_target")) != mean_loglikelihood:
            raise ValueError("stored generated-response outcome does not match its raw row")
        exact_target = contextcite_logit_per_token_from_mean_loglikelihood(
            mean_loglikelihood, generated_token_count
        )
        transformed.append({**outcome, "normalized_target": exact_target})
        corrections.append(exact_target - mean_loglikelihood)
    return transformed, {
        "input_scale": "mean-token-log-probability",
        "output_scale": _TARGET_SCALE,
        "formula": "(log_p_sequence - log(1-p_sequence)) / generated_non_eos_token_count",
        "generated_non_eos_token_count": generated_token_count,
        "correction_exact_minus_input": {
            "minimum": min(corrections),
            "mean": math.fsum(corrections) / len(corrections),
            "maximum": max(corrections),
        },
        "transformed_outcomes_sha256": _canonical_sha256(transformed),
    }


def _condition_summary(
    analysis: dict[str, object], outcomes: list[dict[str, object]], regions: list[dict[str, object]]
) -> dict[str, object]:
    surrogate = analysis["surrogate"]
    fidelity = analysis["fidelity"]
    stability = analysis["stability"]
    design = surrogate["attribution_identity"]
    fit_count = int(surrogate["fit_mask_count"])
    coefficients = surrogate["coefficients"]
    intercept = float(surrogate["intercept"])
    # The complete design is available through the stability refits' source order;
    # predictions use each outcome's matching mask from the analysis input below.
    source_ids = list(stability["source_ids"])
    mask_design = analysis["_analysis_mask_design"]
    fit_masks = mask_design["fit_masks"]
    fit_targets = [float(row["normalized_target"]) for row in outcomes[:fit_count]]
    fit_predictions = [
        intercept
        + sum(
            float(coefficients[source_id]) * float(kept)
            for source_id, kept in zip(source_ids, mask["vector"], strict=True)
        )
        for mask in fit_masks
    ]
    fit_spearman = _spearman_rank_correlation(fit_predictions, fit_targets)
    fit_rmse = _root_mean_square(
        [prediction - target for prediction, target in zip(fit_predictions, fit_targets)]
    )
    selection = whole_region_knapsack(
        regions,
        coefficients={source_id: float(coefficients[source_id]) for source_id in source_ids},
        requested_budget=int(stability["requested_budget"]),
    )
    lds = fidelity["lds_spearman"]
    heldout_rmse = float(fidelity["heldout_rmse"])
    constant_rmse = float(fidelity["constant_rmse"])
    minimum_jaccard = stability["top_selection_summary"]["minimum_defined"]
    criteria = {
        "heldout_lds_at_least_0_5": lds is not None and float(lds) >= 0.5,
        "heldout_rmse_beats_fit_mean_constant": bool(fidelity["surrogate_beats_constant"]),
        "minimum_five_refit_selection_jaccard_at_least_0_8": (
            minimum_jaccard is not None and float(minimum_jaccard) >= 0.8
        ),
    }
    return {
        "boundary": design["forced_boundary"],
        "fit_mask_count": fit_count,
        "holdout_mask_count": int(fidelity["holdout_mask_count"]),
        "source_count": len(source_ids),
        "nonzero_coefficient_count": sum(
            float(coefficients[source_id]) != 0.0 for source_id in source_ids
        ),
        "fit_spearman": fit_spearman,
        "fit_rmse": fit_rmse,
        "heldout_lds_spearman": lds,
        "heldout_rmse": heldout_rmse,
        "constant_rmse": constant_rmse,
        "heldout_to_constant_rmse_ratio": heldout_rmse / constant_rmse,
        "fit_to_holdout_rmse_increase": heldout_rmse - fit_rmse,
        "fit_to_holdout_spearman_drop": (
            float(fit_spearman) - float(lds)
            if fit_spearman is not None and lds is not None
            else None
        ),
        "coefficient_refit_spearman_mean": stability["coefficient_summary"]["mean_defined"],
        "coefficient_refit_spearman_minimum": stability["coefficient_summary"]["minimum_defined"],
        "selection_refit_jaccard_mean": stability["top_selection_summary"]["mean_defined"],
        "selection_refit_jaccard_minimum": minimum_jaccard,
        "canonical_selected_region_count": len(selection["top_source_ids"]),
        "canonical_selected_source_ids": selection["top_source_ids"],
        "canonical_selected_source_ids_sha256": _canonical_sha256(selection["top_source_ids"]),
        "requested_budget": selection["requested_budget"],
        "achieved_budget": selection["achieved_budget"],
        "criteria": criteria,
        "one_question_reliability_pass": all(criteria.values()),
    }


def _diagnosis(b13_summary: dict[str, object], input_summary: dict[str, object]) -> dict[str, str]:
    b13_pass = bool(b13_summary["one_question_reliability_pass"])
    input_pass = bool(input_summary["one_question_reliability_pass"])
    if b13_pass and input_pass:
        return {
            "code": "both-pass-mask-count-likely",
            "plain_language": (
                "Both 256-mask surrogates pass the one-question reliability checks. The leading "
                "explanation for the earlier B13 failure is insufficient fitting masks, although "
                "one development question cannot establish general reliability."
            ),
        }
    if not b13_pass and input_pass:
        return {
            "code": "input-only-pass-b13-mixing",
            "plain_language": (
                "Only decoder-input ablation passes. Information mixing before the B13 deletion "
                "is the leading explanation, rather than mask count alone."
            ),
        }
    if not b13_pass and not input_pass:
        return {
            "code": "both-fail-regional-linearity",
            "plain_language": (
                "Neither condition passes with 256 fitting masks. The regional sources or their "
                "effects are likely too dependent or nonlinear for this sparse linear surrogate."
            ),
        }
    return {
        "code": "b13-only-pass-input-shift",
        "plain_language": (
            "B13 passes but decoder-input ablation does not. The input intervention likely creates "
            "a harder distribution shift or otherwise changes the estimand."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b13-root", type=Path, required=True)
    parser.add_argument("--b13-runtime-commit", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--input-runtime-commit", required=True)
    parser.add_argument("--analysis-runtime-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qid", required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--mapping-sha256", required=True)
    parser.add_argument("--mapping-internal-sha256", required=True)
    parser.add_argument("--geometry-count", type=int, required=True)
    parser.add_argument("--geometry-sha256", required=True)
    parser.add_argument("--original-visual-tokens", type=int, required=True)
    parser.add_argument("--post-btp-visual-tokens", type=int, required=True)
    parser.add_argument("--post-qtp-visual-tokens", type=int, required=True)
    parser.add_argument("--decoder-layer-count", type=int, required=True)
    parser.add_argument("--gpu-substring", required=True)
    parser.add_argument("--requested-budget", type=int, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.output.is_absolute() or args.output.exists() or args.output.is_symlink():
        raise FileExistsError("paired Task 9 analysis output must be a fresh absolute path")
    common = {
        "expected_qid": args.qid,
        "expected_fixture_sha256": args.fixture_sha256,
        "expected_mapping_sha256": args.mapping_sha256,
        "expected_mapping_internal_sha256": args.mapping_internal_sha256,
        "expected_geometry_count": args.geometry_count,
        "expected_geometry_sha256": args.geometry_sha256,
        "expected_trace": (
            args.original_visual_tokens,
            args.post_btp_visual_tokens,
            args.post_qtp_visual_tokens,
        ),
        "expected_decoder_layer_count": args.decoder_layer_count,
        "expected_gpu_substring": args.gpu_substring,
        "expected_fit_mask_count": 256,
        "expected_holdout_mask_count": 64,
    }
    condition_specs = {
        "b13": (args.b13_root, args.b13_runtime_commit, "B_13"),
        "input": (args.input_root, args.input_runtime_commit, "B_input"),
    }
    admissions: dict[str, object] = {}
    raw_payloads: dict[str, dict[str, object]] = {}
    targets: dict[str, dict[str, object]] = {}
    for name, (root, runtime_commit, boundary) in condition_specs.items():
        admissions[name] = admit_task9_regional_development(
            root,
            expected_runtime_commit=runtime_commit,
            expected_boundary=boundary,
            **common,
        )
        raw_payloads[name] = _load_json(root / "raw-result.json")
        targets[name] = _load_json(root / "secondary-target.json")

    generated_ids = raw_payloads["b13"]["generated_response_token_ids"]
    if (
        not isinstance(generated_ids, list)
        or not generated_ids
        or raw_payloads["input"]["generated_response_token_ids"] != generated_ids
    ):
        raise ValueError("paired diagnostics do not bind the same generated response")
    b13_manifest = _load_json(args.b13_root / "run-manifest.json")
    input_manifest = _load_json(args.input_root / "run-manifest.json")
    if b13_manifest["mapping_path"] != input_manifest["mapping_path"]:
        raise ValueError("paired diagnostics do not bind the same mapping path")
    mapping = load_region_mapping(Path(b13_manifest["mapping_path"]), validate_raw_artifacts=True)
    regions = [
        {"source_id": source.source_id, "token_cost": len(source.token_ids)}
        for source in mapping.sources
    ]

    analyses: dict[str, object] = {}
    summaries: dict[str, object] = {}
    transforms: dict[str, object] = {}
    solver_warnings: dict[str, object] = {}
    for name in ("b13", "input"):
        exact_outcomes, transform = _exact_generated_outcomes(targets[name], len(generated_ids))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analysis = analyze_contextcite_development_question(
                targets[name]["design"],
                exact_outcomes,
                regions,
                requested_budget=args.requested_budget,
                target_scale=_TARGET_SCALE,
            )
        # The mask design is attached temporarily for summary prediction replay,
        # then removed so it is not duplicated in the durable JSON.
        analysis["_analysis_mask_design"] = targets[name]["design"]
        summary = _condition_summary(analysis, exact_outcomes, regions)
        analysis.pop("_analysis_mask_design")
        analyses[name] = analysis
        summaries[name] = summary
        transforms[name] = transform
        solver_warnings[name] = [
            {"category": warning.category.__name__, "message": str(warning.message)}
            for warning in caught
        ]

    b13_summary = summaries["b13"]
    input_summary = summaries["input"]
    b13_coefficients = analyses["b13"]["surrogate"]["coefficients"]
    input_coefficients = analyses["input"]["surrogate"]["coefficients"]
    source_ids = list(analyses["b13"]["stability"]["source_ids"])
    b13_selected = set(b13_summary["canonical_selected_source_ids"])
    input_selected = set(input_summary["canonical_selected_source_ids"])
    comparison = {
        "delta_convention": "input_minus_b13",
        "heldout_lds_spearman_delta": (
            float(input_summary["heldout_lds_spearman"])
            - float(b13_summary["heldout_lds_spearman"])
        ),
        "heldout_rmse_delta": float(input_summary["heldout_rmse"])
        - float(b13_summary["heldout_rmse"]),
        "heldout_to_constant_rmse_ratio_delta": (
            float(input_summary["heldout_to_constant_rmse_ratio"])
            - float(b13_summary["heldout_to_constant_rmse_ratio"])
        ),
        "selection_refit_jaccard_minimum_delta": (
            float(input_summary["selection_refit_jaccard_minimum"])
            - float(b13_summary["selection_refit_jaccard_minimum"])
        ),
        "cross_boundary_coefficient_spearman": _spearman_rank_correlation(
            [float(b13_coefficients[source_id]) for source_id in source_ids],
            [float(input_coefficients[source_id]) for source_id in source_ids],
        ),
        "cross_boundary_selected_region_jaccard": len(b13_selected & input_selected)
        / len(b13_selected | input_selected),
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "analyzed-task9-paired-256-mask-diagnostics",
        "scope": "one-question-development-diagnostic-not-holdout",
        "question_id": args.qid,
        "analysis_runtime_commit": args.analysis_runtime_commit,
        "requested_budget": args.requested_budget,
        "generated_response_token_ids": generated_ids,
        "generated_response_token_ids_sha256": _canonical_sha256(generated_ids),
        "target_scale": _TARGET_SCALE,
        "raw_admissions": admissions,
        "source_artifacts": {
            name: {
                "root": str(spec[0]),
                "runtime_commit": spec[1],
                "boundary": spec[2],
                "completion_manifest_file_sha256": _file_sha256(
                    spec[0] / "completion-manifest.json"
                ),
                "secondary_target_file_sha256": _file_sha256(spec[0] / "secondary-target.json"),
            }
            for name, spec in condition_specs.items()
        },
        "target_transforms": transforms,
        "condition_summaries": summaries,
        "comparison": comparison,
        "diagnosis": _diagnosis(b13_summary, input_summary),
        "solver_warnings": solver_warnings,
        "metric_glossary": {
            "heldout_lds_spearman": (
                "How well predicted intervention ordering matches the 64 unseen interventions; "
                "1 is perfect, 0 is no rank relationship, and negative reverses the ordering."
            ),
            "heldout_rmse": "Typical prediction error on unseen interventions; lower is better.",
            "constant_rmse": (
                "Error from always predicting the mean of the 256 fitting outcomes. The surrogate "
                "must be lower to add predictive value."
            ),
            "selection_refit_jaccard_minimum": (
                "Worst overlap among ten pairs of five bootstrap refits; 1 means identical selected "
                "region sets and the frozen diagnostic threshold is 0.8."
            ),
            "fit_to_holdout_gap": (
                "A large training-to-unseen deterioration indicates overfitting or nonlinear effects."
            ),
        },
        "analysis_instructions_for_models": [
            "Start with diagnosis.code and verify it against every boolean in condition_summaries.*.criteria.",
            "Use heldout metrics for reliability; fit metrics diagnose overfitting but do not establish fidelity.",
            "Compare heldout RMSE to constant RMSE within each condition, not raw RMSE across target scales.",
            "Treat B_input as decoder-input deletion after vision encoding and BTP/QTP, not pixel removal.",
            "Do not infer population-level reliability or a confidence interval from this one question.",
            "Full coefficients and five refits are retained under analyses for independent inspection.",
        ],
        "scientific_limits": [
            "one development question",
            "adapted 95-region ContextCite diagnostic",
            "privileged generated-response target",
            "no cross-question confidence interval",
            "not a token oracle or deployable selector",
        ],
        "analyses": analyses,
    }
    result["analysis_sha256"] = _canonical_sha256(result)
    _publish(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "analysis_sha256": result["analysis_sha256"],
                "diagnosis": result["diagnosis"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
