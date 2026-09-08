"""Audited correction-corpus baselines and depth-specific regional oracles.

Model imports are lazy: validation and aggregation run on CPU. Inputs are fixed
fixtures, never retrieval searches. Each completed stage is immutable and bound
by content hashes; unfinished stages may be resumed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from docprune.correction_scoring import score_answer, contract_digest


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def code_tree_sha256():
    root = Path(__file__).resolve().parent
    return digest([(str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest())
                   for path in sorted(root.rglob("*.py"))])


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(payload, artifact_sha256=digest(payload))
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    if path.exists():
        temporary.unlink()
        raise FileExistsError(path)
    temporary.rename(path)
    return value


def resume(path, identity):
    if not Path(path).exists():
        return None
    result = read(path)
    unsigned = dict(result)
    checksum = unsigned.pop("artifact_sha256", None)
    if digest(unsigned) != checksum or result.get("identity") != identity:
        raise ValueError(f"resume identity mismatch: {path}")
    return result


def resource(spec, root):
    path = Path(spec["path"])
    if not path.is_absolute():
        path = (Path(root) / path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"resource hash mismatch: {path}")
    return path


def validate_case(case):
    identifier = case.get("case_id", "")
    if identifier and (Path(identifier).name != identifier or identifier in {".", ".."}):
        raise ValueError("case_id must be a safe single path component")
    if len(case["pages"]) != 4 or not case["question"].strip():
        raise ValueError("case requires a question and exactly four ordered pages")
    answers = case["gold_answers"]
    if not answers or any(not isinstance(x, str) or not x.strip() for x in answers):
        raise ValueError("gold_answers requires complete accepted answer sequences")
    for answer in answers:
        if score_answer(answer, case["answer_contract"])["status"] != "correct":
            raise ValueError("a complete gold sequence fails its answer contract")
    if case["answer_contract"]["kind"] in {"set", "ordered"} and len(answers) != 1:
        raise ValueError("set targets require one reviewed complete serialization")
    return case


def baseline_stratum(case, baseline, adjudications=None):
    """Unknown aliases abstain. Optional adjudication binds the exact generation."""
    score = score_answer(baseline["unpruned"]["answer"], case["answer_contract"])
    review = (adjudications or {}).get(case["case_id"])
    if review is not None:
        if (review.get("baseline_sha256") != baseline["artifact_sha256"]
                or review.get("status") not in {"correct", "incorrect", "review"}
                or not review.get("reviewer") or not review.get("rationale")):
            raise ValueError("adjudication must bind baseline hash, reviewer and rationale")
        score = dict(score, status=review["status"], adjudication=review)
    return score


def boundaries(spec, native):
    result = []
    for value in spec:
        boundary = native if value == "dynamic" else value
        if boundary != "input":
            if isinstance(boundary, bool) or boundary is None:
                raise ValueError("dynamic native boundary unavailable")
            boundary = int(boundary)
            if boundary < 0:
                raise ValueError("decoder index must be nonnegative")
        if boundary not in result:
            result.append(boundary)
    return result


def validate_likelihood_rows(rows, target_count):
    for row in rows:
        if len(row) != target_count or any(isinstance(x, bool) or not math.isfinite(float(x)) for x in row):
            raise ValueError("likelihood row must contain each complete gold sequence plus fixed self, all finite")


def likelihood_effect(values, reference):
    gold, own = max(values[:-1]), values[-1]
    full_gold, full_own = max(reference[:-1]), reference[-1]
    return {"G": gold, "S": own, "G_minus_S": gold-own,
            "delta_G": gold-full_gold, "delta_S": own-full_own,
            "delta_G_minus_S": gold-own-full_gold+full_own}


def _trace(trace):
    return {key: getattr(trace, key) for key in (
        "original_visual_tokens", "post_btp_visual_tokens", "post_qtp_visual_tokens",
        "post_ctp_visual_tokens", "ctp_layer")}


def _record(output, case):
    from docprune.evaluation import list_em, list_f1
    original = case.get("original_answers", case.get("original_gold", []))
    return {"original_gold_f1": list_f1(output.answer, tuple(original)) if original else None,
            "original_gold_em": bool(list_em(output.answer, tuple(original))) if original else None,
            "answer": output.answer, "contract_score": score_answer(output.answer, case["answer_contract"]),
            "original_gold": original,
            "legacy_f1": list_f1(output.answer, tuple(case["gold_answers"])),
            "legacy_em": bool(list_em(output.answer, tuple(case["gold_answers"]))),
            "trace": _trace(output.trace),
            "likelihoods": list(output.teacher_forced_loglikelihoods or []),
            "qa_seconds": output.qa_seconds, "encoder_seconds": output.encoder_seconds,
            "decoder_seconds": output.decoder_seconds,
            "peak_allocated_gpu_bytes": output.peak_allocated_gpu_bytes,
            "policy_selection": None if output.policy_selection is None else output.policy_selection.to_dict(),
            "forced_intervention": None if output.forced_intervention is None else output.forced_intervention.to_dict()}


def _workload(case, resources, root, output, runtime_commit):
    from docprune.config import load_config
    from docprune.ctp_policy import btp_qtp_no_ctp_policy
    from docprune.m3docvqa_factory import build_workload
    from docprune.m3docrag import RetrievalOutput
    paths = {key: resource(resources[key], root) for key in ("config", "run_config", "index_manifest", "fixture")}
    workload = build_workload(operation="evaluate", config=load_config(paths["config"]),
        page_count=4, output=output, mode="docprune", run_config=paths["run_config"],
        index_manifest=paths["index_manifest"], sample_ids=(case["qid"],),
        ctp_policy=btp_qtp_no_ctp_policy(), fixed_page_fixture=paths["fixture"],
        fixed_page_fixture_sha256=resources["fixture"]["sha256"], execution_runtime_commit=runtime_commit)
    samples = tuple(workload.samples)
    if len(samples) != 1 or samples[0].question != case["question"]:
        raise ValueError("fixture question differs from frozen corpus question")
    runner = workload.runner
    runner.warmup(samples[0])
    retrieval = runner.retriever.retrieve(case["question"], 4)
    if not isinstance(retrieval, RetrievalOutput):
        raise ValueError("requires sealed retrieval features")
    expected = [(p["doc_id"], p["page_index"]) for p in case["pages"]]
    if [(p.doc_id, p.page_index) for p in retrieval.pages] != expected:
        raise ValueError("ordered supplied pages differ from corpus")
    images = [runner.page_loader.load_page(p.doc_id, p.page_index) for p in retrieval.pages]
    return runner.answerer, images, retrieval


def _configured(base, *, mapping=None, targets=None, forced=None, native=False):
    from docprune.answerers import DocPruneQwenAnswerer
    from docprune.ctp_policy import aggregate_native_threshold_policy, btp_qtp_no_ctp_policy
    kwargs = dict(page_config=base.page_config, reconstruction=base.reconstruction,
        qa_stage="full", max_new_tokens=base.max_new_tokens,
        teacher_forced_target_token_ids=targets,
        frozen_post_qtp_geometry=None if mapping is None else mapping.geometry)
    if forced is not None:
        kwargs["forced_intervention"] = forced
    else:
        kwargs["ctp_policy"] = aggregate_native_threshold_policy() if native else btp_qtp_no_ctp_policy()
    return DocPruneQwenAnswerer(base.model, base.processor, **kwargs)


def _forced(boundary, ids):
    from docprune.qwen2vl.decoder import ForcedVisualIntervention
    return ForcedVisualIntervention(boundary=boundary, mode="physical_delete", retained_visual_ids=tuple(ids))


def run_baseline(case, resources, root, output, runtime_commit):
    from docprune.answerers import prepare_task7_likelihood_target_for_question
    identity = {"case_sha256": digest(case), "resources_sha256": digest({k:v for k,v in resources.items() if k != "mappings"}),
                "runtime_commit": runtime_commit, "phase": "baseline", "runtime_code_sha256": code_tree_sha256()}
    target_path = output / case["case_id"] / "baseline.json"
    existing = resume(target_path, identity)
    if existing:
        _baseline_reference(target_path, existing)
        return existing
    base, images, retrieval = _workload(case, resources, root, output / case["case_id"] / "bootstrap-baseline", runtime_commit)
    first = base.answer(images, case["question"], retrieval_output=retrieval)
    population = first.trace.post_qtp_visual_tokens
    target = prepare_task7_likelihood_target_for_question(base.processor, page_count=4,
        question=case["question"], accepted_references=tuple(case["gold_answers"]))
    references = tuple(tuple(x) for x in target["target_token_ids"])
    shared = base.score_forced_intervention_likelihoods(images, case["question"], retrieval_output=retrieval,
        forced_interventions=(_forced("input", range(population)),), teacher_forced_target_token_ids=references,
        include_unpruned_generated_response=True)
    own = tuple(shared.unpruned_generated_response_token_ids or ())
    if not own or shared.unpruned_terminal_eos_token_id is None:
        raise ValueError("fixed self generation missing or truncated before EOS")
    decoded = base.processor.tokenizer.decode(list(own), skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    if decoded != first.answer.strip():
        raise ValueError("unpruned generation did not reproduce for fixed-self capture")
    teacher = (*references, own)
    unpruned = _configured(base, targets=teacher).answer(images, case["question"], retrieval_output=retrieval)
    if unpruned.answer.strip() != first.answer.strip():
        raise ValueError("baseline generation changed while scoring")
    values = list(unpruned.teacher_forced_loglikelihoods)
    if len(shared.result.branches) != 1:
        raise ValueError("baseline scoring did not return one all-keep branch")
    continued = list(shared.result.branches[0].teacher_forced_loglikelihoods)
    validate_likelihood_rows([values, continued], len(teacher))
    parity = max(abs(a-b) for a,b in zip(values, continued, strict=True))
    if parity > 1e-4:
        raise ValueError(f"input all-keep continuation likelihood parity failed: {parity}")
    native = _configured(base, targets=teacher, native=True).answer(images, case["question"], retrieval_output=retrieval)
    validate_likelihood_rows([native.teacher_forced_loglikelihoods], len(teacher))
    completed = publish(target_path, {"identity": identity, "case_id": case["case_id"], "qid": case["qid"],
        "question": case["question"], "pages": case["pages"], "gold_answers": case["gold_answers"],
        "answer_contract": case["answer_contract"], "reference_token_ids": [list(x) for x in references],
        "fixed_self_token_ids": list(own), "terminal_eos_token_id": shared.unpruned_terminal_eos_token_id,
        "prompt_sha256": shared.assistant_prompt_sha256, "input_ids_sha256": shared.prefill_input_ids_sha256,
        "unpruned": _record(unpruned, case), "native_docprune": _record(native, case),
        "all_keep_input_parity_max_abs": parity, "retrieval_search_run": False, "global_index_loaded": False})
    _baseline_reference(target_path, completed)
    return completed


def _baseline_reference(path, baseline):
    row = {"question_id": baseline["qid"], "question": baseline["question"],
           "retrieved_pages": [{k: p[k] for k in ("doc_id", "page_index", "score")} for p in baseline["pages"]],
           "trace": baseline["unpruned"]["trace"]}
    destination = path.with_name("reference.jsonl")
    content = json.dumps(row, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text() != content:
            raise ValueError("baseline reference changed")
    else:
        destination.write_text(content)


def run_comparison(case, resources, root, output, runtime_commit, *, depth_spec=("input", "dynamic"), holdout=0, adjudications=None):
    if case.get("protected_overlap_clear") is not True or not case.get("evidence_review"):
        raise ValueError("oracle requires protected-cohort exclusion and recorded visual evidence review")
    if case.get("contract_sha256") != contract_digest(case["answer_contract"]):
        raise ValueError("oracle requires the frozen reviewed answer contract")
    from docprune.segmentation import load_region_mapping
    from docprune.task9_attribution import (build_region_mask_design, build_regional_intervention_plan,
        fit_contextcite_lasso, build_task9_preliminary_arm_selections, build_task9_selected_arm_plan,
        whole_region_knapsack, _canonical_sha256)
    baseline_path = output / case["case_id"] / "baseline.json"
    baseline = read(baseline_path)
    # Authenticate baseline against current case/resources/code, without rerunning it.
    expected_identity = {"case_sha256": digest(case), "resources_sha256": digest({k:v for k,v in resources.items() if k != "mappings"}),
        "runtime_commit": runtime_commit, "phase": "baseline", "runtime_code_sha256": code_tree_sha256()}
    baseline = resume(baseline_path, expected_identity)
    score = baseline_stratum(case, baseline, adjudications)
    if score["status"] == "review":
        return {"case_id": case["case_id"], "status": "needs-baseline-adjudication", "baseline_score": score}
    spec = resources["mappings"][case["qid"]]
    mapping = load_region_mapping(resource(spec, root), validate_raw_artifacts=True)
    if mapping.geometry_count != baseline["unpruned"]["trace"]["post_qtp_visual_tokens"]:
        raise ValueError("mapping population differs from baseline QTP")
    if any(page.fixture_question_id != case["qid"] for artifact in mapping.artifacts for page in artifact.pages):
        raise ValueError("mapping belongs to a different question")
    native = baseline["native_docprune"]
    native_selection = native["policy_selection"]
    if not native_selection or not 0 < native_selection["achieved_budget"] < mapping.geometry_count:
        raise ValueError("native comparator did not produce a nontrivial pruning budget")
    grid = boundaries(depth_spec, native["trace"]["ctp_layer"])
    results = []
    base = None
    for boundary in grid:
        identity = {"baseline_sha256": baseline["artifact_sha256"], "mapping_sha256": spec["sha256"],
            "boundary": boundary, "fit_masks": 256, "holdout_masks": holdout, "baseline_score": score}
        label = "B_input" if boundary == "input" else f"B_{boundary}"
        path = output / case["case_id"] / label / "comparison.json"
        existing = resume(path, identity)
        if existing:
            results.append(existing)
            continue
        if base is None:
            base, images, retrieval = _workload(case, resources, root, output / case["case_id"] / "bootstrap-comparison", runtime_commit)
        references = tuple(tuple(x) for x in baseline["reference_token_ids"])
        own = tuple(baseline["fixed_self_token_ids"])
        teacher = (*references, own)
        scorer = _configured(base, mapping=mapping)
        regions = [{"source_id": x.source_id, "token_cost": len(x.token_ids)} for x in mapping.sources]
        design = build_region_mask_design(regions, question_id=case["qid"], forced_boundary=label,
            mapping_artifact_sha256=spec["sha256"], prompt_input_sha256=baseline["input_ids_sha256"],
            target_kind="max-accepted-reference-mean-loglikelihood", reference_set_token_ids_sha256=digest([list(x) for x in references]),
            generated_response_token_ids_sha256=None, fit_mask_count=256, holdout_mask_count=holdout)
        plan = (*build_regional_intervention_plan(mapping, design, mapping_artifact_sha256=spec["sha256"], split="fit"),
                *build_regional_intervention_plan(mapping, design, mapping_artifact_sha256=spec["sha256"], split="holdout"))
        all_keep = _forced(boundary, range(mapping.geometry_count))
        scored_path = path.with_name("mask-scores.json")
        scored = resume(scored_path, identity)
        if scored is None:
            shared = scorer.score_forced_intervention_likelihoods(images, case["question"], retrieval_output=retrieval,
                forced_interventions=(all_keep, *(x["forced_intervention"] for x in plan)),
                teacher_forced_target_token_ids=teacher, include_unpruned_generated_response=False)
            if shared.prefill_input_ids_sha256 != baseline["input_ids_sha256"] or shared.assistant_prompt_sha256 != baseline["prompt_sha256"]:
                raise ValueError("prompt identity changed across depth")
            raw = [list(x.teacher_forced_loglikelihoods) for x in shared.result.branches]
            if len(raw) != len(plan) + 1 or len(shared.result.branch_decoder_seconds) != len(raw):
                raise ValueError("scoring branch or timing count differs from mask plan")
            validate_likelihood_rows(raw, len(teacher))
            reference = raw[0]
            parity = max(abs(a-b) for a,b in zip(reference, baseline["unpruned"]["likelihoods"], strict=True))
            if parity > 1e-4:
                raise ValueError(f"{label} all-keep likelihood parity failed")
            scored = publish(scored_path, {"identity": identity, "raw": raw,
                "all_keep_parity_max_abs": parity,
                "scoring_cost": {"encoder_seconds": shared.result.encoder_seconds,
                    "prefix_decoder_seconds": shared.result.prefix_decoder_seconds,
                    "branch_decoder_seconds": list(shared.result.branch_decoder_seconds),
                    "checkpoint_cache_lengths": list(shared.result.checkpoint_cache_lengths),
                    "peak_allocated_gpu_bytes": shared.peak_allocated_gpu_bytes}})
        raw = scored["raw"]
        reference = raw[0]
        parity = scored["all_keep_parity_max_abs"]
        parity_path = path.with_name("all-keep-generation.json")
        parity_saved = resume(parity_path, identity)
        if parity_saved is None:
            parity_generation = _record(_configured(base, mapping=mapping, targets=teacher, forced=all_keep).answer(
                images, case["question"], retrieval_output=retrieval), case)
            if parity_generation["answer"].strip() != baseline["unpruned"]["answer"].strip():
                raise ValueError(f"{label} all-keep generation parity failed")
            parity_saved = publish(parity_path, {"identity": identity, "record": parity_generation})
        parity_generation = parity_saved["record"]
        effects = [likelihood_effect(values, reference) for values in raw[1:]]
        surrogates, diagnostics = {}, {}
        self_design = build_region_mask_design(regions, question_id=case["qid"], forced_boundary=label,
            mapping_artifact_sha256=spec["sha256"], prompt_input_sha256=baseline["input_ids_sha256"],
            target_kind="unpruned-generated-response-mean-loglikelihood", reference_set_token_ids_sha256=None,
            generated_response_token_ids_sha256=digest(list(own)), fit_mask_count=256, holdout_mask_count=holdout)
        if self_design["fit_masks"] != design["fit_masks"] or self_design["holdout_masks"] != design["holdout_masks"]:
            raise ValueError("gold and fixed-self fits must share the same masks")
        for name, key in (("gold_support", "G"), ("fixed_self_support", "S"), ("gold_margin", "G_minus_S")):
            fit_design = self_design if name == "fixed_self_support" else design
            outcomes = [{"split": p["split"], "seed": p["seed"], "vector_sha256": p["vector_sha256"],
                "attribution_identity_sha256": fit_design["attribution_identity_sha256"], "normalized_target": effect[key]}
                for p,effect in zip(plan, effects, strict=True)]
            surrogate = fit_contextcite_lasso(fit_design, outcomes[:256])
            # Margin uses the same pinned solver but is explicitly labeled G-S,
            # never exported as a primary gold-support attribution artifact.
            target_provenance = {"definition": key, "design_sha256": fit_design["design_sha256"],
                "reference_token_ids_sha256": digest([list(x) for x in references]),
                "fixed_self_token_ids_sha256": digest(list(own)),
                "mask_vectors_sha256": digest([x["vector"] for x in plan])}
            surrogates[name] = {"target_definition": key, "fit": surrogate,
                "target_provenance": target_provenance, "target_provenance_sha256": digest(target_provenance)}
            if holdout:
                from docprune.task9_attribution import evaluate_contextcite_holdout
                diagnostics[name] = evaluate_contextcite_holdout(fit_design, outcomes[:256], surrogate, outcomes[256:])
            else:
                diagnostics[name] = {"status": "not_measured", "reason": "no held-out masks requested"}
        selections = build_task9_preliminary_arm_selections(regions, question_id=case["qid"],
            requested_token_count=native_selection["achieved_budget"],
            gold_support_scores=surrogates["gold_support"]["fit"]["coefficients"],
            gold_margin_scores=surrogates["gold_margin"]["fit"]["coefficients"])
        gold_fit = surrogates["gold_support"]["fit"]
        self_fit = surrogates["fixed_self_support"]["fit"]
        residual = {"definition": "coefficient(fit G) minus coefficient(fit S); distinct from direct fit(G-S)",
            "coefficients": {key: gold_fit["coefficients"][key] - self_fit["coefficients"][key]
                             for key in gold_fit["coefficients"]},
            "intercept": gold_fit["intercept"] - self_fit["intercept"],
            "gold_fit_sha256": gold_fit["surrogate_sha256"],
            "fixed_self_fit_sha256": self_fit["surrogate_sha256"]}
        residual["residual_sha256"] = digest(residual)
        surrogates["coefficient_residual"] = residual
        for arm, coefficients in (("contextcite_fixed_self_support", self_fit["coefficients"]),
                                  ("contextcite_coefficient_residual", residual["coefficients"])):
            selection = whole_region_knapsack(regions, coefficients=coefficients,
                requested_budget=native_selection["achieved_budget"])
            budget = selections["budgets"][0]
            if selection["achieved_budget"] != budget["achieved_token_count"]:
                raise ValueError("self/residual controls must achieve the same region-token budget")
            budget["arms"].append({"arm": arm, "retained_source_ids": selection["top_source_ids"],
                                   "achieved_token_count": selection["achieved_budget"]})
        selections.pop("selections_sha256")
        selections["selections_sha256"] = _canonical_sha256(selections)
        selected_plan = build_task9_selected_arm_plan(mapping, selections, boundary=label)
        by_arm = {row["arm"]: row for row in selected_plan}
        context_comparison = {"interpretation": "Selected-set overlap is descriptive, not individual causal attribution."}
        for kind, field in (("regions", "retained_source_ids"), ("tokens", "retained_visual_ids")):
            gold_ids = set(by_arm["contextcite_gold_support"][field])
            self_ids = set(by_arm["contextcite_fixed_self_support"][field])
            context_comparison[kind] = {"both": sorted(gold_ids & self_ids),
                "gold_only": sorted(gold_ids - self_ids), "self_only": sorted(self_ids - gold_ids),
                "coefficient_residual_selected": list(by_arm["contextcite_coefficient_residual"][field]),
                "direct_margin_selected": list(by_arm["contextcite_gold_margin"][field])}
        selected = []
        for row in selected_plan:
            arm_path = path.with_name(row["arm"] + ".json")
            arm_identity = dict(identity, selection_sha256=digest({k:v for k,v in row.items() if k != "forced_intervention"}))
            saved_arm = resume(arm_path, arm_identity)
            if saved_arm is None:
                record = _record(_configured(base, mapping=mapping, targets=teacher, forced=row["forced_intervention"]).answer(
                    images, case["question"], retrieval_output=retrieval), case)
                saved_arm = publish(arm_path, {"identity": arm_identity, "record": record})
            record = dict(saved_arm["record"])
            validate_likelihood_rows([record["likelihoods"]], len(teacher))
            record.update({"arm": row["arm"], "retained_source_ids": row["retained_source_ids"],
                "retained_visual_ids": row["retained_visual_ids"], "effects": likelihood_effect(record["likelihoods"], reference),
                "genuine_correction": score["status"] == "incorrect" and record["contract_score"]["status"] == "correct",
                "preserved_correct": score["status"] == "correct" and record["contract_score"]["status"] == "correct"})
            selected.append(record)
        results.append(publish(path, {"identity": identity, "case_id": case["case_id"], "qid": case["qid"],
            "boundary": label, "baseline_stratum": score["status"], "design": design,
            "analysis_schema": "correction-depth-gold-self-residual-v2", "fixed_self_design": self_design,
            "selected_context_comparison": context_comparison,
            "mask_vectors_sha256": digest([x["vector"] for x in plan]),
            "mask_plan": [{k:v for k,v in x.items() if k != "forced_intervention"} for x in plan],
            "raw_likelihoods": raw[1:], "full_context_likelihoods": reference, "effects": effects,
            "surrogates": surrogates, "holdout_diagnostics": diagnostics, "selections": selections,
            "selected_results": selected, "native_docprune": native, "all_keep": parity_generation,
            "all_keep_parity_max_abs": parity, "margin_degenerate_gold_equals_self": own in references,
            "scoring_cost": scored["scoring_cost"]}))
    if len({x["mask_vectors_sha256"] for x in results}) != 1:
        raise ValueError("depths did not use identical region masks")
    return results


def summarize(cases, output):
    rows = []
    for case in cases:
        baseline_path = output / case["case_id"] / "baseline.json"
        if not baseline_path.exists():
            rows.append({"case_id": case["case_id"], "status": "baseline_pending"})
            continue
        baseline = read(baseline_path)
        comparison_paths = sorted((output / case["case_id"]).glob("B_*/comparison.json"))
        native_added = False
        for path in comparison_paths:
            result = read(path)
            if not native_added:
                native = baseline["native_docprune"]
                native_status = native["contract_score"]["status"]
                stratum = result["baseline_stratum"]
                rows.append({"case_id": case["case_id"], "document_component": case.get("document_component"),
                    "boundary": "native_dynamic", "actual_native_boundary": native["trace"]["ctp_layer"],
                    "baseline_status": stratum, "arm": "native_docprune", "answer": native["answer"],
                    "status": native_status, "genuine_correction": stratum == "incorrect" and native_status == "correct",
                    "preserved_correct": stratum == "correct" and native_status == "correct",
                    "retained_tokens": native["trace"]["post_ctp_visual_tokens"], "evidence": str(baseline_path)})
                native_added = True
            for arm in result["selected_results"]:
                rows.append({"case_id": case["case_id"], "document_component": case.get("document_component"),
                    "boundary": result["boundary"], "baseline_status": result["baseline_stratum"],
                    "arm": arm["arm"], "answer": arm["answer"], "status": arm["contract_score"]["status"],
                    "genuine_correction": arm["genuine_correction"], "preserved_correct": arm["preserved_correct"],
                    "retained_tokens": arm["trace"]["post_ctp_visual_tokens"],
                    "native_retained_tokens": baseline["native_docprune"]["trace"]["post_ctp_visual_tokens"],
                    "effects": arm["effects"], "evidence": str(path)})
        if not list((output / case["case_id"]).glob("B_*/comparison.json")):
            rows.append({"case_id": case["case_id"], "status": "comparison_pending",
                         "baseline_score": baseline["unpruned"]["contract_score"]})
    counts = {}
    for row in rows:
        if "arm" not in row:
            continue
        key = (row["boundary"], row["arm"])
        count = counts.setdefault(key, {"boundary": key[0], "arm": key[1],
            "n_verified_incorrect": 0, "n_genuine_corrections": 0,
            "n_verified_correct": 0, "n_preserved": 0, "n_review_outcomes": 0})
        count["n_verified_incorrect"] += int(row["baseline_status"] == "incorrect")
        count["n_genuine_corrections"] += int(row["genuine_correction"])
        count["n_verified_correct"] += int(row["baseline_status"] == "correct")
        count["n_preserved"] += int(row["preserved_correct"])
        count["n_review_outcomes"] += int(row["status"] == "review")
    for count in counts.values():
        count["verified_correction_fraction"] = (count["n_genuine_corrections"] / count["n_verified_incorrect"]
            if count["n_verified_incorrect"] else None)
        count["verified_preservation_fraction"] = (count["n_preserved"] / count["n_verified_correct"]
            if count["n_verified_correct"] else None)
    return {"case_count": len(cases), "rows": rows, "counts": list(counts.values()),
            "fraction_denominators": "All verified-incorrect/correct baselines in each comparison; unresolved outcomes remain in denominators and are reported separately, so verified fractions may increase after adjudication.",
            "interpretation": "Exploratory document-grouped development only; review outcomes are not counted as corrections."}
