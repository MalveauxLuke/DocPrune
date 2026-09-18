"""Fixed-checkpoint evaluation on the exact audited train/dev preference pairs."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "src"), str(Path(__file__).resolve().parents[2])]

import torch

from experiments.training_pilot.run import SELECTOR_REV, load_model, read, sha, stats
from experiments.training_pilot.train64 import Stream
from docprune.stage2.contracts import fingerprint
from docprune.stage2.experiment import ExperimentConfig, Stage1Contract
from docprune.stage2.pilot import CachedPilotSelector
from docprune.stage2.pilot_runtime import TensorCache, restore_parameters
from docprune.stage2.policy import score_candidate_masks
from docprune.stage2.storage import publish, read_record, run_lock
from docprune.stage2.supervision import preference
from docprune.stage2.training import build_selector


FAMILIES = ("single_flip", "two_flip", "broad")
COST_NEAR_FRACTION = 0.05


def pair_family(distance):
    return "single_flip" if distance == 1 else "two_flip" if distance == 2 else "broad"


def preference_reason(bank, first, second, epsilon, margin):
    a, b = bank.outcomes[first], bank.outcomes[second]
    if bank.baseline_correct and bank.correct_preservation == "s":
        return "s_preservation"
    if bank.baseline_correct:
        return "gold_support"
    threshold = bank.reference.g - epsilon
    a_ok, b_ok = a.g >= threshold, b.g >= threshold
    if a_ok != b_ok:
        return "gold_threshold_boundary" if min(abs(a.g - threshold), abs(b.g - threshold)) <= margin else "gold_admissibility"
    return "contrast" if a_ok else "gold_support"


def freeze_pairs(qid, split, example, bank, proposals, *, epsilon, margin, temperature):
    """Freeze all candidate pairs, ties, exact strict-pair order, and metric weights."""
    costs = example.layout.costs.cpu()
    retained = (bank.masks * costs).sum(1).tolist()
    regions = bank.masks.sum(1).tolist()
    rows = []
    strict_by_family = {name: [] for name in FAMILIES}
    for first, a in enumerate(bank.outcomes):
        for second in range(first):
            pref = preference(
                a,
                bank.outcomes[second],
                bank.reference,
                baseline_correct=bank.baseline_correct,
                correct_preservation=bank.correct_preservation,
                mode="gold_aware",
                epsilon=epsilon,
                margin=margin,
            )
            plus, minus = ((first, second) if pref > 0 else (second, first)) if pref else (None, None)
            distance = int((bank.masks[first] != bank.masks[second]).sum())
            family = pair_family(distance)
            teacher_token_delta = None if not pref else int(retained[plus] - retained[minus])
            row = dict(
                pair_id=fingerprint([qid, first, second, epsilon, margin, temperature, "gold_aware"]),
                candidate_indices=[first, second],
                teacher_preference=pref,
                plus_mask_index=plus,
                minus_mask_index=minus,
                tie=pref == 0,
                preference_reason=preference_reason(bank, first, second, epsilon, margin),
                hamming_distance=distance,
                comparison_family=family,
                token_count_difference=int(retained[first] - retained[second]),
                teacher_token_delta=teacher_token_delta,
                teacher_token_delta_fraction=None if not pref else teacher_token_delta / example.budget,
                region_count_difference=int(regions[first] - regions[second]),
                first_acquisition_kind=proposals[first].get("kind") if first < len(proposals) else None,
                second_acquisition_kind=proposals[second].get("kind") if second < len(proposals) else None,
                first_acquisition_reason=proposals[first].get("reason") if first < len(proposals) else None,
                second_acquisition_reason=proposals[second].get("reason") if second < len(proposals) else None,
                within_question_weight=None,
            )
            rows.append(row)
            if pref:
                strict_by_family[family].append(row)
    present = [name for name in FAMILIES if strict_by_family[name]]
    for family in present:
        weight = 1.0 / (len(present) * len(strict_by_family[family]))
        for row in strict_by_family[family]:
            row["within_question_weight"] = weight
    return dict(
        schema="pilot-fixed-pair-manifest-v1",
        qid=qid,
        split=split,
        baseline_correct=bank.baseline_correct,
        mask_count=len(bank.outcomes),
        candidate_pair_count=len(rows),
        strict_pair_count=sum(bool(r["teacher_preference"]) for r in rows),
        tie_count=sum(r["tie"] for r in rows),
        pair_weighting=bank.pair_weighting,
        split_weighting="question",
        supervision="gold_aware",
        epsilon=epsilon,
        margin=margin,
        temperature=temperature,
        full_capacity=example.budget,
        masks=[dict(
            mask_index=i,
            retained_tokens=int(retained[i]),
            retained_regions=int(regions[i]),
            acquisition_kind=proposals[i].get("kind") if i < len(proposals) else None,
            acquisition_reason=proposals[i].get("reason") if i < len(proposals) else None,
        ) for i in range(len(bank.outcomes))],
        pairs=rows,
    )


def tie_credit(margin):
    """Pair accuracy convention shared by every predictor in content analysis."""
    return 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5


def content_pair_rows(scores, manifest):
    """Annotate strict teacher pairs by signed cost and actual mask distance."""
    result = []
    for row in manifest["pairs"]:
        if row["tie"]:
            continue
        delta = row["teacher_token_delta"]
        fraction = abs(delta) / manifest["full_capacity"]
        cost_band = "exact" if delta == 0 else "near" if fraction <= COST_NEAR_FRACTION else "large"
        direction = "equal" if delta == 0 else "more" if delta > 0 else "fewer"
        locality = "local" if row["hamming_distance"] <= 2 else "broad"
        margin = float(scores[row["plus_mask_index"]] - scores[row["minus_mask_index"]])
        result.append(dict(
            pair_id=row["pair_id"],
            margin=margin,
            accuracy_credit=tie_credit(margin),
            cost_band=cost_band,
            teacher_token_direction=direction,
            cost_direction=f"{cost_band}/{direction}",
            locality=locality,
            cost_direction_locality=f"{cost_band}/{direction}/{locality}",
            changed_regions=row["hamming_distance"],
            comparison_family=row["comparison_family"],
            teacher_token_delta=delta,
            teacher_token_delta_fraction=row["teacher_token_delta_fraction"],
        ))
    return result


def weighted_slice(question_rows, field, value):
    """Apply the training contract: balance actual Hamming families per question."""
    selected = [row for row in question_rows if row[field] == value]
    families = []
    for family in FAMILIES:
        members = [row for row in selected if row["comparison_family"] == family]
        if members:
            families.append(sum(row["accuracy_credit"] for row in members) / len(members))
    return None if not families else sum(families) / len(families)


def sliced_accuracy(question_pairs, field):
    values = sorted({pair[field] for pairs in question_pairs.values() for pair in pairs})
    result = {}
    for value in values:
        per_question = [
            (qid, weighted_slice(pairs, field, value)) for qid, pairs in question_pairs.items()
        ]
        per_question = [(qid, score) for qid, score in per_question if score is not None]
        pooled = [pair for pairs in question_pairs.values() for pair in pairs if pair[field] == value]
        result[value] = dict(
            questions=len(per_question),
            pairs=len(pooled),
            question_family_balanced_accuracy=sum(score for _, score in per_question) / len(per_question),
            pooled_pair_accuracy=sum(pair["accuracy_credit"] for pair in pooled) / len(pooled),
        )
    return result


def content_analysis(rows, manifests):
    """Compare selectors and retention on identical pairs, weights, and tie credit."""
    result = {}
    for predictor in ("head1", "combined", "retention"):
        question_pairs = {}
        for row in rows:
            if predictor == "head1":
                scores = torch.tensor([mask["head1"] for mask in row["masks"]])
            elif predictor == "combined":
                scores = torch.tensor([mask["combined"] for mask in row["masks"]])
            else:
                scores = torch.tensor([mask["retained_tokens"] for mask in manifests[row["qid"]]["masks"]], dtype=torch.float64)
            question_pairs[row["qid"]] = content_pair_rows(scores, manifests[row["qid"]])
        result[predictor] = {
            field: sliced_accuracy(question_pairs, field)
            for field in ("cost_band", "teacher_token_direction", "cost_direction", "locality", "cost_direction_locality", "changed_regions")
        }
    result["contract"] = dict(
        near_cost_definition=f"0 < abs(teacher_token_delta) / full_capacity <= {COST_NEAR_FRACTION}",
        large_cost_definition=f"abs(teacher_token_delta) / full_capacity > {COST_NEAR_FRACTION}",
        local_definition="actual changed-region count <= 2",
        broad_definition="actual changed-region count > 2",
        tie_convention="exact predictor-score ties receive 0.5 accuracy credit for every predictor",
        weighting="within each question, average present actual-Hamming families equally; then average questions equally",
    )
    return result


def question_metrics(scores, manifest):
    strict = [row for row in manifest["pairs"] if not row["tie"]]
    if not strict:
        return None, []
    plus = torch.tensor([row["plus_mask_index"] for row in strict], device=scores.device)
    minus = torch.tensor([row["minus_mask_index"] for row in strict], device=scores.device)
    margins = (scores[plus].float() - scores[minus].float()) / manifest["temperature"]
    if not torch.isfinite(margins).all():
        raise ValueError("Nonfinite checkpoint prediction")
    losses = torch.nn.functional.softplus(-margins)
    correct = margins > 0
    result = []
    for index, row in enumerate(strict):
        result.append(dict(
            pair_id=row["pair_id"],
            margin=float(margins[index]),
            logistic_loss=float(losses[index]),
            correct=bool(correct[index]),
            comparison_family=row["comparison_family"],
        ))
    families = {}
    family_losses = []
    family_accuracies = []
    for family in FAMILIES:
        selected = torch.tensor([row["comparison_family"] == family for row in strict], device=scores.device)
        if not selected.any():
            families[family] = None
            continue
        family_loss = losses[selected].mean()
        family_accuracy = correct[selected].float().mean()
        family_losses.append(family_loss)
        family_accuracies.append(family_accuracy)
        families[family] = dict(pairs=int(selected.sum()), accuracy=float(family_accuracy), loss=float(family_loss))
    present = [value for value in families.values() if value is not None]
    if not present:
        return None, result
    return dict(
        pairs=len(result),
        accuracy=float(torch.stack(family_accuracies).mean()),
        loss=float(torch.stack(family_losses).mean()),
        families=families,
    ), result


def result_from_mask_scores(saved, manifest, *, device=None):
    """Rebuild metrics from immutable per-mask scores without another selector forward."""
    masks = saved["masks"]
    head1, head1_pairs = question_metrics(
        torch.tensor([row["head1"] for row in masks], dtype=torch.float32, device=device), manifest
    )
    combined, combined_pairs = question_metrics(
        torch.tensor([row["combined"] for row in masks], dtype=torch.float32, device=device), manifest
    )
    retention, retention_pairs = question_metrics(
        torch.tensor([row["retained_tokens"] for row in masks], dtype=torch.float32, device=device), manifest
    )
    return dict(
        qid=saved["qid"],
        split=saved["split"],
        baseline_correct=saved["baseline_correct"],
        exposures_in_branch_path=saved["exposures_in_branch_path"],
        masks=masks,
        head1=head1,
        head1_pairs=head1_pairs,
        combined=combined,
        combined_pairs=combined_pairs,
        retention=retention,
        retention_pairs=retention_pairs,
    )


def aggregate(rows, key, *, correct=None):
    selected = [row for row in rows if row[key] is not None and (correct is None or row["baseline_correct"] is correct)]
    if not selected:
        return None
    all_pairs = [pair for row in selected for pair in row[key + "_pairs"]]
    families = {}
    for family in FAMILIES:
        values = [row[key]["families"][family] for row in selected if row[key]["families"][family] is not None]
        families[family] = None if not values else dict(
            questions=len(values),
            pairs=sum(value["pairs"] for value in values),
            accuracy=sum(value["accuracy"] for value in values) / len(values),
            loss=sum(value["loss"] for value in values) / len(values),
        )
    return dict(
        questions=len(selected),
        pairs=len(all_pairs),
        question_balanced_accuracy=sum(row[key]["accuracy"] for row in selected) / len(selected),
        question_balanced_loss=sum(row[key]["loss"] for row in selected) / len(selected),
        pooled_pair_accuracy=sum(pair["correct"] for pair in all_pairs) / len(all_pairs),
        pooled_pair_loss=sum(pair["logistic_loss"] for pair in all_pairs) / len(all_pairs),
        families=families,
    )


def summarize(rows):
    return {
        head: {group: aggregate(rows, head, correct=correct) for group, correct in (("all", None), ("correct", True), ("incorrect", False))}
        for head in ("head1", "combined", "retention")
    }


def saved_rows(path):
    record = read_record(path)
    return {row["qid"]: row for row in record["metrics"]["rows"]}


def reproduction(actual, expected, tolerance):
    checks = []
    for row in actual:
        saved = expected[row["qid"]]
        for head in ("head1", "combined", "retention"):
            for metric in ("accuracy", "loss"):
                delta = abs(row[head][metric] - saved[head][metric])
                checks.append(dict(qid=row["qid"], head=head, metric=metric, absolute_delta=delta, passed=delta <= tolerance))
    return dict(tolerance=tolerance, checks=len(checks), maximum_absolute_delta=max(row["absolute_delta"] for row in checks), passed=all(row["passed"] for row in checks))


def proposals(folder, count):
    result = []
    for index in range(count):
        path = Path(folder) / "proposals" / f"{index:02d}.json"
        result.append(read(path) if path.exists() else {})
    return result


def run(args):
    root, training, output = Path(args.root), Path(args.training), Path(args.output)
    audit = read_record(args.audit)
    contract = read_record(training / "contract.json")
    if read_record(training / "training-complete.json")["status"] != "passed" or audit["status"] != "passed":
        raise ValueError("Passed training and bank audit required")
    if contract["audit_sha256"] != sha(args.audit) or Path(args.selector).name != SELECTOR_REV:
        raise ValueError("Training/audit/selector identity mismatch")
    config = ExperimentConfig(
        name="quality-first-pilot64-v1", seed=args.seed, vision_mode="native", head2=True,
        retrieval_dim=129, retrieval_schema=contract["config"]["retrieval_schema"],
        stage1=Stage1Contract(answerer_revision=contract["config"]["stage1"]["answerer_revision"], selector_revision=SELECTOR_REV, epsilon=.1, margin=.05),
    )
    # Training contracts are JSON-sealed, which normalizes tuple fields to lists.
    if json.loads(json.dumps(asdict(config))) != contract["config"]:
        raise ValueError("Evaluation config differs from training contract")
    torch.manual_seed(args.seed); random.seed(args.seed)
    processor, backbone = load_model(args.selector)
    cache_identity = dict(
        selector=SELECTOR_REV,
        runtime=contract["runtime"],
        processor=processor.to_dict() if hasattr(processor, "to_dict") else str(type(processor)),
        sources=contract["sources"],
        audit=contract["audit_sha256"],
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
    )
    cache = TensorCache(training / "cache", cache_identity)
    model = CachedPilotSelector(build_selector(config, answerer_config=backbone.config, selector_model=backbone), disk_cache=cache)
    evaluation_device = next(model.parameters()).device
    stream = Stream(root, audit["rows"], processor, cache)
    identity = fingerprint(contract)
    checkpoints = [
        ("untrained", None, "warmup", training / "untrained.json", 0),
        ("warmup", training / "warmup-best.pt", "warmup", training / "warmup-complete.json", 2),
        ("frozen_best", training / "frozen-best.pt", "frozen", training / "frozen-complete.json", 3),
        ("frozen_final", training / "frozen-latest.pt", "frozen", None, 5),
        ("lora_best", training / "lora-best.pt", "lora", training / "lora-complete.json", 3),
        ("lora_final", training / "lora-latest.pt", "lora", None, 5),
    ]
    manifests = {}
    for row in audit["rows"]:
        manifest_path = output / "pairs" / f'{row["qid"]}.json'
        if manifest_path.exists():
            manifests[row["qid"]] = read_record(manifest_path)
            continue
        batch = stream.batch(row["qid"]); bank = batch["banks"][0]; example = batch["examples"][0]
        manifest = freeze_pairs(row["qid"], row["split"], example, bank, proposals(row["directory"], len(bank.outcomes)), epsilon=.1, margin=.05, temperature=1.)
        publish(manifest_path, manifest); manifests[row["qid"]] = manifest
        del batch, bank, example
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats(); summaries = {}; content = {}; gates = {}
    for name, path, phase, saved, exposures in checkpoints:
        if path is not None:
            restored = restore_parameters(path, model, identity)
            if restored["phase"] != phase:
                raise ValueError("Checkpoint phase mismatch")
        model.set_phase(phase); model.eval(); rows = []
        with torch.inference_mode():
            for audit_row in audit["rows"]:
                qid = audit_row["qid"]
                score_path = output / "scores" / name / f"{qid}.json"
                if score_path.exists():
                    rows.append(result_from_mask_scores(read_record(score_path), manifests[qid], device=evaluation_device))
                    continue
                batch = stream.batch(qid); example, bank = batch["examples"][0], batch["banks"][0]
                encoding = model(example, batch["proxy_prompts"][0], native_layout=batch["native_layouts"][0], return_encoding=True)
                scores = score_candidate_masks(encoding, bank.masks, model.correction)
                head1, head1_pairs = question_metrics(scores.direct, manifests[qid])
                combined, combined_pairs = question_metrics(scores.total, manifests[qid])
                retained = (bank.masks * example.layout.costs.cpu()).sum(1).to(scores.direct)
                retention, retention_pairs = question_metrics(retained, manifests[qid])
                retained = (bank.masks * example.layout.costs.cpu()).sum(1)
                mask_rows = [dict(mask_index=index, retained_tokens=int(retained[index]), head1=float(scores.direct[index]), head2_correction=float(scores.correction[index]), combined=float(scores.total[index])) for index in range(len(bank.outcomes))]
                result = dict(qid=qid, split=audit_row["split"], baseline_correct=bank.baseline_correct, exposures_in_branch_path=exposures,
                              masks=mask_rows, head1=head1, head1_pairs=head1_pairs, combined=combined, combined_pairs=combined_pairs,
                              retention=retention, retention_pairs=retention_pairs)
                publish(score_path, result); rows.append(result)
                del batch, example, bank, encoding, scores
        summaries[name] = {split: summarize([row for row in rows if row["split"] == split]) for split in ("train", "dev")}
        content[name] = {split: content_analysis([row for row in rows if row["split"] == split], manifests) for split in ("train", "dev")}
        if saved is not None:
            gates[name] = reproduction([row for row in rows if row["split"] == "dev"], saved_rows(saved), args.tolerance)
            if not gates[name]["passed"]:
                raise ValueError(f"Saved dev metric reproduction failed for {name}: {gates[name]}")
        publish(output / "checkpoint-summaries" / f"{name}.json", dict(checkpoint=name, phase=phase, path=None if path is None else str(path), exposures_in_branch_path=exposures, summary=summaries[name], content_analysis=content[name], reproduction=gates.get(name)))
    publish(output / "summary.json", dict(schema="pilot-fixed-checkpoint-evaluation-v1", exact_seen_training_pairs=True,
        note="All strict training-bank pairs were consumed once per active question per epoch; no pair subsampling occurred.",
        checkpoints=[name for name, *_ in checkpoints], summaries=summaries, content_analysis=content, reproduction=gates))
    publish(output / "complete.json", dict(status="passed", summary_sha256=sha(output / "summary.json"), audit_sha256=sha(args.audit),
        training_contract_sha256=sha(training / "contract.json"), pair_manifest_files=len(manifests), resources=stats(started, torch),
        scope="Selector-only fixed checkpoint diagnosis; no reader calls, training, temperature fitting, or checkpoint reselection."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for argument in ("root", "training", "output", "audit", "selector"):
        parser.add_argument("--" + argument, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=2e-5)
    options = parser.parse_args()
    with run_lock(options.output):
        run(options)
