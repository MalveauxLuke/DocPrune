"""Conservative, opt-in correction-corpus scoring; never replaces benchmark scores.

Contracts must be evidence-reviewed and frozen before looking at oracle outputs.
Unknown paraphrases abstain; they are not automatically declared wrong.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation

VERSION = "correction-contract-v1"


def normalize_surface(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("answer surfaces must be strings")
    value = unicodedata.normalize("NFC", value).translate(
        str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
    )
    value = " ".join(value.casefold().split()).strip()
    # Presentation punctuation only. Preserve signs, decimal points, units,
    # articles, negation, entity spelling, and word order within each item.
    if value.endswith("."):
        value = value[:-1].rstrip()
    return value


def contract_digest(contract: dict) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _registry(contract: dict) -> dict[str, str]:
    registry: dict[str, str] = {}
    canonicals = set()
    for item in contract.get("items", []):
        canonical = item["canonical"]
        if canonical in canonicals:
            raise ValueError("duplicate canonical item")
        canonicals.add(canonical)
        for alias in [canonical, *item.get("aliases", [])]:
            key = normalize_surface(alias)
            if not key or (key in registry and registry[key] != canonical):
                raise ValueError("empty or ambiguous item alias")
            registry[key] = canonical
    if not registry:
        raise ValueError("contract requires answer items")
    return registry


def score_answer(answer: str | list[str], contract: dict) -> dict:
    """Return correct / incorrect / review under a question-specific contract."""
    kind = contract.get("kind")
    if kind not in {"scalar", "set", "ordered", "number"}:
        raise ValueError("unsupported answer contract kind")
    registry = _registry(contract)
    expected = [x["canonical"] for x in contract["items"]]
    if kind in {"scalar", "number"} and len(expected) != 1:
        raise ValueError("scalar and numeric contracts need one canonical item")
    accepted = {normalize_surface(x) for x in contract.get("accepted_complete", [])}
    rejected = {normalize_surface(x) for x in contract.get("rejected_complete", [])}
    if accepted & rejected or rejected & set(registry):
        raise ValueError("accepted/rejected answer collision")
    digest = contract_digest(contract)

    def result(status: str, reason: str, **extra: object) -> dict:
        return dict(status=status, reason=reason, version=VERSION,
                    contract_sha256=digest, raw_answer=answer, **extra)

    surface = normalize_surface(answer) if isinstance(answer, str) else None
    if surface in accepted:
        return result("correct", "reviewed complete-answer alias")
    if surface in rejected:
        return result("incorrect", "explicitly reviewed incorrect complete answer")
    if kind == "scalar":
        if surface in registry:
            return result("correct", "canonical or reviewed alias")
        if surface in {normalize_surface(x) for x in contract.get("closed_domain", [])}:
            return result("incorrect", "different member of reviewed closed domain")
        return result("review", "unrecognized scalar; possible alias/paraphrase")
    if kind == "number":
        if surface in registry:
            return result("correct", "canonical or reviewed numeric alias")
        if surface is None:
            return result("review", "numeric answer must be a scalar")
        token = surface
        unit = normalize_surface(contract.get("unit", ""))
        if unit:
            suffix = " " + unit
            if token.endswith(suffix):
                token = token[:-len(suffix)]
            elif not contract.get("unit_optional", False):
                return result("review", "unit missing or not the declared unit")
        # Do not strip commas or signs: 1,234 and 1.234 can be locale-dependent.
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", token):
            return result("review", "ambiguous number format or unsupported units")
        try:
            value = Decimal(token)
            target = Decimal(contract["numeric_value"])
            tolerance = Decimal(contract.get("absolute_tolerance", "0"))
        except InvalidOperation as exc:
            raise ValueError("invalid numeric contract") from exc
        if not target.is_finite() or not tolerance.is_finite() or tolerance < 0:
            raise ValueError("invalid numeric target/tolerance")
        correct = abs(value - target) <= tolerance
        return result("correct" if correct else "incorrect", "typed numeric comparison")
    parts = answer
    if isinstance(answer, str):
        # One known item is one item, even if its name contains commas.
        if surface in registry:
            parts = [answer]
        else:
            try:
                parts = json.loads(answer)
            except (ValueError, TypeError):
                separator = contract.get("separator")
                parts = answer.split(separator) if separator else None
    if not isinstance(parts, list) or not all(isinstance(p, str) for p in parts):
        return result("review", "list serialization needs review; no generic comma splitting")
    canonical = []
    for part in parts:
        key = normalize_surface(part)
        if key not in registry:
            return result("review", "unknown list item, qualifier, or extra claim")
        canonical.append(registry[key])
    correct = set(canonical) == set(expected) if kind == "set" else canonical == expected
    return result("correct" if correct else "incorrect", "complete typed answer comparison",
                  canonical_items=canonical, missing_items=sorted(set(expected)-set(canonical)))


def correction_admission(case: dict, contract: dict) -> dict:
    """Fail closed on missing evidence, mismatched reference, or ambiguous baseline."""
    failures = []
    review = case.get("review") or {}
    baseline = case.get("baseline") or {}
    evidence = review.get("evidence", [])
    pages = {(p["doc_id"], p["page_index"]) for p in case.get("pages", [])}
    if len(case.get("pages", [])) != 4:
        failures.append("requires exactly four ordered retrieved pages")
    if case.get("protected_overlap_clear") is not True:
        failures.append("protected-cohort document exclusion not verified")
    if not evidence or any((e.get("doc_id"), e.get("page_index")) not in pages for e in evidence):
        failures.append("evidence locations missing or outside top four")
    for field in ["all_required_evidence_present", "readable_at_input_resolution",
                  "gold_valid", "question_unambiguous", "contract_frozen_before_oracle"]:
        if review.get(field) is not True:
            failures.append(field)
    if not review.get("reviewer") or not review.get("rationale"):
        failures.append("reviewer and evidence rationale required")
    if review.get("contract_sha256") != contract_digest(contract):
        failures.append("review does not bind this answer contract")
    trace = baseline.get("trace", {})
    if (baseline.get("kind") != "btp_qtp_no_ctp"
            or baseline.get("ordered_pages_match") is not True
            or trace.get("ctp_layer") is not None
            or not isinstance(trace.get("post_qtp_visual_tokens"), int)
            or trace.get("post_qtp_visual_tokens") != trace.get("post_ctp_visual_tokens")):
        failures.append("matching post-BTP/QTP no-CTP baseline required")
    if case.get("question_variant", False) and not baseline.get("matches_variant", False):
        failures.append("rewritten question needs its own baseline")
    score = score_answer(baseline.get("answer", ""), contract)
    if score["status"] != "incorrect":
        failures.append("baseline is not definitely incorrect")
    return dict(admitted=not failures, failures=failures, baseline_score=score)


def correction_outcome(case: dict, contract: dict, generated: str | list[str]) -> dict:
    admission = correction_admission(case, contract)
    score = score_answer(generated, contract)
    return dict(eligible=admission["admitted"], baseline=admission,
                generated_score=score,
                true_correction=admission["admitted"] and score["status"] == "correct",
                unresolved=score["status"] == "review")
