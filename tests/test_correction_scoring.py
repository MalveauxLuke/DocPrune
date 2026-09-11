"""Regression examples for false rescues and conservative abstention."""
from copy import deepcopy
from docprune.correction_scoring import (
    contract_digest, correction_admission, correction_outcome, score_answer,
)


def scalar():
    return {"kind": "scalar", "items": [{"canonical": "moustache", "aliases": ["mustache"]}],
            "rejected_complete": ["beard"]}


def case(contract):
    return {"pages": [{"doc_id": "doc", "page_index": i} for i in range(4)],
            "protected_overlap_clear": True,
            "baseline": {"kind": "btp_qtp_no_ctp", "ordered_pages_match": True,
                         "answer": "beard", "trace": {"ctp_layer": None,
                         "post_qtp_visual_tokens": 100, "post_ctp_visual_tokens": 100}},
            "review": {"evidence": [{"doc_id": "doc", "page_index": 0}],
                       "all_required_evidence_present": True, "readable_at_input_resolution": True,
                       "gold_valid": True, "question_unambiguous": True,
                       "contract_frozen_before_oracle": True, "reviewer": "fixture reviewer",
                       "rationale": "Fixture portrait has moustache", "contract_sha256": contract_digest(contract)}}


def test_spelling_alias_and_real_deterioration():
    c=scalar()
    assert score_answer(" Mustache. ", c)["status"] == "correct"
    assert score_answer("Beard", c)["status"] == "incorrect"
    assert score_answer("not a moustache", c)["status"] == "review"
    assert score_answer("moustache and beard", c)["status"] == "review"


def test_alias_is_question_specific():
    c=scalar();c["items"][0]["aliases"]=[]
    assert score_answer("mustache", c)["status"] == "review"


def test_complete_set_beats_single_item_and_handles_comma_in_name():
    c={"kind":"set","items":[{"canonical":"Zhang Ling"},{"canonical":"Tommy"}],
       "accepted_complete":["Zhang Ling, Tommy."]}
    assert score_answer("Zhang Ling, Tommy.",c)["status"]=="correct"
    assert score_answer("Zhang Ling",c)["status"]=="incorrect"
    assert score_answer('["Tommy", "Zhang Ling"]',c)["status"]=="correct"
    assert score_answer('["Tommy", "Zhang Ling", "wrong role"]',c)["status"]=="review"
    assert score_answer("Tommy, Zhang Ling",c)["status"]=="review"
    c={"kind":"set","items":[{"canonical":"Washington, D.C."},{"canonical":"Paris"}]}
    assert score_answer('["Washington, D.C.", "Paris"]',c)["status"]=="correct"


def test_order_and_numeric_meaning_preserved():
    c={"kind":"ordered","items":[{"canonical":"A"},{"canonical":"B"}]}
    assert score_answer('["B","A"]',c)["status"]=="incorrect"
    c={"kind":"number","items":[{"canonical":"3 kg"}],"numeric_value":"3","unit":"kg"}
    assert score_answer("3.00 kg",c)["status"]=="correct"
    assert score_answer("-3 kg",c)["status"]=="incorrect"
    assert score_answer("3 lb",c)["status"]=="review"
    assert score_answer("3",c)["status"]=="review"
    assert score_answer("3,000 kg",c)["status"]=="review"


def test_true_correction_requires_wrong_baseline_and_bound_evidence():
    contract=scalar();c=case(contract)
    assert correction_outcome(c,contract,"Mustache.")["true_correction"]
    c["baseline"]["answer"]="Mustache"
    assert not correction_outcome(c,contract,"moustache")["eligible"]
    c=case(contract);c["review"]["evidence"][0]["page_index"]=9
    assert not correction_admission(c,contract)["admitted"]
    c=case(contract);c["baseline"]["trace"]["post_ctp_visual_tokens"]=50
    assert not correction_admission(c,contract)["admitted"]
    c=case(contract);c["question_variant"]=True
    assert not correction_admission(c,contract)["admitted"]
    c=case(contract);changed=deepcopy(contract);changed["items"][0]["aliases"].append("facial hair")
    assert not correction_admission(c,changed)["admitted"]
    for field in ["gold_valid","all_required_evidence_present","readable_at_input_resolution"]:
        c=case(contract);c["review"][field]=False
        assert not correction_admission(c,contract)["admitted"]


def test_stale_year_cannot_be_rescued_by_normalization():
    c={"kind":"number","items":[{"canonical":"2024"}],"numeric_value":"2024"}
    assert score_answer("1981",c)["status"]=="incorrect"


def test_conflicting_contract_is_rejected():
    c=scalar();c["rejected_complete"].append("Mustache")
    try:score_answer("beard",c)
    except ValueError:pass
    else:raise AssertionError("ambiguous labels accepted")
