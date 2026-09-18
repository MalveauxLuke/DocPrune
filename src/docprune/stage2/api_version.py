"""Opt-in API labels; historical normalized/G-preservation behavior stays default."""

from docprune.answer_adjudication import digest, read


def load_labels(path):
    manifest = read(path)
    if (
        manifest["schema"] != "docprune-answer-api-labels-v1"
        or manifest["correct_preservation"] != "s"
    ):
        raise ValueError("Expected frozen API-version labels")
    rows = manifest["rows"]
    if len({r["qid"] for r in rows}) != len(rows):
        raise ValueError("Duplicate adjudicated IDs")
    return {
        r["qid"]: {
            **r,
            "manifest_identity": digest(manifest),
            "baseline_contract": manifest["baseline_contract"],
        }
        for r in rows
    }


def validate_baseline(label, document):
    record = document.record
    if not label.get("manifest_identity") or not label.get("baseline_contract"):
        raise ValueError("Missing frozen adjudication provenance")
    if label["verdict"] not in ("correct", "incorrect"):
        raise ValueError("Unresolved baseline cannot enter correctness-dependent training")
    if label["qid"] != record.question_id or label["question"] != record.question:
        raise ValueError("Adjudication question mismatch")
    # Source gold denotes a complete set, never a list of interchangeable answers.
    if tuple(label["gold"]) not in record.answer.alternatives:
        raise ValueError("Adjudication reference answer differs from prepared document")
    if not label["model_answer"].strip():
        raise ValueError("Empty adjudicated baseline cannot define S")
