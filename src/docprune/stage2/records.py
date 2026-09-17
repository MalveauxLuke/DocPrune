"""Source annotations and offline answers, separate from selector inputs."""

import json
from collections import Counter
from dataclasses import asdict, dataclass

from .contracts import fingerprint


@dataclass(frozen=True)
class AnswerSpec:
    kind: str
    alternatives: tuple[tuple[str, ...], ...]

    def validate(self):
        if self.kind not in ("single", "list", "unanswerable"):
            raise ValueError("Unknown answer contract")
        if self.kind == "unanswerable":
            if self.alternatives:
                raise ValueError("Abstention strings need a separately approved contract")
            return self
        if not self.alternatives or any(
            not alt or any(not isinstance(x, str) or not x.strip() for x in alt)
            for alt in self.alternatives
        ):
            raise ValueError("Complete answer alternatives must be nonempty")
        if self.kind == "single" and any(len(alt) != 1 for alt in self.alternatives):
            raise ValueError("Single answers cannot contain component lists")
        return self

    def targets(self):
        self.validate()
        if self.kind == "unanswerable":
            raise ValueError("Unanswerable teacher collection needs a separate abstention contract")
        return tuple(
            alt[0] if self.kind == "single" else json.dumps(list(alt), ensure_ascii=False)
            for alt in self.alternatives
        )

    def complete_correct(self, prediction):
        """Strict normalized complete answer, not an official benchmark metric.

        List generation uses an explicit JSON-array answer contract. No guessing
        separators, partial matches, numerical units, or semantic equivalents.
        """
        self.validate()
        if self.kind == "unanswerable":
            raise ValueError("Abstention adjudication is not configured")
        if self.kind == "list":
            try:
                values = json.loads(prediction) if isinstance(prediction, str) else prediction
            except (ValueError, TypeError):
                return False
            if not isinstance(values, list | tuple) or not all(isinstance(x, str) for x in values):
                return False
        else:
            if not isinstance(prediction, str):
                return False
            values = [prediction]

        def normalize(xs):
            return Counter(" ".join(x.casefold().split()) for x in xs)

        return any(normalize(values) == normalize(alt) for alt in self.alternatives)

    @classmethod
    def from_dict(cls, value):
        return cls(value["kind"], tuple(tuple(a) for a in value["alternatives"])).validate()


@dataclass(frozen=True)
class QuestionRecord:
    source: str
    document_family: str
    question_id: str
    question: str
    official_split: str
    page_ids: tuple[str, ...]
    answer: AnswerSpec
    annotation_identity: str
    evidence_sets: tuple[tuple[str, ...], ...] = ()
    evidence_status: str = "unknown"
    tags: tuple[str, ...] = ()

    @property
    def key(self):
        return fingerprint((self.source, self.document_family, self.question_id))

    def validate(self):
        if any(
            not x
            for x in (
                self.source,
                self.document_family,
                self.question_id,
                self.question,
                self.official_split,
                self.annotation_identity,
            )
        ):
            raise ValueError("Question identity and annotation provenance are required")
        if not self.page_ids or len(set(self.page_ids)) != len(self.page_ids):
            raise ValueError("Unique source page identities are required")
        self.answer.validate()
        if self.evidence_status not in (
            "unknown",
            "annotated_path",
            "verified_sufficient_sets",
            "partial",
        ):
            raise ValueError("Unknown evidence provenance")
        if any(not e or not set(e).issubset(self.page_ids) for e in self.evidence_sets):
            raise ValueError("Evidence pages must belong to the source document")
        if self.evidence_status == "unknown" and self.evidence_sets:
            raise ValueError("Evidence needs an explicit provenance label")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["answer"] = AnswerSpec.from_dict(value["answer"])
        for name in ("page_ids", "tags"):
            value[name] = tuple(value.get(name, ()))
        value["evidence_sets"] = tuple(tuple(e) for e in value.get("evidence_sets", ()))
        return cls(**value).validate()
