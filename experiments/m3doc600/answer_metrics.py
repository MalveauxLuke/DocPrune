"""CPU-only official EM/F1 logic copied from src/docprune/evaluation.py.
Kept separate from legacy model imports; normalization and list alignment unchanged.
"""
import re,string
from functools import cache
from word2number.w2n import word_to_num

_PUNCTUATION = set(string.punctuation)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    if word_to_num is None:  # defensive for a corrupted/monkeypatched runtime
        raise RuntimeError("word2number is required for official M3DocVQA answer normalization")
    try:
        return str(float(word_to_num(text)))
    except (TypeError, ValueError):
        pass
    return text


def _normalize_answer(text: object) -> str:
    value = str(text).lower()
    tokens = re.split(" |-", value)
    normalized: list[str] = []
    for token in tokens:
        if not token:
            continue
        if not _is_number(token):
            token = "".join(char for char in token if char not in _PUNCTUATION)
        token = " ".join(token.split())
        token = " ".join(re.sub(r"\b(a|an|the)\b", " ", token).split())
        token = _normalize_number(token)
        if token.strip():
            normalized.append(token.strip())
    return " ".join(normalized).strip()


def _answer_to_bags(answer: object) -> tuple[list[str], list[set[str]]]:
    raw_spans = answer if isinstance(answer, list | tuple) else [answer]
    normalized = [_normalize_answer(span) for span in raw_spans]
    return normalized, [set(span.split()) for span in normalized]


def _match_numbers(gold: set[str], predicted: set[str]) -> bool:
    gold_numbers = {token for token in gold if _is_number(token)}
    predicted_numbers = {token for token in predicted if _is_number(token)}
    return not gold_numbers or bool(gold_numbers.intersection(predicted_numbers))


def _bag_f1(predicted: set[str], gold: set[str]) -> float:
    intersection = len(predicted.intersection(gold))
    precision = 1.0 if not predicted else intersection / len(predicted)
    recall = 1.0 if not gold else intersection / len(gold)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _aligned_f1(predicted: list[set[str]], gold: list[set[str]]) -> float:
    """Match answer bags with the same maximum-weight alignment as the reference."""

    if not gold and not predicted:
        return 0.0
    scores = [
        [_bag_f1(pred, ref) if _match_numbers(ref, pred) else 0.0 for pred in predicted]
        for ref in gold
    ]
    # The official implementation uses scipy's Hungarian algorithm.  Answer
    # lists are short; this exact bitmask DP has the same maximum assignment
    # value without adding scipy as a production dependency.
    if len(predicted) > 20:
        raise ValueError("answer lists longer than 20 spans are unsupported")

    @cache
    def best(row: int, used: int) -> float:
        if row == len(gold):
            return 0.0
        value = best(row + 1, used)  # leave this gold span unmatched
        for column, score in enumerate(scores[row]):
            if not used & (1 << column):
                value = max(value, score + best(row + 1, used | (1 << column)))
        return value

    return round(best(0, 0) / max(len(gold), len(predicted)), 2)


def list_em(predicted: object, gold: object) -> float:
    predicted_spans, _ = _answer_to_bags(predicted)
    gold_spans, _ = _answer_to_bags(gold)
    return float(
        set(predicted_spans) == set(gold_spans) and len(predicted_spans) == len(gold_spans)
    )


def list_f1(predicted: object, gold: object) -> float:
    _, predicted_bags = _answer_to_bags(predicted)
    _, gold_bags = _answer_to_bags(gold)
    return _aligned_f1(predicted_bags, gold_bags)

