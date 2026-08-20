"""Scoring for the evaluation runs.

Answers are generated prose while HotpotQA golds are short spans, so exact
match is near-useless on its own. `contains` (normalized gold appearing in the
normalized prediction) is the workhorse; token F1 catches partial credit.
Both are free. The LLM judge is only needed where these disagree with a human.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize(text: str) -> str:
    """HotpotQA's official normalization: lowercase, strip punctuation,
    articles and extra whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def contains(pred: str, gold: str) -> float:
    """Does the answer state the gold span at all? The most informative cheap
    metric for generated answers, and the closest proxy for 'is it correct'."""
    g = normalize(gold)
    return float(bool(g) and g in normalize(pred))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def abstained(pred: str) -> bool:
    """The system was told to say so rather than guess. That is a distinct
    outcome from a wrong answer and should not be scored as one."""
    p = pred.lower()
    return any(s in p for s in (
        "do not contain", "does not contain", "not contain", "no information",
        "not specified", "not mentioned", "cannot be determined", "couldn't find",
        "could not find", "not provided", "unable to determine", "not found",
    ))
