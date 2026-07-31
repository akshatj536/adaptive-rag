"""Query complexity classification.

One cheap, fast LLM call decides which path a query takes. This is the whole
economy of the system: the agentic path costs many LLM calls, so it must only
run when the query actually needs it.

The classifier is deliberately fail-open. It sits in front of every query, so
a malformed response, a rate limit, or an outage must never be the reason a
question goes unanswered - any failure falls back to the default path.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from src.config import RouterConfig
from src.llm.base import LLMError, Message
from src.llm.router_llm import LLMRouter

logger = logging.getLogger(__name__)

SIMPLE = "simple"
COMPLEX = "complex"

SYSTEM_PROMPT = """You classify how much work a question needs to answer from a \
document collection. Reply with JSON only.

Answer "simple" when the question asks for one fact, definition, or figure that \
would sit in a single passage.
Answer "complex" when answering requires several steps: multi-hop lookups where \
one answer feeds the next, comparisons across entities or documents, aggregation \
or synthesis across many passages, or a question with several distinct parts.

Respond with exactly this JSON and nothing else:
{"complexity": "simple" | "complex", "reason": "<one short clause>"}"""


@dataclass
class RouteDecision:
    """Why a query went where it went. Attached to the result so behaviour is
    inspectable rather than mysterious."""

    path: str                      # path that will actually run
    requested_path: str = ""       # what the classifier picked, before availability
    complexity: str = ""           # simple | complex | ""
    reason: str = ""
    classified: bool = True        # False when the router was off or the call failed

    @property
    def degraded(self) -> bool:
        """True when the chosen path was unavailable and we fell back."""
        return bool(self.requested_path) and self.requested_path != self.path


class QueryClassifier:
    def __init__(self, llm: LLMRouter, cfg: RouterConfig, role: str = "routing") -> None:
        self._llm = llm
        self._cfg = cfg
        self._role = role

    def classify(self, query: str) -> RouteDecision:
        default = self._cfg.default_path
        try:
            response = self._llm.complete(
                self._role,
                [Message("system", SYSTEM_PROMPT), Message("user", query)],
                temperature=0.0,
                max_tokens=120,
            )
        except LLMError as exc:
            logger.warning("classifier unavailable (%s); using default path %r", exc, default)
            return RouteDecision(
                path=default, complexity="", reason=f"classifier unavailable: {type(exc).__name__}",
                classified=False,
            )

        complexity, reason = _parse(response.text)
        if complexity is None:
            logger.warning(
                "classifier returned unparseable output %r; using default path %r",
                response.text[:120], default,
            )
            return RouteDecision(
                path=default, complexity="", reason="unparseable classifier output",
                classified=False,
            )

        path = self._path_for(complexity)
        return RouteDecision(path=path, requested_path=path, complexity=complexity, reason=reason)

    def _path_for(self, complexity: str) -> str:
        """Map a complexity label to a path.

        Kept as a lookup rather than an if/else so a future third label (and its
        path, e.g. 'graph') is a config and prompt change, not a rewrite.
        """
        mapping = {SIMPLE: "naive", COMPLEX: "agentic"}
        path = mapping.get(complexity, self._cfg.default_path)
        return path if path in self._cfg.paths else self._cfg.default_path


def _parse(text: str) -> tuple[str | None, str]:
    """Parse the classifier reply. Small models wrap JSON in prose or fences,
    so accept anything containing the label rather than demanding clean JSON."""
    text = text.strip()
    if not text:
        return None, ""

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            complexity = str(data.get("complexity", "")).strip().lower()
            if complexity in (SIMPLE, COMPLEX):
                return complexity, str(data.get("reason", "")).strip()
            # Valid JSON with an unrecognized label: the model answered, just
            # not with a label we know. Don't guess - guessing "complex" would
            # silently route to the expensive path.
            return None, ""

    # Unstructured reply: look for a whole word, and never inside the key name
    # "complexity" (which contains "complex" and would match every time).
    lowered = re.sub(r"complexity", " ", text.lower())
    has_complex = re.search(rf"\b{COMPLEX}\b", lowered)
    has_simple = re.search(rf"\b{SIMPLE}\b", lowered)
    if has_complex and not has_simple:
        return COMPLEX, "inferred from unstructured reply"
    if has_simple and not has_complex:
        return SIMPLE, "inferred from unstructured reply"
    return None, ""
