"""Node functions for the agentic graph.

Cost discipline runs through all of this. Each node is one rate-limited call,
so prompts stay small (evidence is truncated for grading, not sent whole), and
every node fails *open* - if the LLM is unavailable the graph moves forward
with a safe default instead of looping. A stuck agent burns quota; a
degraded-but-finishing agent still answers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from src.agentic.state import AgenticState, SubQuestion, active
from src.config import Config
from src.generation.context import expand_to_parents, format_context
from src.llm.base import LLMError, Message
from src.llm.router_llm import LLMRouter
from src.retrieval.pipeline import RetrievalPipeline
from src.vectorstore.base import Hit

logger = logging.getLogger(__name__)

# How much of each chunk the grader sees. Grading needs the gist, not the text.
GRADE_SNIPPET_CHARS = 500
GRADE_MAX_SNIPPETS = 4


@dataclass
class NodeDeps:
    config: Config
    retrieval: RetrievalPipeline
    llm: LLMRouter


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

PLANNER_PROMPT = """You plan how to answer a question using a document search tool.

Decide whether the question needs to be broken into sub-questions:
- Break it up when answering requires several lookups: multi-hop chains where one \
answer feeds the next, comparisons between things, or questions with distinct parts.
- Keep it as one question when a single search could find the answer.

Write sub-questions that are self-contained - each is sent to a search engine \
alone, so it must not rely on pronouns or on the answer to a previous one.

Every sub-question must be a *lookup*: something a passage in a document could \
directly state. Never write a sub-question that asks for a comparison, a \
difference, a ranking, or what one thing has "that the other does not" - no \
single passage contains that. Retrieve the facts about each thing separately; \
the comparison is performed later when the answer is composed.

Use at most {max_subs} sub-questions, and prefer fewer.

Respond with JSON only:
{{"strategy": "<one short clause>", "sub_questions": ["...", "..."]}}"""

GRADER_PROMPT = """You judge whether the retrieved excerpts are enough to answer \
a specific question.

Answer "sufficient": true only if the excerpts contain the actual facts needed. \
Related-but-not-answering material is not sufficient.
If it is not sufficient, say briefly what specific information is missing.

Respond with JSON only:
{"sufficient": true | false, "missing": "<what is absent, or empty>"}"""

REFORMULATE_PROMPT = """Rewrite a search query that failed to find what was needed.

You are told the original question and what was missing. Produce one better \
search query: use different or more specific wording, likely synonyms, and \
terms that would literally appear in a document containing the answer.

Respond with JSON only:
{"query": "<the rewritten search query>"}"""

SYNTHESIS_PROMPT = """You answer a question using evidence gathered across several \
searches.

Rules:
- Use only the numbered sources. Do not add outside knowledge.
- Cite inline like [1] or [2][3].
- Where the evidence is incomplete, say so plainly rather than filling the gap.
- Answer the user's original question directly. The sub-questions were only a \
research strategy; do not structure your answer around them unless it helps."""

SELF_CHECK_PROMPT = """You check whether an answer is supported by its sources.

Mark grounded: false only for claims that contradict the sources or that appear \
nowhere in them. An answer that correctly says information is missing is grounded. \
Reasonable paraphrase and summary are grounded.

Respond with JSON only:
{"grounded": true | false, "issue": "<the unsupported claim, or empty>"}"""


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------


class AgenticNodes:
    def __init__(self, deps: NodeDeps) -> None:
        self.cfg = deps.config
        self.retrieval = deps.retrieval
        self.llm = deps.llm
        self.agentic = deps.config.agentic

    # -- planning ----------------------------------------------------------
    def orchestrate(self, state: AgenticState) -> dict:
        query = state["query"]
        trace = list(state["trace"])

        if not self.agentic.enable_decomposition:
            trace.append("orchestrator: decomposition disabled, single hop")
            return {
                "plan": "single retrieval (decomposition disabled)",
                "sub_questions": [SubQuestion(text=query)],
                "current": 0, "trace": trace,
            }

        user = query
        if state["replan_note"]:
            # Second pass after a failed self-check: tell the planner what went
            # wrong so it does not simply repeat the same plan.
            user = (
                f"{query}\n\nA previous attempt produced an answer that was not "
                f"grounded in the evidence. Problem: {state['replan_note']}\n"
                f"Plan differently: target the missing evidence directly."
            )

        text, calls = self._ask(
            "reasoning",
            PLANNER_PROMPT.format(max_subs=max(1, self.agentic.max_sub_questions)),
            user,
            max_tokens=400,
        )
        data = _json(text)
        subs = data.get("sub_questions") if isinstance(data, dict) else None
        strategy = str(data.get("strategy", "")).strip() if isinstance(data, dict) else ""

        if not isinstance(subs, list) or not subs:
            trace.append("orchestrator: no usable decomposition, treating as single hop")
            subs_clean = [query]
        else:
            subs_clean = [str(s).strip() for s in subs if str(s).strip()]
            subs_clean = subs_clean[: max(1, self.agentic.max_sub_questions)] or [query]

        trace.append(
            f"orchestrator: {strategy or 'planned'} -> {len(subs_clean)} sub-question(s)"
        )
        return {
            "plan": strategy or "decomposed",
            "sub_questions": [SubQuestion(text=s) for s in subs_clean],
            "current": 0,
            "trace": trace,
            "llm_calls": state["llm_calls"] + calls,
        }

    # -- retrieval (reuses the shared component pipeline) ------------------
    def retrieve(self, state: AgenticState) -> dict:
        sub = active(state)
        trace = list(state["trace"])
        if sub is None:
            return {"trace": trace}

        ctx = self.retrieval.run(sub.text)
        added = sub.add_evidence(ctx.candidates)
        sub.last_added = added
        trace.append(
            f"retrieve[{state['current'] + 1}] {sub.text[:60]!r}: "
            f"{len(ctx.candidates)} hits, {added} new"
        )
        return {"sub_questions": state["sub_questions"], "trace": trace}

    # -- grading -----------------------------------------------------------
    def grade(self, state: AgenticState) -> dict:
        sub = active(state)
        trace = list(state["trace"])
        if sub is None:
            return {"trace": trace}

        if not sub.evidence:
            sub.sufficient = False
            sub.gap = "no documents retrieved"
            trace.append(f"grade[{state['current'] + 1}]: insufficient (nothing retrieved)")
            return {"sub_questions": state["sub_questions"], "trace": trace}

        text, calls = self._ask(
            "reasoning", GRADER_PROMPT,
            f"Question: {sub.text}\n\nExcerpts:\n{_snippets(sub.evidence)}",
            max_tokens=200,
        )
        data = _json(text)
        if isinstance(data, dict) and "sufficient" in data:
            sub.sufficient = bool(data.get("sufficient"))
            sub.gap = str(data.get("missing", "")).strip()
        else:
            # Unreadable verdict: accept the evidence rather than spend another
            # loop on a grader that is not responding usefully.
            sub.sufficient = True
            sub.gap = ""
            trace.append(f"grade[{state['current'] + 1}]: unparseable, accepting evidence")
            return {"sub_questions": state["sub_questions"], "trace": trace,
                    "llm_calls": state["llm_calls"] + calls}

        verdict = "sufficient" if sub.sufficient else f"insufficient ({sub.gap[:60]})"
        trace.append(f"grade[{state['current'] + 1}]: {verdict}")
        return {"sub_questions": state["sub_questions"], "trace": trace,
                "llm_calls": state["llm_calls"] + calls}

    # -- reformulation -----------------------------------------------------
    def reformulate(self, state: AgenticState) -> dict:
        sub = active(state)
        trace = list(state["trace"])
        if sub is None:
            return {"trace": trace}

        sub.loops += 1
        text, calls = self._ask(
            "reasoning", REFORMULATE_PROMPT,
            f"Original question: {sub.original}\nCurrent query: {sub.text}\n"
            f"Missing: {sub.gap or 'nothing relevant was found'}",
            max_tokens=150,
        )
        data = _json(text)
        new_query = str(data.get("query", "")).strip() if isinstance(data, dict) else ""

        if new_query and new_query != sub.text:
            trace.append(f"reformulate[{state['current'] + 1}] #{sub.loops}: {new_query[:70]!r}")
            sub.text = new_query
        else:
            trace.append(f"reformulate[{state['current'] + 1}] #{sub.loops}: no better query")
        return {"sub_questions": state["sub_questions"], "trace": trace,
                "llm_calls": state["llm_calls"] + calls}

    # -- advance -----------------------------------------------------------
    def next_question(self, state: AgenticState) -> dict:
        sub = active(state)
        trace = list(state["trace"])
        if sub is not None and not sub.sufficient:
            sub.gave_up = True
            trace.append(
                f"sub-question {state['current'] + 1} gave up after "
                f"{sub.loops} reformulation(s); using what was found"
            )
        return {"current": state["current"] + 1, "sub_questions": state["sub_questions"],
                "trace": trace}

    # -- synthesis ---------------------------------------------------------
    def synthesize(self, state: AgenticState) -> dict:
        trace = list(state["trace"])
        all_hits: list[Hit] = []
        seen: set[str] = set()
        for sub in state["sub_questions"]:
            for hit in sub.evidence:
                if hit.id not in seen:
                    seen.add(hit.id)
                    all_hits.append(hit)

        parents = expand_to_parents(all_hits, self.cfg)
        if not parents:
            trace.append("synthesize: no evidence gathered")
            return {"answer": "I couldn't find anything relevant in the indexed "
                              "documents to answer that.", "trace": trace}

        hops = "\n".join(
            f"{i + 1}. {s.original}" + ("  [evidence incomplete]" if s.gave_up else "")
            for i, s in enumerate(state["sub_questions"])
        )
        user = (
            f"Original question: {state['query']}\n\n"
            f"Sub-questions researched:\n{hops}\n\n"
            f"Sources:\n\n{format_context(parents)}\n\n"
            f"Answer the original question."
        )
        text, calls = self._ask(
            "generation", SYNTHESIS_PROMPT, user,
            max_tokens=self.cfg.generation.max_answer_tokens,
        )
        trace.append(f"synthesize: {len(parents)} parent(s) across {len(state['sub_questions'])} hop(s)")
        return {"answer": text or "(synthesis produced no answer)", "trace": trace,
                "llm_calls": state["llm_calls"] + calls}

    # -- self check --------------------------------------------------------
    def self_check(self, state: AgenticState) -> dict:
        trace = list(state["trace"])
        if not self.agentic.enable_self_check:
            return {"grounded": True, "trace": trace}

        all_hits = [h for s in state["sub_questions"] for h in s.evidence]
        if not all_hits or not state["answer"]:
            return {"grounded": True, "trace": trace}

        text, calls = self._ask(
            "reasoning", SELF_CHECK_PROMPT,
            f"Answer:\n{state['answer'][:3000]}\n\nSources:\n{_snippets(all_hits, limit=6)}",
            max_tokens=200,
        )
        data = _json(text)
        grounded = bool(data.get("grounded", True)) if isinstance(data, dict) else True
        issue = str(data.get("issue", "")).strip() if isinstance(data, dict) else ""

        trace.append(f"self-check: {'grounded' if grounded else 'NOT grounded - ' + issue[:70]}")
        return {
            "grounded": grounded,
            "replan_note": issue,
            "self_check_count": state["self_check_count"] + (0 if grounded else 1),
            "trace": trace,
            "llm_calls": state["llm_calls"] + calls,
        }

    # -- helper ------------------------------------------------------------
    def _ask(self, role: str, system: str, user: str, *, max_tokens: int) -> tuple[str, int]:
        """One LLM call. Returns (text, calls_made) - empty text on failure so
        callers apply their own safe default rather than raising mid-graph."""
        try:
            response = self.llm.complete(
                role, [Message("system", system), Message("user", user)],
                temperature=0.0, max_tokens=max_tokens,
            )
            return response.text, 1
        except LLMError as exc:
            logger.warning("agentic node (%s) failed: %s", role, exc)
            return "", 1


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _snippets(hits: list[Hit], limit: int = GRADE_MAX_SNIPPETS) -> str:
    parts = []
    for i, hit in enumerate(hits[:limit], start=1):
        source = hit.metadata.get("source", "?")
        page = hit.metadata.get("page_no", "")
        parts.append(f"[{i}] ({source} p.{page}) {hit.text[:GRADE_SNIPPET_CHARS]}")
    return "\n\n".join(parts)


def _json(text: str):
    """Extract the first JSON object. Small models fence it or pad it with prose."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
