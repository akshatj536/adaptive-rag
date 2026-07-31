"""LangGraph wiring.

    orchestrate -> retrieve -> grade -+-> reformulate -> (back to retrieve)
                                      |
                                      +-> next_question -+-> (back to retrieve)
                                                         |
                                                         +-> synthesize -> self_check -+-> END
                                                                                       |
                                                                        (back to orchestrate)

The conditional edges are where cost is controlled: `max_loops` bounds the
refine cycle per sub-question and `max_self_checks` bounds the outer retry, so
no combination of model outputs can make this run forever.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from src.agentic.nodes import AgenticNodes, NodeDeps
from src.agentic.state import AgenticState, active

logger = logging.getLogger(__name__)


def build_graph(deps: NodeDeps):
    nodes = AgenticNodes(deps)
    agentic = deps.config.agentic

    graph = StateGraph(AgenticState)
    graph.add_node("orchestrate", nodes.orchestrate)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade", nodes.grade)
    graph.add_node("reformulate", nodes.reformulate)
    graph.add_node("next_question", nodes.next_question)
    graph.add_node("synthesize", nodes.synthesize)
    graph.add_node("self_check", nodes.self_check)

    graph.set_entry_point("orchestrate")
    graph.add_edge("orchestrate", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("reformulate", "retrieve")

    def after_grade(state: AgenticState) -> str:
        sub = active(state)
        if sub is None or sub.sufficient:
            return "next_question"
        if sub.loops >= agentic.max_loops:
            # Out of budget for this hop. Move on with partial evidence rather
            # than looping; synthesis is told the evidence was incomplete.
            return "next_question"
        if sub.loops > 0 and sub.last_added == 0:
            # The last rewrite surfaced nothing the corpus had not already
            # returned. Rewriting again spends two more LLM calls to re-read
            # the same chunks - the evidence simply is not there.
            logger.info("no new evidence from the last reformulation; moving on")
            return "next_question"
        return "reformulate"

    graph.add_conditional_edges(
        "grade", after_grade,
        {"reformulate": "reformulate", "next_question": "next_question"},
    )

    def after_next(state: AgenticState) -> str:
        return "retrieve" if state["current"] < len(state["sub_questions"]) else "synthesize"

    graph.add_conditional_edges(
        "next_question", after_next, {"retrieve": "retrieve", "synthesize": "synthesize"},
    )

    if agentic.enable_self_check:
        graph.add_edge("synthesize", "self_check")

        def after_self_check(state: AgenticState) -> str:
            if state["grounded"]:
                return END
            if state["self_check_count"] > agentic.max_self_checks:
                logger.warning("self-check budget exhausted; returning the answer as-is")
                return END
            return "orchestrate"

        graph.add_conditional_edges(
            "self_check", after_self_check, {END: END, "orchestrate": "orchestrate"},
        )
    else:
        graph.add_edge("synthesize", END)

    return graph.compile()


def recursion_limit(config) -> int:
    """A ceiling on total node visits, sized from the configured caps so a
    logic bug surfaces as a clear LangGraph error rather than a runaway loop."""
    a = config.agentic
    per_sub = 2 + a.max_loops * 2          # grade + next_question, plus refine/retrieve pairs
    per_pass = 1 + a.max_sub_questions * per_sub + 2   # plan + hops + synthesize/self-check
    return max(25, per_pass * (a.max_self_checks + 1) + 10)
