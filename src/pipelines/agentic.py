from __future__ import annotations

import logging

from src.agentic.graph import build_graph, recursion_limit
from src.agentic.nodes import NodeDeps
from src.agentic.state import initial_state
from src.config import Config
from src.generation.context import expand_to_parents
from src.llm.router_llm import LLMRouter
from src.pipelines.base import Pipeline, ProgressEvent, RagResult
from src.retrieval.pipeline import RetrievalPipeline
from src.vectorstore.base import Hit

logger = logging.getLogger(__name__)


class AgenticPipeline(Pipeline):
    """Plans, decomposes, retrieves in a loop, grades its own evidence and
    self-checks the answer.

    Takes the *same* RetrievalPipeline instance as the naive path - retrieval
    logic exists in one place, so a config flag flipped for one path applies to
    both.
    """

    name = "agentic"

    def __init__(self, config: Config, retrieval: RetrievalPipeline, llm: LLMRouter) -> None:
        self._config = config
        self._graph = build_graph(NodeDeps(config=config, retrieval=retrieval, llm=llm))
        self._recursion_limit = recursion_limit(config)

    def stream(self, query: str):
        """Emit the graph's progress node by node.

        LangGraph's "values" mode hands back the whole state after each node, so
        newly appended trace lines are the natural progress feed - the same lines
        that end up in the final result, with no separate reporting path to drift.
        """
        state = initial_state(query)
        final = state
        seen_trace = 0
        announced = False

        for snapshot in self._graph.stream(
            state, {"recursion_limit": self._recursion_limit}, stream_mode="values"
        ):
            final = snapshot
            subs = snapshot.get("sub_questions") or []
            if subs and not announced:
                announced = True
                yield ProgressEvent(
                    "sub_questions", sub_questions=[s.original for s in subs]
                )
            trace = snapshot.get("trace") or []
            for line in trace[seen_trace:]:
                yield ProgressEvent("status", message=line)
            seen_trace = len(trace)

        yield ProgressEvent("result", result=self._result(final))

    def _result(self, final) -> RagResult:
        subs = final["sub_questions"]
        hits: list[Hit] = []
        seen: set[str] = set()
        for sub in subs:
            for hit in sub.evidence:
                if hit.id not in seen:
                    seen.add(hit.id)
                    hits.append(hit)

        parents = expand_to_parents(hits, self._config)
        trace = list(final["trace"])
        trace.append(f"agentic: {final['llm_calls']} LLM call(s) total")
        if not final["grounded"]:
            trace.append("warning: answer failed the final grounding check")

        return RagResult(
            answer=final["answer"],
            path=self.name,
            sources=parents,
            sub_questions=[s.original for s in subs],
            trace=trace,
            grounded=final["grounded"],
            llm_calls=final["llm_calls"],
        )
