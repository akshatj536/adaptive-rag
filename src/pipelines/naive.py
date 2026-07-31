from __future__ import annotations

import logging

from src.config import Config
from src.generation.context import expand_to_parents
from src.generation.generate import generate_answer
from src.llm.router_llm import LLMRouter
from src.pipelines.base import Pipeline, ProgressEvent, RagResult
from src.retrieval.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


class NaivePipeline(Pipeline):
    """retrieve -> expand children to parents -> generate. One LLM call."""

    name = "naive"

    def __init__(self, config: Config, retrieval: RetrievalPipeline, llm: LLMRouter) -> None:
        self._config = config
        self._retrieval = retrieval
        self._llm = llm

    def stream(self, query: str):
        yield ProgressEvent("status", message=f"Retrieving ({self._retrieval.describe()})…")
        ctx = self._retrieval.run(query)
        parents = expand_to_parents(ctx.candidates, self._config)
        ctx.log(f"expanded {len(ctx.candidates)} children -> {len(parents)} parents")
        yield ProgressEvent(
            "status",
            message=f"Found {len(ctx.candidates)} chunk(s) in {len(parents)} source passage(s)",
        )

        yield ProgressEvent("status", message="Generating the answer…")
        answer = generate_answer(
            query, parents, self._llm,
            max_tokens=self._config.generation.max_answer_tokens,
        )
        ctx.log(f"generated via {answer.provider}/{answer.model}")
        if answer.truncated:
            ctx.log("answer hit the token cap and was cut off")

        yield ProgressEvent("result", result=RagResult(
            answer=answer.text,
            path=self.name,
            sources=parents,
            trace=ctx.trace,
            provider=answer.provider,
            model=answer.model,
            fell_back=answer.fell_back,
            truncated=answer.truncated,
            llm_calls=1,   # the single generation call
        ))
