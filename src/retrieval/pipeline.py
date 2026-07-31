from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import Config
from src.embeddings.base import Embedder
from src.llm.router_llm import LLMRouter
from src.retrieval.base import RetrievalComponent, RetrievalContext
from src.retrieval.bm25 import BM25Component
from src.retrieval.hyde import HydeComponent
from src.retrieval.rerank import RerankComponent
from src.retrieval.vector import VectorComponent
from src.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalDeps:
    """Everything a component might need. Passed to every builder so adding a
    component never changes the pipeline's construction signature."""

    config: Config
    store: VectorStore
    embedder: Embedder
    llm: LLMRouter


# name -> builder. Adding a component means one class + one entry here + one
# config flag. Nothing else in the codebase changes.
COMPONENT_REGISTRY: dict[str, callable] = {
    "vector": lambda deps: VectorComponent(
        deps.store, deps.embedder, deps.config.retrieval.bm25.rrf_k
    ),
    "bm25": lambda deps: BM25Component(deps.store, deps.config.retrieval.bm25),
    "hyde": lambda deps: HydeComponent(deps.llm, deps.config.retrieval.hyde),
    "rerank": lambda deps: RerankComponent(deps.config.retrieval.rerank),
}


class RetrievalPipeline:
    """Runs the enabled components in config order. Shared by the naive path
    and (slice 4) the agentic path - retrieval logic exists in exactly one place."""

    def __init__(self, components: list[RetrievalComponent], top_k: int, fetch_k: int) -> None:
        self.components = components
        self.top_k = top_k
        self.fetch_k = fetch_k

    @classmethod
    def from_config(cls, deps: RetrievalDeps) -> "RetrievalPipeline":
        cfg = deps.config.retrieval
        components: list[RetrievalComponent] = []
        for name in cfg.enabled_components():
            builder = COMPONENT_REGISTRY.get(name)
            if builder is None:
                logger.warning(
                    "Retrieval component %r is enabled in config but not implemented yet; "
                    "skipping", name,
                )
                continue
            components.append(builder(deps))

        if not components:
            raise ValueError(
                "No retrieval components enabled. Set retrieval.components.vector: true"
            )

        # Stage order is what makes any combination of flags valid: hyde must
        # rewrite before anything searches, rerank can only reorder what exists.
        # sorted() is stable, so config order still decides within a stage.
        components.sort(key=lambda c: c.stage)

        # Reranking only helps if given more than top_k to choose from.
        fetch_k = cfg.top_k * cfg.fetch_multiplier if cfg.is_enabled("rerank") else cfg.top_k
        logger.info(
            "Retrieval pipeline: %s (top_k=%d, fetch_k=%d)",
            " -> ".join(c.name for c in components), cfg.top_k, fetch_k,
        )
        return cls(components, cfg.top_k, fetch_k)

    def run(self, query: str) -> RetrievalContext:
        ctx = RetrievalContext(query=query, top_k=self.top_k, fetch_k=self.fetch_k)
        for component in self.components:
            ctx = component.run(ctx)
        ctx.candidates = ctx.candidates[: self.top_k]
        return ctx

    def describe(self) -> str:
        return " -> ".join(c.name for c in self.components)
