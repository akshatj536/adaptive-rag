"""Single place that wires the object graph together.

The CLI, Streamlit, and (later) the router all build the same stack, so
construction lives here rather than being duplicated per entrypoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import Config, get_config
from src.embeddings.local import build_embedder
from src.llm.router_llm import LLMRouter
from src.pipelines.agentic import AgenticPipeline
from src.pipelines.base import Pipeline
from src.pipelines.naive import NaivePipeline
from src.retrieval.pipeline import RetrievalDeps, RetrievalPipeline
from src.router.classifier import QueryClassifier
from src.router.router import Router
from src.vectorstore.base import VectorStore
from src.vectorstore.chroma_store import build_vector_store


# Third-party libraries that log every HTTP call at INFO and drown our output.
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers",
                  "transformers", "huggingface_hub", "filelock",
                  # litellm logs every call at INFO twice, plus a cost-map
                  # warning per deployment at startup.
                  "LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "litellm")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level, format="%(levelname)s %(name)s: %(message)s", force=False
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


@dataclass
class Stack:
    config: Config
    store: VectorStore
    embedder: object
    llm: LLMRouter
    retrieval: RetrievalPipeline
    pipelines: dict[str, Pipeline]
    router: Router

    def pipeline(self, name: str) -> Pipeline:
        try:
            return self.pipelines[name]
        except KeyError:
            known = ", ".join(sorted(self.pipelines))
            raise KeyError(f"Unknown path {name!r}. Available: {known}") from None

    def run(self, query: str):
        """The single entrypoint callers should use: the router picks the path."""
        return self.router.run(query)

    def stream(self, query: str):
        """Same, but yields ProgressEvents as the work happens."""
        return self.router.stream(query)


def build_stack(config: Config | None = None) -> Stack:
    cfg = config or get_config()
    embedder = build_embedder(cfg.embeddings)
    store = build_vector_store(cfg.vectorstore, cfg)
    llm = LLMRouter(cfg.llm)

    retrieval = RetrievalPipeline.from_config(
        RetrievalDeps(config=cfg, store=store, embedder=embedder, llm=llm)
    )

    # Both paths share one RetrievalPipeline instance - retrieval config
    # applies identically whichever path a query takes.
    pipelines: dict[str, Pipeline] = {
        "naive": NaivePipeline(cfg, retrieval, llm),
        "agentic": AgenticPipeline(cfg, retrieval, llm),
    }

    classifier = QueryClassifier(llm, cfg.router) if cfg.router.enabled else None
    router = Router(cfg, pipelines, classifier)

    return Stack(
        config=cfg,
        store=store,
        embedder=embedder,
        llm=llm,
        retrieval=retrieval,
        pipelines=pipelines,
        router=router,
    )


def resolve_path(cfg: Config) -> str:
    """Deprecated: the router now chooses the path per query. Kept so older
    scripts still work; prefer Stack.run() or Stack.router.run()."""
    return cfg.router.default_path
