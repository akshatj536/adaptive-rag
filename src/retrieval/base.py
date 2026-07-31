"""The pluggable-retrieval extension point.

Every component takes a RetrievalContext and returns it. That uniformity is
what lets slice 2 add BM25 / HyDE / rerank by writing one class and flipping
one config flag, with no change to any caller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.vectorstore.base import Hit


@dataclass
class RetrievalContext:
    query: str
    # Components may rewrite what actually gets searched (HyDE does) without
    # losing the user's original wording, which generation still needs.
    effective_query: str = ""
    candidates: list[Hit] = field(default_factory=list)
    top_k: int = 5
    # How many candidates upstream components should fetch. Rerank raises this
    # so it has a wider pool to reorder.
    fetch_k: int = 5
    trace: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.effective_query:
            self.effective_query = self.query

    def log(self, message: str) -> None:
        self.trace.append(message)


# Execution stages. A component's stage is a property of what it does, not of
# where it sits in config: HyDE must rewrite the query before anything searches,
# and rerank can only reorder candidates that already exist. Ordering by stage
# means any combination of config flags produces a valid pipeline.
PRE_RETRIEVAL = 0     # rewrites the query (hyde)
RETRIEVAL = 1         # produces candidates (vector, bm25)
POST_RETRIEVAL = 2    # reorders/filters candidates (rerank)


class RetrievalComponent(ABC):
    name: str = "component"
    stage: int = RETRIEVAL

    @abstractmethod
    def run(self, ctx: RetrievalContext) -> RetrievalContext:
        ...
