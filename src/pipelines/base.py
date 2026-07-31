"""The stable contract between pipelines and everything above them.

Both the naive path and (slice 4) the agentic path return RagResult, so
Streamlit, the router, and any future eval harness need no branching on path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

from typing import TYPE_CHECKING

from src.generation.context import ParentContext

if TYPE_CHECKING:  # avoid a circular import; router imports this module
    from src.router.classifier import RouteDecision


@dataclass
class RagResult:
    answer: str
    path: str
    sources: list[ParentContext] = field(default_factory=list)
    sub_questions: list[str] | None = None  # populated by the agentic path
    trace: list[str] = field(default_factory=list)
    route: "RouteDecision | None" = None    # set by the router, if enabled
    provider: str = ""
    model: str = ""
    fell_back: bool = False
    truncated: bool = False
    grounded: bool = True                   # agentic self-check verdict
    llm_calls: int = 0                      # rate-limited calls spent on this query


@dataclass
class ProgressEvent:
    """Emitted as a query is processed, so a UI can show work as it happens
    rather than a spinner that hides everything until the end."""

    kind: str                               # route | status | sub_questions | result
    message: str = ""
    sub_questions: list[str] | None = None
    result: RagResult | None = None
    route: "RouteDecision | None" = None


class Pipeline(ABC):
    name: str

    @abstractmethod
    def stream(self, query: str) -> Iterator[ProgressEvent]:
        """Yield progress, ending with exactly one `result` event.

        Streaming is the primitive and `run` drains it, so there is one code
        path - a pipeline cannot behave differently depending on how it is called.
        """

    def run(self, query: str) -> RagResult:
        result = None
        for event in self.stream(query):
            if event.kind == "result":
                result = event.result
        if result is None:
            raise RuntimeError(f"{self.name} pipeline produced no result event")
        return result
