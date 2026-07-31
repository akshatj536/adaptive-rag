"""Typed state for the agentic graph.

Every node reads and writes this one object. Keeping it explicit (rather than
passing ad-hoc dicts between nodes) is what makes the loop auditable: after a
run you can see exactly which sub-question consumed which evidence and how many
times it looped.

Loop counters live here, not in prompts. A model asked nicely to "stop after
three tries" will eventually not; a counter checked on a conditional edge
always will - which matters when every extra iteration spends rate-limited
quota.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from src.vectorstore.base import Hit


@dataclass
class SubQuestion:
    """One hop. Carries its own evidence, loop count and grading verdict."""

    text: str
    original: str = ""              # before any reformulation
    evidence: list[Hit] = field(default_factory=list)
    loops: int = 0                  # retrieve->grade->reformulate cycles spent
    sufficient: bool = False
    gap: str = ""                   # what the grader said was missing
    gave_up: bool = False           # stopped without sufficient evidence
    last_added: int = -1            # new hits from the most recent retrieval

    def __post_init__(self) -> None:
        if not self.original:
            self.original = self.text

    def add_evidence(self, hits: list[Hit]) -> int:
        """Merge new hits, ignoring ones already gathered. Returns how many
        were actually new - a reformulation that finds nothing new is a strong
        signal the corpus simply lacks the answer."""
        seen = {h.id for h in self.evidence}
        fresh = [h for h in hits if h.id not in seen]
        self.evidence.extend(fresh)
        return len(fresh)


class AgenticState(TypedDict):
    """LangGraph state. Nodes return partial dicts that merge into this."""

    query: str                      # the user's original question
    plan: str                       # orchestrator's stated strategy
    sub_questions: list[SubQuestion]
    current: int                    # index into sub_questions
    answer: str
    grounded: bool                  # self-check verdict
    self_check_count: int           # times we have looped back to the orchestrator
    replan_note: str                # feedback carried into a re-plan
    trace: list[str]
    llm_calls: int                  # every rate-limited call, for cost visibility


def initial_state(query: str) -> AgenticState:
    return AgenticState(
        query=query,
        plan="",
        sub_questions=[],
        current=0,
        answer="",
        grounded=True,
        self_check_count=0,
        replan_note="",
        trace=[],
        llm_calls=0,
    )


def active(state: AgenticState) -> SubQuestion | None:
    """The sub-question currently being worked on, if any."""
    subs = state["sub_questions"]
    index = state["current"]
    return subs[index] if 0 <= index < len(subs) else None
