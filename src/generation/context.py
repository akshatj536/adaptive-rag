"""Child -> parent expansion.

This is the single place that knows where parent text lives. Right now it is
carried in child metadata; swapping that for a docstore later means changing
_parent_text() and nothing else in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import Config
from src.vectorstore.base import Hit


@dataclass
class ParentContext:
    parent_id: str
    text: str
    source: str
    page_no: int
    child_scores: list[float] = field(default_factory=list)

    @property
    def best_score(self) -> float:
        return max(self.child_scores, default=0.0)

    @property
    def label(self) -> str:
        return f"{self.source} (p.{self.page_no})" if self.page_no else self.source


def expand_to_parents(hits: list[Hit], cfg: Config) -> list[ParentContext]:
    """Dedupe hits by parent, preserving best-child rank order.

    Multiple retrieved children often share a parent; sending that parent once
    is the whole economy of the parent-child scheme.
    """
    parents: dict[str, ParentContext] = {}
    for hit in hits:
        parent_id = hit.metadata.get("parent_id") or hit.id
        if parent_id in parents:
            parents[parent_id].child_scores.append(hit.score)
            continue
        parents[parent_id] = ParentContext(
            parent_id=parent_id,
            text=_parent_text(hit),
            source=str(hit.metadata.get("source", "unknown")),
            page_no=int(hit.metadata.get("page_no", 0) or 0),
            child_scores=[hit.score],
        )

    return _apply_budget(list(parents.values()), cfg.generation.max_context_chars)


def _parent_text(hit: Hit) -> str:
    """Parent text is stored on the child's metadata. Falls back to the child's
    own text if a chunk predates the parent scheme."""
    return str(hit.metadata.get("parent_text") or hit.text)


def _apply_budget(parents: list[ParentContext], max_chars: int) -> list[ParentContext]:
    """Keep the prompt bounded. Parents arrive in relevance order, so truncation
    drops the weakest evidence first."""
    kept: list[ParentContext] = []
    used = 0
    for parent in parents:
        if used + len(parent.text) > max_chars:
            remaining = max_chars - used
            if remaining > 500:  # a sliver of context is worse than none
                parent.text = parent.text[:remaining]
                kept.append(parent)
            break
        kept.append(parent)
        used += len(parent.text)
    return kept


def format_context(parents: list[ParentContext]) -> str:
    """Numbered sources so the model can cite [1], [2], ..."""
    blocks = []
    for i, parent in enumerate(parents, start=1):
        blocks.append(f"[{i}] Source: {parent.label}\n{parent.text}")
    return "\n\n---\n\n".join(blocks)
