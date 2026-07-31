from __future__ import annotations

from typing import Protocol, runtime_checkable

Vector = list[float]


@runtime_checkable
class Embedder(Protocol):
    """Embedding backend. Kept separate from the vector store so the two can be
    swapped independently."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Embed passages for indexing (no instruction prefix)."""
        ...

    def embed_query(self, text: str) -> Vector:
        """Embed a search query (instruction prefix applied for asymmetric models)."""
        ...
